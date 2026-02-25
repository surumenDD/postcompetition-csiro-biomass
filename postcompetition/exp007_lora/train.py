from utils.timing import measure_time_and_memory
from utils.logger import get_logger
from utils.env import EnvConfig
import os
import re
import sys
import gc

import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import StratifiedGroupKFold

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
wandb.setup(wandb.Settings(reinit="finish_previous"))


# ===== targets =====
ALL_TARGET_COLS = [
    "Dry_Green_g",
    "Dry_Clover_g",
    "Dry_Dead_g",
    "GDM_g",
    "Dry_Total_g",
]
ALL_TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]  # 評価仕様の重み


META_COLS = [
    "sample_id_prefix",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


# ===== config =====
@dataclass
class ExpConfig:
    seed: int = 42
    n_folds: int = 4

    # Model (large固定)
    model_name: str = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size: int = 448

    # Training
    batch_size: int = 4
    accumulate_grad_batches: int = 4
    num_epochs: int = 20
    num_workers: int = 4

    # LR (headとLoRAを分離)
    head_lr: float = 1e-3
    lora_lr: float = 5e-4
    weight_decay: float = 0.05

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_train_last_n_blocks: int = 8  # 0以下なら全ブロック
    lora_target_modules: Tuple[str, ...] = ("attn.qkv", "attn.proj")

    # Loss
    huber_beta: float = 1.0  # SmoothL1のbeta


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


# ===== utils =====
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_wandb_logger(*, exp_group: str, fold_i: int, save_dir: Path) -> WandbLogger:
    run = wandb.init(
        project="csiro-biomass",
        name=f"{exp_group}/fold{fold_i}",  # run名
        group=exp_group,  # group化してfoldごとにまとめる
        job_type=f"cv_fold_{fold_i}",
        tags=[f"fold{fold_i}"],  # 任意
        dir=str(save_dir),
        reinit="finish_previous",  # 既存runがあれば確実に閉じて新規runを作る
    )
    return WandbLogger(experiment=run)


def weighted_r2_score_on_log(
    true_log: np.ndarray,  # (N,5) log1p
    pred_log: np.ndarray,  # (N,5) log1p
) -> Tuple[float, np.ndarray]:
    weight = np.array(ALL_TARGET_WEIGHTS, dtype=np.float64)
    per_target_r2: List[float] = []

    for target_i in range(true_log.shape[1]):
        y = true_log[:, target_i]
        yhat = pred_log[:, target_i]
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        per_target_r2.append(r2)

    per_target_r2_arr = np.array(per_target_r2, dtype=np.float64)
    weighted = float(np.sum(weight * per_target_r2_arr) / np.sum(weight))
    return weighted, per_target_r2_arr


def make_clover_dead_stratified_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["clover_dead_presence"] = (
        (df["Dry_Clover_g"] > 0).astype(int).astype(str)
        + "_"
        + (df["Dry_Dead_g"] > 0).astype(int).astype(str)
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_i, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["clover_dead_presence"],
                   groups=df["sample_id_prefix"])
    ):
        df.loc[val_idx, "fold"] = fold_i
    return df


def make_train_wide(train_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    df[["sample_id_prefix", "sample_id_suffix"]
       ] = df["sample_id"].str.split("__", expand=True)
    assert (df["target_name"] == df["sample_id_suffix"]
            ).all(), "target_nameとsample_id_suffixが一致しない行がある"

    train_wide_df = (
        df.pivot(
            index=META_COLS,
            columns="target_name",
            values="target",
        )
        .reset_index()
    )
    train_wide_df.columns.name = None
    return train_wide_df


# ===== LoRA =====
class LoRALinear(nn.Module):
    """
    timm EVA(DINOv3) は Linear を forward で呼ばずに、
    self.qkv.weight のように weight を直接参照して F.linear を呼ぶ実装がある。

    そのため LoRA で置換したモジュールも .weight/.bias を持つ必要がある。
    (https://chatgpt.com/s/t_699cac133e98819181ec1b8c6d347b8f)
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        # timm が self.qkv.weight を直接参照するので属性として公開する
        self.weight = base_linear.weight
        self.bias = base_linear.bias

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(self.alpha) / float(self.rank)

        # base は凍結（LoRA だけ学習したい）
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.dropout = nn.Dropout(
            p=float(dropout)) if dropout > 0 else nn.Identity()

        # LoRA 行列：A (rank, in), B (out, rank)
        # nn.Linear を使うよりも weight を直接持つ方が形が明確
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))

        # 初期化：B を 0 にして最初は元モデルと等価にする
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base: F.linear を使う（self.weight/self.bias を使う）
        base_out = nn.functional.linear(x, self.weight, self.bias)

        # lora: (x @ A^T) @ B^T
        x_d = self.dropout(x)
        lora_out = nn.functional.linear(
            nn.functional.linear(x_d, self.lora_A, bias=None),
            self.lora_B,
            bias=None,
        )
        return base_out + self.scale * lora_out


def apply_lora_to_dinov3_vit(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Tuple[str, ...],
    train_last_n_blocks: int,
) -> List[nn.Parameter]:
    if not hasattr(backbone, "blocks"):
        raise AttributeError("backbone has no attribute 'blocks'")

    blocks = backbone.blocks  # type: ignore[attr-defined]
    num_blocks = len(blocks)

    if train_last_n_blocks <= 0:
        target_block_indices = set(range(num_blocks))
    else:
        start_i = max(0, num_blocks - train_last_n_blocks)
        target_block_indices = set(range(start_i, num_blocks))

    trainable_params: List[nn.Parameter] = []

    for block_i, block in enumerate(blocks):
        if block_i not in target_block_indices:
            continue

        attn = getattr(block, "attn", None)
        if attn is None:
            continue

        # 現実的に効きやすい最小セット：qkv/proj
        if "attn.qkv" in target_modules:
            base_qkv = getattr(attn, "qkv", None)
            if isinstance(base_qkv, nn.Linear):
                lora_qkv = LoRALinear(
                    base_qkv, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attn, "qkv", lora_qkv)
                trainable_params.append(lora_qkv.lora_A)
                trainable_params.append(lora_qkv.lora_B)

        if "attn.proj" in target_modules:
            base_proj = getattr(attn, "proj", None)
            if isinstance(base_proj, nn.Linear):
                lora_proj = LoRALinear(
                    base_proj, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attn, "proj", lora_proj)
                trainable_params.append(lora_proj.lora_A)
                trainable_params.append(lora_proj.lora_B)

    return trainable_params


# ===== density head =====
class PatchDensityHead(nn.Module):
    """
    patch tokenごとに密度を出すヘッド。
    dens >= 0 を保証するため softplus を使う。
    """

    def __init__(self, embed_dim: int, num_targets: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_targets),
        )

    def forward(self, patch_tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        # patch_tokens: (B, N, D), N=H*W
        batch_size, num_patches, embed_dim = patch_tokens.shape
        grid_h, grid_w = grid_hw
        if num_patches != grid_h * grid_w:
            raise ValueError(
                f"num_patches={num_patches} != H*W={grid_h}*{grid_w}")

        x = patch_tokens.view(batch_size, grid_h, grid_w,
                              embed_dim)      # (B,H,W,D)
        dens = self.proj(x).permute(
            0, 3, 1, 2).contiguous()             # (B,T,H,W)
        dens = nn.functional.softplus(dens)
        return dens


class DualStreamDINOv3DensityLoRARegressor(nn.Module):
    """
    DINOv3 patch tokens -> 密度マップ -> 空間和 -> log1p
    """
    NUM_PREFIX_TOKENS = 5  # 1 CLS + 4 register を仮定

    def __init__(self, cfg: Config, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.exp.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # backboneは基本凍結
        for p in self.backbone.parameters():
            p.requires_grad = False

        # LoRA注入
        self.lora_params: List[nn.Parameter] = []
        if cfg.exp.use_lora:
            self.lora_params = apply_lora_to_dinov3_vit(
                self.backbone,
                rank=cfg.exp.lora_rank,
                alpha=cfg.exp.lora_alpha,
                dropout=cfg.exp.lora_dropout,
                target_modules=tuple(cfg.exp.lora_target_modules),
                train_last_n_blocks=cfg.exp.lora_train_last_n_blocks,
            )

        embed_dim = self.backbone.num_features
        self.density_head = PatchDensityHead(
            embed_dim=embed_dim, num_targets=len(ALL_TARGET_COLS))

    def _get_patch_grid(self, x: torch.Tensor) -> Tuple[int, int]:
        # type: ignore[attr-defined]
        patch_size = self.backbone.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        grid_h = x.shape[-2] // patch_size
        grid_w = x.shape[-1] // patch_size
        return grid_h, grid_w

    def _encode_to_density(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)                  # (B, 5+N, D)
        patch_tokens = feat[:, self.NUM_PREFIX_TOKENS:, :]        # (B, N, D)
        grid_hw = self._get_patch_grid(x)
        dens = self.density_head(patch_tokens, grid_hw)           # (B,5,H,W)
        return dens

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        dens_left = self._encode_to_density(left)
        dens_right = self._encode_to_density(right)
        dens = dens_left + dens_right                             # (B,5,H,W)

        pred_raw = dens.sum(dim=(2, 3))                           # (B,5)
        pred_log = torch.log1p(pred_raw)                          # (B,5)
        return pred_log


# ===== callbacks =====
class SaveLastCheckpointOnTrainEnd(pl.Callback):
    """トレーニング終了時に1回だけlast checkpointを保存する。
    ModelCheckpoint(monitor=None)は毎エポックI/Oが発生するため、
    ViT-Largeのような大きいモデルではコストになる。"""

    def __init__(self, dirpath: Path, fold_i: int):
        self.dirpath = Path(dirpath)
        self.fold_i = fold_i

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"fold{self.fold_i}_last_epoch{epoch:02d}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))


# ===== lightning module =====
class BiomassModule(pl.LightningModule):
    def __init__(self, model: DualStreamDINOv3DensityLoRARegressor, cfg: Config):
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.register_buffer(
            "target_weight_tensor",
            torch.tensor(ALL_TARGET_WEIGHTS, dtype=torch.float32),
        )

        self.val_pred_log_list: List[np.ndarray] = []
        self.val_true_log_list: List[np.ndarray] = []

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model(left, right)

    def _weighted_smooth_l1_loss(self, pred_log: torch.Tensor, true_log: torch.Tensor) -> torch.Tensor:
        # pred_log/true_log: (B,5)
        per_target_loss = nn.functional.smooth_l1_loss(
            pred_log,
            true_log,
            reduction="none",
            beta=self.cfg.exp.huber_beta,
        )  # (B,5)
        loss_per_target = per_target_loss.mean(dim=0)  # (5,)
        loss = (loss_per_target * self.target_weight_tensor).sum()
        return loss

    def training_step(self, batch, batch_idx):
        left, right, true_log = batch
        pred_log = self(left, right)
        loss = self._weighted_smooth_l1_loss(pred_log, true_log)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        left, right, true_log = batch
        pred_log = self(left, right)
        loss = self._weighted_smooth_l1_loss(pred_log, true_log)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_pred_log_list.append(pred_log.detach().cpu().numpy())
        self.val_true_log_list.append(true_log.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        pred_log = np.concatenate(self.val_pred_log_list, axis=0)
        true_log = np.concatenate(self.val_true_log_list, axis=0)
        score, _ = weighted_r2_score_on_log(true_log, pred_log)
        self.log("val_weighted_r2", score, prog_bar=True)

        self.val_pred_log_list.clear()
        self.val_true_log_list.clear()

    def predict_step(self, batch, batch_idx):
        left, right, true_log = batch
        pred_log = self(left, right)
        return pred_log.detach().cpu().numpy(), true_log.detach().cpu().numpy()

    def configure_optimizers(self):
        head_params = list(self.model.density_head.parameters())
        lora_params = list(getattr(self.model, "lora_params", []))

        param_groups = []
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": self.cfg.exp.head_lr})
        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": self.cfg.exp.lora_lr})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.exp.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.exp.num_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ===== transforms =====
def get_transforms(img_size: int, is_train: bool) -> A.Compose:
    """
    left/rightへ同一の乱数変換を適用するため additional_targets を使う。
    """
    if is_train:
        transform_list = [
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.60, 1.00),
                ratio=(0.90, 1.10),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        ]
    else:
        transform_list = [
            A.Resize(img_size, img_size),
        ]

    transform_list += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    return A.Compose(transform_list, additional_targets={"right": "image"})


# ===== dataset =====
class BiomassDataset(Dataset):
    """
    1000×2000画像を左右に分割し、両方をモデルに入れる。
    targetsはlog1p空間で保持する（評価もlog空間で行う）。
    """

    def __init__(self, df: pd.DataFrame, img_root: Path, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform
        self.targets_log = np.log1p(
            df[ALL_TARGET_COLS].values.astype(np.float32))  # (N,5)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.img_root / self.df.loc[idx, "image_path"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"cv2.imread failed: {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        left_img = img[:, :1000, :]
        right_img = img[:, 1000:, :]

        out = self.transform(image=left_img, right=right_img)
        left_t = out["image"]
        right_t = out["right"]

        target_log = torch.tensor(self.targets_log[idx], dtype=torch.float32)
        return left_t, right_t, target_log


# ===== main =====
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    # TODO: loraの厳密化（loraの仕組みと密度マップの部分よく見る）
    # TODO: post precessing
    # TODO: 解像度あげる
    # TODO: 検討：幅で予測、密度マップ

    project_root = Path(HydraConfig.get().runtime.cwd)
    input_dir_path = (project_root / cfg.env.input_dir).resolve()
    train_csv_path = input_dir_path / "train.csv"

    output_root = (project_root / cfg.env.output_dir).resolve()
    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"
    output_dir_path = output_root / exp_name
    os.makedirs(output_dir_path, exist_ok=True)

    global LOGGER
    LOGGER = get_logger(__name__, output_dir_path)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Input dir: %s", input_dir_path)
    LOGGER.info("Output dir: %s", output_dir_path)
    LOGGER.info("Config: %s", cfg)

    set_seed(cfg.exp.seed)
    torch.set_float32_matmul_precision("high")

    with measure_time_and_memory("train.csvをlongからwide形式にして読み込む処理"):
        train_wide_df = make_train_wide(train_csv_path)

    train_wide_df = make_clover_dead_stratified_folds(
        train_wide_df, cfg.exp.n_folds, cfg.exp.seed)
    LOGGER.info("[CV] clover_dead_stratified_folds: %d folds", cfg.exp.n_folds)
    LOGGER.info("\n%s", pd.crosstab(
        train_wide_df["fold"], train_wide_df["clover_dead_presence"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU (CUDA) is required but not available.")
    LOGGER.info("Using device: %s", device)

    all_oof_pred_log_list: List[np.ndarray] = []
    all_oof_true_log_list: List[np.ndarray] = []
    all_oof_indices_list: List[np.ndarray] = []

    # 各foldのスコアとbest epochを記録するリスト
    fold_scores: List[float] = []
    fold_best_epochs: List[int] = []

    for fold_i in range(cfg.exp.n_folds):
        LOGGER.info("======== Fold %d / %d ========", fold_i, cfg.exp.n_folds)

        train_df = train_wide_df[train_wide_df["fold"] != fold_i]
        val_df = train_wide_df[train_wide_df["fold"] == fold_i]

        train_loader = DataLoader(
            BiomassDataset(train_df, input_dir_path,
                           get_transforms(cfg.exp.img_size, True)),
            batch_size=cfg.exp.batch_size,
            shuffle=True,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            BiomassDataset(val_df, input_dir_path,
                           get_transforms(cfg.exp.img_size, False)),
            batch_size=cfg.exp.batch_size,
            shuffle=False,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )

        model = DualStreamDINOv3DensityLoRARegressor(cfg, pretrained=True)
        module = BiomassModule(model, cfg)

        # trainable paramsをログ
        trainable_count = sum(p.numel()
                              for p in module.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in module.parameters())
        LOGGER.info("Trainable params: %d / Total params: %d",
                    trainable_count, total_count)

        fold_output_dir = output_dir_path / f"fold{fold_i}"

        # best checkpoint：val_weighted_r2が最高のエポックを保存
        best_ckpt = ModelCheckpoint(
            dirpath=fold_output_dir,
            monitor="val_weighted_r2",
            mode="max",
            save_top_k=1,
            filename=f"fold{fold_i}_best_epoch{{epoch:02d}}",
        )
        # last checkpoint：トレーニング終了時に1回だけ保存（毎エポックI/Oを避ける）
        callbacks = [
            best_ckpt,
            SaveLastCheckpointOnTrainEnd(fold_output_dir, fold_i),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        wandb_logger = build_wandb_logger(
            exp_group=exp_name, fold_i=fold_i, save_dir=output_dir_path
        )

        trainer = pl.Trainer(
            max_epochs=cfg.exp.num_epochs,
            accumulate_grad_batches=cfg.exp.accumulate_grad_batches,
            callbacks=callbacks,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(module, train_loader, val_loader)

        # best checkpointのepoch番号とスコアをログ
        best_model_path = best_ckpt.best_model_path
        best_val_score = float(best_ckpt.best_model_score)
        m = re.search(r"epoch(\d+)", Path(best_model_path).stem)
        best_epoch_num = int(m.group(1)) if m else -1
        LOGGER.info(
            "Fold %d best checkpoint: %s (best_epoch=%d, val_weighted_r2=%.4f)",
            fold_i, best_model_path, best_epoch_num, best_val_score,
        )

        wandb.finish()

        # best checkpointでOOF
        predictions = trainer.predict(module, val_loader, ckpt_path="best")
        pred_log_fold = np.concatenate(
            [p[0] for p in predictions], axis=0)  # (N,5) log1p
        true_log_fold = np.concatenate(
            [p[1] for p in predictions], axis=0)  # (N,5) log1p

        all_oof_pred_log_list.append(pred_log_fold)
        all_oof_true_log_list.append(true_log_fold)
        all_oof_indices_list.append(val_df["sample_id_prefix"].values)

        fold_score, per_target = weighted_r2_score_on_log(
            true_log_fold, pred_log_fold)
        LOGGER.info("Fold %d OOF weighted_r2 (log-space): %.4f",
                    fold_i, fold_score)
        for col, r2 in zip(ALL_TARGET_COLS, per_target):
            LOGGER.info("  %s: %.4f", col, r2)

        # fold毎のスコアとbest epochを記録
        fold_scores.append(fold_score)
        fold_best_epochs.append(best_epoch_num)

        # fold 間でメモリ掃除（fold0で止まる/OOMを避ける）
        del predictions, trainer, wandb_logger, module, model
        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    pred_log_all = np.concatenate(all_oof_pred_log_list, axis=0)
    true_log_all = np.concatenate(all_oof_true_log_list, axis=0)
    indices_all = np.concatenate(all_oof_indices_list, axis=0)

    # ======== CV Summary ========
    LOGGER.info("======== CV Summary ========")
    for fold_index, (score, best_epoch) in enumerate(zip(fold_scores, fold_best_epochs)):
        LOGGER.info(
            "  Fold %d: val_weighted_r2=%.4f, best_epoch=%d", fold_index, score, best_epoch)
    scores_arr = np.array(fold_scores, dtype=np.float64)
    LOGGER.info("  Mean: %.4f, Std: %.4f", scores_arr.mean(), scores_arr.std())

    overall_score, per_target = weighted_r2_score_on_log(
        true_log_all, pred_log_all)
    LOGGER.info("Overall OOF weighted_r2 (log-space): %.4f", overall_score)
    for col, r2 in zip(ALL_TARGET_COLS, per_target):
        LOGGER.info("  %s: %.4f", col, r2)

    # 保存（logとrawの両方）
    pred_raw_all = np.expm1(pred_log_all).clip(min=0.0)
    true_raw_all = np.expm1(true_log_all).clip(min=0.0)

    oof_df = pd.DataFrame({"sample_id_prefix": indices_all})
    for i, col in enumerate(ALL_TARGET_COLS):
        oof_df[f"pred_log_{col}"] = pred_log_all[:, i]
        oof_df[f"true_log_{col}"] = true_log_all[:, i]
        oof_df[f"pred_raw_{col}"] = pred_raw_all[:, i]
        oof_df[f"true_raw_{col}"] = true_raw_all[:, i]

    oof_csv_path = output_dir_path / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path)
    LOGGER.info("OOF predictions saved to %s", oof_csv_path)


if __name__ == "__main__":
    main()
