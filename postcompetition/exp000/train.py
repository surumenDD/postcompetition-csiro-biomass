import os
import sys
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import measure_time_and_memory, timer

TARGET_COLS = [
    "Dry_Green_g",
    "Dry_Clover_g",
    "Dry_Dead_g",
    "GDM_g",
    "Dry_Total_g",
]
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]


META_COLS = [
    "sample_id_prefix",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


@dataclass
class ExpConfig:
    seed: int = 42
    n_folds: int = 4
    # Model
    model_name: str = "vit_small_patch16_dinov3_qkvb.lvd1689m"
    img_size: int = 448          # 448/16=28 → 28×28=784 patches
    # Training
    batch_size: int = 16
    num_epochs: int = 20
    num_workers: int = 4
    lr: float = 1e-3
    backbone_lr_ratio: float = 0.1
    # Progressive unfreeze
    unfreeze_epoch: int = 5
    unfreeze_ratio: float = 0.5
    # Fusion head
    fusion_hidden_dim: int = 512


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


# hydra用にdefaultを設定
# YAMLで両者を合成する
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_r2_score(
    true_target_matrix: np.ndarray,
    predicted_target_matrix: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    shape: (num_samples, 5)
    targets = [Green, Clover, Dead, GDM, Total]

    Returns
    -------
    weighted_r2_score_value : float
        各ターゲットR²を重み付き平均したスコア
    per_target_r2_scores : np.ndarray
        ターゲットごとのR²
    """
    # 各ターゲットの重要度（コンペ評価仕様に合わせた重み）
    target_importance_weights = np.array(
        [0.1, 0.1, 0.1, 0.2, 0.5],
        dtype=np.float64,
    )
    # 各ターゲットのR²を格納
    per_target_r2_scores = []
    # 列方向（=ターゲット単位）にR²を計算
    num_targets = true_target_matrix.shape[1]

    for target_index in range(num_targets):
        # 全サンプルにおける1ターゲット分の値を取り出す
        target_true_values = true_target_matrix[:, target_index]
        target_predicted_values = predicted_target_matrix[:, target_index]
        # 残差平方和 Σ(y - ŷ)^2
        residual_sum_of_squares = np.sum(
            (target_true_values - target_predicted_values) ** 2
        )
        # 全変動 Σ(y - mean(y))^2
        total_sum_of_squares = np.sum(
            (target_true_values - np.mean(target_true_values)) ** 2
        )
        # R²計算
        target_r2_score = (
            1.0 - residual_sum_of_squares / total_sum_of_squares
            if total_sum_of_squares > 0
            else 0.0  # 分散0（定数ターゲット）の場合の安全処理
        )
        per_target_r2_scores.append(target_r2_score)

    per_target_r2_scores = np.array(per_target_r2_scores)
    # 重み付き平均R²
    weighted_r2_score_value = np.sum(
        per_target_r2_scores * target_importance_weights
    ) / np.sum(target_importance_weights)
    return weighted_r2_score_value, per_target_r2_scores


def make_clover_dead_stratified_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Clover/Deadの存在有無で層化した StratifiedGroupKFold を作成する。"""
    df["clover_dead_presence"] = (
        (df["Dry_Clover_g"] > 0).astype(int).astype(str)
        + "_"
        + (df["Dry_Dead_g"] > 0).astype(int).astype(str)
    )
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf.split(
            df,
            y=df["clover_dead_presence"],
            groups=df["sample_id_prefix"],
        )
    ):
        df.loc[val_idx, "fold"] = fold
    return df


def make_train_wide(train_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    # sample_id: "ID1011485656__Dry_Clover_g"をIDとtarget_nameに分割
    # sample_id_prefix: "ID1011485656"
    # sample_id_suffix: "Dry_Clover_g" target_nameと同じ内容
    df[["sample_id_prefix", "sample_id_suffix"]
       ] = df["sample_id"].str.split("__", expand=True)

    # 　target_nameとsample_id_suffixが同じであることを確認
    assert (df["target_name"] == df["sample_id_suffix"]
            ).all(), "target_nameとsample_id_suffixが一致しない行があります"

    train_csv_wide_df = (
        # target_name（例: Dry_Green_g など）を列方向へ展開し、long形式 → wide形式へ変換
        df.pivot(
            # 同一サンプル（画像1枚）を識別するメタ情報列を行インデックスにする
            index=META_COLS,
            # target_name の値を新しい列名として横展開する
            columns="target_name",
            # 各 target_name に対応する target 値をセルに配置する
            values="target",
        )
        # pivot後にインデックス化された META_COLS を通常の列へ戻す
        .reset_index()
    )
    # pivotで自動付与された列インデックス名を削除
    train_csv_wide_df.columns.name = None

    return train_csv_wide_df


class SpatialAttentionPool(nn.Module):
    """patch tokens 上の soft attention weight を学習し空間的特徴を抽出する。
    attn_weights を reshape すると空間的重要度マップ（密度マップ相当）として可視化可能。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens: torch.Tensor):
        # patch_tokens: (B, N_patches, D)
        attn_weights = self.attn(patch_tokens).softmax(dim=1)  # (B, N, 1)
        pooled = (patch_tokens * attn_weights).sum(dim=1)       # (B, D)
        return pooled, attn_weights


class DualStreamDINOv3Regressor(nn.Module):
    NUM_PREFIX_TOKENS = 5  # 1 CLS + 4 Register (DINOv3 固定)

    def __init__(
        self,
        model_name: str,
        num_targets: int = 5,
        fusion_hidden_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=''
        )
        # バックボーン全体を freeze
        for p in self.backbone.parameters():
            p.requires_grad = False

        D = self.backbone.num_features  # 384 for ViT-Small
        self.spatial_pool_left = SpatialAttentionPool(D)
        self.spatial_pool_right = SpatialAttentionPool(D)
        # 2ストリーム × (CLS + spatial_pool) = 4D → fusion_hidden_dim
        self.fusion_trunk = nn.Sequential(
            nn.Linear(D * 4, fusion_hidden_dim),
            nn.GELU(),
        )
        # ターゲットごとに独立したヘッド
        self.target_heads = nn.ModuleList([
            nn.Linear(fusion_hidden_dim, 1) for _ in range(num_targets)
        ])

    def _encode(self, x: torch.Tensor, spatial_pool: SpatialAttentionPool):
        feat = self.backbone.forward_features(x)          # (B, 5+N, D)
        cls = feat[:, 0, :]                           # (B, D)
        patches = feat[:, self.NUM_PREFIX_TOKENS:, :]     # (B, N, D)
        spatial, attn = spatial_pool(patches)             # (B, D), (B, N, 1)
        return cls, spatial, attn

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cls_L, sp_L, _ = self._encode(left,  self.spatial_pool_left)
        cls_R, sp_R, _ = self._encode(right, self.spatial_pool_right)
        fused = torch.cat([cls_L, sp_L, cls_R, sp_R], dim=-1)  # (B, D*4)
        # (B, hidden_dim)
        trunk = self.fusion_trunk(fused)
        # 各ヘッドの出力を concat → (B, num_targets)
        return torch.cat([h(trunk) for h in self.target_heads], dim=-1)


class GradualUnfreezeCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int, unfreeze_ratio: float, backbone_lr: float):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_ratio = unfreeze_ratio
        self.backbone_lr = backbone_lr

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        blocks = pl_module.model.backbone.blocks   # nn.ModuleList (depth=12)
        n_unfreeze = int(len(blocks) * self.unfreeze_ratio)
        # 出力側（末尾）n_unfreeze ブロック + 最終 norm を unfreeze
        target_modules = list(blocks[-n_unfreeze:]) + \
            [pl_module.model.backbone.norm]
        new_params = [p for m in target_modules for p in m.parameters()]
        for p in new_params:
            p.requires_grad = True
        trainer.optimizers[0].add_param_group(
            {"params": new_params, "lr": self.backbone_lr}
        )
        print(
            f"[GradualUnfreeze] Epoch {trainer.current_epoch}: "
            f"unfreezing last {n_unfreeze}/{len(blocks)} blocks "
            f"+ norm with lr={self.backbone_lr:.2e}"
        )
        trainer.strategy.barrier()


class BiomassModule(pl.LightningModule):
    def __init__(self, model: DualStreamDINOv3Regressor, cfg: "Config"):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.register_buffer(
            "target_weights",
            torch.tensor(TARGET_WEIGHTS, dtype=torch.float32)
        )
        self._val_preds: List[np.ndarray] = []
        self._val_targets: List[np.ndarray] = []

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model(left, right)

    def _weighted_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ターゲットごとの SmoothL1Loss を評価重みで加重平均する。"""
        per_target_loss = torch.stack([
            nn.functional.smooth_l1_loss(pred[:, i], y[:, i])
            for i in range(pred.shape[1])
        ])  # (5,)
        return (per_target_loss * self.target_weights).sum()

    def training_step(self, batch, _):
        left, right, y = batch
        pred = self(left, right)
        loss = self._weighted_loss(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        left, right, y = batch
        pred = self(left, right)
        loss = self._weighted_loss(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # OOF 収集（log1p 逆変換して評価）
        self._val_preds.append(torch.expm1(pred).cpu().numpy())
        self._val_targets.append(torch.expm1(y).cpu().numpy())

    def on_validation_epoch_end(self):
        preds = np.concatenate(self._val_preds,   axis=0)
        targets = np.concatenate(self._val_targets, axis=0)
        score, _ = weighted_r2_score(targets, preds)
        self.log("val_weighted_r2", score, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def predict_step(self, batch, batch_idx):
        left, right, y = batch
        pred = self(left, right)
        return torch.expm1(pred).cpu().numpy(), torch.expm1(y).cpu().numpy()

    def configure_optimizers(self):
        # 初期は backbone 以外（spatial_pool, trunk, 各ヘッド）のみ学習
        trainable_params = (
            list(self.model.spatial_pool_left.parameters())
            + list(self.model.spatial_pool_right.parameters())
            + list(self.model.fusion_trunk.parameters())
            + list(self.model.target_heads.parameters())
        )
        return torch.optim.AdamW(trainable_params, lr=self.cfg.exp.lr)


def get_transforms(img_size: int, is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class BiomassDataset(Dataset):
    """1000×2000 画像を left/right に分割して返す"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_root: Path,
        transform: A.Compose,
        target_cols: List[str],
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform
        self.targets = np.log1p(df[target_cols].values.astype(np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_root / self.df.loc[idx, "image_path"]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        left = img[:, :1000, :]   # (1000, 1000, 3)
        right = img[:, 1000:, :]   # (1000, 1000, 3)
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return left_t, right_t, targets


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    # 各pathの設定
    project_root = Path(HydraConfig.get().runtime.cwd)
    input_dir_path = (project_root / cfg.env.input_dir).resolve()
    train_csv_path = input_dir_path / "train.csv"
    output_root = (project_root / cfg.env.output_dir).resolve()
    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"
    output_dir_path = output_root / exp_name
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"project root: {project_root}")
    print(f"input dir: {input_dir_path}")
    print(f"output dir: {output_dir_path}")

    # Loggerの設定
    global LOGGER
    LOGGER = get_logger(__name__, output_dir_path)
    LOGGER.info(f"Project root: {project_root}")
    LOGGER.info(f"Input dir: {input_dir_path}")
    LOGGER.info(f"Output dir: {output_dir_path}")
    LOGGER.info("Config: %s", cfg)

    # seedの設定
    set_seed(cfg.exp.seed)

    # 精度は少し下がるが、計算速度が上昇する設定
    torch.set_float32_matmul_precision('high')

    # データを読みこむ
    with measure_time_and_memory("train.csvをlongからwide形式にして読み込む処理"):
        train_csv_wide_df = make_train_wide(train_csv_path)

    # CloverとDeadの分布を均すCVを採用
    n_folds = cfg.exp.n_folds
    train_csv_wide_df = make_clover_dead_stratified_folds(
        train_csv_wide_df, n_folds, cfg.exp.seed)
    LOGGER.info(f"[CV-strategy]:clover_dead_stratified_foldsを使用して{n_folds}foldを作成しました。")
    LOGGER.info("\n%s", pd.crosstab(
        train_csv_wide_df["fold"], train_csv_wide_df["clover_dead_presence"]))

    # GPUが使えることの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    if device.type != "cuda":
        LOGGER.error("CUDA is not available. Training requires GPU.")
        raise RuntimeError("GPU (CUDA) is required but not available.")

    # fold ループ
    all_oof_preds: List[np.ndarray] = []
    all_oof_targets: List[np.ndarray] = []
    all_oof_indices: List[np.ndarray] = []

    for fold in range(n_folds):
        LOGGER.info(f"======== Fold {fold} / {n_folds} ========")
        train_df = train_csv_wide_df[train_csv_wide_df["fold"] != fold]
        val_df = train_csv_wide_df[train_csv_wide_df["fold"] == fold]

        train_loader = DataLoader(
            BiomassDataset(
                train_df, input_dir_path,
                get_transforms(cfg.exp.img_size, True),
                TARGET_COLS,
            ),
            batch_size=cfg.exp.batch_size,
            shuffle=True,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            BiomassDataset(
                val_df, input_dir_path,
                get_transforms(cfg.exp.img_size, False),
                TARGET_COLS,
            ),
            batch_size=cfg.exp.batch_size,
            shuffle=False,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )

        model = DualStreamDINOv3Regressor(
            cfg.exp.model_name,
            fusion_hidden_dim=cfg.exp.fusion_hidden_dim,
        )
        module = BiomassModule(model, cfg)
        backbone_lr = cfg.exp.lr * cfg.exp.backbone_lr_ratio

        fold_output_dir = output_dir_path / f"fold{fold}"
        callbacks = [
            GradualUnfreezeCallback(
                cfg.exp.unfreeze_epoch,
                cfg.exp.unfreeze_ratio,
                backbone_lr,
            ),
            ModelCheckpoint(
                dirpath=fold_output_dir,
                monitor="val_weighted_r2",
                mode="max",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        wandb_logger = WandbLogger(
            project="csiro-biomass",
            name=f"{exp_name}/fold{fold}",
        )
        trainer = pl.Trainer(
            max_epochs=cfg.exp.num_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(module, train_loader, val_loader)

        # ベストチェックポイントで OOF 予測
        predictions = trainer.predict(module, val_loader, ckpt_path="best")
        oof_preds_fold = np.concatenate([p[0] for p in predictions], axis=0)
        oof_targets_fold = np.concatenate([p[1] for p in predictions], axis=0)
        all_oof_preds.append(oof_preds_fold)
        all_oof_targets.append(oof_targets_fold)
        all_oof_indices.append(val_df.index.values)

        fold_score, per_target = weighted_r2_score(
            oof_targets_fold, oof_preds_fold)
        LOGGER.info(f"Fold {fold} OOF weighted_r2: {fold_score:.4f}")
        for col, r2 in zip(TARGET_COLS, per_target):
            LOGGER.info(f"  {col}: {r2:.4f}")

    # 全 fold の OOF を結合して CSV に保存
    all_preds = np.concatenate(all_oof_preds,   axis=0)
    all_targets = np.concatenate(all_oof_targets, axis=0)
    all_indices = np.concatenate(all_oof_indices, axis=0)

    overall_score, per_target = weighted_r2_score(all_targets, all_preds)
    LOGGER.info(f"Overall OOF weighted_r2: {overall_score:.4f}")
    for col, r2 in zip(TARGET_COLS, per_target):
        LOGGER.info(f"  {col}: {r2:.4f}")

    oof_df = pd.DataFrame(
        all_preds,
        columns=[f"pred_{c}" for c in TARGET_COLS],
        index=all_indices,
    )
    for i, col in enumerate(TARGET_COLS):
        oof_df[f"true_{col}"] = all_targets[:, i]
    oof_csv_path = output_dir_path / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path)
    LOGGER.info(f"OOF predictions saved to {oof_csv_path}")


if __name__ == "__main__":
    main()
