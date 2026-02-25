# infer_code1_kaggle.py
# Code1 (DualStreamDINOv3DensityLoRARegressor) 用 Kaggle 推論コード
# - test.csv から submission.csv を作成（sample_submission.csv 不要）
# - Lightning .ckpt を直接ロード
# - foldアンサンブル対応
# - DINOv3 + LoRA + Density head + log1p出力 に対応

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =====================
# 競技ターゲット
# =====================
ALL_TARGET_COLS = [
    "Dry_Green_g",
    "Dry_Clover_g",
    "Dry_Dead_g",
    "GDM_g",
    "Dry_Total_g",
]
TARGET5_ORDER = ALL_TARGET_COLS


# =====================
# 推論設定（学習時と一致させる）
# =====================
@dataclass
class InferenceConfig:
    model_name: str = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size: int = 448

    # DataLoader
    batch_size: int = 4
    num_workers: int = 2

    # LoRA (学習時設定と一致必須)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_train_last_n_blocks: int = 8
    lora_target_modules: Tuple[str, ...] = ("attn.qkv", "attn.proj")

    # モデル実装依存
    num_prefix_tokens: int = 5  # code1と同じ仮定（1 CLS + 4 register）


# =====================
# test.csv 整形
# =====================
def make_test_tables(test_csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    test.csv を読み、
    - test_img: 画像1枚=1行（推論用）
    - test_long: 提出行ベース（sample_id単位）
    を返す
    """
    test_long = pd.read_csv(test_csv_path)

    if "image_path" not in test_long.columns:
        raise ValueError("test.csv に image_path 列が見つかりません。")

    # sample_id_suffix を作る
    if "sample_id_suffix" in test_long.columns:
        pass
    elif "target_name" in test_long.columns:
        test_long["sample_id_suffix"] = test_long["target_name"].astype(str)
    elif "sample_id" in test_long.columns:
        split_df = test_long["sample_id"].astype(str).str.split("__", n=1, expand=True)
        if split_df.shape[1] != 2:
            raise ValueError("sample_id を '__' で分割できません。prefix__suffix形式を想定しています。")
        test_long["sample_id_prefix"] = split_df[0]
        test_long["sample_id_suffix"] = split_df[1]
    else:
        raise ValueError("test.csv に sample_id / target_name / sample_id_suffix が見つかりません。")

    # sample_id を作る
    if "sample_id" not in test_long.columns:
        if "sample_id_prefix" not in test_long.columns:
            raise ValueError("sample_id を作れません。sample_id_prefix がありません。")
        test_long["sample_id"] = (
            test_long["sample_id_prefix"].astype(str) + "__" + test_long["sample_id_suffix"].astype(str)
        )

    # sample_id_prefix を作る
    if "sample_id_prefix" not in test_long.columns:
        split_df = test_long["sample_id"].astype(str).str.split("__", n=1, expand=True)
        if split_df.shape[1] != 2:
            raise ValueError("sample_id から sample_id_prefix を作れません。")
        test_long["sample_id_prefix"] = split_df[0]

    # 推論は画像ごとに1回だけ行う
    test_img = (
        test_long[["image_path", "sample_id_prefix"]]
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )

    # 型を明示（merge事故回避）
    test_img["image_path"] = test_img["image_path"].astype(str)
    test_img["sample_id_prefix"] = test_img["sample_id_prefix"].astype(str)
    test_long["image_path"] = test_long["image_path"].astype(str)
    test_long["sample_id_prefix"] = test_long["sample_id_prefix"].astype(str)
    test_long["sample_id_suffix"] = test_long["sample_id_suffix"].astype(str)
    test_long["sample_id"] = test_long["sample_id"].astype(str)

    return test_img, test_long


# =====================
# Transform (code1のvalid相当)
# =====================
def get_test_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        additional_targets={"right": "image"},
    )


# =====================
# Dataset (code1の左右分割に対応)
# =====================
class BiomassTestDataset(Dataset):
    """
    code1 と同じく 1000x2000 画像を左右に分割して入力する。
    test 用なので target は返さない。
    """

    def __init__(self, test_img_df: pd.DataFrame, input_dir: Path, transform: A.Compose):
        self.test_img_df = test_img_df.reset_index(drop=True)
        self.input_dir = Path(input_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.test_img_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.test_img_df.iloc[index]
        image_path = self.input_dir / str(row["image_path"])

        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            raise FileNotFoundError(f"cv2.imread failed: {image_path}")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # code1の実装に合わせて左右1000pxで分割
        left_rgb = rgb_image[:, :1000, :]
        right_rgb = rgb_image[:, 1000:, :]

        transformed = self.transform(image=left_rgb, right=right_rgb)
        left_tensor = transformed["image"]
        right_tensor = transformed["right"]
        return left_tensor, right_tensor


# =====================
# LoRA (code1互換)
# =====================
class LoRALinear(nn.Module):
    """
    code1互換:
    timm EVA/DINOv3 系で self.qkv.weight を直接参照する実装に対応するため、
    .weight / .bias を属性として持つ。
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.weight = base_linear.weight
        self.bias = base_linear.bias

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(self.alpha) / float(self.rank)

        # baseは凍結
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))

        # 初期化（学習前相当）
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = nn.functional.linear(x, self.weight, self.bias)

        dropped_x = self.dropout(x)
        lora_output = nn.functional.linear(
            nn.functional.linear(dropped_x, self.lora_A, bias=None),
            self.lora_B,
            bias=None,
        )
        return base_output + self.scale * lora_output


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

    transformer_blocks = backbone.blocks
    num_blocks = len(transformer_blocks)

    if train_last_n_blocks <= 0:
        target_block_indices = set(range(num_blocks))
    else:
        start_block_index = max(0, num_blocks - train_last_n_blocks)
        target_block_indices = set(range(start_block_index, num_blocks))

    lora_trainable_params: List[nn.Parameter] = []

    for block_index, transformer_block in enumerate(transformer_blocks):
        if block_index not in target_block_indices:
            continue

        attention_module = getattr(transformer_block, "attn", None)
        if attention_module is None:
            continue

        if "attn.qkv" in target_modules:
            base_qkv = getattr(attention_module, "qkv", None)
            if isinstance(base_qkv, nn.Linear):
                lora_qkv = LoRALinear(base_qkv, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attention_module, "qkv", lora_qkv)
                lora_trainable_params.append(lora_qkv.lora_A)
                lora_trainable_params.append(lora_qkv.lora_B)

        if "attn.proj" in target_modules:
            base_proj = getattr(attention_module, "proj", None)
            if isinstance(base_proj, nn.Linear):
                lora_proj = LoRALinear(base_proj, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attention_module, "proj", lora_proj)
                lora_trainable_params.append(lora_proj.lora_A)
                lora_trainable_params.append(lora_proj.lora_B)

    return lora_trainable_params


# =====================
# Density head / model (code1互換)
# =====================
class PatchDensityHead(nn.Module):
    """
    patch token -> density map
    dens >= 0 を保証するため softplus を使う
    """

    def __init__(self, embed_dim: int, num_targets: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_targets),
        )

    def forward(self, patch_tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        # patch_tokens: (B, N, D), N = H*W
        batch_size, num_patches, embed_dim = patch_tokens.shape
        grid_h, grid_w = grid_hw

        if num_patches != grid_h * grid_w:
            raise ValueError(f"num_patches={num_patches} != H*W={grid_h}*{grid_w}")

        patch_grid = patch_tokens.view(batch_size, grid_h, grid_w, embed_dim)  # (B,H,W,D)
        density_map = self.proj(patch_grid).permute(0, 3, 1, 2).contiguous()   # (B,T,H,W)
        density_map = nn.functional.softplus(density_map)
        return density_map


class DualStreamDINOv3DensityLoRARegressor(nn.Module):
    """
    code1互換:
    DINOv3 patch tokens -> density map -> spatial sum -> log1p
    """

    def __init__(self, infer_cfg: InferenceConfig, pretrained_backbone: bool):
        super().__init__()
        self.infer_cfg = infer_cfg
        self.num_prefix_tokens = infer_cfg.num_prefix_tokens

        self.backbone = timm.create_model(
            infer_cfg.model_name,
            pretrained=pretrained_backbone,  # Kaggleでは通常False推奨（重みはckptから読む）
            num_classes=0,
            global_pool="",
        )

        # 学習時と同じく一旦凍結（推論ではどのみち grad 使わない）
        for backbone_param in self.backbone.parameters():
            backbone_param.requires_grad = False

        self.lora_params: List[nn.Parameter] = []
        if infer_cfg.use_lora:
            self.lora_params = apply_lora_to_dinov3_vit(
                self.backbone,
                rank=infer_cfg.lora_rank,
                alpha=infer_cfg.lora_alpha,
                dropout=infer_cfg.lora_dropout,
                target_modules=tuple(infer_cfg.lora_target_modules),
                train_last_n_blocks=infer_cfg.lora_train_last_n_blocks,
            )

        embed_dim = self.backbone.num_features
        self.density_head = PatchDensityHead(embed_dim=embed_dim, num_targets=len(ALL_TARGET_COLS))

    def _get_patch_grid(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        patch_size = self.backbone.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        grid_h = input_tensor.shape[-2] // patch_size
        grid_w = input_tensor.shape[-1] // patch_size
        return grid_h, grid_w

    def _encode_to_density(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(input_tensor)          # (B, prefix+N, D)
        patch_tokens = features[:, self.num_prefix_tokens :, :]          # (B, N, D)
        grid_hw = self._get_patch_grid(input_tensor)
        density_map = self.density_head(patch_tokens, grid_hw)          # (B,5,H,W)
        return density_map

    def forward(self, left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> torch.Tensor:
        left_density = self._encode_to_density(left_tensor)
        right_density = self._encode_to_density(right_tensor)

        total_density = left_density + right_density                     # (B,5,H,W)
        pred_raw = total_density.sum(dim=(2, 3))                        # (B,5)
        pred_log = torch.log1p(pred_raw)                                # (B,5)
        return pred_log


# =====================
# ckpt ロード（Lightning .ckpt 対応）
# =====================
def strip_state_dict_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    stripped_state_dict: Dict[str, torch.Tensor] = {}

    for param_name, param_tensor in state_dict.items():
        new_name = param_name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
        stripped_state_dict[new_name] = param_tensor

    return stripped_state_dict


def load_model_from_checkpoint(
    checkpoint_path: Path,
    infer_cfg: InferenceConfig,
    device: torch.device,
) -> nn.Module:
    """
    code1のLightningModule(BiomassModule)のckptから、
    内部 model (DualStream...) に対応する state_dict をロードする。
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint_obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if "state_dict" in checkpoint_obj:
        checkpoint_state_dict = checkpoint_obj["state_dict"]
    else:
        checkpoint_state_dict = checkpoint_obj

    # code1のBiomassModule内の self.model.xxx を想定して "model." を剥がす
    # 例: model.backbone.blocks.0.... / model.density_head....
    model_state_dict = strip_state_dict_prefix(
        checkpoint_state_dict,
        prefixes=["model.", "net.", "module."],
    )

    # Kaggleのオフライン推論では pretrained_backbone=False 推奨
    model = DualStreamDINOv3DensityLoRARegressor(
        infer_cfg=infer_cfg,
        pretrained_backbone=False,
    )

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    # 重要な不一致を見逃しにくくする
    if len(unexpected_keys) > 0:
        print(f"[warn] unexpected keys (first 20): {unexpected_keys[:20]} (total={len(unexpected_keys)})")
    if len(missing_keys) > 0:
        print(f"[warn] missing keys (first 20): {missing_keys[:20]} (total={len(missing_keys)})")

    model.to(device)
    model.eval()
    return model


# =====================
# 推論本体
# =====================
@torch.no_grad()
def predict_log_ensemble(
    ckpt_paths: List[Path],
    infer_cfg: InferenceConfig,
    test_img_df: pd.DataFrame,
    input_dir: Path,
    device: torch.device,
) -> np.ndarray:
    """
    戻り値:
        pred_log_mean: (N,5)  log1p予測
    """
    test_transform = get_test_transform(infer_cfg.img_size)
    test_dataset = BiomassTestDataset(test_img_df, input_dir=input_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=infer_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_log_sum: Optional[np.ndarray] = None

    for fold_index, checkpoint_path in enumerate(ckpt_paths):
        print(f"[info] loading checkpoint {fold_index}: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, infer_cfg=infer_cfg, device=device)

        fold_pred_log_batches: List[np.ndarray] = []
        for left_tensor, right_tensor in test_loader:
            left_tensor = left_tensor.to(device, non_blocking=True)
            right_tensor = right_tensor.to(device, non_blocking=True)

            pred_log_tensor = model(left_tensor, right_tensor).float()  # (B,5) log1p
            fold_pred_log_batches.append(pred_log_tensor.cpu().numpy().astype(np.float32))

        fold_pred_log = np.concatenate(fold_pred_log_batches, axis=0)  # (N,5)
        pred_log_sum = fold_pred_log if pred_log_sum is None else (pred_log_sum + fold_pred_log)

        # メモリ解放
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_log_mean = pred_log_sum / max(1, len(ckpt_paths))
    return pred_log_mean.astype(np.float32)


# =====================
# submission 作成
# =====================
def build_submission_from_testcsv(
    pred_log5: np.ndarray,   # (N,5) log1p
    test_img_df: pd.DataFrame,
    test_long_df: pd.DataFrame,
) -> pd.DataFrame:
    if pred_log5.ndim != 2 or pred_log5.shape[1] != 5:
        raise ValueError(f"pred_log5 の形が想定外: {pred_log5.shape} (N,5) を想定")

    # code1 の保存方針に合わせて raw 空間へ戻す
    pred_raw5 = np.expm1(pred_log5)
    pred_raw5 = np.clip(pred_raw5, 0.0, None).astype(np.float32)

    pred_wide_df = test_img_df.copy()
    for target_index, target_name in enumerate(TARGET5_ORDER):
        pred_wide_df[target_name] = pred_raw5[:, target_index]

    # test.csv の行順そのままで提出を作る
    submission_df = test_long_df[["sample_id", "sample_id_suffix", "image_path"]].copy()
    submission_df = submission_df.merge(
        pred_wide_df[["image_path"] + TARGET5_ORDER],
        on="image_path",
        how="left",
    )

    # suffixに応じて target を選ぶ
    submission_df["target"] = np.nan
    for target_name in TARGET5_ORDER:
        row_mask = submission_df["sample_id_suffix"].astype(str) == target_name
        submission_df.loc[row_mask, "target"] = submission_df.loc[row_mask, target_name].astype(float)

    # 異常検知
    if submission_df["target"].isna().any():
        bad_rows = submission_df[submission_df["target"].isna()][["sample_id", "sample_id_suffix"]].head(10)
        raise ValueError(
            "submission target に NaN が発生した。suffix が想定外の可能性あり。例:\n"
            f"{bad_rows}"
        )

    return submission_df[["sample_id", "target"]]


# =====================
# main (Kaggle Notebook用)
# =====================
def main() -> None:
    # ===== ここを自分のKaggle環境に合わせて変更 =====
    COMP_DATASET_DIR = Path("/kaggle/input/competitions/csiro-biomass")  # 競技データ
    WEIGHTS_DATASET_DIR = Path("/kaggle/input/datasets/surumendd/csiro-latesub-exp007-000-dinov3-lora-dataset")  # あなたの重みdataset名に変更

    CKPT_PATHS = [
        WEIGHTS_DATASET_DIR / "fold0" / "epoch=...-step=....ckpt",  # 実ファイル名に変更
        WEIGHTS_DATASET_DIR / "fold1" / "epoch=...-step=....ckpt",
        WEIGHTS_DATASET_DIR / "fold2" / "epoch=...-step=....ckpt",
        WEIGHTS_DATASET_DIR / "fold3" / "epoch=...-step=....ckpt",
    ]

    infer_cfg = InferenceConfig(
        model_name="vit_large_patch16_dinov3_qkvb.lvd1689m",
        img_size=448,
        batch_size=4,
        num_workers=2,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_train_last_n_blocks=8,
        lora_target_modules=("attn.qkv", "attn.proj"),
        num_prefix_tokens=5,
    )
    # ================================================

    test_csv_path = COMP_DATASET_DIR / "test.csv"
    input_dir = COMP_DATASET_DIR  # image_path が "test/xxx.jpg" 前提

    if not test_csv_path.exists():
        raise FileNotFoundError(f"test.csv が見つかりません: {test_csv_path}")

    for checkpoint_path in CKPT_PATHS:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint が見つかりません: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")


    test_img_df, test_long_df = make_test_tables(test_csv_path)
    print(f"[info] test images: {len(test_img_df)}")
    print(f"[info] test rows  : {len(test_long_df)}")

    pred_log5 = predict_log_ensemble(
        ckpt_paths=CKPT_PATHS,
        infer_cfg=infer_cfg,
        test_img_df=test_img_df,
        input_dir=input_dir,
        device=device,
    )

    submission_df = build_submission_from_testcsv(
        pred_log5=pred_log5,
        test_img_df=test_img_df,
        test_long_df=test_long_df,
    )

    output_path = Path("submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"[info] saved: {output_path} rows={len(submission_df)}")
    print(submission_df.head())


if __name__ == "__main__":
    main()