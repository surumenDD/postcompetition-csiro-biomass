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
# ===== config =====
@dataclass
class ExpConfig:
    seed: int = 42
    n_folds: int = 4

    # Model
    model_name: str = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size: int = 512

    # Training
    batch_size: int = 4
    accumulate_grad_batches: int = 4
    num_epochs: int = 20
    num_workers: int = 4

    # Optimizer
    head_lr: float = 1e-3
    lora_lr: float = 5e-4
    uw_lr: float = 1e-3
    weight_decay: float = 0.05

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_train_last_n_blocks: int = 16
    lora_target_modules: Tuple[str, ...] = ("attn.qkv",)  # projは外す

    # Intermediate features (ViT block indices; negative index可)
    intermediate_block_indices: Tuple[int, ...] = (-1, -5, -9, -13)

    # Pyramid / shared representation
    pyramid_hidden_channels: int = 256
    pyramid_scales: Tuple[float, ...] = (2.0, 1.0, 0.5, 0.25)
    shared_z_dim: int = 512

    # Loss
    huber_beta: float = 1.0
    eps: float = 1e-6

    # MTL
    use_uncertainty_weighting: bool = True
    use_pcgrad: bool = True
    pcgrad_eps: float = 1e-12

    # 5D reconstruction weights (LB重み)
    recon5_weights: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.2, 0.5)


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

# PCGrad（headだけ、LoRA/UWをPCgradの対象から除外）実装補助


def _flatten_grad_list(grad_list: List[torch.Tensor], param_list: List[nn.Parameter]) -> torch.Tensor:
    flat_chunks: List[torch.Tensor] = []
    for grad_tensor, param in zip(grad_list, param_list):
        if grad_tensor is None:
            flat_chunks.append(torch.zeros_like(
                param, memory_format=torch.contiguous_format).view(-1))
        else:
            flat_chunks.append(grad_tensor.contiguous().view(-1))
    return torch.cat(flat_chunks, dim=0)


def _split_flat_grad_to_param_shapes(flat_grad: torch.Tensor, param_list: List[nn.Parameter]) -> List[torch.Tensor]:
    split_grads: List[torch.Tensor] = []
    start_index = 0
    for param in param_list:
        numel = param.numel()
        grad_view = flat_grad[start_index:start_index + numel].view_as(param)
        split_grads.append(grad_view)
        start_index += numel
    return split_grads


def _project_if_conflict(
    source_grad: torch.Tensor,
    reference_grad: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    reference_norm_sq = torch.dot(reference_grad, reference_grad)
    if float(reference_norm_sq.detach().item()) <= eps:
        return source_grad

    dot_value = torch.dot(source_grad, reference_grad)
    if float(dot_value.detach().item()) < 0.0:
        projected = source_grad - \
            (dot_value / (reference_norm_sq + eps)) * reference_grad
        return projected
    return source_grad


def build_grouped_pcgrad_head_gradient(
    primary_term_grad_map: Dict[str, torch.Tensor],
    auxiliary_term_grad_map: Dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """
    primary_sum と auxiliary各勾配の衝突だけを見る。
    """
    if len(primary_term_grad_map) == 0:
        raise ValueError("primary_term_grad_map is empty")

    primary_grad_sum = None
    for primary_grad in primary_term_grad_map.values():
        primary_grad_sum = primary_grad if primary_grad_sum is None else (
            primary_grad_sum + primary_grad)

    assert primary_grad_sum is not None
    projected_aux_grad_sum = torch.zeros_like(primary_grad_sum)

    for aux_grad in auxiliary_term_grad_map.values():
        projected_aux_grad = _project_if_conflict(
            source_grad=aux_grad,
            reference_grad=primary_grad_sum,
            eps=eps,
        )
        projected_aux_grad_sum = projected_aux_grad_sum + projected_aux_grad

    final_head_grad = primary_grad_sum + projected_aux_grad_sum
    return final_head_grad


# Uncertainty Weighting
class MultiTaskUncertaintyWeighting(nn.Module):
    """
    taskごとに log_variance s_t を学習する。
    term_t = exp(-s_t) * L_t + s_t
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = list(task_names)
        self.task_to_index = {task_name: task_i for task_i,
                              task_name in enumerate(self.task_names)}
        self.log_variances = nn.Parameter(
            torch.zeros(len(task_names), dtype=torch.float32))

    def make_weighted_term(self, task_name: str, raw_task_loss: torch.Tensor) -> torch.Tensor:
        task_index = self.task_to_index[task_name]
        log_variance = self.log_variances[task_index]
        weighted_term = torch.exp(-log_variance) * raw_task_loss + log_variance
        return weighted_term


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


# ===== debug C: LoRA injection確認 =====
def log_lora_injection_summary(backbone: nn.Module, logger) -> None:
    """LoRAが注入されたモジュール一覧をログ出力する（1回だけ呼ぶ）。"""
    injected_module_names = []
    for module_name, module in backbone.named_modules():
        if isinstance(module, LoRALinear):
            injected_module_names.append(module_name)

    logger.info("Injected LoRA modules count: %d", len(injected_module_names))
    for module_name in injected_module_names[:100]:
        logger.info("  LoRA: %s", module_name)


# ===== density head =====
class ConvBnGelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding_size = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidTokenizerHead(nn.Module):
    """
    中間層特徴を統合し、4段ピラミッド -> GAP -> concat -> shared z を作る。
    入力は list[(B,C,H,W)] を想定。
    """

    def __init__(
        self,
        backbone_channels: int,
        num_intermediate_levels: int,
        pyramid_hidden_channels: int,
        pyramid_scales: Tuple[float, ...],
        shared_z_dim: int,
    ):
        super().__init__()
        self.pyramid_scales = tuple(float(scale_factor)
                                    for scale_factor in pyramid_scales)

        fused_in_channels = backbone_channels * num_intermediate_levels
        self.fuse_intermediates = nn.Sequential(
            nn.Conv2d(fused_in_channels, pyramid_hidden_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(pyramid_hidden_channels),
            nn.GELU(),
            ConvBnGelu(pyramid_hidden_channels,
                       pyramid_hidden_channels, kernel_size=3),
        )

        self.pyramid_refine_blocks = nn.ModuleList([
            ConvBnGelu(pyramid_hidden_channels,
                       pyramid_hidden_channels, kernel_size=3)
            for _ in self.pyramid_scales
        ])

        concat_feature_dim = pyramid_hidden_channels * len(self.pyramid_scales)
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(concat_feature_dim),
            nn.Linear(concat_feature_dim, shared_z_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(shared_z_dim, shared_z_dim),
        )

    def forward(self, intermediate_feature_maps: List[torch.Tensor]) -> torch.Tensor:
        if len(intermediate_feature_maps) == 0:
            raise ValueError("intermediate_feature_maps is empty")

        reference_height, reference_width = intermediate_feature_maps[0].shape[-2:]
        resized_feature_maps = []
        for feature_map in intermediate_feature_maps:
            if feature_map.shape[-2:] != (reference_height, reference_width):
                feature_map = nn.functional.interpolate(
                    feature_map,
                    size=(reference_height, reference_width),
                    mode="bilinear",
                    align_corners=False,
                )
            resized_feature_maps.append(feature_map)

        fused_feature_map = self.fuse_intermediates(
            torch.cat(resized_feature_maps, dim=1))

        pyramid_vectors: List[torch.Tensor] = []
        for pyramid_i, scale_factor in enumerate(self.pyramid_scales):
            if scale_factor == 1.0:
                scaled_feature_map = fused_feature_map
            else:
                scaled_feature_map = nn.functional.interpolate(
                    fused_feature_map,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
            scaled_feature_map = self.pyramid_refine_blocks[pyramid_i](
                scaled_feature_map)
            pooled_vector = scaled_feature_map.mean(dim=(2, 3))  # GAP
            pyramid_vectors.append(pooled_vector)

        concat_vector = torch.cat(pyramid_vectors, dim=1)
        shared_z = self.shared_mlp(concat_vector)
        return shared_z


class DualStreamDINOv3PyramidMTL(nn.Module):
    """
    - DINOv3凍結 + LoRA(QKVのみ)
    - 中間層複数ブロック特徴を使用
    - 2Dピラミッドで shared representation z を作る
    - Total + Ratio + Aux (Height/NDVI/State)
    """

    def __init__(self, cfg: Config, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.eps = float(cfg.exp.eps)

        self.backbone = timm.create_model(
            cfg.exp.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # base backbone freeze
        for backbone_param in self.backbone.parameters():
            backbone_param.requires_grad = False

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

        self.intermediate_block_indices = tuple(
            cfg.exp.intermediate_block_indices)
        backbone_channels = int(self.backbone.num_features)

        self.pyramid_head = PyramidTokenizerHead(
            backbone_channels=backbone_channels,
            num_intermediate_levels=len(self.intermediate_block_indices),
            pyramid_hidden_channels=cfg.exp.pyramid_hidden_channels,
            pyramid_scales=cfg.exp.pyramid_scales,
            shared_z_dim=cfg.exp.shared_z_dim,
        )

        fused_stream_dim = cfg.exp.shared_z_dim * 2
        self.stream_fusion = nn.Sequential(
            nn.LayerNorm(fused_stream_dim),
            nn.Linear(fused_stream_dim, cfg.exp.shared_z_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        head_input_dim = cfg.exp.shared_z_dim

        # main heads
        # Dry_Total_g (raw positive via softplus)
        self.total_head = nn.Linear(head_input_dim, 1)
        # [Clover, Dead, Green] logits
        self.ratio_head = nn.Linear(head_input_dim, 3)

        # aux heads
        self.height_head = nn.Linear(head_input_dim, 1)
        self.ndvi_head = nn.Linear(head_input_dim, 1)
        self.state_head = nn.Linear(
            head_input_dim, 16)  # 実際のstate数で置換（後でsetする）

    def set_num_state_classes(self, num_state_classes: int) -> None:
        current_input_dim = self.state_head.in_features
        self.state_head = nn.Linear(current_input_dim, int(num_state_classes))

    def _get_patch_grid_hw(self, image_tensor: torch.Tensor) -> Tuple[int, int]:
        patch_size = self.backbone.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = int(patch_size)
        grid_h = image_tensor.shape[-2] // patch_h
        grid_w = image_tensor.shape[-1] // patch_w
        return int(grid_h), int(grid_w)

    def _extract_intermediate_feature_maps(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        timm forward_intermediates を利用。
        ViTでは indices は block index を指す。docs参照。:contentReference[oaicite:4]{index=4}
        戻りの型差異に備えて吸収する。
        """
        if not hasattr(self.backbone, "forward_intermediates"):
            raise RuntimeError(
                "This timm model does not expose forward_intermediates(). Update timm or use hook-based extraction."
            )

        forward_result = self.backbone.forward_intermediates(
            image_tensor,
            indices=self.intermediate_block_indices,
            intermediates_only=True,
        )

        if isinstance(forward_result, tuple):
            maybe_intermediates = forward_result[-1]
        else:
            maybe_intermediates = forward_result

        if not isinstance(maybe_intermediates, (list, tuple)):
            raise TypeError(
                f"Unexpected forward_intermediates return type: {type(maybe_intermediates)}")

        patch_grid_h, patch_grid_w = self._get_patch_grid_hw(image_tensor)
        patch_token_count = patch_grid_h * patch_grid_w

        intermediate_feature_maps: List[torch.Tensor] = []
        for intermediate_feature in maybe_intermediates:
            if intermediate_feature.ndim == 4:
                # 想定: (B, C, H, W)
                intermediate_feature_maps.append(intermediate_feature)

            elif intermediate_feature.ndim == 3:
                batch_size, num_tokens, channel_dim = intermediate_feature.shape

                if num_tokens == patch_token_count:
                    patch_tokens = intermediate_feature
                elif num_tokens > patch_token_count:
                    # prefix tokenが先頭に付いているケースを吸収（後ろ patch_token_count 個を採用）
                    patch_tokens = intermediate_feature[:, -
                                                        patch_token_count:, :]
                else:
                    raise ValueError(
                        f"num_tokens={num_tokens} < patch_token_count={patch_token_count}; cannot recover patch grid"
                    )

                feature_map = patch_tokens.transpose(1, 2).contiguous().view(
                    batch_size, channel_dim, patch_grid_h, patch_grid_w
                )
                intermediate_feature_maps.append(feature_map)

            else:
                raise ValueError(
                    f"Unexpected intermediate feature ndim={intermediate_feature.ndim}")

        return intermediate_feature_maps

    def _encode_single_view(self, image_tensor: torch.Tensor) -> torch.Tensor:
        intermediate_feature_maps = self._extract_intermediate_feature_maps(
            image_tensor)
        shared_z = self.pyramid_head(intermediate_feature_maps)
        return shared_z

    def _reconstruct_raw5_from_total_ratio(
        self,
        total_raw: torch.Tensor,        # (B,)
        ratio_logits: torch.Tensor,     # (B,3) [Clover, Dead, Green]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ratio_probs = torch.softmax(ratio_logits, dim=-1)  # (B,3)

        clover_raw = total_raw * ratio_probs[:, 0]
        dead_raw = total_raw * ratio_probs[:, 1]
        green_raw = total_raw * ratio_probs[:, 2]

        # 仮定: GDM = Green + Clover
        gdm_raw = green_raw + clover_raw
        total_raw_out = total_raw

        # ALL_TARGET_COLS = [Green, Clover, Dead, GDM, Total]
        reconstructed_raw5 = torch.stack(
            [green_raw, clover_raw, dead_raw, gdm_raw, total_raw_out],
            dim=1,
        )
        return reconstructed_raw5, ratio_probs

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        left_z = self._encode_single_view(left)
        right_z = self._encode_single_view(right)

        fused_z = self.stream_fusion(torch.cat([left_z, right_z], dim=1))

        total_raw = nn.functional.softplus(
            self.total_head(fused_z).squeeze(1))  # (B,)
        ratio_logits = self.ratio_head(
            fused_z)                                   # (B,3)

        reconstructed_raw5, ratio_probs = self._reconstruct_raw5_from_total_ratio(
            total_raw, ratio_logits)
        reconstructed_log5 = torch.log1p(reconstructed_raw5.clamp_min(0.0))
        total_log = torch.log1p(total_raw)

        output_dict = {
            "shared_z": fused_z,
            "pred_total_raw": total_raw,
            "pred_total_log": total_log,
            "pred_ratio_logits": ratio_logits,
            "pred_ratio_probs": ratio_probs,
            "pred_recon_raw5": reconstructed_raw5,
            "pred_recon_log5": reconstructed_log5,

            # aux (test時は捨てる)
            "pred_height": self.height_head(fused_z).squeeze(1),
            "pred_ndvi": self.ndvi_head(fused_z).squeeze(1),
            "pred_state_logits": self.state_head(fused_z),
        }
        return output_dict

    def get_head_params_for_pcgrad(self) -> List[nn.Parameter]:
        head_modules = [
            self.pyramid_head,
            self.stream_fusion,
            self.total_head,
            self.ratio_head,
            self.height_head,
            self.ndvi_head,
            self.state_head,
        ]
        head_params: List[nn.Parameter] = []
        for head_module in head_modules:
            for param in head_module.parameters():
                if param.requires_grad:
                    head_params.append(param)
        return head_params


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
        ckpt_path = self.dirpath / \
            f"fold{self.fold_i}_last_epoch{epoch:02d}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))


# ===== lightning module =====
class BiomassMTLModule(pl.LightningModule):
    def __init__(self, model: DualStreamDINOv3PyramidMTL, cfg: Config):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.automatic_optimization = False  # PCGradのため手動化

        self.register_buffer(
            "recon5_weight_tensor",
            torch.tensor(cfg.exp.recon5_weights, dtype=torch.float32),
        )

        self.primary_task_names = ["primary_total",
                                   "primary_ratio", "primary_recon5"]
        self.aux_task_names = ["aux_height", "aux_ndvi", "aux_state"]
        self.all_task_names = self.primary_task_names + self.aux_task_names

        self.uncertainty_weighting = MultiTaskUncertaintyWeighting(
            self.all_task_names)

        self.val_pred_log_list: List[np.ndarray] = []
        self.val_true_log_list: List[np.ndarray] = []

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(left, right)

    def _soft_cross_entropy_with_soft_targets(
        self,
        logits: torch.Tensor,        # (B, C)
        soft_targets: torch.Tensor,  # (B, C)
        sample_mask: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_targets * log_probs).sum(dim=-1)  # (B,)
        masked_loss_sum = (per_sample_loss * sample_mask).sum()
        mask_sum = sample_mask.sum().clamp_min(1.0)
        return masked_loss_sum / mask_sum

    def _masked_smooth_l1_loss(
        self,
        pred_value: torch.Tensor,    # (B,)
        true_value: torch.Tensor,    # (B,)
        valid_mask: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        per_sample_loss = nn.functional.smooth_l1_loss(
            pred_value, true_value, reduction="none", beta=self.cfg.exp.huber_beta
        )
        masked_loss_sum = (per_sample_loss * valid_mask).sum()
        valid_count = valid_mask.sum().clamp_min(1.0)
        return masked_loss_sum / valid_count

    def _masked_cross_entropy_ignore_unknown(
        self,
        logits: torch.Tensor,          # (B, C)
        targets: torch.Tensor,         # (B,)
        ignore_index: int,
    ) -> torch.Tensor:
        valid_mask = (targets != ignore_index)
        if not bool(valid_mask.any()):
            # 勾配が流れる0を返す（dtype/deviceを合わせる）
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        return nn.functional.cross_entropy(valid_logits, valid_targets)

    def _weighted_recon5_log_loss(self, pred_log_5d: torch.Tensor, true_log_5d: torch.Tensor) -> torch.Tensor:
        per_element_loss = nn.functional.smooth_l1_loss(
            pred_log_5d,
            true_log_5d,
            reduction="none",
            beta=self.cfg.exp.huber_beta,
        )  # (B,5)
        per_target_loss = per_element_loss.mean(dim=0)  # (5,)
        return (per_target_loss * self.recon5_weight_tensor).sum()

    def _compute_raw_task_losses(
        self,
        batch_dict: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raw_loss_dict: Dict[str, torch.Tensor] = {}

        # Primary 1: Total (log-space regression)
        raw_loss_dict["primary_total"] = nn.functional.smooth_l1_loss(
            output_dict["pred_total_log"],
            batch_dict["target_total_log"],
            reduction="mean",
            beta=self.cfg.exp.huber_beta,
        )

        # Primary 2: Ratio (soft-label CE, [Clover, Dead, Green])
        raw_loss_dict["primary_ratio"] = self._soft_cross_entropy_with_soft_targets(
            logits=output_dict["pred_ratio_logits"],
            soft_targets=batch_dict["target_ratio_soft"],
            sample_mask=batch_dict["target_ratio_valid_mask"],
        )

        # Primary 3: 5D reconstruction (log-space, LB weight)
        raw_loss_dict["primary_recon5"] = self._weighted_recon5_log_loss(
            output_dict["pred_recon_log5"],
            batch_dict["target_log_5d"],
        )

        # Aux 1: Height
        raw_loss_dict["aux_height"] = self._masked_smooth_l1_loss(
            pred_value=output_dict["pred_height"],
            true_value=batch_dict["target_height"],
            valid_mask=batch_dict["target_height_mask"],
        )

        # Aux 2: NDVI
        raw_loss_dict["aux_ndvi"] = self._masked_smooth_l1_loss(
            pred_value=output_dict["pred_ndvi"],
            true_value=batch_dict["target_ndvi"],
            valid_mask=batch_dict["target_ndvi_mask"],
        )

        # Aux 3: State (ignore unknown)
        raw_loss_dict["aux_state"] = nn.functional.cross_entropy(
            output_dict["pred_state_logits"],
            batch_dict["target_state"],
            ignore_index=BiomassMultiTaskDataset.STATE_UNKNOWN_INDEX,
        )

        return raw_loss_dict

    def _make_weighted_terms(self, raw_loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        weighted_term_dict: Dict[str, torch.Tensor] = {}
        for task_name, raw_loss in raw_loss_dict.items():
            if self.cfg.exp.use_uncertainty_weighting:
                weighted_term_dict[task_name] = self.uncertainty_weighting.make_weighted_term(
                    task_name, raw_loss)
            else:
                weighted_term_dict[task_name] = raw_loss
        return weighted_term_dict

    def _collect_head_term_gradients(
        self,
        weighted_term_dict: Dict[str, torch.Tensor],
        head_param_list: List[nn.Parameter],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        primary_term_grad_map: Dict[str, torch.Tensor] = {}
        auxiliary_term_grad_map: Dict[str, torch.Tensor] = {}

        task_order = self.primary_task_names + self.aux_task_names
        for task_i, task_name in enumerate(task_order):
            weighted_term = weighted_term_dict[task_name]
            grad_list = torch.autograd.grad(
                weighted_term,
                head_param_list,
                retain_graph=True,   # 複数taskで使い回す
                allow_unused=True,
            )
            flat_grad = _flatten_grad_list(list(grad_list), head_param_list)

            if task_name in self.primary_task_names:
                primary_term_grad_map[task_name] = flat_grad
            else:
                auxiliary_term_grad_map[task_name] = flat_grad

        return primary_term_grad_map, auxiliary_term_grad_map

    def _clone_current_head_grads(self, head_param_list: List[nn.Parameter]) -> List[torch.Tensor | None]:
        cloned_grad_list: List[torch.Tensor | None] = []
        for head_param in head_param_list:
            if head_param.grad is None:
                cloned_grad_list.append(None)
            else:
                cloned_grad_list.append(head_param.grad.detach().clone())
        return cloned_grad_list

    # ===== debug A: NaN early detection =====
    def _assert_finite_tensor(self, tensor_value: torch.Tensor, tensor_name: str) -> None:
        if not torch.isfinite(tensor_value).all():
            raise RuntimeError(
                f"{tensor_name} has non-finite values: {tensor_value.detach().cpu()}")

    def training_step(self, batch_dict, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        accumulation_steps = int(self.cfg.exp.accumulate_grad_batches)
        if accumulation_steps <= 0:
            raise ValueError("accumulate_grad_batches must be >= 1")

        is_accumulation_start = (batch_idx % accumulation_steps) == 0
        is_accumulation_boundary = ((batch_idx + 1) % accumulation_steps) == 0
        is_last_batch = bool(self.trainer.is_last_batch)
        should_step_optimizer = is_accumulation_boundary or is_last_batch

        # 蓄積開始時だけ zero_grad
        if is_accumulation_start:
            optimizer.zero_grad(set_to_none=True)

        left_image = batch_dict["left"]
        right_image = batch_dict["right"]

        output_dict = self(left_image, right_image)
        raw_loss_dict = self._compute_raw_task_losses(batch_dict, output_dict)
        weighted_term_dict = self._make_weighted_terms(raw_loss_dict)

        # ログ用（人間が見る値）
        total_loss = sum(weighted_term_dict.values())

        # backward用（勾配蓄積のためにスケール）
        scaled_total_loss = total_loss / float(accumulation_steps)

        # 先にheadのtask別勾配を取得（term_tベース）
        head_param_list = self.model.get_head_params_for_pcgrad()
        should_apply_pcgrad = bool(
            self.cfg.exp.use_pcgrad and len(head_param_list) > 0)

        if should_apply_pcgrad:
            # PCGrad対象のtermも accumulation に合わせてスケールする
            scaled_weighted_term_dict = {
                task_name: task_term / float(accumulation_steps)
                for task_name, task_term in weighted_term_dict.items()
            }
            primary_term_grad_map, auxiliary_term_grad_map = self._collect_head_term_gradients(
                weighted_term_dict=scaled_weighted_term_dict,
                head_param_list=head_param_list,
            )

        # LoRA/UW含む全体勾配（scaled）
        self.manual_backward(scaled_total_loss)

        # headだけPCGradで上書き（これも scaled 勾配）
        if should_apply_pcgrad:
            final_head_flat_grad = build_grouped_pcgrad_head_gradient(
                primary_term_grad_map=primary_term_grad_map,
                auxiliary_term_grad_map=auxiliary_term_grad_map,
                eps=float(self.cfg.exp.pcgrad_eps),
            )
            split_head_grads = _split_flat_grad_to_param_shapes(
                final_head_flat_grad, head_param_list)
            for head_param, pcgrad_tensor in zip(head_param_list, split_head_grads):
                head_param.grad = pcgrad_tensor

        # accumulation境界（または最終batch）のみstep
        if should_step_optimizer:
            optimizer.step()

            # epoch末でだけ scheduler.step（今の設計のままでよい）
            if is_last_batch:
                scheduler.step()

        # logging（scaledではなく元の値を記録）
        self.log("train_loss", total_loss.detach(),
                 on_step=False, on_epoch=True)
        for task_name, raw_loss in raw_loss_dict.items():
            self.log(f"train_raw/{task_name}",
                     raw_loss.detach(), on_step=False, on_epoch=True)
        for task_name, weighted_term in weighted_term_dict.items():
            self.log(
                f"train_term/{task_name}", weighted_term.detach(), on_step=False, on_epoch=True)

        if self.cfg.exp.use_uncertainty_weighting:
            for task_i, task_name in enumerate(self.all_task_names):
                self.log(
                    f"train_log_var/{task_name}",
                    self.uncertainty_weighting.log_variances[task_i].detach(),
                    on_step=False,
                    on_epoch=True,
                )

        return total_loss.detach()

    def validation_step(self, batch_dict, batch_idx):
        output_dict = self(batch_dict["left"], batch_dict["right"])
        raw_loss_dict = self._compute_raw_task_losses(batch_dict, output_dict)
        weighted_term_dict = self._make_weighted_terms(raw_loss_dict)
        total_loss = sum(weighted_term_dict.values())

        self.log("val_loss", total_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        pred_log_5d = output_dict["pred_recon_log5"]
        true_log_5d = batch_dict["target_log_5d"]

        self.val_pred_log_list.append(pred_log_5d.detach().cpu().numpy())
        self.val_true_log_list.append(true_log_5d.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        if len(self.val_pred_log_list) == 0:
            return
        pred_log = np.concatenate(self.val_pred_log_list, axis=0)
        true_log = np.concatenate(self.val_true_log_list, axis=0)
        score_value, _ = weighted_r2_score_on_log(true_log, pred_log)
        self.log("val_weighted_r2", score_value, prog_bar=True)

        self.val_pred_log_list.clear()
        self.val_true_log_list.clear()

    def predict_step(self, batch_dict, batch_idx):
        """
        テスト時は補助タスクを捨てる前提。主タスク（5D再構成）のみ返す。
        """
        output_dict = self(batch_dict["left"], batch_dict["right"])
        pred_log_5d = output_dict["pred_recon_log5"].detach().cpu().numpy()

        if "target_log_5d" in batch_dict:
            true_log_5d = batch_dict["target_log_5d"].detach().cpu().numpy()
            return pred_log_5d, true_log_5d
        return pred_log_5d

    def configure_optimizers(self):
        head_params = self.model.get_head_params_for_pcgrad()
        lora_params = list(getattr(self.model, "lora_params", []))
        uw_params = list(self.uncertainty_weighting.parameters()
                         ) if self.cfg.exp.use_uncertainty_weighting else []

        param_groups = []
        if len(head_params) > 0:
            param_groups.append(
                {"params": head_params, "lr": self.cfg.exp.head_lr})
        if len(lora_params) > 0:
            param_groups.append(
                {"params": lora_params, "lr": self.cfg.exp.lora_lr})
        if len(uw_params) > 0:
            param_groups.append(
                {"params": uw_params, "lr": self.cfg.exp.uw_lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.exp.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.exp.num_epochs,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]


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
class BiomassMultiTaskDataset(Dataset):
    """
    左右画像 + 主タスク(5D, Total, Ratio) + 補助タスク(Height, NDVI, State) を返す。
    ratio は [Clover, Dead, Green] 順の soft target。
    """
    STATE_UNKNOWN_INDEX = -100

    def __init__(
        self,
        df: pd.DataFrame,
        img_root: Path,
        transform: A.Compose,
        state_to_index: Dict[str, int],
        *,
        eps: float = 1e-6,
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform
        self.state_to_index = state_to_index
        self.eps = float(eps)

        target_raw_5d = self.df[ALL_TARGET_COLS].values.astype(
            np.float32)  # [Green,Clover,Dead,GDM,Total]
        self.target_raw_5d = target_raw_5d
        self.target_log_5d = np.log1p(target_raw_5d).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _build_ratio_target(self, row_index: int) -> Tuple[np.ndarray, np.float32]:
        # ALL_TARGET_COLS = [Green, Clover, Dead, GDM, Total]
        green_value = float(self.target_raw_5d[row_index, 0])
        clover_value = float(self.target_raw_5d[row_index, 1])
        dead_value = float(self.target_raw_5d[row_index, 2])
        total_value = float(self.target_raw_5d[row_index, 4])

        ratio_denominator = max(total_value, self.eps)
        ratio_target = np.array(
            [
                clover_value / ratio_denominator,  # Clover
                dead_value / ratio_denominator,    # Dead
                green_value / ratio_denominator,   # Green
            ],
            dtype=np.float32,
        )
        ratio_target = ratio_target / max(float(ratio_target.sum()), self.eps)
        ratio_valid_mask = np.float32(total_value > self.eps)
        return ratio_target, ratio_valid_mask

    def _safe_float_value(self, row_index: int, column_name: str) -> Tuple[np.float32, np.float32]:
        raw_value = self.df.loc[row_index, column_name]
        if pd.isna(raw_value):
            return np.float32(0.0), np.float32(0.0)
        return np.float32(raw_value), np.float32(1.0)

    def _state_label(self, row_index: int) -> int:
        raw_state = self.df.loc[row_index, "State"]
        if pd.isna(raw_state):
            return self.STATE_UNKNOWN_INDEX
        state_text = str(raw_state)
        return int(self.state_to_index.get(state_text, self.STATE_UNKNOWN_INDEX))

    def __getitem__(self, row_index: int):
        image_path = self.img_root / self.df.loc[row_index, "image_path"]
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            raise FileNotFoundError(f"cv2.imread failed: {image_path}")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        left_image = rgb_image[:, :1000, :]
        right_image = rgb_image[:, 1000:, :]

        transformed = self.transform(image=left_image, right=right_image)
        left_tensor = transformed["image"]
        right_tensor = transformed["right"]

        target_log_5d_tensor = torch.tensor(
            self.target_log_5d[row_index], dtype=torch.float32)
        target_raw_5d_tensor = torch.tensor(
            self.target_raw_5d[row_index], dtype=torch.float32)

        total_raw_value = torch.tensor(
            float(self.target_raw_5d[row_index, 4]), dtype=torch.float32)
        total_log_value = torch.log1p(total_raw_value)

        ratio_target_np, ratio_valid_mask_np = self._build_ratio_target(
            row_index)
        ratio_target_tensor = torch.tensor(
            ratio_target_np, dtype=torch.float32)
        ratio_valid_mask_tensor = torch.tensor(
            ratio_valid_mask_np, dtype=torch.float32)

        height_value, height_mask = self._safe_float_value(
            row_index, "Height_Ave_cm")
        ndvi_value, ndvi_mask = self._safe_float_value(
            row_index, "Pre_GSHH_NDVI")
        state_index = self._state_label(row_index)

        batch_dict = {
            "left": left_tensor,
            "right": right_tensor,

            # main targets
            "target_log_5d": target_log_5d_tensor,
            "target_raw_5d": target_raw_5d_tensor,
            "target_total_log": total_log_value,
            # [Clover, Dead, Green]
            "target_ratio_soft": ratio_target_tensor,
            "target_ratio_valid_mask": ratio_valid_mask_tensor,

            # aux targets
            "target_height": torch.tensor(height_value, dtype=torch.float32),
            "target_height_mask": torch.tensor(height_mask, dtype=torch.float32),
            "target_ndvi": torch.tensor(ndvi_value, dtype=torch.float32),
            "target_ndvi_mask": torch.tensor(ndvi_mask, dtype=torch.float32),
            "target_state": torch.tensor(state_index, dtype=torch.long),
        }
        return batch_dict


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
    # folds作成後
    unique_state_values = sorted([
        str(state_value)
        for state_value in train_wide_df["State"].dropna().unique().tolist()
    ])
    state_to_index = {state_text: state_i for state_i,
                      state_text in enumerate(unique_state_values)}
    LOGGER.info("Num state classes: %d", len(state_to_index))

    for fold_i in range(cfg.exp.n_folds):
        LOGGER.info("======== Fold %d / %d ========", fold_i, cfg.exp.n_folds)

        train_df = train_wide_df[train_wide_df["fold"] != fold_i]
        val_df = train_wide_df[train_wide_df["fold"] == fold_i]

        train_loader = DataLoader(
            BiomassMultiTaskDataset(
                train_df,
                input_dir_path,
                get_transforms(cfg.exp.img_size, True),
                state_to_index=state_to_index,
                eps=cfg.exp.eps,
            ),
            batch_size=cfg.exp.batch_size,
            shuffle=True,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            BiomassMultiTaskDataset(
                val_df,
                input_dir_path,
                get_transforms(cfg.exp.img_size, False),
                state_to_index=state_to_index,
                eps=cfg.exp.eps,
            ),
            batch_size=cfg.exp.batch_size,
            shuffle=False,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )

        model = DualStreamDINOv3PyramidMTL(cfg, pretrained=True)
        model.set_num_state_classes(len(state_to_index))
        module = BiomassMTLModule(model, cfg)

        # ===== debug C: LoRA injection確認（fold 0のみ、毎fold同じ構造なので1回で十分） =====
        if fold_i == 0:
            log_lora_injection_summary(model.backbone, LOGGER)

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

        # accumulate_grad_batchesはmanual optimization内で自前実装済みのため
        # Trainerには渡さない (渡すとMisconfigurationExceptionになる)
        trainer = pl.Trainer(
            max_epochs=cfg.exp.num_epochs,
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
