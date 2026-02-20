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
    test_csv_path = input_dir_path / "test.csv"
    sample_submission_csv = input_dir_path / "sample_submission.csv"
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

    # OOFをcsvで保存


if __name__ == "__main__":
    main()
