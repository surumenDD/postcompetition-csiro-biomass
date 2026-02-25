# /home/ryo52/workspace/postcompetition-csiro-biomass/tests/postcompetition/exp008_lora/test_train_components.py

import os
os.environ.setdefault("WANDB_MODE", "disabled")

import math
import numpy as np
import torch
import pandas as pd

# 実際の配置に合わせて import を調整
from postcompetition.exp008_lora.train import (
    _project_if_conflict,
    build_grouped_pcgrad_head_gradient,
    MultiTaskUncertaintyWeighting,
    BiomassMultiTaskDataset,
    DualStreamDINOv3PyramidMTL,
    Config,
    ExpConfig,
)


def test_project_if_conflict_removes_negative_component():
    source_grad = torch.tensor([1.0, -2.0], dtype=torch.float32)
    reference_grad = torch.tensor([0.0, 1.0], dtype=torch.float32)

    # 内積は -2.0 で衝突
    projected_grad = _project_if_conflict(source_grad, reference_grad, eps=1e-12)

    # 射影後は reference と直交（ほぼ0）
    dot_value = torch.dot(projected_grad, reference_grad).item()
    assert abs(dot_value) < 1e-6


def test_build_grouped_pcgrad_head_gradient_keeps_primary_and_projects_aux():
    primary_term_grad_map = {
        "primary_total": torch.tensor([1.0, 0.0]),
        "primary_ratio": torch.tensor([0.0, 1.0]),
    }
    # primary_sum = [1,1]
    auxiliary_term_grad_map = {
        # 内積 = -2.0 (衝突)
        "aux_height": torch.tensor([-1.0, -1.0]),
        # 内積 = +1.0 (衝突なし)
        "aux_ndvi": torch.tensor([1.0, 0.0]),
    }

    final_head_grad = build_grouped_pcgrad_head_gradient(
        primary_term_grad_map=primary_term_grad_map,
        auxiliary_term_grad_map=auxiliary_term_grad_map,
        eps=1e-12,
    )

    # aux_height は primary_sum に平行な負方向なので射影で消える
    # primary_sum=[1,1], aux_ndvi=[1,0] はそのまま加算
    expected = torch.tensor([2.0, 1.0])  # [1,1] + [1,0]
    assert torch.allclose(final_head_grad, expected, atol=1e-6)


def test_uncertainty_weighting_formula_matches_definition():
    task_names = ["task_a", "task_b"]
    module = MultiTaskUncertaintyWeighting(task_names)

    with torch.no_grad():
        module.log_variances[:] = torch.tensor([0.0, math.log(2.0)], dtype=torch.float32)

    raw_loss_a = torch.tensor(3.0)
    raw_loss_b = torch.tensor(4.0)

    term_a = module.make_weighted_term("task_a", raw_loss_a)
    term_b = module.make_weighted_term("task_b", raw_loss_b)

    # s_a=0 => exp(-0)*3 + 0 = 3
    assert torch.allclose(term_a, torch.tensor(3.0), atol=1e-6)

    # s_b=log2 => exp(-log2)*4 + log2 = 2 + log2
    expected_b = torch.tensor(2.0 + math.log(2.0), dtype=torch.float32)
    assert torch.allclose(term_b, expected_b, atol=1e-6)


def test_dataset_ratio_target_is_normalized_when_total_positive(tmp_path):
    # 画像は使わないので __getitem__ は呼ばず _build_ratio_target を直接使う
    df = pd.DataFrame({
        "image_path": ["dummy.png"],
        "State": ["A"],
        "Height_Ave_cm": [10.0],
        "Pre_GSHH_NDVI": [0.3],
        "Dry_Green_g": [2.0],
        "Dry_Clover_g": [3.0],
        "Dry_Dead_g": [5.0],
        "GDM_g": [5.0],         # Green + Clover (このテストでは値自体は重要でない)
        "Dry_Total_g": [10.0],
    })

    dataset = BiomassMultiTaskDataset(
        df=df,
        img_root=tmp_path,
        transform=None,  # __getitem__ を呼ばないので未使用
        state_to_index={"A": 0},
        eps=1e-6,
    )

    ratio_target, ratio_valid_mask = dataset._build_ratio_target(0)
    assert ratio_valid_mask == np.float32(1.0)
    assert np.isclose(ratio_target.sum(), 1.0, atol=1e-6)
    # 順序は [Clover, Dead, Green]
    assert np.allclose(ratio_target, np.array([0.3, 0.5, 0.2], dtype=np.float32), atol=1e-6)


def test_cross_entropy_all_ignore_would_be_nan_in_plain_pytorch():
    logits = torch.randn(4, 3)
    targets = torch.full((4,), -100, dtype=torch.long)

    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.isnan(loss), "PyTorch plain cross_entropy(all ignored) は NaN になることを確認する"