from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from r6.engine.evaluator import evaluate
from r6.utils.visualization import save_diagnostic_grid


class _TinySegDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 4, 4),
            "mask": torch.tensor(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 2, 2, 2],
                    [0, 2, 2, 2],
                ],
                dtype=torch.long,
            ),
            "id": "case0",
        }


class _FixedModel(torch.nn.Module):
    def forward(self, image):
        logits = torch.zeros(image.shape[0], 3, image.shape[-2], image.shape[-1], device=image.device)
        logits[:, 0] = 3.0
        logits[:, 1, :2, 2:] = 5.0
        logits[:, 2, 2:, 1:] = 5.0
        return logits


class _TopologyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        mask = torch.zeros(6, 6, dtype=torch.long)
        mask[:2, :2] = 1
        mask[:2, 4:] = 1
        mask[4:, 2:5] = 2
        return {"image": torch.zeros(3, 6, 6), "mask": mask, "id": "topology_case"}


class _TopologyOverPredictModel(torch.nn.Module):
    def forward(self, image):
        logits = torch.zeros(image.shape[0], 3, image.shape[-2], image.shape[-1], device=image.device)
        logits[:, 0] = 3.0
        logits[:, 1, :2, :2] = 7.0
        logits[:, 1, :2, 4:] = 7.0
        logits[:, 1, 3, 0] = 5.0
        logits[:, 2, 4:, 2:5] = 7.0
        logits[:, 2, 0, 3] = 5.0
        return logits


def test_evaluator_reports_area_drift_metrics():
    metrics = evaluate(
        _FixedModel(),
        DataLoader(_TinySegDataset(), batch_size=1),
        num_classes=3,
        device="cpu",
        compute_hd95=False,
    )

    assert metrics["class_pred_ratio"][1] == metrics["class_gt_ratio"][1]
    assert metrics["class_2_pred_to_gt_ratio"] == 1.0
    assert "foreground_area_abs_error" in metrics
    assert "class_overseg_ratio" in metrics


def test_evaluator_topology_postprocess_removes_extra_components():
    metrics = evaluate(
        _TopologyOverPredictModel(),
        DataLoader(_TopologyDataset(), batch_size=1),
        num_classes=3,
        device="cpu",
        compute_hd95=False,
        topology_postprocess={
            "enabled": True,
            "max_components_per_class": [0, 2, 1],
            "min_component_area": 1,
        },
    )

    assert metrics["topology_postprocess_active"] == 1.0
    assert metrics["topology_removed_ratio_class1"] > 0.0
    assert metrics["topology_removed_ratio_class2"] > 0.0
    assert metrics["topology_dropped_components_class1"] == 1.0
    assert metrics["topology_dropped_components_class2"] == 1.0
    assert metrics["class_1_pred_to_gt_ratio"] == 1.0
    assert metrics["class_2_pred_to_gt_ratio"] == 1.0


def test_diagnostic_grid_writes_png(tmp_path):
    path = tmp_path / "diag.png"
    image = torch.zeros(3, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.long)
    mask[2:5, 2:5] = 1

    save_diagnostic_grid(image, [("mask", mask, "mask"), ("score", mask.float(), "heatmap")], path, num_classes=3)

    assert path.exists()
    assert path.stat().st_size > 0


def test_diagnostic_grid_writes_prompt_overlay(tmp_path):
    path = tmp_path / "prompt_diag.png"
    image = torch.zeros(3, 16, 16)
    prompts = {
        "boxes_xyxy": torch.tensor([[0.20, 0.20, 0.60, 0.60], [0.70, 0.70, 0.76, 0.76]]),
        "point_coords": torch.tensor([[[0.40, 0.40]], [[0.73, 0.73]]]),
        "image_index": torch.tensor([0, 0]),
        "class_ids": torch.tensor([1, 2]),
        "prompt_valid": torch.tensor([[1.0, 1.0, 0.0]]),
    }

    save_diagnostic_grid(
        image,
        [("sam_prompt", {"prompts": prompts, "sample_index": 0}, "prompt_overlay")],
        path,
        num_classes=3,
    )

    assert path.exists()
    assert path.stat().st_size > 0
