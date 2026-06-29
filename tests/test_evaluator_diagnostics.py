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


def test_diagnostic_grid_writes_png(tmp_path):
    path = tmp_path / "diag.png"
    image = torch.zeros(3, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.long)
    mask[2:5, 2:5] = 1

    save_diagnostic_grid(image, [("mask", mask, "mask"), ("score", mask.float(), "heatmap")], path, num_classes=3)

    assert path.exists()
    assert path.stat().st_size > 0
