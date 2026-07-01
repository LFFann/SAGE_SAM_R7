from __future__ import annotations

import pytest
import torch

from r6.ssl.anatomical_copy_paste import build_labeled_foreground_copy_paste, copy_paste_replay_weight


def test_labeled_foreground_copy_paste_supervises_only_foreground():
    labeled = torch.zeros(1, 3, 4, 4)
    unlabeled = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[:, 1:3, 1:3] = 1
    labeled[:, :, 1:3, 1:3] = 0.7

    mixed, target, paste_mask, stats = build_labeled_foreground_copy_paste(
        labeled,
        mask,
        unlabeled,
        foreground_classes=[1, 2],
        ignore_index=255,
    )

    assert paste_mask.sum().item() == 4
    assert torch.allclose(mixed[:, :, 1:3, 1:3], labeled[:, :, 1:3, 1:3])
    assert torch.allclose(mixed[:, :, 0, 0], unlabeled[:, :, 0, 0])
    assert torch.equal(target[paste_mask], mask[paste_mask])
    assert torch.all(target[~paste_mask] == 255)
    assert stats["copy_paste_active"] == 1.0
    assert stats["copy_paste_class1_ratio"] > 0.0
    assert stats["copy_paste_class2_ratio"] == 0.0


def test_labeled_foreground_copy_paste_respects_area_bounds():
    labeled = torch.zeros(1, 3, 4, 4)
    unlabeled = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[:, 1, 1] = 1

    mixed, target, paste_mask, stats = build_labeled_foreground_copy_paste(
        labeled,
        mask,
        unlabeled,
        foreground_classes=[1],
        ignore_index=255,
        min_foreground_ratio=0.20,
    )

    assert paste_mask.sum().item() == 0
    assert torch.allclose(mixed, unlabeled)
    assert torch.all(target == 255)
    assert stats["copy_paste_active"] == 0.0


def test_copy_paste_replay_weight_boosts_when_foreground_coverage_is_low():
    weight, logs = copy_paste_replay_weight(
        1000,
        {
            "start_iter": 800,
            "weight": 0.12,
            "ramp_iterations": 400,
            "max_effective_weight": 0.16,
            "coverage_boost": {
                "enabled": True,
                "start_iter": 800,
                "min_class_ratio": [0.0, 0.004, 0.003],
                "max_boost": 1.5,
            },
        },
        {"per_class_foreground_participation_ratio": [0.0, 0.001, 0.0005]},
        foreground_classes=[1, 2],
    )

    base = (201 / 400) * 0.12
    assert weight > base
    assert logs["copy_paste_coverage_boost"] > 1.0
    assert logs["copy_paste_coverage_deficit"] > 0.0


def test_copy_paste_replay_weight_decays_coverage_boost_late():
    cfg = {
        "start_iter": 800,
        "weight": 0.12,
        "ramp_iterations": 400,
        "max_effective_weight": 0.16,
        "coverage_boost": {
            "enabled": True,
            "start_iter": 800,
            "min_class_ratio": [0.0, 0.004, 0.003],
            "max_boost": 1.5,
            "decay_start_iter": 1200,
            "decay_iterations": 400,
        },
    }
    stats = {"per_class_foreground_participation_ratio": [0.0, 0.0, 0.0]}

    early, early_logs = copy_paste_replay_weight(1200, cfg, stats, foreground_classes=[1, 2])
    late, late_logs = copy_paste_replay_weight(1600, cfg, stats, foreground_classes=[1, 2])

    assert early > late
    assert early_logs["copy_paste_coverage_boost"] > late_logs["copy_paste_coverage_boost"]
    assert late == pytest.approx(0.12)
