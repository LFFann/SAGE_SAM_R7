from __future__ import annotations

import torch

from r6.ssl.anatomical_copy_paste import build_labeled_foreground_copy_paste


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
