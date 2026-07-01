from __future__ import annotations

import torch

from r6.losses.prototype_anchor import foreground_prototype_anchor_loss


def _targets():
    candidate_set = torch.zeros(1, 3, 2, 2, dtype=torch.bool)
    candidate_set[:, 1, 0, 0] = True
    candidate_set[:, 2, 0, 1] = True
    candidate_set[:, 1, 1, 0] = True
    candidate_set[:, 2, 1, 0] = True
    soft_target = torch.zeros(1, 3, 2, 2)
    soft_target[:, 1, 0, 0] = 1.0
    soft_target[:, 2, 0, 1] = 1.0
    soft_target[:, 1, 1, 0] = 0.5
    soft_target[:, 2, 1, 0] = 0.5
    return {
        "candidate_set": candidate_set,
        "candidate_weight": torch.ones(1, 2, 2),
        "soft_target": soft_target,
    }


def test_foreground_prototype_anchor_prefers_matching_class_features():
    labeled_feature = torch.zeros(1, 2, 2, 2)
    labeled_feature[:, 0, 0, 0] = 1.0
    labeled_feature[:, 1, 0, 1] = 1.0
    labeled_mask = torch.zeros(1, 2, 2, dtype=torch.long)
    labeled_mask[:, 0, 0] = 1
    labeled_mask[:, 0, 1] = 2
    good_unlabeled = labeled_feature.clone()
    bad_unlabeled = torch.flip(good_unlabeled, dims=[1])

    good_loss, good_stats = foreground_prototype_anchor_loss(
        labeled_feature,
        labeled_mask,
        good_unlabeled,
        _targets(),
        foreground_classes=[1, 2],
        min_labeled_pixels=1,
        min_unlabeled_pixels=1,
        entropy_discount=0.0,
    )
    bad_loss, _ = foreground_prototype_anchor_loss(
        labeled_feature,
        labeled_mask,
        bad_unlabeled,
        _targets(),
        foreground_classes=[1, 2],
        min_labeled_pixels=1,
        min_unlabeled_pixels=1,
        entropy_discount=0.0,
    )

    assert good_stats["prototype_anchor_active"] == 1.0
    assert good_stats["prototype_anchor_valid_class_count"] == 2.0
    assert float(good_loss) < float(bad_loss)


def test_foreground_prototype_anchor_skips_missing_labeled_class():
    labeled_feature = torch.randn(1, 2, 2, 2)
    labeled_mask = torch.zeros(1, 2, 2, dtype=torch.long)

    loss, stats = foreground_prototype_anchor_loss(
        labeled_feature,
        labeled_mask,
        torch.randn(1, 2, 2, 2),
        _targets(),
        foreground_classes=[1, 2],
    )

    assert float(loss) == 0.0
    assert stats["prototype_anchor_active"] == 0.0
