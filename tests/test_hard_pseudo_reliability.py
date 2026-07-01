from __future__ import annotations

import torch

from r6.ssl.hard_pseudo_reliability import apply_hard_pseudo_reliability


def _targets() -> dict:
    candidate = torch.zeros(1, 3, 2, 2, dtype=torch.bool)
    candidate[:, 1, 0, 0] = True
    candidate[:, 2, 1, 0] = True
    singleton = torch.zeros(1, 2, 2, dtype=torch.bool)
    singleton[:, 0, 0] = True
    singleton[:, 0, 1] = True
    singleton[:, 1, 0] = True
    labels = torch.tensor([[[1, 0], [2, 0]]])
    teacher = torch.tensor(
        [[
            [[0.20, 0.74], [0.25, 0.90]],
            [[0.50, 0.13], [0.25, 0.05]],
            [[0.30, 0.13], [0.50, 0.05]],
        ]]
    )
    sam_support = torch.zeros_like(teacher)
    sam_support[:, 1, 0, 0] = 0.85
    sam_support[:, 2, 1, 0] = 0.02
    return {
        "singleton_label": labels,
        "singleton_mask": singleton,
        "candidate_set": candidate,
        "ambiguous_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "fuzzy_region": torch.zeros(1, 2, 2, dtype=torch.bool),
        "teacher_weight": teacher.max(dim=1).values,
        "teacher_only_soft_target": teacher,
        "sam_support": sam_support,
        "sam_verifier_score": torch.ones(1, 2, 2),
        "sam_foreground_support": sam_support[:, 1:].max(dim=1).values,
    }


def test_hard_pseudo_reliability_demotes_uncertain_foreground_singleton():
    out, stats = apply_hard_pseudo_reliability(
        _targets(),
        {
            "enabled": True,
            "start_iter": 1,
            "demote_low_foreground": True,
            "demote_confidence_threshold": 0.55,
            "demote_entropy_threshold": 0.55,
        },
        iteration=10,
    )

    assert stats["hard_pseudo_reliability_active"] == 1.0
    assert out["singleton_mask"][0, 1, 0] == 0
    assert out["ambiguous_mask"][0, 1, 0] == 1
    assert out["candidate_set"][0, 2, 1, 0] == 1
    assert out["singleton_weight"][0, 1, 0] == 0


def test_hard_pseudo_reliability_keeps_sam_supported_foreground_and_downweights_background():
    out, stats = apply_hard_pseudo_reliability(
        _targets(),
        {
            "enabled": True,
            "start_iter": 1,
            "background_scale": 0.5,
            "sam_bonus": 0.10,
            "demote_low_foreground": True,
        },
        iteration=10,
    )

    assert out["singleton_mask"][0, 0, 0] == 1
    assert out["singleton_weight"][0, 0, 0] > out["singleton_weight"][0, 0, 1]
    assert 0.0 < stats["singleton_weight_bg_mean"] < stats["singleton_weight_fg_mean"]
