from __future__ import annotations

import torch

from r6.engine.trainer import SAGESAMR6Trainer


class _FakeLabeledDataset:
    def __init__(self, masks):
        self.masks = masks
        self.records = [{"mask_path": idx} for idx in range(len(masks))]

    def _load_mask(self, path):
        return self.masks[int(path)]


def test_labeled_foreground_prior_tightens_candidate_caps(tmp_path):
    mask = torch.zeros(10, 10, dtype=torch.long)
    mask[0, 0] = 1
    mask[0, 1] = 2
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)
    trainer.num_classes = 3
    trainer.ignore_index = 255
    trainer.output_dir = tmp_path
    trainer.config = {
        "pseudo": {
            "use_labeled_foreground_prior": True,
            "foreground_prior_cap_multiplier": 4.0,
            "foreground_prior_min_cap": 0.005,
            "foreground_prior_max_cap": 0.05,
            "max_fg_candidate_ratio_per_class": [0.0, 0.12, 0.08],
        }
    }

    trainer._configure_labeled_foreground_prior(_FakeLabeledDataset([mask]))

    caps = trainer.config["pseudo"]["max_fg_candidate_ratio_per_class"]
    assert abs(caps[1] - 0.04) < 1e-6
    assert abs(caps[2] - 0.04) < 1e-6
    assert caps[1] < 0.12
    assert caps[2] < 0.08


def test_dynamic_trust_catches_pre_ceiling_flood_and_sam_overgate():
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)
    trainer.num_classes = 3
    trainer.config = {
        "pseudo": {"foreground_classes": [1, 2]},
        "trust": {
            "enabled": True,
            "start_iter": 1000,
            "min_candidate_foreground_ratio": 0.02,
            "max_candidate_foreground_ratio": 0.20,
            "min_class_foreground_ratio": 0.004,
            "max_class_foreground_ratio": [0.0, 0.08, 0.06],
            "max_pre_ceiling_foreground_ratio": [0.0, 0.30, 0.20],
            "min_sam_foreground_support_ratio": 0.02,
            "max_sam_gate_without_support": 0.50,
            "max_sam_gate_to_support_ratio": 6.0,
            "sam_support_ratio_floor": 0.005,
            "low_support_sam_scale": 0.0,
            "unsafe_unsup_scale": 0.05,
            "unsafe_sam_scale": 0.10,
            "unsafe_negative_scale": 0.0,
        },
    }
    targets = {
        "safe_negative_weight": torch.ones(1, 2, 2),
        "negative_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "stats": {
            "candidate_foreground_ratio": 0.05,
            "safe_negative_pixel_ratio": 0.02,
            "background_hard_ratio": 0.0,
            "per_class_foreground_participation_ratio": [0.0, 0.03, 0.02],
            "per_class_safe_negative_ratio": [0.0, 0.01, 0.01],
            "sam_foreground_support_ratio": 0.01,
            "sam_train_gate_ratio": 0.12,
            "foreground_ceiling_before_ratio_class1": 0.95,
            "foreground_ceiling_before_ratio_class2": 0.08,
        },
    }
    _, weights, logs = trainer._apply_dynamic_trust(
        2000,
        targets,
        {"unsup": 1.0, "sam": 1.0, "correlation": 1.0, "locality": 1.0},
    )

    assert logs["trust_unsafe"] == 1.0
    assert logs["trust_pre_ceiling_flood"] == 1.0
    assert logs["trust_sam_gate_too_wide"] == 1.0
    assert logs["trust_sam_overgate"] == 1.0
    assert weights["unsup"] == 0.05
    assert weights["sam"] == 0.0
