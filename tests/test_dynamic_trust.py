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


def test_labeled_foreground_prior_calibrates_pseudo_budget_by_class(tmp_path):
    mask = torch.zeros(10, 10, dtype=torch.long)
    mask[0, :2] = 1
    mask[1, 0] = 2
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)
    trainer.num_classes = 3
    trainer.ignore_index = 255
    trainer.output_dir = tmp_path
    trainer.config = {
        "pseudo": {
            "use_labeled_foreground_prior": True,
            "prior_calibrated_foreground_budget": True,
            "foreground_prior_cap_multiplier": 1.5,
            "foreground_prior_min_cap": [0.0, 0.005, 0.003],
            "foreground_prior_max_cap": [0.0, 0.05, 0.04],
            "foreground_prior_min_ratio_multiplier": 0.5,
            "foreground_prior_collapse_min_multiplier": 0.4,
            "foreground_prior_collapse_force_multiplier": 0.7,
            "min_fg_pixels_per_class_ratio": [0.0, 0.02, 0.02],
            "collapse_min_fg_ratio_per_class": 0.02,
            "collapse_force_fg_ratio_per_class": 0.02,
            "max_fg_candidate_ratio_per_class": [0.0, 0.12, 0.08],
        }
    }

    trainer._configure_labeled_foreground_prior(_FakeLabeledDataset([mask]))

    pseudo = trainer.config["pseudo"]
    assert pseudo["max_fg_candidate_ratio_per_class"][1] == 0.03
    assert pseudo["max_fg_candidate_ratio_per_class"][2] == 0.015
    assert pseudo["min_fg_pixels_per_class_ratio"][1] == 0.01
    assert pseudo["min_fg_pixels_per_class_ratio"][2] == 0.005
    assert pseudo["collapse_min_fg_ratio_per_class"][1] == 0.008
    assert pseudo["collapse_min_fg_ratio_per_class"][2] == 0.004
    assert abs(pseudo["collapse_force_fg_ratio_per_class"][1] - 0.014) < 1e-8
    assert abs(pseudo["collapse_force_fg_ratio_per_class"][2] - 0.007) < 1e-8


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


def test_dynamic_trust_scales_sam_when_support_is_low_even_without_overgate():
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
            "max_safe_negative_pixel_ratio": 0.65,
            "max_class_safe_negative_ratio": 0.50,
            "max_background_hard_ratio": 0.45,
            "min_sam_foreground_support_ratio": 0.006,
            "max_sam_gate_without_support": 0.03,
            "max_sam_gate_to_support_ratio": 10.0,
            "sam_support_ratio_floor": 0.004,
            "low_support_sam_scale": 0.10,
            "unsafe_unsup_scale": 0.05,
            "unsafe_sam_scale": 0.10,
            "unsafe_negative_scale": 0.0,
        },
    }
    targets = {
        "stats": {
            "candidate_foreground_ratio": 0.04,
            "safe_negative_pixel_ratio": 0.02,
            "background_hard_ratio": 0.0,
            "per_class_foreground_participation_ratio": [0.0, 0.02, 0.02],
            "per_class_safe_negative_ratio": [0.0, 0.01, 0.01],
            "sam_foreground_support_ratio": 0.003,
            "sam_train_gate_ratio": 0.02,
            "foreground_ceiling_before_ratio_class1": 0.02,
            "foreground_ceiling_before_ratio_class2": 0.02,
        },
    }

    _, weights, logs = trainer._apply_dynamic_trust(
        2000,
        targets,
        {"unsup": 1.0, "sam": 1.0, "correlation": 1.0, "locality": 1.0},
    )

    assert logs["trust_low_sam_support"] == 1.0
    assert logs["trust_sam_overgate"] == 0.0
    assert weights["unsup"] == 1.0
    assert weights["sam"] == 0.10


def test_sam_floor_scale_respects_trust_and_blocks_overgate():
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)

    scale, logs = trainer._sam_floor_scale_from_trust(
        {
            "trust_conditioned_floor": True,
            "floor_unsafe_scale": 0.5,
            "floor_block_on_low_sam_support": True,
            "floor_block_on_sam_overgate": True,
        },
        {
            "trust_sam_scale": 0.1,
            "trust_unsafe": 1.0,
            "trust_low_sam_support": 0.0,
            "trust_sam_overgate": 0.0,
            "trust_sam_gate_too_wide": 0.0,
        },
    )

    assert scale == 0.05
    assert logs["sam_floor_trust_conditioned"] == 1.0
    assert logs["sam_floor_unsafe"] == 1.0
    assert logs["sam_floor_blocked"] == 0.0

    blocked_scale, blocked_logs = trainer._sam_floor_scale_from_trust(
        {"trust_conditioned_floor": True, "floor_block_on_sam_overgate": True},
        {
            "trust_sam_scale": 1.0,
            "trust_unsafe": 0.0,
            "trust_low_sam_support": 0.0,
            "trust_sam_overgate": 0.0,
            "trust_sam_gate_too_wide": 1.0,
        },
    )

    assert blocked_scale == 0.0
    assert blocked_logs["sam_floor_blocked"] == 1.0
    assert blocked_logs["sam_floor_blocked_overgate"] == 1.0


def test_sam_floor_scale_can_be_disabled_for_ablation():
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)

    scale, logs = trainer._sam_floor_scale_from_trust(
        {"trust_conditioned_floor": False},
        {"trust_sam_scale": 0.0, "trust_low_sam_support": 1.0},
    )

    assert scale == 1.0
    assert logs["sam_floor_trust_conditioned"] == 0.0


def test_student_prior_feedback_downscales_unsup_when_student_foreground_drifts():
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)
    trainer.num_classes = 3
    trainer.student_prior_ema = None
    trainer.config = {
        "pseudo": {
            "foreground_classes": [1, 2],
            "labeled_class_prior": [0.9900, 0.0060, 0.0040],
        },
        "prior_feedback": {
            "enabled": True,
            "start_iter": 1500,
            "ema_decay": 0.0,
            "max_class_prior_multiplier": 1.25,
            "max_foreground_prior_multiplier": 1.25,
            "min_class_prior_multiplier": 0.45,
            "min_unsup_scale": 0.45,
            "drift_scale_strength": 3.0,
            "sam_scale_coupling": 0.35,
            "min_sam_scale": 0.70,
        },
    }
    prob = torch.zeros(2, 3, 4, 4)
    prob[:, 0] = 0.96
    prob[:, 1] = 0.03
    prob[:, 2] = 0.01

    weights, logs = trainer._apply_student_prior_feedback(
        2000,
        prob,
        {"unsup": 1.0, "sam": 1.0, "correlation": 1.0, "locality": 1.0},
    )

    assert logs["prior_feedback_active"] == 1.0
    assert logs["prior_feedback_drift"] > 0.0
    assert weights["unsup"] < 1.0
    assert weights["sam"] < 1.0
    assert weights["correlation"] == weights["unsup"]


def test_supervised_class_weights_use_labeled_prior_without_pseudo_labels():
    trainer = SAGESAMR6Trainer.__new__(SAGESAMR6Trainer)
    trainer.num_classes = 3
    trainer.config = {
        "pseudo": {
            "foreground_classes": [1, 2],
            "labeled_class_prior": [0.9900, 0.0060, 0.0040],
        },
        "losses": {
            "class_balanced_ce": {
                "enabled": True,
                "background_weight": 0.25,
                "foreground_scale": 1.15,
                "foreground_power": 0.5,
                "min_foreground_weight": 0.90,
                "max_foreground_weight": 1.35,
            }
        },
    }

    weights, logs = trainer._supervised_class_weights(torch.device("cpu"), torch.float32)

    assert logs["class_balanced_ce_active"] == 1.0
    assert float(weights[0]) == 0.25
    assert float(weights[2]) > float(weights[1])
    assert float(weights[2]) <= 1.35
