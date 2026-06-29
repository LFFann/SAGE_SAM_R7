from __future__ import annotations

import torch

from r6.calibration.prompt_reliability_calibrator import PromptReliabilityCalibrator
from r6.ssl.target_builder import build_set_valued_targets


def test_target_builder_always_returns_set_and_safe_negative_shapes():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True, min_participation_ratio=0.25)
    cal.teacher_q = torch.tensor([0.95, 0.95, 0.95])
    cal.sam_q = torch.tensor([0.95, 0.95, 0.95])
    teacher_prob = torch.full((2, 3, 5, 5), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {"max_candidate_set_size": 2, "safe_negative_threshold": 0.05, "min_teacher_confidence": 0.5},
    )

    assert targets["candidate_set"].shape == teacher_prob.shape
    assert targets["safe_negative_set"].shape == teacher_prob.shape
    assert targets["candidate_set"].sum(dim=1).min() >= 1
    assert targets["sam_train_gate"].any()
    assert torch.isfinite(targets["candidate_weight"]).all()
    assert "safe_negative_pixel_ratio" in targets["stats"]
    assert len(targets["stats"]["per_class_safe_negative_ratio"]) == 3


def test_calibrator_coverage_fallback_keeps_soft_participation_nonzero():
    cal = PromptReliabilityCalibrator(
        2,
        min_pixels_per_class=1,
        use_soft_gate=True,
        min_participation_ratio=0.50,
        coverage_target=0.50,
        temperature=0.05,
    )
    cal.teacher_q = torch.tensor([1.0, 1.0])
    cal.sam_q = torch.tensor([1.0, 1.0])
    prob = torch.tensor([[[[0.60, 0.55], [0.50, 0.45]], [[0.40, 0.45], [0.50, 0.55]]]])

    gates = cal.gates(prob, prob)

    assert float(gates["sam_train_weight"].mean()) > 0.05
    assert float(gates["sam_train_gate"].float().mean()) >= 0.50


def test_r6_sam_foreground_support_does_not_create_background_hard_label():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 4, 4), 0.05)
    teacher_prob[:, 0] = 0.90
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.95

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_grounding_start": 1200,
            "disable_background_unsup_until": 1200,
            "foreground_classes": [1, 2],
            "min_teacher_confidence": 0.5,
            "min_sam_confidence": 0.5,
        },
    )

    assert targets["stats"]["background_hard_ratio"] == 0.0
    assert targets["candidate_set"][:, 0].sum() == 0
    assert targets["candidate_set"][:, 1].sum() > 0
    assert targets["sam_train_gate"].any()


def test_r6_emergency_mode_disables_background_when_foreground_absent():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 4, 4), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 0] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_grounding_start": 1200,
            "disable_background_unsup_until": 1200,
            "foreground_classes": [1, 2],
            "min_teacher_confidence": 0.5,
            "min_sam_confidence": 0.5,
            "disable_bg_if_no_fg": True,
            "collapse_sentinel_enabled": False,
        },
    )

    assert targets["stats"]["emergency_mode"] == 1.0
    assert targets["stats"]["background_hard_ratio"] == 0.0
    assert targets["singleton_mask"].sum() == 0
    assert targets["candidate_set"][:, 0].sum() == 0
    assert targets["candidate_set"].sum() == 0


def test_r6_rank_negative_keeps_unreliable_pixels_useful_with_weak_sam_veto():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    teacher_prob = torch.tensor(
        [[
            [[0.65, 0.65], [0.65, 0.65]],
            [[0.30, 0.30], [0.30, 0.30]],
            [[0.05, 0.05], [0.05, 0.05]],
        ]]
    )
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.10
    sam_prob[:, 2] = 0.02

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_classes": [1, 2],
            "disable_bg_if_no_fg": True,
            "empty_candidate_topk_foreground": 1,
            "min_empty_foreground_score": 0.02,
            "safe_negative_rank_low": 2,
            "safe_negative_sam_threshold": 0.30,
            "safe_negative_max_prob": 0.35,
        },
    )

    assert targets["candidate_set"][:, 1].any()
    assert targets["safe_negative_set"][:, 2].any()
    assert targets["stats"]["safe_negative_pixel_ratio"] > 0.0


def test_r7_bounded_negative_caps_rare_class_suppression():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    teacher_prob = torch.full((1, 3, 10, 10), 0.01)
    teacher_prob[:, 0] = 0.90
    teacher_prob[:, 1] = 0.09
    teacher_prob[:, 2] = 0.01
    teacher_prob[:, 1, :2, :2] = 0.65
    teacher_prob[:, 0, :2, :2] = 0.30
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1, :2, :2] = 0.80

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob, "prompt_quality": torch.ones(1, 3), "sam_iou": torch.ones(1, 3)},
        cal,
        {
            "_iteration": 2000,
            "foreground_classes": [1, 2],
            "sam_role": "verifier",
            "min_sam_verifier_score": 0.20,
            "bounded_safe_negative": True,
            "safe_negative_to_positive_ratio": 1.0,
            "max_safe_negative_ratio_per_class": 0.10,
            "safe_negative_rank_low": 2,
            "safe_negative_sam_threshold": 0.30,
            "safe_negative_max_prob": 0.35,
            "min_fg_pixels_per_class_ratio": 0.02,
            "disable_bg_if_no_fg": True,
        },
    )

    assert targets["stats"]["safe_negative_budget_active"] == 1.0
    assert targets["stats"]["safe_negative_ratio_class2"] <= 0.10
    assert targets["stats"]["safe_negative_ratio_class2"] < targets["stats"]["safe_negative_raw_ratio_class2"]
    assert targets["stats"]["per_class_foreground_participation_ratio"][2] >= 0.0199


def test_r7_foreground_ceiling_blocks_class1_candidate_flooding():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    teacher_prob = torch.full((1, 3, 10, 10), 0.02)
    teacher_prob[:, 0] = 0.35
    teacher_prob[:, 1] = 0.60
    teacher_prob[:, 2] = 0.05
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.02

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob, "prompt_quality": torch.ones(1, 3), "sam_iou": torch.ones(1, 3)},
        cal,
        {
            "_iteration": 1800,
            "foreground_classes": [1, 2],
            "sam_role": "verifier",
            "bounded_foreground_candidates": True,
            "max_fg_candidate_ratio_per_class": [0.0, 0.12, 0.08],
            "min_fg_pixels_per_class_ratio": 0.01,
            "use_background_from_foreground_ceiling": True,
            "background_candidate_min_confidence": 0.30,
            "max_background_from_ceiling_ratio": 0.25,
            "bounded_safe_negative": True,
            "safe_negative_to_positive_ratio": 1.0,
            "max_safe_negative_ratio_per_class": 0.10,
        },
    )

    assert targets["stats"]["foreground_ceiling_active"] == 1.0
    assert targets["stats"]["foreground_ceiling_flood_class_count"] >= 1.0
    assert targets["stats"]["soft_fg_ratio_class1"] <= 0.12
    assert targets["stats"]["candidate_foreground_ratio"] <= 0.20
    assert targets["stats"]["background_from_ceiling_ratio"] > 0.0


def test_r7_prior_alignment_and_bounded_recovery_stop_pre_ceiling_flood():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.90, 0.90, 0.90])
    teacher_prob = torch.full((1, 3, 10, 10), 0.05)
    teacher_prob[:, 0] = 0.35
    teacher_prob[:, 1] = 0.60
    sam_prob = torch.full_like(teacher_prob, 0.01)

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob, "prompt_quality": torch.ones(1, 3), "sam_iou": torch.ones(1, 3)},
        cal,
        {
            "_iteration": 1500,
            "foreground_classes": [1, 2],
            "disable_background_unsup_until": 9999,
            "sam_role": "verifier",
            "use_labeled_prior_distribution_alignment": True,
            "labeled_class_prior": [0.989, 0.006, 0.005],
            "prior_alignment_strength": 0.35,
            "prior_alignment_min_ratio": 0.10,
            "prior_alignment_max_ratio": 5.0,
            "bounded_empty_foreground_fallback": True,
            "bounded_empty_candidate_recovery": True,
            "bounded_foreground_candidates": True,
            "max_fg_candidate_ratio_per_class": [0.0, 0.06, 0.04],
            "max_foreground_candidate_ratio": 0.06,
            "min_fg_pixels_per_class_ratio": 0.01,
            "min_empty_foreground_score": 0.01,
            "use_background_from_foreground_ceiling": False,
            "bounded_safe_negative": True,
            "safe_negative_to_positive_ratio": 1.0,
            "max_safe_negative_ratio_per_class": 0.10,
        },
    )

    stats = targets["stats"]
    assert stats["prior_alignment_active"] == 1.0
    assert stats["prior_alignment_after_mean_class1"] < stats["prior_alignment_before_mean_class1"]
    assert stats["empty_candidate_recovery_raw_ratio"] > stats["empty_candidate_recovered_ratio"]
    assert stats["bounded_empty_candidate_recovery_active"] == 1.0
    assert stats["foreground_ceiling_before_ratio_class1"] <= 0.061
    assert stats["candidate_foreground_ratio"] <= 0.101


def test_r7_verifier_score_alone_does_not_open_global_sam_gate():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    teacher_prob = torch.full((1, 3, 10, 10), 0.02)
    teacher_prob[:, 0] = 0.38
    teacher_prob[:, 1] = 0.60
    teacher_prob[:, 2] = 0.02
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.02

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob, "prompt_quality": torch.ones(1, 3), "sam_iou": torch.ones(1, 3)},
        cal,
        {
            "_iteration": 1800,
            "foreground_classes": [1, 2],
            "sam_role": "verifier",
            "min_sam_verifier_score": 0.20,
            "sam_foreground_low": 0.12,
            "sam_structure_mask_min_support": 0.08,
            "bounded_foreground_candidates": True,
            "max_fg_candidate_ratio_per_class": [0.0, 0.12, 0.08],
            "min_fg_pixels_per_class_ratio": 0.01,
            "use_background_from_foreground_ceiling": True,
            "background_candidate_min_confidence": 0.70,
            "max_background_from_ceiling_ratio": 0.08,
        },
    )

    assert targets["stats"]["sam_verifier_gate_ratio"] >= 0.99
    assert targets["stats"]["sam_foreground_support_ratio"] == 0.0
    assert targets["stats"]["sam_train_gate_ratio"] <= 0.13
    assert targets["stats"]["sam_structure_support_mask_ratio"] <= 0.13
    assert float(targets["sam_train_gate"].float().mean()) < float(targets["sam_verifier_score"].ge(0.20).float().mean())


def test_r7_sam_kd_gate_requires_classwise_sam_teacher_agreement():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 10, 10), 0.02)
    teacher_prob[:, 0] = 0.38
    teacher_prob[:, 1] = 0.60
    teacher_prob[:, 2] = 0.02
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.02

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob, "prompt_quality": torch.ones(1, 3), "sam_iou": torch.ones(1, 3)},
        cal,
        {
            "_iteration": 1800,
            "foreground_classes": [1, 2],
            "sam_role": "verifier",
            "bounded_foreground_candidates": True,
            "max_fg_candidate_ratio_per_class": [0.0, 0.12, 0.08],
            "min_fg_pixels_per_class_ratio": 0.01,
            "sam_train_gate_use_kd_agreement": True,
            "sam_kd_min_support": 0.08,
            "sam_kd_min_teacher_confidence": 0.03,
            "sam_kd_min_verifier_score": 0.30,
        },
    )

    assert targets["stats"]["sam_region_gate_ratio"] > 0.0
    assert targets["stats"]["sam_kd_agreement_gate_ratio"] == 0.0
    assert targets["stats"]["sam_train_gate_ratio"] == 0.0
    assert targets["sam_kd_gate"].sum() == 0
    assert targets["sam_kd_weight"].sum() == 0


def test_r6_collapse_sentinel_blocks_background_takeover_and_forces_fg_candidates():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 10, 10), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 0] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_grounding_start": 800,
            "disable_background_unsup_until": 800,
            "foreground_classes": [1, 2],
            "min_teacher_confidence": 0.5,
            "min_sam_confidence": 0.5,
            "disable_bg_if_no_fg": True,
            "collapse_sentinel_enabled": True,
            "collapse_min_fg_ratio_per_class": 0.05,
            "collapse_force_fg_ratio_per_class": 0.05,
            "collapse_max_background_hard_ratio": 0.20,
            "collapse_disable_background_hard": True,
        },
    )

    assert targets["stats"]["collapse_sentinel_active"] == 1.0
    assert targets["stats"]["collapse_disabled_background"] == 1.0
    assert targets["stats"]["background_hard_ratio"] == 0.0
    assert targets["stats"]["collapse_forced_fg_ratio"] > 0.0
    assert targets["candidate_set"][:, 0].sum() == 0
    assert targets["candidate_set"][:, 1:].any()
    assert targets["fuzzy_region"].any()
