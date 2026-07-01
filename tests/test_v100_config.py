from __future__ import annotations

from pathlib import Path

from r6.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_v100_tuned_config_matches_server_defaults():
    cfg = load_yaml(ROOT / "configs/r6_3class_v100_tuned.yaml")

    assert cfg["data"]["root"] == "/root/autodl-tmp/echoData"
    assert cfg["data"]["dataset_name"] == "260513_data_labeled30pct"
    assert cfg["data"]["image_size"] == 256
    assert cfg["sam"]["checkpoint"] == "/root/autodl-tmp/sam_vit_b_01ec64.pth"
    assert cfg["sam"]["use_sam"] is True
    assert cfg["sam"]["image_size"] == 1024
    assert cfg["train"]["device"] == "cuda"
    assert cfg["sam"]["device"] == "cuda"
    assert cfg["train"]["amp"] is True
    assert cfg["train"]["batch_size_labeled"] == 4
    assert cfg["train"]["batch_size_unlabeled"] == 4
    assert cfg["train"]["gradient_accumulation"] == 2
    assert cfg["train"]["num_workers"] == 8
    assert cfg["train"]["max_iterations"] == 8000
    assert cfg["train"]["warmup_iterations"] == 1200
    assert cfg["r6"]["foreground_grounding_start"] == 800
    assert cfg["r6"]["correlation_locality_start"] == 2000
    assert cfg["r6"]["self_reliance_start"] == 5000
    assert cfg["pseudo"]["collapse_sentinel_enabled"] is True
    assert cfg["pseudo"]["collapse_disable_background_hard"] is True
    assert cfg["pseudo"]["safe_negative_rank_low"] == 2
    assert cfg["pseudo"]["safe_negative_sam_threshold"] == 0.30


def test_r7_v100_config_uses_adapter_only_verifier_and_trust_gate():
    cfg = load_yaml(ROOT / "configs/r7_3class_v100_tuned.yaml")

    assert cfg["experiment"]["name"] == "SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP"
    assert cfg["data"]["root"] == "/root/autodl-tmp/echoData"
    assert cfg["train"]["lr_schedule"] == "cosine"
    assert cfg["train"]["lr_decay_start_iteration"] <= 750
    assert 0.0 < cfg["train"]["min_lr_ratio"] <= 0.15
    assert cfg["train"]["deploy_best_checkpoint"] is True
    assert cfg["train"]["stop_on_val_collapse"] is True
    assert cfg["sam"]["peft_type"] == "adapter"
    assert cfg["sam"]["train_mask_decoder"] is False
    assert cfg["sam"]["freeze_prompt_encoder"] is True
    assert cfg["pseudo"]["sam_role"] == "verifier"
    assert cfg["pseudo"]["bounded_safe_negative"] is True
    assert cfg["pseudo"]["bounded_foreground_candidates"] is True
    assert cfg["pseudo"]["use_labeled_foreground_prior"] is True
    assert cfg["pseudo"]["use_labeled_prior_distribution_alignment"] is True
    assert 0.0 < cfg["pseudo"]["prior_alignment_strength"] <= 0.5
    assert cfg["pseudo"]["bounded_empty_foreground_fallback"] is True
    assert cfg["pseudo"]["bounded_empty_candidate_recovery"] is True
    assert cfg["pseudo"]["prior_calibrated_foreground_budget"] is True
    assert cfg["pseudo"]["foreground_prior_cap_multiplier"] <= 1.35
    assert cfg["pseudo"]["foreground_prior_min_cap"][2] <= 0.0035
    assert cfg["pseudo"]["foreground_prior_min_ratio_multiplier"] <= 0.55
    assert cfg["pseudo"]["foreground_prior_collapse_force_multiplier"] <= 0.70
    assert cfg["pseudo"]["empty_foreground_fallback_cap_scale"] <= 0.75
    assert cfg["pseudo"]["empty_candidate_recovery_cap_scale"] <= 0.75
    assert cfg["pseudo"]["max_foreground_candidate_ratio"] <= 0.035
    assert cfg["pseudo"]["max_safe_negative_ratio_per_class"] <= 0.25
    assert cfg["pseudo"]["max_fg_candidate_ratio_per_class"][1] <= 0.04
    assert cfg["pseudo"]["min_fg_pixels_per_class_ratio"][1] <= 0.004
    assert cfg["pseudo"]["min_fg_pixels_per_class_ratio"][2] <= 0.003
    assert cfg["pseudo"]["collapse_min_fg_ratio_per_class"][2] <= 0.0022
    assert cfg["pseudo"]["collapse_force_fg_ratio_per_class"][2] <= 0.0032
    assert cfg["pseudo"]["max_background_from_ceiling_ratio"] <= 0.10
    assert cfg["pseudo"]["background_candidate_min_confidence"] >= 0.65
    assert cfg["pseudo"]["sam_structure_mask_min_support"] >= 0.08
    assert cfg["pseudo"]["sam_train_gate_use_kd_agreement"] is True
    assert cfg["pseudo"]["sam_kd_require_teacher_agreement"] is True
    assert cfg["pseudo"]["sam_kd_allow_boundary_without_support"] is False
    assert cfg["pseudo"]["sam_kd_min_verifier_score"] >= 0.30
    assert cfg["pseudo"]["sam_guided_pseudo_enabled"] is False
    assert cfg["pseudo"]["sam_guided_support_min"] >= 0.08
    assert cfg["pseudo"]["sam_guided_candidate_cap_scale"] <= 0.70
    assert cfg["pseudo"]["sam_guided_singleton_cap_scale"] <= 0.30
    assert cfg["pseudo"]["sam_disagreement_suppression_enabled"] is True
    assert cfg["pseudo"]["sam_disagreement_support_max"] <= 0.04
    assert cfg["pseudo"]["sam_disagreement_verifier_max"] <= 0.55
    assert cfg["pseudo"]["sam_disagreement_teacher_max"] <= 0.42
    assert cfg["sam"]["prompt"]["use_point_prompt"] is False
    assert cfg["sam"]["prompt"]["max_box_area_ratio"] <= 0.12
    assert cfg["sam"]["prompt"]["fallback_box_half_size"] <= 0.035
    assert cfg["sam"]["prompt"]["max_components_per_class"] == [0, 2, 1]
    assert cfg["sam"]["losses"]["sam_sup_weight"] <= 0.30
    assert cfg["sam"]["losses"]["sam_unsup_weight"] == 0.0
    assert cfg["sam"]["losses"]["sam_student_kd_weight"] <= 0.015
    assert cfg["sam"]["losses"]["sam_extent_weight"] == 0.0
    assert 0.50 <= cfg["sam"]["losses"]["sam_extent_target_mix"] <= 0.75
    assert 0.0 < cfg["sam"]["losses"]["sam_kd_min_effective_weight"] <= 0.00025
    assert cfg["sam"]["losses"]["sam_kd_min_effective_after"] >= 1200
    assert cfg["sam"]["losses"]["sam_kd_min_effective_gate_ratio"] >= 0.004
    assert 0.0 < cfg["sam"]["losses"]["sam_agreement_weight"] <= 0.04
    assert cfg["sam"]["losses"]["sam_agreement_min_support"] <= 0.06
    assert cfg["sam"]["losses"]["sam_agreement_min_verifier"] >= 0.45
    assert cfg["sam"]["losses"]["sam_agreement_min_effective_weight"] >= 0.001
    assert cfg["trust"]["max_candidate_foreground_ratio"] <= 0.25
    assert cfg["trust"]["min_candidate_foreground_ratio"] <= 0.025
    assert cfg["trust"]["prior_calibrated_min_foreground"] is True
    assert cfg["trust"]["prior_calibrated_max_foreground"] is True
    assert cfg["trust"]["min_candidate_prior_multiplier"] <= 0.85
    assert cfg["trust"]["min_candidate_foreground_floor"] <= 0.006
    assert cfg["trust"]["min_class_prior_multiplier"] <= 0.55
    assert cfg["trust"]["min_class_foreground_ratio"][1] <= 0.004
    assert cfg["trust"]["min_class_foreground_ratio"][2] <= 0.004
    assert cfg["trust"]["max_class_foreground_ratio"][1] <= 0.14
    assert cfg["trust"]["min_sam_foreground_support_ratio"] >= 0.006
    assert cfg["trust"]["max_sam_gate_without_support"] <= 0.030
    assert cfg["trust"]["max_sam_gate_to_support_ratio"] <= 3.0
    assert cfg["trust"]["low_support_sam_scale"] <= 0.10
    assert cfg["trust"]["max_pre_ceiling_foreground_ratio"][1] <= 0.12
    assert cfg["trust"]["enabled"] is True
    assert cfg["trust"]["disable_correlation_when_unsafe"] is True
    assert cfg["prior_feedback"]["enabled"] is True
    assert cfg["prior_feedback"]["start_iter"] >= 1200
    assert 0.0 < cfg["prior_feedback"]["loss_weight"] <= 0.10
    assert max(cfg["prior_feedback"]["max_class_prior_multiplier"][1:]) <= 1.25
    assert cfg["prior_feedback"]["max_class_prior_multiplier"][2] <= 1.15
    assert cfg["prior_feedback"]["max_foreground_prior_multiplier"] <= 1.22
    assert 0.5 <= cfg["prior_feedback"]["monitor_temperature"] <= 1.0
    assert 0.5 <= cfg["prior_feedback"]["loss_temperature"] <= 1.0
    assert cfg["prior_feedback"]["min_unsup_scale"] >= 0.40
    assert cfg["prior_feedback"]["min_sam_scale"] >= 0.70
    assert cfg["copy_paste"]["enabled"] is True
    assert cfg["copy_paste"]["start_iter"] >= cfg["r6"]["foreground_grounding_start"]
    assert 0.0 < cfg["copy_paste"]["weight"] <= 0.15
    assert cfg["copy_paste"]["max_foreground_ratio"] <= 0.06
    assert 1200 <= cfg["r6"]["foreground_grounding_start"] <= 1600
    assert 0.10 < cfg["r6"]["stage1_unsup_max_scale"] <= 0.20
    assert 0.06 < cfg["r6"]["stage1_sam_max_scale"] <= 0.10
    assert cfg["eval"]["baseline"]["avg_dice"] > 0.760
    assert cfg["diagnostics"]["train_visualize_every"] == 250


def test_v100_launch_scripts_are_parameterized():
    train_script = (ROOT / "scripts/train_r6_v100_tuned.sh").read_text(encoding="utf-8")
    test_script = (ROOT / "scripts/test_r6_v100_tuned.sh").read_text(encoding="utf-8")

    for token in ("CONFIG=", "OUTPUT_DIR=", "MAX_ITERATIONS=", "RESUME="):
        assert token in train_script
    assert "tools/validate_dataset.py" in train_script
    assert "tools/verify_real_sam.py" in train_script
    assert 'python train_r6.py "${train_args[@]}" "$@"' in train_script

    for token in ("OUTPUT_DIR=", "CONFIG=", "CHECKPOINT="):
        assert token in test_script
    assert "best_val_dice.pth" in test_script
    assert "latest.pth" in test_script
    assert "export_deploy_checkpoint.py" in test_script


def test_r7_launch_scripts_are_parameterized():
    train_script = (ROOT / "scripts/train_r7_v100_tuned.sh").read_text(encoding="utf-8")
    test_script = (ROOT / "scripts/test_r7_v100_tuned.sh").read_text(encoding="utf-8")

    assert "configs/r7_3class_v100_tuned.yaml" in train_script
    assert "SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP" in train_script
    assert "SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP" in test_script
    assert 'python train_r7.py "${train_args[@]}" "$@"' in train_script
    assert "validate_r7.py" in test_script
    assert "test_r7.py" in test_script
