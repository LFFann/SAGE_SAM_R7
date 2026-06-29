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

    assert cfg["experiment"]["name"] == "SAGE_SAM_R7_3Class_V100_Tuned_MultiPrompt"
    assert cfg["data"]["root"] == "/root/autodl-tmp/echoData"
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
    assert cfg["pseudo"]["foreground_prior_cap_multiplier"] <= 1.6
    assert cfg["pseudo"]["foreground_prior_min_cap"] <= 0.006
    assert cfg["pseudo"]["empty_foreground_fallback_cap_scale"] <= 0.75
    assert cfg["pseudo"]["empty_candidate_recovery_cap_scale"] <= 0.75
    assert cfg["pseudo"]["max_foreground_candidate_ratio"] <= 0.05
    assert cfg["pseudo"]["max_safe_negative_ratio_per_class"] <= 0.25
    assert cfg["pseudo"]["max_fg_candidate_ratio_per_class"][1] <= 0.06
    assert cfg["pseudo"]["min_fg_pixels_per_class_ratio"][1] <= 0.006
    assert cfg["pseudo"]["min_fg_pixels_per_class_ratio"][2] >= 0.008
    assert cfg["pseudo"]["max_background_from_ceiling_ratio"] <= 0.10
    assert cfg["pseudo"]["background_candidate_min_confidence"] >= 0.65
    assert cfg["pseudo"]["sam_structure_mask_min_support"] >= 0.08
    assert cfg["pseudo"]["sam_train_gate_use_kd_agreement"] is True
    assert cfg["pseudo"]["sam_kd_require_teacher_agreement"] is True
    assert cfg["pseudo"]["sam_kd_allow_boundary_without_support"] is False
    assert cfg["pseudo"]["sam_kd_min_verifier_score"] >= 0.30
    assert cfg["sam"]["prompt"]["use_point_prompt"] is False
    assert cfg["sam"]["prompt"]["max_box_area_ratio"] <= 0.12
    assert cfg["sam"]["prompt"]["fallback_box_half_size"] <= 0.035
    assert cfg["sam"]["prompt"]["max_components_per_class"] >= 2
    assert cfg["sam"]["losses"]["sam_sup_weight"] <= 0.30
    assert cfg["sam"]["losses"]["sam_unsup_weight"] == 0.0
    assert cfg["sam"]["losses"]["sam_student_kd_weight"] <= 0.015
    assert cfg["trust"]["max_candidate_foreground_ratio"] <= 0.25
    assert cfg["trust"]["min_candidate_foreground_ratio"] <= 0.025
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
    assert 1200 <= cfg["r6"]["foreground_grounding_start"] <= 1600
    assert cfg["r6"]["stage1_unsup_max_scale"] <= 0.15
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
    assert "SAGE_SAM_R7_3Class_V100_Tuned_MultiPrompt" in train_script
    assert 'python train_r7.py "${train_args[@]}" "$@"' in train_script
    assert "validate_r7.py" in test_script
    assert "test_r7.py" in test_script
