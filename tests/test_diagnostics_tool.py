from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.check_r6_diagnostics import evaluate_diagnostics, load_metric_rows
from tools.analyze_r7_run import analyze
from tools.compare_r7_runs import compare_runs


ROOT = Path(__file__).resolve().parents[1]


def _row(iteration: int, **updates):
    row = {
        "iteration": iteration,
        "phase": "train",
        "candidate_foreground_ratio": 0.20,
        "safe_negative_pixel_ratio": 0.03,
        "background_hard_ratio": 0.10,
        "sam_valid_ratio": 1.0,
        "sam_kd_gate_ratio": 0.12,
        "sam_kd_gate_weight_mean": 0.08,
        "loss_sam_kd": 0.01,
    }
    row.update(updates)
    return row


def test_diagnostics_pass_for_healthy_sam_window():
    report = evaluate_diagnostics([_row(i) for i in range(1, 6)], config={"sam": {"use_sam": True}})

    assert report["status"] == "pass"
    assert {check["name"]: check["status"] for check in report["checks"]}["sam_kd_gate_ratio"] == "pass"


def test_diagnostics_fail_when_sam_expected_but_gate_is_zero():
    rows = [
        _row(i, sam_kd_gate_ratio=0.0, sam_kd_gate_weight_mean=0.0, loss_sam_kd=0.0)
        for i in range(1, 6)
    ]

    report = evaluate_diagnostics(rows, config={"sam": {"use_sam": True}})

    assert report["status"] == "fail"
    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert {"sam_kd_gate_ratio", "sam_kd_gate_weight_mean", "loss_sam_kd"} <= failed


def test_diagnostics_cli_reads_output_dir(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "resolved_config.yaml").write_text("sam:\n  use_sam: false\n", encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for i in range(1, 4):
            row = _row(i, sam_valid_ratio=0.0, sam_kd_gate_ratio=0.0, sam_kd_gate_weight_mean=0.0, loss_sam_kd=0.0)
            f.write(json.dumps(row) + "\n")

    rows = load_metric_rows(output / "metrics.jsonl")
    assert len(rows) == 3

    result = subprocess.run(
        [
            sys.executable,
            "tools/check_r6_diagnostics.py",
            "--output-dir",
            str(output),
            "--expect-sam",
            "no",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"status": "pass"' in result.stdout


def test_analyze_r7_run_reports_best_and_drop(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    rows = [
        {"iteration": 1, "phase": "train", "r6_unsup_scale": 0.0, "trust_unsafe": 0.0},
        {"iteration": 250, "phase": "val", "avg_dice": 0.70, "class_dice": [0.99, 0.60, 0.80]},
        {
            "iteration": 500,
            "phase": "val",
            "avg_dice": 0.60,
            "class_dice": [0.99, 0.40, 0.80],
            "foreground_pred_ratio": 0.02,
            "foreground_gt_ratio": 0.03,
            "class_pred_to_gt_ratio": [1.0, 0.5, 1.0],
        },
    ]
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = analyze(output, baseline_avg_dice=0.65)

    assert report["best_iteration"] == 250
    assert report["best_baseline_gap_avg_dice"] > 0
    assert report["worst_dropped_class"] == 1


def test_compare_r7_runs_flags_late_overexpansion_and_ssl_starvation(tmp_path):
    output = tmp_path / "run_a"
    output.mkdir()
    rows = [
        {"iteration": 100, "phase": "train", "r6_unsup_scale": 0.0, "sam_train_gate_ratio": 0.0},
        {
            "iteration": 250,
            "phase": "val",
            "avg_dice": 0.76,
            "class_dice": [0.99, 0.72, 0.80],
            "class_1_pred_to_gt_ratio": 1.10,
            "class_2_pred_to_gt_ratio": 1.05,
        },
        {"iteration": 500, "phase": "train", "r6_unsup_scale": 0.0, "sam_train_gate_ratio": 0.0},
        {
            "iteration": 500,
            "phase": "val",
            "avg_dice": 0.70,
            "class_dice": [0.99, 0.60, 0.75],
            "class_1_pred_to_gt_ratio": 1.50,
            "class_2_pred_to_gt_ratio": 1.35,
            "foreground_pred_ratio": 0.015,
            "foreground_gt_ratio": 0.010,
        },
    ]
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = compare_runs([output], baseline_avg_dice=0.7601364254951477)
    run = report["runs"][0]

    assert report["best_run"] == "run_a"
    assert "late_val_decline" in run["blockers"]
    assert "class1_overexpansion" in run["blockers"]
    assert "class1_dice_collapse" in run["blockers"]
    assert "best_before_effective_ssl" in run["blockers"]
    assert run["verdict"] == "ssl_not_contributing_to_peak"
    assert report["best_run_verdict"] == "ssl_not_contributing_to_peak"


def test_compare_r7_runs_reports_class2_below_baseline_verdict(tmp_path):
    output = tmp_path / "run_b"
    output.mkdir()
    rows = [
        {"iteration": 100, "phase": "train", "r6_unsup_scale": 0.02, "sam_train_gate_ratio": 0.02},
        {
            "iteration": 250,
            "phase": "val",
            "avg_dice": 0.755,
            "class_dice": [0.99, 0.73, 0.78],
            "class_1_pred_to_gt_ratio": 1.05,
            "class_2_pred_to_gt_ratio": 1.08,
        },
        {"iteration": 500, "phase": "train", "r6_unsup_scale": 0.03, "sam_train_gate_ratio": 0.02},
        {
            "iteration": 500,
            "phase": "val",
            "avg_dice": 0.752,
            "class_dice": [0.99, 0.72, 0.784],
            "class_1_pred_to_gt_ratio": 1.04,
            "class_2_pred_to_gt_ratio": 1.10,
        },
    ]
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = compare_runs([output], baseline_avg_dice=0.7601364254951477)
    run = report["runs"][0]

    assert "class2_below_baseline" in run["blockers"]
    assert run["c2_best_baseline_gap"] < 0
    assert run["verdict"] == "class2_quality_limited"
