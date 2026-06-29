from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.check_r6_diagnostics import evaluate_diagnostics, load_metric_rows
from tools.analyze_r7_run import analyze


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
