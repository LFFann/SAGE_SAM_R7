from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row, key) for row in rows if row.get(key) is not None]
    return float(mean(values)) if values else 0.0


def _window_summary(rows: list[dict[str, Any]], lo: int, hi: int) -> dict[str, Any]:
    win = [row for row in rows if lo <= int(row.get("iteration", 0)) <= hi]
    keys = [
        "r6_unsup_scale",
        "trust_unsafe",
        "trust_pre_ceiling_flood",
        "candidate_foreground_ratio",
        "soft_fg_ratio_class1",
        "soft_fg_ratio_class2",
        "sam_region_gate_ratio",
        "sam_kd_agreement_gate_ratio",
        "sam_train_gate_ratio",
        "sam_foreground_support_ratio",
        "sam_gate_to_support_ratio",
        "sam_region_to_support_ratio",
        "sam_kd_effective_weight",
        "prior_feedback_active",
        "prior_feedback_drift",
        "prior_feedback_unsup_scale",
        "prior_feedback_sam_scale",
        "prior_feedback_student_fg_ratio",
        "prior_feedback_fg_over",
        "loss_prior_feedback",
        "prior_feedback_effective_weight",
        "loss_copy_paste",
        "copy_paste_effective_weight",
        "copy_paste_active",
        "copy_paste_fg_ratio",
        "class_balanced_ce_active",
        "class_balanced_ce_weight_class1",
        "class_balanced_ce_weight_class2",
        "loss_sup_boundary",
        "loss_sam_kd",
        "sam_adapter_grad_norm",
        "prompt_quality",
        "sam_prompt_valid_mean",
        "sam_prompt_valid_class1",
        "sam_prompt_valid_class2",
        "sam_prompt_area_ratio_mean",
        "sam_prompt_area_ratio_class1",
        "sam_prompt_area_ratio_class2",
        "sam_prompt_box_area_ratio_mean",
        "sam_prompt_box_area_ratio_class1",
        "sam_prompt_box_area_ratio_class2",
        "sam_prompt_component_count_mean",
        "sam_prompt_component_count_class1",
        "sam_prompt_component_count_class2",
    ]
    return {"range": [lo, hi], "rows": len(win), **{key: _mean(win, key) for key in keys}}


def analyze(output_dir: Path, baseline_avg_dice: float | None = None) -> dict[str, Any]:
    rows = _load_rows(output_dir / "metrics.jsonl")
    train = [row for row in rows if row.get("phase") == "train"]
    val = [row for row in rows if row.get("phase") == "val"]
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "train_rows": len(train),
        "val_rows": len(val),
    }
    if val:
        best = max(val, key=lambda row: _float(row, "avg_dice", -1.0))
        last = val[-1]
        best_class = best.get("class_dice", [])
        last_class = last.get("class_dice", [])
        class_drop = []
        for idx, value in enumerate(best_class):
            if idx < len(last_class):
                class_drop.append(float(value) - float(last_class[idx]))
        report.update(
            {
                "best_iteration": best.get("iteration"),
                "best_avg_dice": best.get("avg_dice"),
                "best_class_dice": best_class,
                "last_iteration": last.get("iteration"),
                "last_avg_dice": last.get("avg_dice"),
                "last_class_dice": last_class,
                "best_to_last_class_dice_drop": class_drop,
                "worst_dropped_class": int(max(range(len(class_drop)), key=lambda i: class_drop[i])) if class_drop else None,
                "last_foreground_pred_ratio": last.get("foreground_pred_ratio"),
                "last_foreground_gt_ratio": last.get("foreground_gt_ratio"),
                "last_class_pred_to_gt_ratio": last.get("class_pred_to_gt_ratio"),
                "last_class_underseg_ratio": last.get("class_underseg_ratio"),
                "last_class_overseg_ratio": last.get("class_overseg_ratio"),
            }
        )
        if baseline_avg_dice is not None:
            report["best_baseline_gap_avg_dice"] = float(best.get("avg_dice", 0.0)) - float(baseline_avg_dice)
    if train:
        max_iter = max(int(row.get("iteration", 0)) for row in train)
        width = max(250, max_iter // 4)
        report["train_windows"] = [
            _window_summary(train, 1, width),
            _window_summary(train, width + 1, 2 * width),
            _window_summary(train, 2 * width + 1, 3 * width),
            _window_summary(train, 3 * width + 1, max_iter),
        ]
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize R7 training collapse and pseudo-label health.")
    parser.add_argument("--output-dir", required=True, help="Run output directory containing metrics.jsonl.")
    parser.add_argument("--baseline-avg-dice", type=float, default=0.7601364254951477)
    parser.add_argument("--report", help="Optional JSON path to save the report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(Path(args.output_dir), baseline_avg_dice=args.baseline_avg_dice)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
