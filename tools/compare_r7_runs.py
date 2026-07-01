from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_FIELDS = [
    "unsup_weight",
    "r6_unsup_scale",
    "loss_set",
    "loss_branch_ssl",
    "candidate_foreground_ratio",
    "safe_negative_pixel_ratio",
    "sam_train_gate_ratio",
    "sam_foreground_support_ratio",
    "sam_agreement_gate_ratio",
    "sam_agreement_effective_weight",
    "prior_feedback_drift",
    "prior_feedback_unsup_scale",
    "loss_prior_feedback",
    "copy_paste_effective_weight",
    "copy_paste_fg_ratio",
    "loss_copy_paste",
    "class_balanced_ce_weight_class1",
    "class_balanced_ce_weight_class2",
    "loss_sup_boundary",
    "loss_strong_consistency",
    "strong_view_consistency_mask_ratio",
    "topology_postprocess_active",
    "topology_removed_pixel_ratio",
    "topology_removed_ratio_class1",
    "topology_removed_ratio_class2",
    "topology_dropped_components_class1",
    "topology_dropped_components_class2",
    "sam_kd_effective_weight",
    "loss_sam_kd",
    "trust_unsafe",
]

DEFAULT_BASELINE_CLASS_DICE = [None, 0.7218010774222662, 0.7984717713624256]
STABLE_DROP_THRESHOLD = 0.03
CLASS_GAP_TOLERANCE = 0.005
EFFECTIVE_SSL_THRESHOLD = 0.005


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float(row.get(key), float("nan")) for row in rows if row.get(key) is not None]
    values = [value for value in values if math.isfinite(value)]
    return float(mean(values)) if values else None


def _class_value(row: dict[str, Any], prefix: str, cls: int) -> float | None:
    key = f"class_{cls}_{prefix}"
    if row.get(key) is not None:
        return _float(row[key])
    values = row.get(f"class_{prefix}")
    if isinstance(values, list) and cls < len(values) and values[cls] is not None:
        return _float(values[cls])
    return None


def _window(train: list[dict[str, Any]], center: int, radius: int = 125) -> list[dict[str, Any]]:
    rows = [row for row in train if center - radius <= int(row.get("iteration", 0)) <= center]
    if rows or not train:
        return rows
    return [min(train, key=lambda row: abs(int(row.get("iteration", 0)) - center))]


def _class_gap(value: float | None, cls: int, baseline_class_dice: list[float | None]) -> float | None:
    if value is None or cls >= len(baseline_class_dice) or baseline_class_dice[cls] is None:
        return None
    return float(value) - float(baseline_class_dice[cls])


def _recommendation(blockers: list[str], baseline_best_gap: float, stable: bool) -> tuple[str, str]:
    if baseline_best_gap >= 0.0 and stable:
        return "baseline_beaten_stable", "Proceed to test best checkpoint and run ablations for prior_feedback, copy_paste, boundary, and strong_view_consistency."
    if baseline_best_gap >= 0.0:
        return "baseline_peak_unstable", "Keep the best checkpoint for test, then reduce late SSL drift by checking c1_pg/c2_pg, prior_feedback_drift, and trust_unsafe after the best iteration."
    if "best_before_effective_ssl" in blockers:
        return "ssl_not_contributing_to_peak", "The best point appears before effective SSL; verify that copy_paste, strong_view_consistency, SAM agreement, and prior_feedback are nonzero after SSL starts."
    if "class2_below_baseline" in blockers:
        return "class2_quality_limited", "Focus on class2: inspect class_balanced_ce_weight_class2, copy_paste_class2_ratio, prompt_c2 validity, and c2_pred_to_gt."
    if "class1_overexpansion" in blockers or "class2_overexpansion" in blockers or "late_val_decline" in blockers:
        return "late_foreground_drift_limited", "Tune only drift controls next: prior_feedback upper bounds, trust max foreground ratios, and strong-view consistency mask ratio."
    if "sam_gate_starvation" in blockers:
        return "sam_signal_too_sparse", "Check prompt validity, SAM support, verifier thresholds, and sam_agreement_gate_ratio before increasing SAM weights."
    return "below_baseline_unclassified", "Use the report fields to identify whether the gap is class-specific, stability-related, or caused by inactive new mechanisms."


def summarize_run(
    output_dir: Path,
    baseline_avg_dice: float = 0.7601364254951477,
    baseline_class_dice: list[float | None] | None = None,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {
            "run": output_dir.name,
            "output_dir": str(output_dir),
            "train_rows": 0,
            "val_rows": 0,
            "status": "missing_metrics",
        }
    rows = _load_rows(metrics_path)
    train = [row for row in rows if row.get("phase") == "train"]
    val = [row for row in rows if row.get("phase") == "val"]
    if not val:
        return {"run": output_dir.name, "output_dir": str(output_dir), "train_rows": len(train), "val_rows": 0}

    best = max(val, key=lambda row: _float(row.get("avg_dice"), -1.0))
    final = val[-1]
    best_iter = int(best.get("iteration", 0))
    best_window = _window(train, best_iter)
    final_window = _window(train, int(final.get("iteration", 0)))
    c1_pg_best = _class_value(best, "pred_to_gt_ratio", 1)
    c2_pg_best = _class_value(best, "pred_to_gt_ratio", 2)
    c1_pg_final = _class_value(final, "pred_to_gt_ratio", 1)
    c2_pg_final = _class_value(final, "pred_to_gt_ratio", 2)
    c1_drop = _class_value(best, "dice", 1)
    c2_drop = _class_value(best, "dice", 2)
    c1_final = _class_value(final, "dice", 1)
    c2_final = _class_value(final, "dice", 2)
    c1_best = c1_drop
    c2_best = c2_drop
    c1_drop = None if c1_drop is None or c1_final is None else c1_drop - c1_final
    c2_drop = None if c2_drop is None or c2_final is None else c2_drop - c2_final
    best_avg = _float(best.get("avg_dice"), -1.0)
    final_avg = _float(final.get("avg_dice"), -1.0)
    baseline_class_dice = baseline_class_dice or DEFAULT_BASELINE_CLASS_DICE
    c1_best_gap = _class_gap(c1_best, 1, baseline_class_dice)
    c2_best_gap = _class_gap(c2_best, 2, baseline_class_dice)
    c1_final_gap = _class_gap(c1_final, 1, baseline_class_dice)
    c2_final_gap = _class_gap(c2_final, 2, baseline_class_dice)

    blockers: list[str] = []
    if final_avg > 0 and best_avg - final_avg >= 0.03:
        blockers.append("late_val_decline")
    if c1_pg_final is not None and c1_pg_final >= 1.30:
        blockers.append("class1_overexpansion")
    if c2_pg_final is not None and c2_pg_final >= 1.25:
        blockers.append("class2_overexpansion")
    if c1_drop is not None and c1_drop >= 0.05:
        blockers.append("class1_dice_collapse")
    if c2_drop is not None and c2_drop >= 0.05:
        blockers.append("class2_dice_collapse")
    best_unsup_for_blocker = _mean(best_window, "r6_unsup_scale")
    if best_unsup_for_blocker is None or best_unsup_for_blocker < EFFECTIVE_SSL_THRESHOLD:
        blockers.append("best_before_effective_ssl")
    if (_mean(best_window, "sam_train_gate_ratio") or 0.0) < 0.003:
        blockers.append("sam_gate_starvation")
    if (_mean(final_window, "prior_feedback_drift") or 0.0) > 0.0:
        blockers.append("student_prior_feedback_active")
    if c1_best_gap is not None and c1_best_gap < -CLASS_GAP_TOLERANCE:
        blockers.append("class1_below_baseline")
    if c2_best_gap is not None and c2_best_gap < -CLASS_GAP_TOLERANCE:
        blockers.append("class2_below_baseline")

    best_unsup = _mean(best_window, "r6_unsup_scale")
    stable = best_avg - final_avg <= STABLE_DROP_THRESHOLD
    ssl_active_at_best = best_unsup is not None and best_unsup >= EFFECTIVE_SSL_THRESHOLD
    verdict, next_action = _recommendation(blockers, best_avg - float(baseline_avg_dice), stable)
    mechanism_coverage = {
        "prior_feedback_logged": _mean(final_window, "prior_feedback_drift") is not None,
        "copy_paste_logged": _mean(final_window, "copy_paste_effective_weight") is not None,
        "strong_view_consistency_logged": _mean(final_window, "loss_strong_consistency") is not None,
        "supervised_boundary_logged": _mean(final_window, "loss_sup_boundary") is not None,
        "topology_postprocess_logged": _mean([final], "topology_removed_pixel_ratio") is not None,
    }

    return {
        "run": output_dir.name,
        "output_dir": str(output_dir),
        "train_rows": len(train),
        "val_rows": len(val),
        "best_iteration": best_iter,
        "best_avg_dice": best_avg,
        "best_baseline_gap": best_avg - float(baseline_avg_dice),
        "final_iteration": int(final.get("iteration", 0)),
        "final_avg_dice": final_avg,
        "best_to_final_drop": best_avg - final_avg,
        "c1_dice_best": c1_best,
        "c2_dice_best": c2_best,
        "c1_dice_final": c1_final,
        "c2_dice_final": c2_final,
        "c1_best_baseline_gap": c1_best_gap,
        "c2_best_baseline_gap": c2_best_gap,
        "c1_final_baseline_gap": c1_final_gap,
        "c2_final_baseline_gap": c2_final_gap,
        "c1_dice_drop": c1_drop,
        "c2_dice_drop": c2_drop,
        "c1_pred_to_gt_best": c1_pg_best,
        "c2_pred_to_gt_best": c2_pg_best,
        "c1_pred_to_gt_final": c1_pg_final,
        "c2_pred_to_gt_final": c2_pg_final,
        "foreground_pred_ratio_final": final.get("foreground_pred_ratio"),
        "foreground_gt_ratio_final": final.get("foreground_gt_ratio"),
        "topology_removed_pixel_ratio_final": final.get("topology_removed_pixel_ratio"),
        "topology_removed_ratio_class1_final": final.get("topology_removed_ratio_class1"),
        "topology_removed_ratio_class2_final": final.get("topology_removed_ratio_class2"),
        "topology_dropped_components_class1_final": final.get("topology_dropped_components_class1"),
        "topology_dropped_components_class2_final": final.get("topology_dropped_components_class2"),
        "stable_within_drop_threshold": stable,
        "ssl_active_at_best": ssl_active_at_best,
        "verdict": verdict,
        "recommended_next_action": next_action,
        "mechanism_coverage": mechanism_coverage,
        "best_window": {key: _mean(best_window, key) for key in DEFAULT_FIELDS},
        "final_window": {key: _mean(final_window, key) for key in DEFAULT_FIELDS},
        "blockers": blockers,
    }


def compare_runs(output_dirs: list[Path], baseline_avg_dice: float = 0.7601364254951477) -> dict[str, Any]:
    runs = [summarize_run(path, baseline_avg_dice=baseline_avg_dice) for path in output_dirs]
    finished = [run for run in runs if run.get("val_rows", 0) > 0]
    best_run = max(finished, key=lambda run: _float(run.get("best_avg_dice"), -1.0)) if finished else None
    most_stable = min(finished, key=lambda run: _float(run.get("best_to_final_drop"), float("inf"))) if finished else None
    blocker_counts: dict[str, int] = {}
    for run in finished:
        for blocker in run.get("blockers", []):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    return {
        "baseline_avg_dice": float(baseline_avg_dice),
        "runs": sorted(runs, key=lambda run: _float(run.get("best_avg_dice"), -1.0), reverse=True),
        "best_run": best_run["run"] if best_run else None,
        "best_run_verdict": best_run.get("verdict") if best_run else None,
        "best_run_recommended_next_action": best_run.get("recommended_next_action") if best_run else None,
        "most_stable_run": most_stable["run"] if most_stable else None,
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare R7 output folders and expose validation Dice blockers.")
    parser.add_argument("output_dirs", nargs="+", help="R7 output directories containing metrics.jsonl.")
    parser.add_argument("--baseline-avg-dice", type=float, default=0.7601364254951477)
    parser.add_argument("--report", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_runs([Path(path) for path in args.output_dirs], baseline_avg_dice=args.baseline_avg_dice)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
