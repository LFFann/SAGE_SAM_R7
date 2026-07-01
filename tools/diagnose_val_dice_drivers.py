from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_r7_runs import DEFAULT_FIELDS


EXTRA_FIELDS = [
    "foreground_pred_ratio",
    "foreground_gt_ratio",
    "class_1_pred_to_gt_ratio",
    "class_2_pred_to_gt_ratio",
]

TARGET_FIELDS = {
    "avg_dice",
    "avg_iou",
    "avg_hd95",
    "best_dice",
    "best_avg_dice",
    "is_best_dice",
    "drop_from_best",
    "baseline_gap",
    "class_1_dice",
    "class_2_dice",
}


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


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _class_value(row: dict[str, Any], prefix: str, cls: int) -> float | None:
    direct = _float(row.get(f"class_{cls}_{prefix}"))
    if direct is not None:
        return direct
    values = row.get(f"class_{prefix}")
    if isinstance(values, list) and cls < len(values):
        return _float(values[cls])
    return None


def _field_value(row: dict[str, Any], field: str) -> float | None:
    if field == "class_1_pred_to_gt_ratio":
        return _class_value(row, "pred_to_gt_ratio", 1)
    if field == "class_2_pred_to_gt_ratio":
        return _class_value(row, "pred_to_gt_ratio", 2)
    return _float(row.get(field))


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return float(mean(clean)) if clean else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(value * value for value in dx)
    var_y = sum(value * value for value in dy)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    return float(sum(x * y for x, y in zip(dx, dy)) / math.sqrt(var_x * var_y))


def _window(train_rows: list[dict[str, Any]], iteration: int, radius: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in train_rows
        if iteration - radius <= int(row.get("iteration", 0)) <= iteration
    ]
    if rows or not train_rows:
        return rows
    return [min(train_rows, key=lambda row: abs(int(row.get("iteration", 0)) - iteration))]


def _collect_fields(rows: list[dict[str, Any]], requested: list[str] | None) -> list[str]:
    fields = list(dict.fromkeys(DEFAULT_FIELDS + EXTRA_FIELDS))
    if requested:
        fields.extend(requested)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"iteration", "epoch"} and key not in TARGET_FIELDS:
                fields.append(key)
    return sorted(field for field in set(fields) if field not in TARGET_FIELDS)


def _run_pairs(output_dir: Path, fields: list[str], radius: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_rows(output_dir / "metrics.jsonl")
    train_rows = [row for row in rows if row.get("phase") == "train"]
    val_rows = [row for row in rows if row.get("phase") == "val" and _float(row.get("avg_dice")) is not None]
    pairs: list[dict[str, Any]] = []
    best_so_far = -float("inf")
    for val in sorted(val_rows, key=lambda row: int(row.get("iteration", 0))):
        iteration = int(val.get("iteration", 0))
        avg_dice = _float(val.get("avg_dice"))
        if avg_dice is None:
            continue
        window = _window(train_rows, iteration, radius)
        pair = {
            "run": output_dir.name,
            "iteration": iteration,
            "avg_dice": avg_dice,
            "drop_from_best": max(0.0, best_so_far - avg_dice) if best_so_far > -float("inf") else 0.0,
            "baseline_gap": avg_dice - 0.7601364254951477,
        }
        best_so_far = max(best_so_far, avg_dice)
        for field in fields:
            if field in TARGET_FIELDS:
                continue
            source = window if window else [val]
            pair[field] = _mean([_field_value(row, field) for row in source])
        pairs.append(pair)

    summary = {
        "run": output_dir.name,
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "pairs": len(pairs),
        "best_avg_dice": max((_float(row.get("avg_dice")) or -1.0 for row in val_rows), default=None),
        "final_avg_dice": _float(val_rows[-1].get("avg_dice")) if val_rows else None,
    }
    return pairs, summary


def _driver_note(field: str, corr_dice: float | None, corr_drop: float | None) -> str:
    corr_dice = corr_dice or 0.0
    corr_drop = corr_drop or 0.0
    if corr_dice < -0.45 and corr_drop > 0.35:
        return "higher_when_val_dice_drops"
    if corr_dice > 0.45 and corr_drop < -0.25:
        return "higher_when_val_dice_improves"
    if field in {"r6_unsup_scale", "loss_set", "sam_kd_effective_weight"} and corr_dice < -0.25:
        return "ssl_signal_correlates_with_lower_val_dice"
    if "topology" in field and corr_drop > 0.25:
        return "topology_removal_rises_during_decline"
    if "prompt_consistency" in field and corr_drop > 0.25:
        return "prompt_consistency_active_during_decline"
    return "monitor"


def diagnose(
    output_dirs: list[Path],
    radius: int = 250,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            all_rows.extend(_load_rows(metrics_path))
    all_fields = _collect_fields(all_rows, fields)

    pairs: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        metrics_path = output_dir / "metrics.jsonl"
        if not metrics_path.exists():
            runs.append(
                {
                    "run": output_dir.name,
                    "output_dir": str(output_dir),
                    "status": "missing_metrics",
                    "pairs": 0,
                }
            )
            continue
        run_pairs, run_summary = _run_pairs(output_dir, all_fields, radius)
        pairs.extend(run_pairs)
        runs.append(run_summary)

    drivers: list[dict[str, Any]] = []
    for field in all_fields:
        xy_dice = [
            (pair[field], pair["avg_dice"])
            for pair in pairs
            if pair.get(field) is not None
        ]
        xy_drop = [
            (pair[field], pair["drop_from_best"])
            for pair in pairs
            if pair.get(field) is not None
        ]
        if len(xy_dice) < 3:
            continue
        xs_dice, ys_dice = zip(*xy_dice)
        xs_drop, ys_drop = zip(*xy_drop)
        corr_dice = _pearson(list(xs_dice), list(ys_dice))
        corr_drop = _pearson(list(xs_drop), list(ys_drop))
        if corr_dice is None and corr_drop is None:
            continue
        mean_value = _mean([pair.get(field) for pair in pairs])
        drivers.append(
            {
                "field": field,
                "n": len(xy_dice),
                "mean": mean_value,
                "corr_with_val_dice": corr_dice,
                "corr_with_drop_from_best": corr_drop,
                "note": _driver_note(field, corr_dice, corr_drop),
            }
        )

    drivers.sort(
        key=lambda item: max(
            abs(item["corr_with_val_dice"] or 0.0),
            abs(item["corr_with_drop_from_best"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "window_radius": int(radius),
        "runs": runs,
        "num_pairs": len(pairs),
        "drivers": drivers,
        "top_negative_drivers": [
            item
            for item in drivers
            if (item["corr_with_val_dice"] or 0.0) < -0.25
        ][:12],
        "top_drop_drivers": [
            item
            for item in drivers
            if (item["corr_with_drop_from_best"] or 0.0) > 0.25
        ][:12],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank training diagnostics by association with later validation Dice in R7 runs."
    )
    parser.add_argument("output_dirs", nargs="+", help="Output folders containing metrics.jsonl.")
    parser.add_argument("--radius", type=int, default=250, help="Training-iteration window before each validation point.")
    parser.add_argument("--field", action="append", default=None, help="Additional numeric metric field to include.")
    parser.add_argument("--report", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = diagnose([Path(path) for path in args.output_dirs], radius=args.radius, fields=args.field)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
