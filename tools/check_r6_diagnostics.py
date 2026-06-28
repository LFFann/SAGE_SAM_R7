from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r6.utils.io import load_yaml, save_json


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _train_rows(rows: list[dict[str, Any]], min_iteration: int, tail: int) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("phase", "train") == "train" and int(row.get("iteration", 0)) >= int(min_iteration)
    ]
    if tail > 0:
        selected = selected[-tail:]
    return selected


def _positive_mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(_float(row.get(key)) for row in rows)


def _latest(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return _float(rows[-1].get(key))


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, value: float | str, threshold: str, detail: str):
    checks.append(
        {
            "name": name,
            "status": "pass" if passed else "fail",
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def _add_skip(checks: list[dict[str, Any]], name: str, detail: str):
    checks.append({"name": name, "status": "skip", "value": "n/a", "threshold": "n/a", "detail": detail})


def _expect_sam(config: dict[str, Any] | None, mode: str) -> bool:
    if mode == "yes":
        return True
    if mode == "no":
        return False
    if not config:
        return False
    return bool(config.get("sam", {}).get("use_sam", False))


def evaluate_diagnostics(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    min_iteration: int = 0,
    tail: int = 200,
    min_train_rows: int = 1,
    expect_sam: str = "auto",
    min_candidate_foreground: float = 1e-6,
    min_safe_negative: float = 1e-6,
    max_background_hard: float = 0.70,
    min_sam_gate: float = 1e-6,
    min_sam_weight: float = 1e-6,
    min_sam_kd_loss: float = 1e-8,
    correlation_after: int = 2000,
    min_correlation: float = 1e-8,
    min_locality: float = 1e-8,
) -> dict[str, Any]:
    train = _train_rows(rows, min_iteration=min_iteration, tail=tail)
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "train_rows",
        len(train) >= min_train_rows,
        len(train),
        f">= {min_train_rows}",
        "enough train metric rows in the selected diagnostic window",
    )
    if not train:
        return {
            "status": "fail",
            "selected_rows": 0,
            "max_iteration": 0,
            "checks": checks,
        }

    cand_mean = _positive_mean(train, "candidate_foreground_ratio")
    _add_check(
        checks,
        "candidate_foreground_ratio",
        cand_mean > min_candidate_foreground,
        cand_mean,
        f"> {min_candidate_foreground}",
        "foreground candidates must stay active",
    )

    safe_neg_mean = _positive_mean(train, "safe_negative_pixel_ratio")
    _add_check(
        checks,
        "safe_negative_pixel_ratio",
        safe_neg_mean > min_safe_negative,
        safe_neg_mean,
        f"> {min_safe_negative}",
        "rank-based negative supervision should not stay zero",
    )

    bg_mean = _positive_mean(train, "background_hard_ratio")
    bg_latest = _latest(train, "background_hard_ratio")
    _add_check(
        checks,
        "background_hard_ratio",
        bg_mean <= max_background_hard and bg_latest <= max_background_hard,
        {"mean": bg_mean, "latest": bg_latest},
        f"mean/latest <= {max_background_hard}",
        "background hard CE should not take over the SSL target",
    )

    if _expect_sam(config, expect_sam):
        sam_valid = _positive_mean(train, "sam_valid_ratio")
        sam_gate = _positive_mean(train, "sam_kd_gate_ratio")
        sam_weight = _positive_mean(train, "sam_kd_gate_weight_mean")
        sam_kd = _positive_mean(train, "loss_sam_kd")
        _add_check(checks, "sam_valid_ratio", sam_valid > 0.0, sam_valid, "> 0", "real SAM path must be valid")
        _add_check(
            checks,
            "sam_kd_gate_ratio",
            sam_gate > min_sam_gate,
            sam_gate,
            f"> {min_sam_gate}",
            "SAM KD gate must not depend on hard seed only",
        )
        _add_check(
            checks,
            "sam_kd_gate_weight_mean",
            sam_weight > min_sam_weight,
            sam_weight,
            f"> {min_sam_weight}",
            "structure-weighted SAM KD route should carry nonzero weight",
        )
        _add_check(
            checks,
            "loss_sam_kd",
            sam_kd > min_sam_kd_loss,
            sam_kd,
            f"> {min_sam_kd_loss}",
            "SAM semantic KD should not remain zero through the diagnostic window",
        )
    else:
        _add_skip(checks, "sam_kd_gate", "SAM checks skipped because sam.use_sam is false or --expect-sam=no")

    max_iter = max(int(row.get("iteration", 0)) for row in train)
    if max_iter >= correlation_after:
        corr = _positive_mean(train, "loss_correlation")
        propagated = _positive_mean(train, "foreground_propagated_ratio")
        locality = _positive_mean(train, "masked_locality_ratio")
        _add_check(
            checks,
            "foreground_propagated_or_loss_correlation",
            corr > min_correlation or propagated > min_correlation,
            {"loss_correlation": corr, "foreground_propagated_ratio": propagated},
            f"either > {min_correlation}",
            "correlation should participate after the configured stage",
        )
        _add_check(
            checks,
            "masked_locality_ratio",
            locality > min_locality,
            locality,
            f"> {min_locality}",
            "locality masking should participate after the configured stage",
        )
    else:
        _add_skip(
            checks,
            "correlation_and_locality",
            f"selected max iteration {max_iter} is before correlation_after={correlation_after}",
        )

    failed = [check for check in checks if check["status"] == "fail"]
    return {
        "status": "fail" if failed else "pass",
        "selected_rows": len(train),
        "max_iteration": max_iter,
        "checks": checks,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Check SAGE-SAM R6 short-run diagnostic metrics.")
    p.add_argument("--metrics", help="Path to metrics.jsonl. Defaults to <output-dir>/metrics.jsonl.")
    p.add_argument("--output-dir", help="Run output directory containing metrics.jsonl and resolved_config.yaml.")
    p.add_argument("--config", help="Resolved config path. Defaults to <output-dir>/resolved_config.yaml when present.")
    p.add_argument("--report", help="Optional JSON report path.")
    p.add_argument("--min-iteration", type=int, default=0)
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("--min-train-rows", type=int, default=1)
    p.add_argument("--expect-sam", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--min-candidate-foreground", type=float, default=1e-6)
    p.add_argument("--min-safe-negative", type=float, default=1e-6)
    p.add_argument("--max-background-hard", type=float, default=0.70)
    p.add_argument("--min-sam-gate", type=float, default=1e-6)
    p.add_argument("--min-sam-weight", type=float, default=1e-6)
    p.add_argument("--min-sam-kd-loss", type=float, default=1e-8)
    p.add_argument("--correlation-after", type=int, default=2000)
    p.add_argument("--min-correlation", type=float, default=1e-8)
    p.add_argument("--min-locality", type=float, default=1e-8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    metrics = Path(args.metrics) if args.metrics else (output_dir / "metrics.jsonl" if output_dir else None)
    if metrics is None:
        raise SystemExit("--metrics or --output-dir is required")
    config_path = Path(args.config) if args.config else (output_dir / "resolved_config.yaml" if output_dir else None)
    config = load_yaml(config_path) if config_path and config_path.exists() else None
    rows = load_metric_rows(metrics)
    report = evaluate_diagnostics(
        rows,
        config=config,
        min_iteration=args.min_iteration,
        tail=args.tail,
        min_train_rows=args.min_train_rows,
        expect_sam=args.expect_sam,
        min_candidate_foreground=args.min_candidate_foreground,
        min_safe_negative=args.min_safe_negative,
        max_background_hard=args.max_background_hard,
        min_sam_gate=args.min_sam_gate,
        min_sam_weight=args.min_sam_weight,
        min_sam_kd_loss=args.min_sam_kd_loss,
        correlation_after=args.correlation_after,
        min_correlation=args.min_correlation,
        min_locality=args.min_locality,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        save_json(report, args.report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
