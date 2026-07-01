from __future__ import annotations

import torch


def _class_value(value, cls: int, default: float) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[cls]) if 0 <= cls < len(value) else float(default)
    if isinstance(value, dict):
        return float(value.get(cls, value.get(str(cls), default)))
    return float(value if value is not None else default)


def copy_paste_replay_weight(
    iteration: int,
    config: dict,
    target_stats: dict | None = None,
    foreground_classes: list[int] | tuple[int, ...] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the labeled foreground replay weight.

    The base schedule warms up copy-paste from labeled foreground masks.  An
    optional coverage boost raises the weight when pseudo-target foreground
    participation is below anatomical class floors, so the student sees labeled
    foreground in unlabeled ultrasound context before noisy pseudo labels are
    trusted.
    """

    start_iter = int(config.get("start_iter", 0))
    base_weight = float(config.get("weight", 0.0))
    if iteration < start_iter or base_weight <= 0.0:
        return 0.0, {
            "copy_paste_base_weight": 0.0,
            "copy_paste_coverage_boost": 1.0,
            "copy_paste_coverage_deficit": 0.0,
            "copy_paste_effective_cap": float(config.get("max_effective_weight", base_weight)),
        }

    ramp_iterations = max(1, int(config.get("ramp_iterations", 1)))
    ramp = min(1.0, max(0, iteration - start_iter + 1) / ramp_iterations)
    base = ramp * base_weight
    boost = 1.0
    max_deficit = 0.0
    boost_cfg = config.get("coverage_boost", {})
    if bool(boost_cfg.get("enabled", False)) and iteration >= int(boost_cfg.get("start_iter", start_iter)):
        stats = target_stats or {}
        ratios = stats.get("per_class_foreground_participation_ratio", [])
        min_ratio_cfg = boost_cfg.get("min_class_ratio", config.get("min_fg_pixels_per_class_ratio", 0.0))
        fg_classes = [int(cls) for cls in (foreground_classes or []) if int(cls) > 0]
        for cls in fg_classes:
            observed = float(ratios[cls]) if isinstance(ratios, (list, tuple)) and cls < len(ratios) else 0.0
            target = max(0.0, _class_value(min_ratio_cfg, cls, 0.0))
            if target > 0.0:
                max_deficit = max(max_deficit, max(0.0, target - observed) / max(target, 1e-6))
        max_boost = max(1.0, float(boost_cfg.get("max_boost", 1.0)))
        boost = 1.0 + (max_boost - 1.0) * min(1.0, max_deficit)
        decay_start = int(boost_cfg.get("decay_start_iter", 0))
        if decay_start > 0 and iteration > decay_start:
            decay_iters = max(1, int(boost_cfg.get("decay_iterations", 1)))
            decay = min(1.0, max(0.0, (iteration - decay_start) / decay_iters))
            boost = 1.0 + (boost - 1.0) * (1.0 - decay)

    cap = float(config.get("max_effective_weight", max(base_weight, base * boost)))
    effective = min(cap, base * boost)
    return effective, {
        "copy_paste_base_weight": float(base),
        "copy_paste_coverage_boost": float(boost),
        "copy_paste_coverage_deficit": float(max_deficit),
        "copy_paste_effective_cap": float(cap),
    }


def _foreground_mask(mask: torch.Tensor, foreground_classes: list[int], ignore_index: int) -> torch.Tensor:
    valid = mask != int(ignore_index)
    fg = torch.zeros_like(valid, dtype=torch.bool)
    for cls in foreground_classes:
        fg = fg | (mask == int(cls))
    return fg & valid


def build_labeled_foreground_copy_paste(
    labeled_image: torch.Tensor,
    labeled_mask: torch.Tensor,
    unlabeled_image: torch.Tensor,
    foreground_classes: list[int] | None = None,
    ignore_index: int = 255,
    min_foreground_ratio: float = 0.0,
    max_foreground_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Paste labeled foreground anatomy into unlabeled context.

    The returned target supervises only pasted foreground pixels.  Background and
    unlabeled context stay ignored, so this branch anchors rare foreground
    semantics without turning unlabeled background into hard negatives.
    """
    if labeled_image.shape != unlabeled_image.shape:
        raise ValueError(
            "labeled_image and unlabeled_image must share BCHW shape, "
            f"got {tuple(labeled_image.shape)} and {tuple(unlabeled_image.shape)}"
        )
    if labeled_mask.shape != labeled_image.shape[:1] + labeled_image.shape[-2:]:
        raise ValueError(
            "labeled_mask must share BHW with labeled_image, "
            f"got {tuple(labeled_mask.shape)} for image {tuple(labeled_image.shape)}"
        )

    if foreground_classes is None:
        valid_labels = torch.unique(labeled_mask[labeled_mask != int(ignore_index)])
        fg_classes = [int(label.item()) for label in valid_labels if int(label.item()) > 0]
    else:
        fg_classes = [int(cls) for cls in foreground_classes]
    paste_mask = _foreground_mask(labeled_mask, fg_classes, ignore_index)
    b, _, h, w = labeled_image.shape
    area = paste_mask.flatten(1).float().mean(dim=1)
    keep = (area >= float(min_foreground_ratio)) & (area <= float(max_foreground_ratio))
    paste_mask = paste_mask & keep.view(b, 1, 1)

    mixed = unlabeled_image.clone()
    mixed = torch.where(paste_mask.unsqueeze(1), labeled_image, mixed)
    target = labeled_mask.new_full((b, h, w), int(ignore_index))
    target = torch.where(paste_mask, labeled_mask, target)

    class_ratios = {}
    total_pixels = float(max(1, b * h * w))
    for cls in fg_classes:
        class_ratios[f"copy_paste_class{int(cls)}_ratio"] = float(((target == int(cls)).sum().detach()).item() / total_pixels)
    stats = {
        "copy_paste_active": 1.0 if bool(paste_mask.any()) else 0.0,
        "copy_paste_fg_ratio": float(paste_mask.float().mean().detach()),
        "copy_paste_kept_samples": float(keep.float().sum().detach()),
        **class_ratios,
    }
    return mixed, target, paste_mask, stats
