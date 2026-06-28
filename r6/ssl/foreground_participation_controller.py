from __future__ import annotations

import torch


def _topk_mask(score: torch.Tensor, eligible: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(eligible, dtype=torch.bool)
    if k <= 0 or int(eligible.sum()) == 0:
        return out
    flat_score = score[eligible]
    keep = min(int(k), int(flat_score.numel()))
    if keep <= 0:
        return out
    _, order = flat_score.topk(keep)
    flat_idx = eligible.flatten().nonzero(as_tuple=False).squeeze(1)[order]
    out.flatten()[flat_idx] = True
    return out


def _foreground_participation(
    *,
    singleton_label: torch.Tensor,
    singleton_mask: torch.Tensor,
    candidate_set: torch.Tensor,
    ambiguous_mask: torch.Tensor,
    fg_classes: list[int],
) -> tuple[torch.Tensor, dict[int, int]]:
    fg_participation = torch.zeros_like(singleton_mask, dtype=torch.bool)
    per_class = {}
    for cls in fg_classes:
        if not (0 < cls < candidate_set.shape[1]):
            continue
        hard_cls = singleton_mask & (singleton_label == cls)
        soft_cls = ambiguous_mask & candidate_set[:, cls]
        cls_mask = hard_cls | soft_cls
        fg_participation = fg_participation | cls_mask
        per_class[cls] = int(cls_mask.sum())
    return fg_participation, per_class


def apply_foreground_budget(
    *,
    singleton_label: torch.Tensor,
    singleton_mask: torch.Tensor,
    candidate_set: torch.Tensor,
    ambiguous_mask: torch.Tensor,
    foreground_score: torch.Tensor,
    teacher_prob: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Guarantee foreground has a training route and cap hard background CE."""

    num_classes = candidate_set.shape[1]
    total_pixels = int(singleton_mask.numel())
    fg_classes = list(config.get("foreground_classes", list(range(1, num_classes))))
    min_ratio = float(config.get("min_fg_pixels_per_class_ratio", config.get("min_fg_ratio_per_class", 0.02)))
    min_pixels_cfg = int(config.get("min_fg_pixels_per_class", 0))
    min_pixels = max(min_pixels_cfg, int(round(total_pixels * min_ratio)))
    min_pixels = max(0, min(min_pixels, total_pixels))
    promoted_any = torch.zeros_like(singleton_mask, dtype=torch.bool)
    fg_budget_violation = 0

    for cls in fg_classes:
        if not (0 < cls < num_classes):
            continue
        hard_cls = singleton_mask & (singleton_label == cls)
        soft_cls = ambiguous_mask & candidate_set[:, cls]
        current = int((hard_cls | soft_cls).sum())
        if current >= min_pixels:
            continue
        need = min_pixels - current
        eligible = (~hard_cls) & (~singleton_mask | (singleton_label == 0) | ambiguous_mask)
        eligible = eligible & ((candidate_set[:, cls]) | (foreground_score[:, cls] > 0))
        promote = _topk_mask(foreground_score[:, cls], eligible, need)
        if int(promote.sum()) == 0:
            continue
        candidate_set[:, cls] = candidate_set[:, cls] | promote
        ambiguous_mask = ambiguous_mask | promote
        singleton_mask = singleton_mask & ~promote
        promoted_any = promoted_any | promote
        fg_budget_violation += 1

    fg_participation, per_class_counts = _foreground_participation(
        singleton_label=singleton_label,
        singleton_mask=singleton_mask,
        candidate_set=candidate_set,
        ambiguous_mask=ambiguous_mask,
        fg_classes=fg_classes,
    )

    sentinel_enabled = bool(config.get("collapse_sentinel_enabled", True))
    sentinel_ratio = float(config.get("collapse_min_fg_ratio_per_class", min_ratio))
    sentinel_pixels = max(int(config.get("collapse_min_fg_pixels_per_class", 0)), int(round(total_pixels * sentinel_ratio)))
    sentinel_pixels = max(0, min(sentinel_pixels, total_pixels))
    bg_hard = singleton_mask & (singleton_label == 0)
    bg_ratio_before = float(bg_hard.float().mean().detach())
    fg_count_before = int(fg_participation.sum())
    fg_target_total = sentinel_pixels * max(1, len([c for c in fg_classes if 0 < c < num_classes]))
    low_all_fg = bool(fg_classes) and all(per_class_counts.get(cls, 0) < sentinel_pixels for cls in fg_classes if 0 < cls < num_classes)
    low_total_fg = fg_count_before < fg_target_total
    bg_takeover = bg_ratio_before > float(config.get("collapse_max_background_hard_ratio", 0.50))
    collapse_active = sentinel_enabled and (low_all_fg or (low_total_fg and bg_takeover))
    collapse_forced = torch.zeros_like(singleton_mask, dtype=torch.bool)
    collapse_fg_budget_violation = 0

    if collapse_active and sentinel_pixels > 0:
        force_ratio = float(config.get("collapse_force_fg_ratio_per_class", max(min_ratio, sentinel_ratio)))
        force_pixels_cfg = int(config.get("collapse_force_fg_pixels_per_class", 0))
        force_pixels = max(force_pixels_cfg, int(round(total_pixels * force_ratio)), sentinel_pixels)
        force_pixels = max(0, min(force_pixels, total_pixels))
        teacher_weight = float(config.get("collapse_teacher_fallback_weight", 1.0))
        min_score = float(config.get("collapse_min_candidate_score", 0.0))
        for cls in fg_classes:
            if not (0 < cls < num_classes):
                continue
            hard_cls = singleton_mask & (singleton_label == cls)
            soft_cls = ambiguous_mask & candidate_set[:, cls]
            current = int((hard_cls | soft_cls).sum())
            if current >= force_pixels:
                continue
            need = force_pixels - current
            score = torch.maximum(foreground_score[:, cls], teacher_prob[:, cls] * teacher_weight)
            eligible = (~hard_cls) & (score > min_score)
            promote = _topk_mask(score, eligible, need)
            if int(promote.sum()) == 0:
                continue
            candidate_set[:, cls] = candidate_set[:, cls] | promote
            ambiguous_mask = ambiguous_mask | promote
            singleton_mask = singleton_mask & ~promote
            promoted_any = promoted_any | promote
            collapse_forced = collapse_forced | promote
            collapse_fg_budget_violation += 1

        fg_participation, per_class_counts = _foreground_participation(
            singleton_label=singleton_label,
            singleton_mask=singleton_mask,
            candidate_set=candidate_set,
            ambiguous_mask=ambiguous_mask,
            fg_classes=fg_classes,
        )

    emergency_mode = bool(config.get("disable_bg_if_no_fg", True)) and int(fg_participation.sum()) == 0
    background_cap_active = False
    bg_hard = singleton_mask & (singleton_label == 0)
    collapse_disable_background = collapse_active and bool(config.get("collapse_disable_background_hard", True))
    if emergency_mode or collapse_disable_background:
        singleton_mask = singleton_mask & ~bg_hard
        background_cap_active = bool(int(bg_hard.sum()) > 0)
    else:
        max_bg_ratio = float(config.get("max_background_hard_ratio", 0.70))
        max_bg_to_fg_ratio = float(config.get("max_bg_to_fg_ratio", 4.0))
        max_by_total = int(round(total_pixels * max_bg_ratio))
        max_by_fg = int(round(max_bg_to_fg_ratio * max(1, int(fg_participation.sum()))))
        max_bg = max(0, min(max_by_total, max_by_fg))
        bg_count = int(bg_hard.sum())
        if bg_count > max_bg:
            keep_bg = _topk_mask(teacher_prob[:, 0], bg_hard, max_bg)
            singleton_mask = torch.where(bg_hard, keep_bg, singleton_mask)
            background_cap_active = True

    kept_background = singleton_mask & (singleton_label == 0)
    candidate_set[:, 0] = candidate_set[:, 0] & kept_background
    empty_candidate = candidate_set.sum(dim=1) == 0
    empty_recovered = torch.zeros_like(singleton_mask, dtype=torch.bool)
    if empty_candidate.any() and num_classes > 1:
        min_empty_fg = float(config.get("min_empty_foreground_score", config.get("min_foreground_score", 0.02)))
        fg_score, fg_idx = foreground_score[:, 1:].max(dim=1)
        has_fg_hint = empty_candidate & (fg_score >= min_empty_fg)
        for cls in fg_classes:
            if not (0 < cls < num_classes):
                continue
            promote = has_fg_hint & ((fg_idx + 1) == cls)
            if int(promote.sum()) == 0:
                continue
            candidate_set[:, cls] = candidate_set[:, cls] | promote
            ambiguous_mask = ambiguous_mask | promote
            singleton_mask = singleton_mask & ~promote
            promoted_any = promoted_any | promote
            empty_recovered = empty_recovered | promote
        empty_candidate = candidate_set.sum(dim=1) == 0
    ambiguous_mask = ambiguous_mask & (candidate_set.sum(dim=1) > 0)

    hard_fg_ratios = {}
    soft_fg_ratios = {}
    for cls in fg_classes:
        if 0 < cls < num_classes:
            hard_fg_ratios[f"hard_fg_ratio_class{cls}"] = float(((singleton_mask & (singleton_label == cls)).float().mean()).detach())
            soft_fg_ratios[f"soft_fg_ratio_class{cls}"] = float(((ambiguous_mask & candidate_set[:, cls]).float().mean()).detach())

    stats = {
        **hard_fg_ratios,
        **soft_fg_ratios,
        "background_hard_ratio": float(((singleton_mask & (singleton_label == 0)).float().mean()).detach()),
        "background_cap_active": 1.0 if background_cap_active else 0.0,
        "pseudo_set_size_mean": float(candidate_set.float().sum(dim=1).mean().detach()),
        "empty_candidate_after_budget_ratio": float(empty_candidate.float().mean().detach()),
        "fg_budget_violation": float(fg_budget_violation),
        "emergency_mode": 1.0 if emergency_mode else 0.0,
        "foreground_promoted_ratio": float(promoted_any.float().mean().detach()),
        "empty_candidate_recovered_ratio": float(empty_recovered.float().mean().detach()),
        "collapse_sentinel_active": 1.0 if collapse_active else 0.0,
        "collapse_disabled_background": 1.0 if collapse_disable_background else 0.0,
        "collapse_bg_hard_ratio_before": bg_ratio_before,
        "collapse_fg_ratio_before": float(fg_count_before / max(1, total_pixels)),
        "collapse_forced_fg_ratio": float(collapse_forced.float().mean().detach()),
        "collapse_fg_budget_violation": float(collapse_fg_budget_violation),
    }
    return singleton_label, singleton_mask, candidate_set, ambiguous_mask, stats
