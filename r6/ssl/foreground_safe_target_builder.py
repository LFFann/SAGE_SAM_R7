from __future__ import annotations

import torch

from .foreground_participation_controller import apply_foreground_budget
from .sam_structural_support import build_sam_structural_support


def _threshold_vec(calibrator, attr: str, default: float, num_classes: int, device, dtype):
    value = getattr(calibrator, attr, None)
    if value is None:
        return torch.full((num_classes,), float(default), device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.numel() != num_classes:
        return torch.full((num_classes,), float(default), device=device, dtype=dtype)
    return value


def _foreground_classes(config: dict, num_classes: int) -> list[int]:
    return [int(c) for c in config.get("foreground_classes", list(range(1, num_classes))) if 0 < int(c) < num_classes]


def _topk_foreground_candidates(score: torch.Tensor, k: int) -> torch.Tensor:
    if score.ndim != 4:
        raise ValueError(f"foreground score must be BCHW, got {tuple(score.shape)}")
    bsz, num_fg, height, width = score.shape
    if num_fg == 0 or k <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    keep = min(int(k), num_fg)
    _, topi = score.topk(k=keep, dim=1)
    out = torch.zeros_like(score, dtype=torch.bool)
    out.scatter_(1, topi, True)
    return out


def _rank_positions(prob: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(prob, dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(prob.shape[1], device=prob.device, dtype=order.dtype).view(1, -1, 1, 1)
    return ranks.scatter_(1, order, rank_values.expand_as(order))


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


def _class_value(config: dict, key: str, cls: int, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, dict):
        return float(value.get(cls, value.get(str(cls), default)))
    if isinstance(value, (list, tuple)):
        if cls < len(value):
            return float(value[cls])
        return float(value[-1]) if value else float(default)
    return float(value)


def _align_teacher_to_labeled_prior(teacher_prob: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    if not bool(config.get("use_labeled_prior_distribution_alignment", False)):
        return teacher_prob, {"prior_alignment_active": 0.0}

    prior_value = config.get("labeled_class_prior")
    if prior_value is None:
        return teacher_prob, {"prior_alignment_active": 0.0}
    prior = torch.as_tensor(prior_value, device=teacher_prob.device, dtype=teacher_prob.dtype)
    if prior.numel() != teacher_prob.shape[1]:
        return teacher_prob, {"prior_alignment_active": 0.0}

    eps = teacher_prob.new_tensor(float(config.get("prior_alignment_eps", 1e-6)))
    prior = prior.clamp_min(eps)
    prior = prior / prior.sum().clamp_min(eps)
    batch_mean = teacher_prob.mean(dim=(0, 2, 3)).clamp_min(eps)
    min_ratio = float(config.get("prior_alignment_min_ratio", 0.10))
    max_ratio = float(config.get("prior_alignment_max_ratio", 5.0))
    strength = float(config.get("prior_alignment_strength", 0.35))
    ratio = (prior / batch_mean).clamp(min_ratio, max_ratio)
    adjust = ratio.pow(strength)
    if not bool(config.get("prior_alignment_include_background", True)) and adjust.numel() > 0:
        adjust = adjust.clone()
        adjust[0] = 1.0
    aligned = teacher_prob * adjust.view(1, -1, 1, 1)
    aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(eps)
    aligned_mean = aligned.mean(dim=(0, 2, 3))

    stats = {
        "prior_alignment_active": 1.0,
        "prior_alignment_strength": strength,
    }
    for cls in range(teacher_prob.shape[1]):
        stats[f"prior_alignment_before_mean_class{cls}"] = float(batch_mean[cls].detach())
        stats[f"prior_alignment_after_mean_class{cls}"] = float(aligned_mean[cls].detach())
        stats[f"prior_alignment_ratio_class{cls}"] = float(ratio[cls].detach())
    return aligned, stats


def _foreground_participation_stats(singleton_label, singleton_mask, candidate_set, ambiguous_mask, fg_classes):
    stats: dict[str, float] = {
        "background_hard_ratio": float(((singleton_mask & (singleton_label == 0)).float().mean()).detach()),
    }
    for cls in fg_classes:
        if 0 < cls < candidate_set.shape[1]:
            stats[f"hard_fg_ratio_class{cls}"] = float(((singleton_mask & (singleton_label == cls)).float().mean()).detach())
            stats[f"soft_fg_ratio_class{cls}"] = float(((ambiguous_mask & candidate_set[:, cls]).float().mean()).detach())
    return stats


def _cap_foreground_candidate_set(
    candidate_set: torch.Tensor,
    singleton_label: torch.Tensor,
    singleton_mask: torch.Tensor,
    ambiguous_mask: torch.Tensor,
    foreground_score: torch.Tensor,
    teacher_prob: torch.Tensor,
    verifier_score: torch.Tensor,
    fg_classes: list[int],
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if not bool(config.get("bounded_foreground_candidates", False)):
        return candidate_set, singleton_label, singleton_mask, ambiguous_mask, {"foreground_ceiling_active": 0.0}

    total_pixels = int(singleton_mask.numel())
    min_ratio_default = float(config.get("min_fg_ratio_per_class", 0.0))
    min_pixels_cfg = int(config.get("min_fg_pixels_per_class", 0))
    old_fg_any = candidate_set[:, 1:].any(dim=1) if candidate_set.shape[1] > 1 else torch.zeros_like(singleton_mask)
    ceiling_stats: dict[str, float] = {"foreground_ceiling_active": 1.0}
    flood_classes = 0
    verifier_weight = float(config.get("foreground_ceiling_verifier_weight", 0.15))

    for cls in fg_classes:
        if not (0 < cls < candidate_set.shape[1]):
            continue
        hard_cls = singleton_mask & (singleton_label == cls)
        soft_cls = ambiguous_mask & candidate_set[:, cls]
        current = int((hard_cls | soft_cls).sum())
        min_ratio = _class_value(config, "min_fg_pixels_per_class_ratio", cls, min_ratio_default)
        min_pixels = max(min_pixels_cfg, int(round(total_pixels * min_ratio)))
        max_ratio = _class_value(config, "max_fg_candidate_ratio_per_class", cls, float(config.get("max_foreground_candidate_ratio", 1.0)))
        max_pixels = max(min_pixels, int(round(total_pixels * max_ratio)))
        max_pixels = max(0, min(max_pixels, total_pixels))
        hard_count = int(hard_cls.sum())
        score = foreground_score[:, cls] + verifier_weight * verifier_score
        if hard_count > max_pixels and bool(config.get("foreground_ceiling_demote_hard", True)):
            keep_hard = _topk_mask(score, hard_cls, max_pixels)
            drop_hard = hard_cls & ~keep_hard
            singleton_mask = singleton_mask & ~drop_hard
            candidate_set[:, cls] = candidate_set[:, cls] & ~drop_hard
            hard_cls = keep_hard
            hard_count = int(hard_cls.sum())
            flood_classes += 1
        keep_soft = max(0, max_pixels - hard_count)
        if current > max_pixels:
            keep = _topk_mask(score, soft_cls, keep_soft)
            drop = soft_cls & ~keep
            candidate_set[:, cls] = candidate_set[:, cls] & ~drop
            flood_classes += 1
        ceiling_stats[f"foreground_ceiling_max_pixels_class{cls}"] = float(max_pixels)
        ceiling_stats[f"foreground_ceiling_before_ratio_class{cls}"] = float(((hard_cls | soft_cls).float().mean()).detach())
        ceiling_stats[f"foreground_ceiling_after_ratio_class{cls}"] = float(((hard_cls | (ambiguous_mask & candidate_set[:, cls])).float().mean()).detach())

    fg_any = candidate_set[:, 1:].any(dim=1) if candidate_set.shape[1] > 1 else torch.zeros_like(singleton_mask)
    orphan = old_fg_any & ~fg_any
    background_from_ceiling = torch.zeros_like(singleton_mask, dtype=torch.bool)
    if bool(config.get("use_background_from_foreground_ceiling", True)) and int(orphan.sum()) > 0:
        bg_min_conf = float(config.get("background_candidate_min_confidence", config.get("min_teacher_confidence", 0.50)))
        max_bg_ratio = float(config.get("max_background_from_ceiling_ratio", config.get("max_background_hard_ratio", 0.35)))
        max_bg = int(round(total_pixels * max(0.0, max_bg_ratio)))
        bg_eligible = orphan & (teacher_prob[:, 0] >= bg_min_conf)
        background_from_ceiling = _topk_mask(teacher_prob[:, 0], bg_eligible, max_bg)
        if int(background_from_ceiling.sum()) > 0:
            candidate_set[:, 0] = candidate_set[:, 0] | background_from_ceiling
            singleton_label = torch.where(background_from_ceiling, torch.zeros_like(singleton_label), singleton_label)
            singleton_mask = (singleton_mask & ~background_from_ceiling) | background_from_ceiling
            ambiguous_mask = ambiguous_mask & ~background_from_ceiling

    ambiguous_mask = ambiguous_mask & (candidate_set.sum(dim=1) > 0) & ~singleton_mask
    ceiling_stats["foreground_ceiling_flood_class_count"] = float(flood_classes)
    ceiling_stats["candidate_foreground_after_ceiling_ratio"] = float(fg_any.float().mean().detach())
    ceiling_stats["background_from_ceiling_ratio"] = float(background_from_ceiling.float().mean().detach())
    ceiling_stats["empty_candidate_after_ceiling_ratio"] = float((candidate_set.sum(dim=1) == 0).float().mean().detach())
    ceiling_stats.update(_foreground_participation_stats(singleton_label, singleton_mask, candidate_set, ambiguous_mask, fg_classes))
    return candidate_set, singleton_label, singleton_mask, ambiguous_mask, ceiling_stats


def _cap_safe_negative_set(
    raw_negative_set: torch.Tensor,
    candidate_set: torch.Tensor,
    singleton_label: torch.Tensor,
    singleton_mask: torch.Tensor,
    ambiguous_mask: torch.Tensor,
    teacher_prob: torch.Tensor,
    sam_support: torch.Tensor,
    fg_classes: list[int],
    config: dict,
) -> tuple[torch.Tensor, dict]:
    if not bool(config.get("bounded_safe_negative", False)):
        return raw_negative_set, {"safe_negative_budget_active": 0.0}

    bounded = torch.zeros_like(raw_negative_set, dtype=torch.bool)
    total_pixels = int(singleton_mask.numel())
    max_ratio = float(config.get("max_safe_negative_ratio_per_class", 0.35))
    neg_to_pos = float(config.get("safe_negative_to_positive_ratio", config.get("max_negative_to_positive_ratio", 2.0)))
    min_neg_cfg = int(config.get("safe_negative_min_pixels_per_class", 0))
    allow_absent = bool(config.get("allow_negative_without_positive", False))
    stats: dict[str, float] = {"safe_negative_budget_active": 1.0}

    for cls in fg_classes:
        if not (0 < cls < raw_negative_set.shape[1]):
            continue
        positive = (singleton_mask & (singleton_label == cls)) | (ambiguous_mask & candidate_set[:, cls])
        positive_count = int(positive.sum())
        if positive_count <= 0 and not allow_absent:
            max_keep = 0
        else:
            max_by_total = int(round(total_pixels * max_ratio))
            max_by_positive = int(round(max(1, positive_count) * neg_to_pos))
            max_keep = min(max_by_total, max_by_positive)
            if positive_count > 0:
                max_keep = max(max_keep, min_neg_cfg)
        eligible = raw_negative_set[:, cls]
        negative_reliability = (1.0 - teacher_prob[:, cls]).clamp(0.0, 1.0) * (1.0 - sam_support[:, cls]).clamp(0.0, 1.0)
        keep = _topk_mask(negative_reliability, eligible, max_keep)
        bounded[:, cls] = keep
        stats[f"safe_negative_budget_class{cls}"] = float(max_keep)
        stats[f"safe_negative_positive_pixels_class{cls}"] = float(positive_count)
        stats[f"safe_negative_raw_ratio_class{cls}"] = float(eligible.float().mean().detach())
        stats[f"safe_negative_kept_ratio_class{cls}"] = float(keep.float().mean().detach())
    return bounded, stats


def build_foreground_safe_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    teacher_prob = teacher_out["mean_prob"].detach()
    if teacher_prob.ndim != 4:
        raise ValueError(f"teacher mean_prob must be BCHW, got {tuple(teacher_prob.shape)}")
    device = teacher_prob.device
    dtype = teacher_prob.dtype
    bsz, num_classes, height, width = teacher_prob.shape
    fg_classes = _foreground_classes(config, num_classes)
    iter_now = int(config.get("_iteration", 0))
    disable_bg_until = int(config.get("disable_background_unsup_until", config.get("foreground_grounding_start", 1200)))
    use_background_hard = iter_now >= disable_bg_until
    teacher_prob, prior_alignment_stats = _align_teacher_to_labeled_prior(teacher_prob, config)

    teacher_thresh = _threshold_vec(calibrator, "teacher_q", config.get("min_teacher_confidence", 0.5), num_classes, device, dtype)
    sam_thresh = _threshold_vec(calibrator, "sam_q", config.get("min_sam_confidence", 0.5), num_classes, device, dtype)
    sam_struct = build_sam_structural_support(
        sam_out,
        teacher_prob,
        foreground_classes=fg_classes,
        min_support=float(config.get("min_sam_structural_support", 0.0)),
    )
    sam_support = sam_struct["support"]
    sam_valid = bool(sam_struct["valid"])
    fg_support = sam_struct["foreground_support"]
    verifier_score = sam_struct.get("verifier_score", fg_support).to(device=device, dtype=dtype)
    min_verifier_score = float(config.get("min_sam_verifier_score", config.get("min_structural_verifier_score", 0.20)))
    sam_as_verifier = str(config.get("sam_role", "teacher")).lower() in {"verifier", "structural_verifier", "save"}

    teacher_candidate, teacher_low = calibrator.prediction_sets(teacher_prob)
    candidate_set = torch.zeros_like(teacher_candidate, dtype=torch.bool)
    foreground_score = teacher_prob.clone()
    if sam_valid and num_classes > 1:
        verified_sam_support = sam_support[:, 1:] * verifier_score.unsqueeze(1)
        foreground_score[:, 1:] = torch.maximum(teacher_prob[:, 1:], verified_sam_support)

    min_sam_conf = float(config.get("min_sam_confidence", 0.5))
    min_teacher_conf = float(config.get("min_teacher_confidence", 0.5))
    min_fg_score = float(config.get("min_foreground_score", 0.02))
    for cls in fg_classes:
        teacher_fg = teacher_candidate[:, cls] | (teacher_prob[:, cls] >= min_teacher_conf)
        if sam_valid:
            sam_fg = (sam_support[:, cls] >= min_sam_conf) & (verifier_score >= min_verifier_score)
            candidate_set[:, cls] = teacher_fg | sam_fg
        else:
            candidate_set[:, cls] = teacher_fg

    reliable_fg = torch.zeros_like(candidate_set)
    for cls in fg_classes:
        if sam_valid and sam_as_verifier:
            reliable_fg[:, cls] = (
                (teacher_prob[:, cls] >= torch.maximum(teacher_thresh[cls], teacher_prob.new_tensor(min_teacher_conf)))
                & (verifier_score >= min_verifier_score)
                & (foreground_score[:, cls] >= min_fg_score)
            ) | (
                (sam_support[:, cls] >= torch.maximum(sam_thresh[cls], teacher_prob.new_tensor(min_sam_conf)))
                & (teacher_prob[:, cls] >= min_fg_score)
                & (verifier_score >= min_verifier_score)
            )
        elif sam_valid:
            reliable_fg[:, cls] = (
                (teacher_prob[:, cls] >= torch.maximum(teacher_thresh[cls], teacher_prob.new_tensor(min_teacher_conf)))
                & (sam_support[:, cls] >= torch.maximum(sam_thresh[cls], teacher_prob.new_tensor(min_sam_conf)))
                & (foreground_score[:, cls] >= min_fg_score)
            )
        else:
            reliable_fg[:, cls] = teacher_prob[:, cls] >= min_teacher_conf

    reliable_fg_any = reliable_fg[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros((bsz, height, width), device=device, dtype=torch.bool)
    fg_score_max, fg_label = foreground_score[:, 1:].max(dim=1) if num_classes > 1 else (teacher_prob.new_zeros((bsz, height, width)), torch.zeros((bsz, height, width), device=device, dtype=torch.long))
    fg_label = fg_label + 1
    teacher_conf, teacher_label = teacher_prob.max(dim=1)
    fallback_fg_label = torch.where(teacher_label > 0, teacher_label, fg_label)
    singleton_label = torch.where(reliable_fg_any, fallback_fg_label, teacher_label)

    fg_low = float(config.get("sam_foreground_low", 0.15))
    bg_thresh = float(config.get("background_confidence", config.get("min_teacher_confidence", 0.5)))
    has_fg_candidate = candidate_set[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(teacher_label, dtype=torch.bool)
    reliable_background = (
        use_background_hard
        & (teacher_prob[:, 0] >= bg_thresh)
        & (fg_support < fg_low)
        & (verifier_score < float(config.get("background_max_verifier_score", min_verifier_score)))
        & ~has_fg_candidate
        & ~reliable_fg_any
    )
    candidate_set[:, 0] = reliable_background
    singleton_label = torch.where(reliable_background, torch.zeros_like(singleton_label), singleton_label)
    singleton_mask = reliable_fg_any | reliable_background

    max_set = int(config.get("max_candidate_set_size", 2))
    if max_set > 0 and num_classes > 1:
        score_for_topk = teacher_prob.clone()
        support_boost = float(config.get("sam_fuzzy_support_weight", 0.25))
        score_for_topk[:, 1:] = torch.maximum(score_for_topk[:, 1:], sam_support[:, 1:] * support_boost)
        _, topi = score_for_topk.topk(k=min(max_set, num_classes), dim=1)
        top_candidate = torch.zeros_like(candidate_set)
        top_candidate.scatter_(1, topi, True)
        candidate_set = candidate_set & top_candidate
        candidate_set[:, 0] = candidate_set[:, 0] & reliable_background

    empty_before_fallback = candidate_set.sum(dim=1) == 0
    empty_fg_fallback = torch.zeros((bsz, height, width), device=device, dtype=torch.bool)
    empty_fg_fallback_raw = torch.zeros((bsz, height, width), device=device, dtype=torch.bool)
    if empty_before_fallback.any() and num_classes > 1:
        support_boost = float(config.get("sam_fuzzy_support_weight", 0.25))
        fallback_score = torch.maximum(teacher_prob[:, 1:], sam_support[:, 1:] * support_boost)
        fallback_conf, _ = fallback_score.max(dim=1)
        min_empty_fg = float(config.get("min_empty_foreground_score", config.get("min_foreground_score", 0.02)))
        empty_fg_fallback_raw = empty_before_fallback & (fallback_conf >= min_empty_fg)
        topk_fg = _topk_foreground_candidates(
            fallback_score,
            int(config.get("empty_candidate_topk_foreground", 1)),
        )
        if bool(config.get("bounded_empty_foreground_fallback", False)):
            total_pixels = int(empty_before_fallback.numel())
            scale = float(config.get("empty_foreground_fallback_cap_scale", 1.0))
            for cls in fg_classes:
                rel_cls = cls - 1
                if rel_cls < 0 or rel_cls >= topk_fg.shape[1]:
                    continue
                max_ratio = _class_value(
                    config,
                    "max_fg_candidate_ratio_per_class",
                    cls,
                    float(config.get("max_foreground_candidate_ratio", 1.0)),
                )
                max_pixels = max(0, min(total_pixels, int(round(total_pixels * max_ratio * scale))))
                eligible = empty_fg_fallback_raw & topk_fg[:, rel_cls]
                keep = _topk_mask(fallback_score[:, rel_cls], eligible, max_pixels)
                candidate_set[:, cls] = candidate_set[:, cls] | keep
                empty_fg_fallback = empty_fg_fallback | keep
        else:
            empty_fg_fallback = empty_fg_fallback_raw
            candidate_set[:, 1:] = candidate_set[:, 1:] | (topk_fg & empty_fg_fallback.unsqueeze(1))
    has_fg_candidate = candidate_set[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(teacher_label, dtype=torch.bool)
    candidate_count = candidate_set.sum(dim=1)
    ambiguous_mask = ((candidate_count > 1) | (~singleton_mask & has_fg_candidate) | teacher_low) & (candidate_count > 0)
    ambiguous_mask = ambiguous_mask & ~reliable_fg_any
    conflict_mask = (teacher_label == 0) & has_fg_candidate & ~reliable_background

    singleton_label, singleton_mask, candidate_set, ambiguous_mask, budget_stats = apply_foreground_budget(
        singleton_label=singleton_label,
        singleton_mask=singleton_mask,
        candidate_set=candidate_set,
        ambiguous_mask=ambiguous_mask,
        foreground_score=torch.maximum(foreground_score, teacher_prob * candidate_set.float()),
        teacher_prob=teacher_prob,
        config=config,
    )
    candidate_set, singleton_label, singleton_mask, ambiguous_mask, ceiling_stats = _cap_foreground_candidate_set(
        candidate_set=candidate_set,
        singleton_label=singleton_label,
        singleton_mask=singleton_mask,
        ambiguous_mask=ambiguous_mask,
        foreground_score=torch.maximum(foreground_score, teacher_prob * candidate_set.float()),
        teacher_prob=teacher_prob,
        verifier_score=verifier_score,
        fg_classes=fg_classes,
        config=config,
    )
    has_fg_candidate = candidate_set[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(teacher_label, dtype=torch.bool)
    ambiguous_mask = ambiguous_mask & (candidate_set.sum(dim=1) > 0)
    conflict_mask = (teacher_label == 0) & has_fg_candidate & ~reliable_background

    raw_negative_set = torch.zeros_like(candidate_set, dtype=torch.bool)
    rank_pos = _rank_positions(teacher_prob)
    rank_low = int(config.get("safe_negative_rank_low", max(1, max_set)))
    rank_high = int(config.get("safe_negative_rank_high", num_classes - 1))
    negative_max_prob = float(config.get("safe_negative_max_prob", config.get("safe_negative_threshold", 0.40)))
    sam_veto_threshold = float(config.get("safe_negative_sam_threshold", config.get("safe_negative_sam_veto_threshold", 0.30)))
    for cls in fg_classes:
        weak_sam_veto = sam_support[:, cls] < sam_veto_threshold if sam_valid else torch.ones_like(candidate_set[:, cls], dtype=torch.bool)
        raw_negative_set[:, cls] = (
            ~candidate_set[:, cls]
            & (rank_pos[:, cls] >= rank_low)
            & (rank_pos[:, cls] <= rank_high)
            & (teacher_prob[:, cls] <= negative_max_prob)
            & weak_sam_veto
        )
    negative_set, negative_budget_stats = _cap_safe_negative_set(
        raw_negative_set,
        candidate_set,
        singleton_label,
        singleton_mask,
        ambiguous_mask,
        teacher_prob,
        sam_support,
        fg_classes,
        config,
    )
    negative_mask = negative_set.any(dim=1) | conflict_mask

    soft_score = teacher_prob.clone()
    soft_score[:, 1:] = torch.maximum(soft_score[:, 1:], sam_support[:, 1:] * float(config.get("sam_fuzzy_support_weight", 0.25)))
    soft_score = soft_score * candidate_set.float()
    soft_empty = soft_score.sum(dim=1, keepdim=True) <= 1e-6
    soft_score = torch.where(soft_empty, teacher_prob, soft_score)
    soft_target = soft_score / soft_score.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_only_soft_target = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

    candidate_weight = torch.maximum(teacher_conf, fg_support).clamp(0.05, 1.0)
    safe_negative_weight = negative_mask.float().clamp(0.0, 1.0)
    foreground_seed = reliable_fg.bool()
    foreground_seed_mask = foreground_seed[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(singleton_mask)
    fuzzy_region = ambiguous_mask & (candidate_set[:, 1:].any(dim=1) if num_classes > 1 else ambiguous_mask)
    candidate_foreground_mask = candidate_set[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(singleton_mask)
    boundary = sam_struct["boundary"]
    boundary_score = boundary.max(dim=1).values if boundary.ndim == 4 else boundary
    boundary_uncertain = boundary_score >= float(config.get("sam_boundary_uncertain_threshold", 0.10))
    semantic_gate = singleton_mask | ambiguous_mask | conflict_mask
    structure_support_min = float(config.get("sam_structure_mask_min_support", fg_low))
    sam_support_gate = fg_support >= structure_support_min
    sam_region_gate = (
        candidate_foreground_mask
        | fuzzy_region
        | conflict_mask
        | sam_support_gate
        | boundary_uncertain
    ) & bool(sam_valid)
    sam_train_gate = sam_region_gate
    structure_gate = sam_support_gate | foreground_seed_mask | fuzzy_region | conflict_mask | boundary_uncertain
    sam_structure_support_mask = structure_gate & (
        candidate_foreground_mask | sam_support_gate | fuzzy_region | conflict_mask | boundary_uncertain
    )
    sam_weight = (foreground_score[:, 1:].max(dim=1).values if (sam_valid and num_classes > 1) else teacher_prob.new_zeros((bsz, height, width))).clamp(0.0, 1.0)
    sam_weight = torch.maximum(sam_weight, fg_support).clamp(0.0, 1.0)
    sam_weight = torch.maximum(sam_weight, verifier_score * candidate_foreground_mask.float()).clamp(0.0, 1.0)
    min_sam_region_weight = float(config.get("sam_region_min_weight", 0.05))
    sam_weight = torch.where(sam_region_gate, sam_weight.clamp_min(min_sam_region_weight), sam_weight.new_zeros(sam_weight.shape))
    structure_weight = torch.maximum(sam_weight, fg_support).clamp(0.0, 1.0)

    per_class_participation = [0.0 for _ in range(num_classes)]
    per_class_foreground_participation = [0.0 for _ in range(num_classes)]
    per_class_safe_negative = [0.0 for _ in range(num_classes)]
    for cls in fg_classes:
        fg_participates = foreground_seed[:, cls] | (fuzzy_region & candidate_set[:, cls])
        per_class_foreground_participation[cls] = float(fg_participates.float().mean().detach())
        if sam_valid:
            per_class_participation[cls] = float(((sam_support[:, cls] >= min_sam_conf) & fg_participates).float().mean().detach())
        per_class_safe_negative[cls] = float(negative_set[:, cls].float().mean().detach())

    stats = {
        "singleton_ratio": float(singleton_mask.float().mean().detach()),
        "singleton_pixel_ratio": float(singleton_mask.float().mean().detach()),
        "ambiguous_ratio": float(ambiguous_mask.float().mean().detach()),
        "ambiguous_pixel_ratio": float(ambiguous_mask.float().mean().detach()),
        "conflict_ratio": float(conflict_mask.float().mean().detach()),
        "negative_ratio": float(negative_mask.float().mean().detach()),
        "safe_negative_pixel_ratio": float(negative_set.any(dim=1).float().mean().detach()),
        "per_class_safe_negative_ratio": per_class_safe_negative,
        "avg_set_size": float(candidate_set.float().sum(dim=1).mean().detach()),
        "empty_candidate_ratio": float((candidate_set.sum(dim=1) == 0).float().mean().detach()),
        "empty_foreground_fallback_ratio": float(empty_fg_fallback.float().mean().detach()),
        "empty_foreground_fallback_raw_ratio": float(empty_fg_fallback_raw.float().mean().detach()),
        "candidate_foreground_ratio": float(candidate_foreground_mask.float().mean().detach()),
        "boundary_uncertain_ratio": float(boundary_uncertain.float().mean().detach()),
        "sam_semantic_gate_ratio": float(semantic_gate.float().mean().detach()),
        "sam_structure_gate_ratio": float(structure_gate.float().mean().detach()),
        "sam_support_gate_ratio": float(sam_support_gate.float().mean().detach()),
        "sam_structure_support_mask_ratio": float(sam_structure_support_mask.float().mean().detach()),
        "sam_train_gate_ratio": float(sam_train_gate.float().mean().detach()),
        "sam_soft_weight_mean": float(sam_weight.mean().detach()),
        "sam_soft_weight_p25": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.25)),
        "sam_soft_weight_p50": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.50)),
        "sam_soft_weight_p75": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.75)),
        "sam_participation_ratio": float(sam_train_gate.float().mean().detach()),
        "per_class_sam_participation_ratio": per_class_participation,
        "per_class_foreground_participation_ratio": per_class_foreground_participation,
        "sam_teacher_agreement": float(((teacher_label == singleton_label) | ambiguous_mask).float().mean().detach()),
        "sam_foreground_support_ratio": float((fg_support >= fg_low).float().mean().detach()),
        "sam_verifier_score_mean": float(verifier_score.mean().detach()),
        "sam_verifier_gate_ratio": float((verifier_score >= min_verifier_score).float().mean().detach()),
        **budget_stats,
        **ceiling_stats,
        **negative_budget_stats,
        **prior_alignment_stats,
    }
    for cls in fg_classes:
        stats[f"safe_negative_ratio_class{cls}"] = per_class_safe_negative[cls]

    return {
        "singleton_label": singleton_label.detach(),
        "singleton_mask": singleton_mask.detach(),
        "candidate_set": candidate_set.detach(),
        "candidate_weight": candidate_weight.detach(),
        "ambiguous_mask": ambiguous_mask.detach(),
        "fuzzy_region": fuzzy_region.detach(),
        "conflict_mask": conflict_mask.detach(),
        "negative_set": negative_set.detach(),
        "safe_negative_set": negative_set.detach(),
        "negative_mask": negative_mask.detach(),
        "safe_negative_weight": safe_negative_weight.detach(),
        "semantic_gate": semantic_gate.detach(),
        "sam_train_gate": sam_train_gate.detach(),
        "sam_region_gate": sam_region_gate.detach(),
        "structure_gate": structure_gate.detach(),
        "sam_structure_support_mask": sam_structure_support_mask.detach(),
        "sam_weight": sam_weight.detach(),
        "teacher_weight": teacher_conf.detach(),
        "semantic_weight": candidate_weight.detach(),
        "structure_weight": structure_weight.detach(),
        "teacher_reliable_mask": foreground_seed_mask.detach(),
        "foreground_seed": foreground_seed.detach(),
        "foreground_seed_mask": foreground_seed_mask.detach(),
        "sam_support": sam_support.detach(),
        "sam_foreground_support": fg_support.detach(),
        "sam_verifier_score": verifier_score.detach(),
        "sam_boundary": sam_struct["boundary"].detach(),
        "reliable_background_mask": reliable_background.detach(),
        "soft_target": soft_target.detach(),
        "teacher_only_soft_target": teacher_only_soft_target.detach(),
        "stats": stats,
    }


def build_set_valued_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    return build_foreground_safe_targets(teacher_out, sam_out, calibrator, config)
