from __future__ import annotations

import math

import torch


def _stats_defaults() -> dict[str, float]:
    return {
        "hard_pseudo_reliability_active": 0.0,
        "singleton_weight_mean": 1.0,
        "singleton_weight_fg_mean": 1.0,
        "singleton_weight_bg_mean": 1.0,
        "singleton_entropy_mean": 0.0,
        "hard_pseudo_fg_demoted_ratio": 0.0,
        "hard_pseudo_bg_conflict_ratio": 0.0,
        "hard_pseudo_sam_agree_fg_ratio": 0.0,
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, default: float = 0.0) -> float:
    selected = value[mask]
    if selected.numel() == 0:
        return float(default)
    return float(selected.float().mean().detach())


def _normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    classes = max(2, int(prob.shape[1]))
    entropy = -(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(dim=1)
    return (entropy / math.log(classes)).clamp(0.0, 1.0)


def _gather_label_value(value: torch.Tensor | None, labels: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    if value is None or value.ndim != 4:
        return default
    index = labels.clamp(min=0, max=value.shape[1] - 1).unsqueeze(1)
    return value.gather(1, index).squeeze(1)


def apply_hard_pseudo_reliability(
    targets: dict,
    config: dict,
    iteration: int,
) -> tuple[dict, dict[str, float]]:
    """Downweight or relax brittle hard pseudo labels before SSL losses.

    The target builder already separates hard singleton, ambiguous set-valued,
    and safe-negative pixels. This module does not create new positives; it only
    reduces hard-label pressure when uncertainty or SAM disagreement suggests
    that a singleton label should remain soft.
    """

    stats = _stats_defaults()
    if not bool(config.get("enabled", False)):
        return targets, stats
    if iteration < int(config.get("start_iter", 0)):
        return targets, stats

    singleton_mask = targets.get("singleton_mask")
    labels = targets.get("singleton_label")
    candidate_set = targets.get("candidate_set")
    ambiguous_mask = targets.get("ambiguous_mask")
    if singleton_mask is None or labels is None or candidate_set is None or ambiguous_mask is None:
        return targets, stats

    out = dict(targets)
    singleton_mask = singleton_mask.bool().clone()
    labels = labels.long()
    candidate_set = candidate_set.bool().clone()
    ambiguous_mask = ambiguous_mask.bool().clone()
    device = singleton_mask.device
    dtype = targets.get("teacher_weight", singleton_mask.float()).float().dtype

    selected = singleton_mask
    if not bool(selected.any()):
        out["singleton_weight"] = singleton_mask.float()
        stats["hard_pseudo_reliability_active"] = 1.0
        return out, stats

    teacher_weight = targets.get("teacher_weight")
    if teacher_weight is None:
        teacher_weight = torch.ones_like(singleton_mask, dtype=torch.float32, device=device)
    teacher_weight = teacher_weight.to(device=device).float().clamp(0.0, 1.0)

    teacher_prob = targets.get("teacher_only_soft_target", targets.get("soft_target"))
    if teacher_prob is not None and teacher_prob.ndim == 4:
        teacher_prob = teacher_prob.to(device=device).float()
        entropy = _normalized_entropy(teacher_prob)
    else:
        entropy = torch.zeros_like(teacher_weight)

    entropy_discount = float(config.get("entropy_discount", 0.45))
    min_entropy_scale = float(config.get("min_entropy_scale", 0.45))
    uncertainty_scale = (1.0 - entropy_discount * entropy).clamp(min=min_entropy_scale, max=1.0)
    weight = teacher_weight * uncertainty_scale

    verifier = targets.get("sam_verifier_score")
    if verifier is None:
        verifier = torch.zeros_like(weight)
    verifier = verifier.to(device=device).float().clamp(0.0, 1.0)
    sam_support = targets.get("sam_support")
    sam_label_support = _gather_label_value(sam_support.to(device=device).float() if sam_support is not None else None, labels, weight.new_zeros(weight.shape))
    sam_label_quality = (sam_label_support * verifier).clamp(0.0, 1.0)

    foreground = selected & (labels > 0)
    background = selected & (labels == 0)
    sam_agree_min = float(config.get("sam_agree_min_quality", 0.18))
    sam_agree = foreground & (sam_label_quality >= sam_agree_min)
    sam_bonus = float(config.get("sam_bonus", 0.15))
    if foreground.any():
        fg_weight = torch.maximum(weight, sam_label_quality)
        fg_weight = (fg_weight + sam_bonus * sam_label_quality).clamp(0.0, 1.0)
        weight = torch.where(foreground, fg_weight, weight)

    bg_scale = float(config.get("background_scale", 0.65))
    bg_decay_start = int(config.get("background_decay_start", 0))
    bg_decay_iters = max(1, int(config.get("background_decay_iterations", 1)))
    bg_min_scale = float(config.get("background_min_scale", bg_scale))
    if iteration > bg_decay_start:
        progress = min(1.0, max(0.0, (iteration - bg_decay_start) / bg_decay_iters))
        bg_scale = bg_scale + progress * (bg_min_scale - bg_scale)
    if background.any():
        weight = torch.where(background, weight * bg_scale, weight)

    fg_support = targets.get("sam_foreground_support")
    if fg_support is None:
        fg_support = torch.zeros_like(weight)
    fg_support = fg_support.to(device=device).float()
    bg_conflict = background & (fg_support >= float(config.get("background_sam_conflict_support", 0.08)))
    if bg_conflict.any():
        weight = torch.where(bg_conflict, weight * float(config.get("background_sam_conflict_scale", 0.35)), weight)

    disagreement = targets.get("sam_disagreement_mask")
    if disagreement is not None:
        disagreement = disagreement.to(device=device).bool()
        weight = torch.where(disagreement & selected, weight * float(config.get("sam_disagreement_scale", 0.55)), weight)

    raw_weight = weight.clone()
    min_weight = float(config.get("min_weight", 0.12))
    max_weight = float(config.get("max_weight", 1.0))
    fg_min_weight = float(config.get("foreground_min_weight", min_weight))
    weight = weight.clamp(min=min_weight, max=max_weight)
    weight = torch.where(foreground, weight.clamp_min(fg_min_weight), weight)

    demote = torch.zeros_like(singleton_mask)
    if bool(config.get("demote_low_foreground", True)) and foreground.any():
        demote = foreground & ~sam_agree & (
            (teacher_weight < float(config.get("demote_confidence_threshold", 0.55)))
            | (entropy > float(config.get("demote_entropy_threshold", 0.55)))
            | (raw_weight < float(config.get("demote_weight_threshold", 0.25)))
        )
        if demote.any():
            label_candidate = torch.zeros_like(candidate_set)
            label_candidate.scatter_(1, labels.clamp(min=0, max=candidate_set.shape[1] - 1).unsqueeze(1), True)
            candidate_set = candidate_set | (label_candidate & demote.unsqueeze(1))
            singleton_mask = singleton_mask & ~demote
            ambiguous_mask = ambiguous_mask | demote
            candidate_weight = out.get("candidate_weight")
            if candidate_weight is not None:
                relaxed_weight = torch.minimum(candidate_weight.to(device=device).float(), raw_weight.clamp(min=0.05, max=1.0))
                out["candidate_weight"] = torch.where(demote, relaxed_weight, candidate_weight.to(device=device).float()).detach()
                if out.get("semantic_weight") is not None:
                    out["semantic_weight"] = out["candidate_weight"]

    weight = torch.where(singleton_mask, weight, weight.new_zeros(weight.shape))
    out["singleton_mask"] = singleton_mask.detach()
    out["candidate_set"] = candidate_set.detach()
    ambiguous_mask = ambiguous_mask & (candidate_set.sum(dim=1) > 0) & ~singleton_mask
    out["ambiguous_mask"] = ambiguous_mask.detach()
    fg_candidate = candidate_set[:, 1:].any(dim=1) if candidate_set.shape[1] > 1 else ambiguous_mask
    out["fuzzy_region"] = (ambiguous_mask & fg_candidate).detach()
    out["singleton_weight"] = weight.to(dtype=dtype).detach()

    updated_target_stats = dict(targets.get("stats", {}))
    updated_target_stats.update(
        {
            "singleton_ratio": float(singleton_mask.float().mean().detach()),
            "singleton_pixel_ratio": float(singleton_mask.float().mean().detach()),
            "ambiguous_ratio": float(ambiguous_mask.float().mean().detach()),
            "ambiguous_pixel_ratio": float(ambiguous_mask.float().mean().detach()),
            "avg_set_size": float(candidate_set.float().sum(dim=1).mean().detach()),
            "pseudo_set_size_mean": float(candidate_set.float().sum(dim=1).mean().detach()),
            "candidate_foreground_ratio": float(fg_candidate.float().mean().detach()),
            "background_hard_ratio": float(((singleton_mask & (labels == 0)).float().mean()).detach()),
        }
    )
    for cls in range(candidate_set.shape[1]):
        if cls == 0:
            continue
        updated_target_stats[f"hard_fg_ratio_class{cls}"] = float(((singleton_mask & (labels == cls)).float().mean()).detach())
        updated_target_stats[f"soft_fg_ratio_class{cls}"] = float(((ambiguous_mask & candidate_set[:, cls]).float().mean()).detach())
    stats.update(
        {
            "hard_pseudo_reliability_active": 1.0,
            "singleton_weight_mean": _masked_mean(weight, selected, 0.0),
            "singleton_weight_fg_mean": _masked_mean(weight, foreground, 0.0),
            "singleton_weight_bg_mean": _masked_mean(weight, background, 0.0),
            "singleton_entropy_mean": _masked_mean(entropy, selected, 0.0),
            "hard_pseudo_fg_demoted_ratio": float(demote.float().mean().detach()),
            "hard_pseudo_bg_conflict_ratio": float(bg_conflict.float().mean().detach()),
            "hard_pseudo_sam_agree_fg_ratio": float(sam_agree.float().mean().detach()),
        }
    )
    updated_target_stats.update(stats)
    out["stats"] = updated_target_stats
    return out, stats
