from __future__ import annotations

import torch
import torch.nn.functional as F

from .correlation_propagation import correlation_propagation_loss, propagate_correlation_targets


def _as_spatial_bool(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    out = value.bool()
    if out.ndim == 4:
        out = out.any(dim=1)
    return out


def build_foreground_structure_mask(targets: dict) -> torch.Tensor | None:
    """Shared broad foreground route for SAM KD, locality, and propagation.

    Hard foreground seeds are intentionally only one possible source.  Candidate
    foreground, fuzzy set-valued pixels, and SAM structure gates must also keep
    the structural training path alive.
    """

    seed = None
    candidate_set = targets.get("candidate_set")
    if candidate_set is not None and candidate_set.ndim == 4 and candidate_set.shape[1] > 1:
        seed = candidate_set[:, 1:].any(dim=1).bool()

    for key in ("foreground_seed_mask", "fuzzy_region", "structure_gate"):
        value = _as_spatial_bool(targets.get(key))
        if value is None:
            continue
        seed = value if seed is None else (seed | value.to(device=seed.device))

    return seed


@torch.no_grad()
def propagate_foreground_correlation_targets(
    feature_fusion: torch.Tensor,
    prob_fusion: torch.Tensor,
    targets: dict,
    sam_shape: torch.Tensor | None = None,
    resolution: int = 16,
    topk: int = 8,
    temperature: float = 0.2,
    min_weight: float = 0.15,
):
    prob_fg = prob_fusion.detach().clone()
    if prob_fg.shape[1] > 1:
        prob_fg[:, 0] = 0.0
    seed = build_foreground_structure_mask(targets)
    if seed is None:
        seed = targets.get("singleton_mask", None)
    propagated = propagate_correlation_targets(
        feature_fusion,
        prob_fg,
        sam_shape=sam_shape,
        reliable_mask=seed,
        resolution=resolution,
        topk=topk,
        temperature=temperature,
        min_weight=min_weight,
    )
    foreground = propagated["propagated_label"] > 0
    propagated["expanded_reliable_mask"] = propagated["expanded_reliable_mask"] & foreground
    propagated["propagated_weight"] = propagated["propagated_weight"] * foreground.float()
    return propagated


@torch.no_grad()
def expand_targets_with_correlation(targets: dict, propagated: dict, min_weight: float = 0.15) -> dict:
    labels = propagated["propagated_label"].to(targets["candidate_set"].device).long()
    weights = propagated["propagated_weight"].to(targets["candidate_set"].device).float()
    mask = propagated["expanded_reliable_mask"].to(targets["candidate_set"].device).bool()
    mask = mask & (weights >= float(min_weight)) & (labels > 0)
    if int(mask.sum()) == 0:
        out = dict(targets)
        stats = dict(targets.get("stats", {}))
        stats.setdefault("foreground_propagated_ratio", 0.0)
        out["stats"] = stats
        return out

    out = dict(targets)
    candidate_set = targets["candidate_set"].clone().bool()
    add = torch.zeros_like(candidate_set)
    add.scatter_(1, labels.unsqueeze(1).clamp(min=0, max=candidate_set.shape[1] - 1), True)
    add[:, 0] = False
    add = add & mask.unsqueeze(1)
    candidate_set = candidate_set | add

    ambiguous = (targets["ambiguous_mask"].bool() | (mask & ~targets["singleton_mask"].bool())) & (candidate_set.sum(dim=1) > 0)
    fuzzy = ambiguous & (candidate_set[:, 1:].any(dim=1) if candidate_set.shape[1] > 1 else ambiguous)
    negative_set = targets["negative_set"].bool() & ~candidate_set
    negative_mask = negative_set.any(dim=1) | targets["conflict_mask"].bool()

    teacher_soft = targets["teacher_only_soft_target"]
    soft_score = teacher_soft * candidate_set.float()
    soft_empty = soft_score.sum(dim=1, keepdim=True) <= 1e-6
    soft_score = torch.where(soft_empty, teacher_soft, soft_score)
    soft_target = soft_score / soft_score.sum(dim=1, keepdim=True).clamp_min(1e-6)

    candidate_weight = torch.maximum(targets["candidate_weight"].float(), weights.clamp(0.0, 1.0))
    old_sam_gate = targets.get("sam_region_gate", targets.get("sam_train_gate")).bool()
    sam_gate = old_sam_gate | mask
    structure_gate = targets["structure_gate"].bool() | mask
    sam_weight = torch.where(mask, torch.maximum(targets["sam_weight"].float(), weights), targets["sam_weight"].float())

    out.update(
        {
            "candidate_set": candidate_set.detach(),
            "candidate_weight": candidate_weight.detach(),
            "ambiguous_mask": ambiguous.detach(),
            "fuzzy_region": fuzzy.detach(),
            "negative_set": negative_set.detach(),
            "safe_negative_set": negative_set.detach(),
            "negative_mask": negative_mask.detach(),
            "safe_negative_weight": negative_mask.float().detach(),
            "soft_target": soft_target.detach(),
            "sam_train_gate": sam_gate.detach(),
            "sam_region_gate": sam_gate.detach(),
            "structure_gate": structure_gate.detach(),
            "sam_weight": sam_weight.detach(),
            "structure_weight": torch.maximum(out["structure_weight"].float(), sam_weight).clamp(0.0, 1.0).detach(),
        }
    )
    stats = dict(targets.get("stats", {}))
    stats["foreground_propagated_ratio"] = float(mask.float().mean().detach())
    stats["avg_set_size"] = float(candidate_set.float().sum(dim=1).mean().detach())
    stats["pseudo_set_size_mean"] = stats["avg_set_size"]
    stats["sam_train_gate_ratio"] = float(sam_gate.float().mean().detach())
    stats["sam_participation_ratio"] = stats["sam_train_gate_ratio"]
    stats["negative_ratio"] = float(negative_mask.float().mean().detach())
    stats["safe_negative_pixel_ratio"] = float(negative_set.any(dim=1).float().mean().detach())
    out["stats"] = stats
    return out


def foreground_correlation_loss(logits: torch.Tensor, propagated: dict):
    return correlation_propagation_loss(logits, propagated)


@torch.no_grad()
def build_masked_locality_view(
    images: torch.Tensor,
    foreground_seed_mask: torch.Tensor | None,
    mask_ratio: float = 0.30,
    patch_size: int = 16,
    fill: str = "mean",
) -> tuple[torch.Tensor, dict]:
    """Mask a random subset of reliable foreground-local patches.

    This is intentionally target-aware: the mask is sampled only from foreground
    seed patches so the auxiliary task asks the model to recover calibrated
    local structure instead of hiding already-unreliable background regions.
    """

    if foreground_seed_mask is None:
        return images, {"masked_locality_ratio": 0.0, "foreground_masked_ratio": 0.0}
    if images.ndim != 4:
        raise ValueError(f"images must be BCHW, got {tuple(images.shape)}")
    bsz, _, height, width = images.shape
    fg = foreground_seed_mask.to(device=images.device).bool()
    if fg.ndim == 4:
        fg = fg.any(dim=1)
    if fg.shape[-2:] != (height, width):
        fg = F.interpolate(fg.float().unsqueeze(1), size=(height, width), mode="nearest").squeeze(1).bool()
    if int(fg.sum()) == 0 or mask_ratio <= 0.0:
        return images, {"masked_locality_ratio": 0.0, "foreground_masked_ratio": 0.0}

    patch = max(1, int(patch_size))
    pooled = F.max_pool2d(fg.float().unsqueeze(1), kernel_size=patch, stride=patch, ceil_mode=True).squeeze(1).bool()
    rand = torch.rand_like(pooled.float())
    selected_low = pooled & (rand < float(mask_ratio))
    if int(selected_low.sum()) == 0:
        for bi in range(bsz):
            eligible = pooled[bi].flatten().nonzero(as_tuple=False).squeeze(1)
            if eligible.numel() > 0:
                selected_low[bi].flatten()[eligible[torch.randint(eligible.numel(), (1,), device=eligible.device)]] = True
    selected = F.interpolate(selected_low.float().unsqueeze(1), size=(height, width), mode="nearest").squeeze(1).bool()
    selected = selected & fg
    if int(selected.sum()) == 0:
        return images, {"masked_locality_ratio": 0.0, "foreground_masked_ratio": 0.0}

    if fill == "zero":
        fill_value = torch.zeros_like(images)
    else:
        fill_value = images.mean(dim=(2, 3), keepdim=True).expand_as(images)
    masked = torch.where(selected.unsqueeze(1), fill_value, images)
    return masked, {
        "masked_locality_ratio": float(selected.float().mean().detach()),
        "foreground_masked_ratio": float((selected.float().sum() / fg.float().sum().clamp_min(1.0)).detach()),
    }


def masked_locality_proxy_loss(logits: torch.Tensor, targets: dict, rank_margin: float = 0.5):
    from r6.losses.tri_state_pseudo_loss import tri_state_pseudo_supervision_loss

    losses = tri_state_pseudo_supervision_loss(logits, targets, rank_margin)
    return losses["loss_hard_fg"] + losses["loss_fuzzy"] + losses["loss_set"]
