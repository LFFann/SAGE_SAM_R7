from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _resize_spatial(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    if value.ndim == 3:
        value = value.unsqueeze(1)
        out = F.interpolate(value.float(), size=size, mode=mode)
        return out[:, 0]
    return F.interpolate(value.float(), size=size, mode=mode)


def foreground_prototype_anchor_loss(
    labeled_feature: torch.Tensor,
    labeled_mask: torch.Tensor,
    unlabeled_feature: torch.Tensor,
    targets: dict,
    *,
    foreground_classes: list[int] | tuple[int, ...] | None = None,
    temperature: float = 0.25,
    entropy_discount: float = 0.20,
    min_labeled_pixels: int = 8,
    min_unlabeled_pixels: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align reliable unlabeled foreground features to labeled class prototypes."""

    stats = {
        "prototype_anchor_active": 0.0,
        "prototype_anchor_valid_class_count": 0.0,
        "prototype_anchor_mask_ratio": 0.0,
        "prototype_anchor_weight_mean": 0.0,
        "prototype_anchor_entropy_mean": 0.0,
    }
    if labeled_feature.ndim != 4 or unlabeled_feature.ndim != 4:
        raise ValueError("labeled_feature and unlabeled_feature must be BCHW tensors")
    if labeled_feature.shape[1] != unlabeled_feature.shape[1]:
        raise ValueError(
            "labeled_feature and unlabeled_feature must share channel count, "
            f"got {labeled_feature.shape[1]} and {unlabeled_feature.shape[1]}"
        )
    candidate_set = targets.get("candidate_set")
    if candidate_set is None or candidate_set.ndim != 4 or candidate_set.shape[1] <= 1:
        return unlabeled_feature.new_tensor(0.0), stats

    num_classes = int(candidate_set.shape[1])
    fg_classes = [
        int(cls)
        for cls in (foreground_classes if foreground_classes is not None else range(1, num_classes))
        if 0 < int(cls) < num_classes
    ]
    if not fg_classes:
        return unlabeled_feature.new_tensor(0.0), stats

    size = tuple(unlabeled_feature.shape[-2:])
    labeled_mask_small = _resize_spatial(labeled_mask.long(), size, mode="nearest").long().to(unlabeled_feature.device)
    candidate = _resize_spatial(candidate_set.float(), size, mode="nearest").bool().to(unlabeled_feature.device)
    candidate_weight = targets.get("candidate_weight")
    if candidate_weight is None:
        weight_map = unlabeled_feature.new_ones(unlabeled_feature.shape[0], *size)
    else:
        weight_map = _resize_spatial(candidate_weight.float(), size, mode="bilinear").to(unlabeled_feature.device)
    soft_target = targets.get("soft_target")
    if soft_target is None or soft_target.shape[1] != num_classes:
        soft = candidate.float()
    else:
        soft = _resize_spatial(soft_target.float(), size, mode="bilinear").to(unlabeled_feature.device)
        soft = soft * candidate.float()

    labeled_norm = F.normalize(labeled_feature, dim=1)
    unlabeled_norm = F.normalize(unlabeled_feature, dim=1)
    prototypes: list[torch.Tensor] = []
    valid_classes: list[int] = []
    min_labeled = max(1, int(min_labeled_pixels))
    for cls in fg_classes:
        cls_mask = labeled_mask_small == cls
        pixel_count = int(cls_mask.sum())
        stats[f"prototype_anchor_labeled_pixels_class{cls}"] = float(pixel_count)
        if pixel_count < min_labeled:
            continue
        proto = labeled_norm.permute(0, 2, 3, 1)[cls_mask].mean(dim=0)
        prototypes.append(F.normalize(proto, dim=0))
        valid_classes.append(cls)

    if not prototypes:
        return unlabeled_feature.new_tensor(0.0), stats

    valid_index = torch.as_tensor(valid_classes, device=unlabeled_feature.device, dtype=torch.long)
    fg_candidate = candidate[:, valid_index]
    target_score = soft[:, valid_index].clamp_min(0.0) * fg_candidate.float()
    score_sum = target_score.sum(dim=1)
    active = score_sum > 1e-6
    min_unlabeled = max(1, int(min_unlabeled_pixels))
    if int(active.sum()) < min_unlabeled:
        return unlabeled_feature.new_tensor(0.0), stats
    target = target_score / score_sum.unsqueeze(1).clamp_min(1e-6)

    proto_tensor = torch.stack(prototypes, dim=0).to(device=unlabeled_feature.device, dtype=unlabeled_feature.dtype)
    logits = torch.einsum("bchw,kc->bkhw", unlabeled_norm, proto_tensor) / max(float(temperature), 1e-6)
    log_prob = F.log_softmax(logits, dim=1)
    ce = -(target.detach() * log_prob).sum(dim=1)

    entropy = -(target.clamp_min(1e-6) * target.clamp_min(1e-6).log()).sum(dim=1)
    if len(valid_classes) > 1:
        entropy = entropy / math.log(len(valid_classes))
    entropy_scale = (1.0 - float(entropy_discount) * entropy).clamp(0.0, 1.0)
    weight = (weight_map * entropy_scale * active.float()).clamp_min(0.0)
    denom = weight.sum().clamp_min(1e-6)
    loss = (ce * weight).sum() / denom

    stats.update(
        {
            "prototype_anchor_active": 1.0,
            "prototype_anchor_valid_class_count": float(len(valid_classes)),
            "prototype_anchor_mask_ratio": float(active.float().mean().detach()),
            "prototype_anchor_weight_mean": float((weight.sum() / active.float().sum().clamp_min(1.0)).detach()),
            "prototype_anchor_entropy_mean": float((entropy * active.float()).sum().detach() / active.float().sum().detach().clamp_min(1.0)),
        }
    )
    for cls in fg_classes:
        stats[f"prototype_anchor_valid_class{cls}"] = 1.0 if cls in valid_classes else 0.0
    return loss, stats
