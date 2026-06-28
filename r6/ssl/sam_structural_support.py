from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(tensor: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if tensor.shape[-2:] == ref.shape[-2:]:
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=ref.shape[-2:], mode=mode)
    return F.interpolate(tensor, size=ref.shape[-2:], mode=mode, align_corners=False)


def _class_scores_to_spatial(scores: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"class scores must be shaped BxC, got {tuple(scores.shape)}")
    return scores[:, :, None, None].expand(-1, -1, height, width)


def _spatial_quality(scores: torch.Tensor | None, num_classes: int, height: int, width: int, ref: torch.Tensor) -> torch.Tensor:
    if scores is None:
        return ref.new_ones((ref.shape[0], num_classes, height, width))
    scores = scores.detach().to(device=ref.device, dtype=ref.dtype)
    if scores.ndim == 2:
        usable = min(num_classes, scores.shape[1])
        out = ref.new_ones((ref.shape[0], num_classes, height, width))
        out[:, :usable] = _class_scores_to_spatial(scores[:, :usable].clamp(0.0, 1.0), height, width)
        return out
    return ref.new_ones((ref.shape[0], num_classes, height, width))


def build_sam_structural_support(
    sam_out: dict | None,
    teacher_prob: torch.Tensor,
    foreground_classes: list[int] | tuple[int, ...] | None = None,
    min_support: float = 0.0,
) -> dict:
    """Convert SAM outputs into foreground-only structural support.

    SAM is deliberately not allowed to create a background channel here.  The
    returned support tensor is C-channel for shape compatibility, but channel 0
    is always zero and downstream code must derive background from the task
    teacher plus foreground exclusion.
    """

    if teacher_prob.ndim != 4:
        raise ValueError(f"teacher_prob must be BCHW, got {tuple(teacher_prob.shape)}")
    device = teacher_prob.device
    dtype = teacher_prob.dtype
    bsz, num_classes, height, width = teacher_prob.shape
    fg_classes = list(foreground_classes or range(1, num_classes))
    support = teacher_prob.new_zeros((bsz, num_classes, height, width))
    boundary = teacher_prob.new_zeros((bsz, 1, height, width))
    valid = bool(sam_out and sam_out.get("valid") and sam_out.get("sam_prob") is not None)
    if not valid:
        return {
            "valid": False,
            "support": support,
            "foreground_support": support[:, 1:].max(dim=1).values if num_classes > 1 else support[:, 0],
            "boundary": boundary,
            "verifier_score": support[:, 0],
            "prompt_stability": support,
            "teacher_sam_agreement": support[:, 0],
        }

    sam_prob = sam_out["sam_prob"].detach().to(device=device, dtype=dtype)
    if sam_prob.ndim != 4:
        raise ValueError(f"sam_prob must be BCHW, got {tuple(sam_prob.shape)}")
    sam_prob = _resize_like(sam_prob, teacher_prob).clamp(0.0, 1.0)
    usable_classes = min(num_classes, sam_prob.shape[1])
    for cls in fg_classes:
        if 0 < cls < usable_classes:
            support[:, cls] = sam_prob[:, cls]

    prompt_quality = _spatial_quality(sam_out.get("prompt_quality"), num_classes, height, width, teacher_prob)
    support = support * prompt_quality

    sam_iou_quality = _spatial_quality(sam_out.get("sam_iou"), num_classes, height, width, teacher_prob)
    support = support * sam_iou_quality

    if min_support > 0.0:
        support = torch.where(support >= float(min_support), support, torch.zeros_like(support))
    support[:, 0] = 0.0

    sam_boundary = sam_out.get("sam_boundary")
    if sam_boundary is not None:
        boundary = sam_boundary.detach().to(device=device, dtype=dtype)
        if boundary.ndim == 3:
            boundary = boundary.unsqueeze(1)
        boundary = _resize_like(boundary, teacher_prob[:, :1]).clamp(0.0, 1.0)

    fg_support = support[:, 1:].max(dim=1).values if num_classes > 1 else support[:, 0]
    teacher_fg = teacher_prob[:, 1:].max(dim=1).values if num_classes > 1 else teacher_prob[:, 0]
    denom = (teacher_fg + fg_support - teacher_fg * fg_support).clamp_min(1e-6)
    soft_overlap = (teacher_fg * fg_support) / denom
    confidence_agreement = (1.0 - (teacher_fg - fg_support).abs()).clamp(0.0, 1.0)
    prompt_stability = prompt_quality[:, 1:].max(dim=1).values if num_classes > 1 else prompt_quality[:, 0]
    iou_stability = sam_iou_quality[:, 1:].max(dim=1).values if num_classes > 1 else sam_iou_quality[:, 0]
    verifier_score = (
        0.35 * confidence_agreement
        + 0.25 * soft_overlap
        + 0.20 * prompt_stability
        + 0.20 * iou_stability
    ).clamp(0.0, 1.0)
    return {
        "valid": True,
        "support": support,
        "foreground_support": fg_support,
        "boundary": boundary,
        "verifier_score": verifier_score,
        "prompt_stability": prompt_quality,
        "teacher_sam_agreement": confidence_agreement,
    }
