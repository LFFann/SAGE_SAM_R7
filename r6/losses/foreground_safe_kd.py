from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground_target(prob: torch.Tensor) -> torch.Tensor:
    if prob.shape[1] <= 1:
        return prob
    fg = prob[:, 1:].detach().clamp_min(0.0)
    denom = fg.sum(dim=1, keepdim=True)
    normalized = fg / denom.clamp_min(1e-6)
    uniform = fg.new_full(fg.shape, 1.0 / max(1, fg.shape[1]))
    return torch.where(denom > 1e-6, normalized, uniform)


def _valid_weight(logits: torch.Tensor, gate: torch.Tensor | None, foreground_mask: torch.Tensor | None):
    weight = logits.new_ones(logits.shape[0], logits.shape[2], logits.shape[3])
    if gate is not None:
        weight = gate.to(device=logits.device, dtype=logits.dtype)
    if foreground_mask is not None:
        weight = weight * foreground_mask.to(device=logits.device, dtype=logits.dtype)
    return weight.clamp_min(0.0)


def foreground_safe_sam_kd_loss(
    student_logits: torch.Tensor,
    sam_prob: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    temperature: float = 1.0,
):
    if student_logits.shape[1] <= 1:
        return student_logits.new_tensor(0.0)
    weight = _valid_weight(student_logits, gate, foreground_mask)
    if weight.sum() <= 0:
        return student_logits.new_tensor(0.0)
    log_student = F.log_softmax(student_logits[:, 1:] / temperature, dim=1)
    target = _foreground_target(sam_prob.to(student_logits.device, student_logits.dtype))
    kd = F.kl_div(log_student, target, reduction="none").sum(dim=1) * (temperature**2)
    return ((kd.clamp_min(0.0)) * weight).sum() / weight.sum().clamp_min(1e-6)


def foreground_safe_sam_consistency_loss(
    sam_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
):
    if sam_prob.shape[1] <= 1:
        return sam_prob.new_tensor(0.0)
    weight = _valid_weight(sam_prob, gate, foreground_mask)
    if weight.sum() <= 0:
        return sam_prob.new_tensor(0.0)
    target = _foreground_target(teacher_prob.to(sam_prob.device, sam_prob.dtype))
    log_prob = torch.log(_foreground_target(sam_prob).clamp_min(1e-6))
    ce = -(target * log_prob).sum(dim=1)
    return (ce * weight).sum() / weight.sum().clamp_min(1e-6)


def sam_guided_extent_kd_loss(
    student_logits: torch.Tensor,
    sam_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    gate: torch.Tensor | None = None,
    temperature: float = 1.0,
    sam_mix: float = 0.65,
):
    """Distill SAM's foreground extent only inside vetted SAM-guided pixels."""

    if student_logits.shape != sam_prob.shape or student_logits.shape != teacher_prob.shape:
        raise ValueError(
            "student_logits, sam_prob, and teacher_prob must share BCHW shape, "
            f"got {tuple(student_logits.shape)}, {tuple(sam_prob.shape)}, {tuple(teacher_prob.shape)}"
        )
    weight = _valid_weight(student_logits, gate, None)
    if weight.sum() <= 0:
        return student_logits.new_tensor(0.0)
    mix = max(0.0, min(1.0, float(sam_mix)))
    target = (mix * sam_prob.to(student_logits.device, student_logits.dtype) + (1.0 - mix) * teacher_prob.to(student_logits.device, student_logits.dtype)).detach()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(log_student, target, reduction="none").sum(dim=1) * (temperature**2)
    return (kd.clamp_min(0.0) * weight).sum() / weight.sum().clamp_min(1e-6)
