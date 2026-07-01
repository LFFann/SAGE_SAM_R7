from __future__ import annotations

import torch
import torch.nn.functional as F


def _gate_to_weight(gate, shape, device, dtype):
    if gate is None:
        return torch.ones(shape, device=device, dtype=dtype)
    weight = gate.to(device)
    if weight.dtype == torch.bool:
        weight = weight.float()
    return weight.to(dtype=dtype).clamp_min(0.0)


def sam_ce_dice_loss(sam_prob: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    valid = target != ignore_index
    if valid.sum() == 0:
        return sam_prob.new_tensor(0.0)
    ce_target = target.clone()
    safe_target = target.clamp(0, num_classes - 1)
    log_prob = torch.log(sam_prob.clamp_min(1e-6))
    ce = F.nll_loss(log_prob, ce_target, ignore_index=ignore_index)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    dice_terms = []
    for c in range(1, num_classes):
        p = sam_prob[:, c] * valid.float()
        t = one_hot[:, c] * valid.float()
        dice_terms.append(1.0 - (2.0 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6))
    dice = torch.stack(dice_terms).mean() if dice_terms else sam_prob.new_tensor(0.0)
    return ce + dice * valid_f.mean().clamp_min(1e-6) / valid_f.mean().clamp_min(1e-6)


def gated_soft_sam_loss(sam_prob: torch.Tensor, soft_target: torch.Tensor, gate: torch.Tensor | None = None):
    soft_target = soft_target.detach()
    weight = _gate_to_weight(gate, (sam_prob.shape[0], sam_prob.shape[2], sam_prob.shape[3]), sam_prob.device, sam_prob.dtype)
    if weight.sum() <= 0:
        return sam_prob.new_tensor(0.0)
    log_prob = torch.log(sam_prob.clamp_min(1e-6))
    ce = -(soft_target * log_prob).sum(dim=1)
    return (ce * weight).sum() / weight.sum().clamp_min(1e-6)


def sam_student_kd_loss(student_logits: torch.Tensor, sam_prob: torch.Tensor, gate: torch.Tensor | None = None, temperature: float = 1.0):
    target = sam_prob.detach()
    weight = _gate_to_weight(gate, (student_logits.shape[0], student_logits.shape[2], student_logits.shape[3]), student_logits.device, student_logits.dtype)
    if weight.sum() <= 0:
        return student_logits.new_tensor(0.0)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(log_student, target, reduction="none").sum(dim=1) * (temperature**2)
    return (kd * weight).sum() / weight.sum().clamp_min(1e-6)


def sam_prompt_consistency_loss(
    soft_prompt: torch.Tensor,
    sam_prob: torch.Tensor,
    teacher_prob: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    prompt_valid: torch.Tensor | None = None,
    prompt_quality: torch.Tensor | None = None,
    target_mix: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the prompt generator to match trusted SAM/teacher foreground support.

    The SAM mask decoder is not called a second time. `soft_prompt` remains
    differentiable, while SAM and teacher probabilities are detached targets.
    """

    if soft_prompt.ndim != 4 or sam_prob.ndim != 4:
        raise ValueError("soft_prompt and sam_prob must be BCHW tensors")
    if sam_prob.shape[1] <= 1:
        return soft_prompt.new_tensor(0.0), {
            "prompt_consistency_active": 0.0,
            "prompt_consistency_mask_ratio": 0.0,
            "prompt_consistency_weight_mean": 0.0,
            "prompt_consistency_abs_gap": 0.0,
        }
    fg_prompt = soft_prompt.float().clamp(1e-4, 1.0 - 1e-4)
    fg_target = sam_prob.detach().float()[:, 1 : 1 + fg_prompt.shape[1]]
    if teacher_prob is not None and teacher_prob.shape[1] > 1:
        teacher_fg = teacher_prob.detach().float()[:, 1 : 1 + fg_prompt.shape[1]]
        mix = min(1.0, max(0.0, float(target_mix)))
        fg_target = mix * fg_target + (1.0 - mix) * teacher_fg
    if fg_target.shape[-2:] != fg_prompt.shape[-2:]:
        fg_target = F.interpolate(fg_target, size=fg_prompt.shape[-2:], mode="bilinear", align_corners=False)

    if gate is None:
        weight = torch.ones_like(fg_prompt)
    else:
        weight = gate.detach().to(device=fg_prompt.device, dtype=fg_prompt.dtype)
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
        if weight.shape[1] == 1 and fg_prompt.shape[1] > 1:
            weight = weight.expand(-1, fg_prompt.shape[1], -1, -1)
        if weight.shape[-2:] != fg_prompt.shape[-2:]:
            weight = F.interpolate(weight, size=fg_prompt.shape[-2:], mode="nearest")
        weight = weight[:, : fg_prompt.shape[1]].clamp(0.0, 1.0)
    if prompt_valid is not None:
        valid = prompt_valid.detach().to(device=fg_prompt.device, dtype=fg_prompt.dtype)
        valid = valid[:, 1 : 1 + fg_prompt.shape[1]] if valid.ndim == 2 and valid.shape[1] > fg_prompt.shape[1] else valid[:, : fg_prompt.shape[1]]
        weight = weight * valid.view(valid.shape[0], valid.shape[1], 1, 1).clamp(0.0, 1.0)
    if prompt_quality is not None:
        quality = prompt_quality.detach().to(device=fg_prompt.device, dtype=fg_prompt.dtype)
        quality = quality[:, 1 : 1 + fg_prompt.shape[1]] if quality.ndim == 2 and quality.shape[1] > fg_prompt.shape[1] else quality[:, : fg_prompt.shape[1]]
        weight = weight * quality.view(quality.shape[0], quality.shape[1], 1, 1).clamp(0.0, 1.0)

    if float(weight.sum().detach()) <= 0.0:
        return soft_prompt.new_tensor(0.0), {
            "prompt_consistency_active": 0.0,
            "prompt_consistency_mask_ratio": 0.0,
            "prompt_consistency_weight_mean": 0.0,
            "prompt_consistency_abs_gap": 0.0,
        }

    bce = F.binary_cross_entropy(fg_prompt, fg_target.clamp(0.0, 1.0), reduction="none")
    loss = (bce * weight).sum() / weight.sum().clamp_min(1e-6)
    active = weight > 0
    gap = (fg_prompt.detach() - fg_target).abs()
    return loss, {
        "prompt_consistency_active": 1.0,
        "prompt_consistency_mask_ratio": float(active.float().mean().detach()),
        "prompt_consistency_weight_mean": float(weight.mean().detach()),
        "prompt_consistency_abs_gap": float((gap * weight).sum().detach() / weight.sum().detach().clamp_min(1e-6)),
        "prompt_consistency_prompt_fg_mean": float((fg_prompt.detach() * weight).sum().detach() / weight.sum().detach().clamp_min(1e-6)),
        "prompt_consistency_target_fg_mean": float((fg_target * weight).sum().detach() / weight.sum().detach().clamp_min(1e-6)),
    }
