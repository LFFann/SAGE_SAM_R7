from __future__ import annotations

import torch
import torch.nn.functional as F


def strong_view_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric consistency between two spatially aligned strong views."""
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"logits_a and logits_b must share BCHW shape, got {tuple(logits_a.shape)} and {tuple(logits_b.shape)}")
    if logits_a.ndim != 4:
        raise ValueError(f"logits must be BCHW, got {tuple(logits_a.shape)}")
    if mask is None:
        mask = torch.ones(logits_a.shape[0], logits_a.shape[2], logits_a.shape[3], device=logits_a.device, dtype=torch.bool)
    else:
        mask = mask.to(device=logits_a.device).bool()
    if mask.sum() == 0:
        return logits_a.new_tensor(0.0), {"strong_view_consistency_mask_ratio": 0.0, "strong_view_consistency_weight_mean": 0.0}
    temp = max(float(temperature), 1e-6)
    prob_a = torch.softmax(logits_a.detach() / temp, dim=1)
    prob_b = torch.softmax(logits_b.detach() / temp, dim=1)
    log_a = F.log_softmax(logits_a / temp, dim=1)
    log_b = F.log_softmax(logits_b / temp, dim=1)
    loss_map = 0.5 * (
        F.kl_div(log_a, prob_b, reduction="none").sum(dim=1)
        + F.kl_div(log_b, prob_a, reduction="none").sum(dim=1)
    ) * (temp**2)
    if weight is None:
        pixel_weight = mask.float()
    else:
        pixel_weight = weight.to(device=logits_a.device, dtype=logits_a.dtype).clamp_min(0.0) * mask.float()
    denom = pixel_weight.sum().clamp_min(1e-6)
    loss = (loss_map.clamp_min(0.0) * pixel_weight).sum() / denom
    return loss, {
        "strong_view_consistency_mask_ratio": float(mask.float().mean().detach()),
        "strong_view_consistency_weight_mean": float(pixel_weight.mean().detach()),
    }
