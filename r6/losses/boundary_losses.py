from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_bce_loss(boundary_logits, boundary_target):
    if boundary_logits is None or boundary_target is None:
        return torch.tensor(0.0)
    if boundary_target.ndim == 3:
        boundary_target = boundary_target.unsqueeze(1)
    if boundary_target.ndim == 4 and boundary_logits.shape[1] == 1 and boundary_target.shape[1] != 1:
        boundary_target = boundary_target.max(dim=1, keepdim=True).values
    boundary_target = F.interpolate(boundary_target.float(), size=boundary_logits.shape[-2:], mode="nearest")
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
