from __future__ import annotations

import torch.nn.functional as F

from .dice import multiclass_dice_loss


def supervised_loss(logits, target, num_classes: int, ignore_index: int = 255, class_weights=None):
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, weight=weight)
    dice = multiclass_dice_loss(logits, target, num_classes, ignore_index)
    return ce + dice, {"loss_sup_ce": float(ce.detach()), "loss_sup_dice": float(dice.detach())}
