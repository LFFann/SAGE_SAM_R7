from __future__ import annotations

import torch
import torch.nn.functional as F


def _foreground_classes(num_classes: int, foreground_classes: list[int] | tuple[int, ...] | None = None) -> list[int]:
    if foreground_classes is None:
        return list(range(1, num_classes))
    return [int(cls) for cls in foreground_classes if 0 < int(cls) < num_classes]


def _class_value(value, cls: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(cls, value.get(str(cls), default)))
    if isinstance(value, (list, tuple)):
        if cls < len(value):
            return float(value[cls])
        return float(value[-1]) if value else float(default)
    return float(value if value is not None else default)


def student_prior_feedback_loss(
    logits: torch.Tensor,
    class_prior,
    foreground_classes: list[int] | tuple[int, ...] | None = None,
    min_class_multiplier=0.45,
    max_class_multiplier=1.25,
    min_class_floor=0.0,
    max_class_floor=0.0,
    over_weight: float = 1.0,
    under_weight: float = 0.15,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize foreground mass drift against the labeled anatomical prior.

    The loss is hinge-shaped: it is zero while the student foreground area stays
    inside a loose prior interval and grows only when rare foreground classes
    over-expand or collapse. This avoids forcing every unlabeled mini-batch to
    exactly match the global prior.
    """

    if logits.ndim != 4:
        raise ValueError(f"logits must be BCHW, got {tuple(logits.shape)}")
    num_classes = int(logits.shape[1])
    fg_classes = _foreground_classes(num_classes, foreground_classes)
    stats: dict[str, float] = {"prior_feedback_loss_active": 0.0}
    if not fg_classes:
        return logits.new_tensor(0.0), stats

    prior = torch.as_tensor(class_prior, device=logits.device, dtype=logits.dtype)
    if prior.numel() != num_classes:
        raise ValueError(f"class_prior must have {num_classes} entries, got {int(prior.numel())}")
    prior = prior.clamp_min(float(eps))
    prior = prior / prior.sum().clamp_min(float(eps))
    temp = max(float(temperature), float(eps))
    mean_prob = F.softmax(logits / temp, dim=1).mean(dim=(0, 2, 3))

    losses = []
    over_values = []
    under_values = []
    for cls in fg_classes:
        min_multiplier = _class_value(min_class_multiplier, cls, 0.45)
        max_multiplier = _class_value(max_class_multiplier, cls, 1.25)
        lower_floor = _class_value(min_class_floor, cls, 0.0)
        upper_floor = _class_value(max_class_floor, cls, 0.0)
        lower = torch.clamp(prior[cls] * min_multiplier, min=lower_floor)
        upper = torch.clamp(prior[cls] * max_multiplier, min=upper_floor)
        upper = torch.maximum(upper, lower + float(eps))
        over = F.relu(mean_prob[cls] - upper) / upper.clamp_min(float(eps))
        under = F.relu(lower - mean_prob[cls]) / lower.clamp_min(float(eps))
        losses.append(float(over_weight) * over.square() + float(under_weight) * under.square())
        over_values.append(over.detach())
        under_values.append(under.detach())
        stats[f"prior_feedback_student_ratio_class{cls}"] = float(mean_prob[cls].detach())
        stats[f"prior_feedback_lower_class{cls}"] = float(lower.detach())
        stats[f"prior_feedback_upper_class{cls}"] = float(upper.detach())
        stats[f"prior_feedback_over_class{cls}"] = float(over.detach())
        stats[f"prior_feedback_under_class{cls}"] = float(under.detach())

    loss = torch.stack(losses).mean()
    stats["prior_feedback_loss_active"] = 1.0 if float(loss.detach()) > 0.0 else 0.0
    stats["prior_feedback_over_max"] = float(torch.stack(over_values).max()) if over_values else 0.0
    stats["prior_feedback_under_max"] = float(torch.stack(under_values).max()) if under_values else 0.0
    stats["prior_feedback_student_fg_ratio"] = float(mean_prob[fg_classes].sum().detach())
    stats["prior_feedback_target_fg_ratio"] = float(prior[fg_classes].sum().detach())
    return loss, stats
