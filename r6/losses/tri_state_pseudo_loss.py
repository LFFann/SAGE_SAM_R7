from __future__ import annotations

import torch

from .set_valued_losses import (
    rank_margin_loss,
    safe_negative_loss,
    set_cross_entropy_loss,
    singleton_ce_loss,
    soft_fuzzy_positive_loss,
)


def tri_state_pseudo_supervision_loss(logits: torch.Tensor, targets: dict, rank_margin: float = 0.5):
    labels = targets["singleton_label"]
    singleton_mask = targets["singleton_mask"].bool()
    candidate_set = targets["candidate_set"].bool()
    ambiguous_mask = targets.get("fuzzy_region", targets["ambiguous_mask"]).bool()
    negative_set = targets["safe_negative_set"].bool()
    negative_mask = targets["negative_mask"].bool()
    soft_target = targets.get("soft_target")
    candidate_weight = targets.get("candidate_weight")
    negative_weight = targets.get("safe_negative_weight")

    loss_hard = singleton_ce_loss(logits, labels, singleton_mask)
    loss_fuzzy = soft_fuzzy_positive_loss(logits, soft_target, ambiguous_mask, candidate_weight)
    loss_set = set_cross_entropy_loss(logits, candidate_set, ambiguous_mask, candidate_weight)
    loss_rank = rank_margin_loss(logits, candidate_set, ambiguous_mask, rank_margin, candidate_weight)
    loss_neg = safe_negative_loss(logits, negative_set, negative_mask, negative_weight)
    return {
        "loss_singleton": loss_hard,
        "loss_hard": loss_hard,
        "loss_hard_fg": singleton_ce_loss(logits, labels, singleton_mask & (labels > 0)),
        "loss_set": loss_set,
        "loss_rank": loss_rank,
        "loss_negative": loss_neg,
        "loss_fuzzy": loss_fuzzy,
    }
