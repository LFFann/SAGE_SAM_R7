from __future__ import annotations

import torch

from r6.losses.foreground_safe_kd import sam_guided_extent_kd_loss
from r6.losses.set_valued_losses import rank_margin_loss, safe_negative_loss, set_cross_entropy_loss, singleton_ce_loss


def test_singleton_empty_mask_no_nan():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    loss = singleton_ce_loss(logits, labels, torch.zeros(1, 4, 4, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_set_cross_entropy_backward():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    candidate = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    candidate[:, :2] = True
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    loss = set_cross_entropy_loss(logits, candidate, mask)
    loss.backward()
    assert logits.grad is not None


def test_rank_and_negative_losses():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    candidate = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    candidate[:, 1] = True
    negative = torch.zeros_like(candidate)
    negative[:, 2] = True
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    loss = rank_margin_loss(logits, candidate, mask) + safe_negative_loss(logits, negative, mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_sam_guided_extent_kd_uses_full_distribution_under_gate():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    teacher = torch.full((1, 3, 2, 2), 1.0 / 3.0)
    sam = teacher.clone()
    sam[:, 0] = 0.10
    sam[:, 1] = 0.85
    sam[:, 2] = 0.05
    gate = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    loss = sam_guided_extent_kd_loss(logits, sam, teacher, gate=gate, sam_mix=0.8)

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[:, 0].abs().sum() > 0
