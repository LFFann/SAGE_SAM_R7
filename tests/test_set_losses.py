from __future__ import annotations

import torch

from r6.losses.foreground_safe_kd import sam_guided_extent_kd_loss, student_anchored_sam_agreement_loss
from r6.losses.prior_feedback import student_prior_feedback_loss
from r6.losses.set_valued_losses import rank_margin_loss, safe_negative_loss, set_cross_entropy_loss, singleton_ce_loss
from r6.losses.supervised import supervised_loss


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


def test_student_anchored_sam_agreement_loss_uses_reliable_sam_without_teacher_gate():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    logits.data[:, 1] = 0.8
    sam = torch.full((1, 3, 2, 2), 0.05)
    sam[:, 0] = 0.10
    sam[:, 1] = 0.85
    support = torch.zeros_like(sam)
    support[:, 1] = 0.70
    verifier = torch.full((1, 2, 2), 0.80)

    loss, stats = student_anchored_sam_agreement_loss(
        logits,
        sam,
        support,
        verifier,
        min_support=0.06,
        min_verifier=0.45,
    )

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert stats["sam_agreement_gate_ratio"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_student_anchored_sam_agreement_loss_ignores_unsupported_sam():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    sam = torch.full((1, 3, 2, 2), 1.0 / 3.0)
    support = torch.zeros_like(sam)
    verifier = torch.full((1, 2, 2), 0.80)

    loss, stats = student_anchored_sam_agreement_loss(logits, sam, support, verifier)

    assert float(loss.detach()) == 0.0
    assert stats["sam_agreement_gate_ratio"] == 0.0


def test_student_prior_feedback_loss_is_zero_inside_prior_band():
    prior = torch.tensor([0.9900, 0.0060, 0.0040])
    logits = prior.log().view(1, 3, 1, 1).expand(1, 3, 4, 4).clone().requires_grad_(True)

    loss, stats = student_prior_feedback_loss(
        logits,
        prior,
        foreground_classes=[1, 2],
        min_class_multiplier=0.45,
        max_class_multiplier=1.25,
    )

    assert float(loss.detach()) == 0.0
    assert stats["prior_feedback_loss_active"] == 0.0


def test_student_prior_feedback_loss_penalizes_foreground_overexpansion():
    prior = torch.tensor([0.9900, 0.0060, 0.0040])
    logits = torch.zeros(1, 3, 4, 4, requires_grad=True)
    logits.data[:, 1] = 4.0

    loss, stats = student_prior_feedback_loss(
        logits,
        prior,
        foreground_classes=[1, 2],
        min_class_multiplier=0.45,
        max_class_multiplier=1.25,
    )

    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert stats["prior_feedback_over_class1"] > 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[:, 1].mean() > 0.0


def test_student_prior_feedback_loss_supports_class_specific_upper_bounds():
    prior = torch.tensor([0.9900, 0.0060, 0.0040])
    logits = prior.log().view(1, 3, 1, 1).expand(1, 3, 4, 4).clone().requires_grad_(True)
    logits.data[:, 2] += 4.00

    loss, stats = student_prior_feedback_loss(
        logits,
        prior,
        foreground_classes=[1, 2],
        max_class_multiplier=[0.0, 1.25, 1.02],
        temperature=0.70,
    )

    assert torch.isfinite(loss)
    assert stats["prior_feedback_over_class2"] > 0.0


def test_supervised_loss_accepts_class_weights():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target = torch.tensor([[[0, 1], [2, 2]]])
    weights = torch.tensor([0.25, 1.0, 1.3])

    loss, stats = supervised_loss(logits, target, num_classes=3, class_weights=weights)

    assert torch.isfinite(loss)
    assert stats["loss_sup_ce"] > 0.0
    loss.backward()
    assert logits.grad is not None
