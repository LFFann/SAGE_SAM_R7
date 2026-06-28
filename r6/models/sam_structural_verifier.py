from __future__ import annotations

import torch.nn as nn

from r6.ssl.sam_structural_support import build_sam_structural_support


class SAMStructuralVerifier(nn.Module):
    """Thin module wrapper around foreground-only SAM structural support."""

    def __init__(self, foreground_classes=None, min_support: float = 0.0):
        super().__init__()
        self.foreground_classes = foreground_classes
        self.min_support = float(min_support)

    def forward(self, sam_out, teacher_prob):
        return build_sam_structural_support(
            sam_out,
            teacher_prob,
            foreground_classes=self.foreground_classes,
            min_support=self.min_support,
        )
