from __future__ import annotations

import torch


class ClassConditionalSAMCalibrator:
    """Foreground-only score calibrator for SAM structural support.

    The score follows the R6 design: teacher semantic probability times SAM
    structural support.  Background is intentionally not calibrated here.
    """

    def __init__(
        self,
        num_classes: int,
        foreground_quantile: float = 0.30,
        min_pixels_per_class: int = 128,
        momentum: float = 0.8,
        default_threshold: float = 0.10,
    ):
        self.num_classes = int(num_classes)
        self.foreground_quantile = float(foreground_quantile)
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.momentum = float(momentum)
        self.q_score = torch.full((self.num_classes,), float(default_threshold))
        self.fitted = False

    @torch.no_grad()
    def update_from_batch(self, teacher_prob: torch.Tensor, sam_support: torch.Tensor, gt: torch.Tensor):
        teacher_prob = teacher_prob.detach().float().cpu()
        sam_support = sam_support.detach().float().cpu()
        gt = gt.detach().cpu()
        new_q = self.q_score.clone()
        for cls in range(1, self.num_classes):
            mask = gt == cls
            if int(mask.sum()) < self.min_pixels_per_class:
                continue
            score = (teacher_prob[:, cls] * sam_support[:, cls])[mask]
            if score.numel() > 0:
                new_q[cls] = torch.quantile(score, max(0.0, min(1.0, self.foreground_quantile)))
        if self.fitted:
            self.q_score = self.momentum * self.q_score + (1.0 - self.momentum) * new_q
        else:
            self.q_score = new_q
            self.fitted = True
        self.q_score[0] = 1.0
        return self

    def thresholds(self, device=None, dtype=None):
        q = self.q_score
        if device is not None or dtype is not None:
            q = q.to(device=device, dtype=dtype)
        return q

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "foreground_quantile": self.foreground_quantile,
            "min_pixels_per_class": self.min_pixels_per_class,
            "momentum": self.momentum,
            "q_score": self.q_score.tolist(),
            "fitted": self.fitted,
        }

    def load_state_dict(self, state):
        self.num_classes = int(state["num_classes"])
        self.foreground_quantile = float(state.get("foreground_quantile", self.foreground_quantile))
        self.min_pixels_per_class = int(state.get("min_pixels_per_class", self.min_pixels_per_class))
        self.momentum = float(state.get("momentum", self.momentum))
        self.q_score = torch.tensor(state.get("q_score", [0.10] * self.num_classes)).float()
        self.fitted = bool(state.get("fitted", True))
