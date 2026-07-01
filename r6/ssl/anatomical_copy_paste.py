from __future__ import annotations

import torch


def _foreground_mask(mask: torch.Tensor, foreground_classes: list[int], ignore_index: int) -> torch.Tensor:
    valid = mask != int(ignore_index)
    fg = torch.zeros_like(valid, dtype=torch.bool)
    for cls in foreground_classes:
        fg = fg | (mask == int(cls))
    return fg & valid


def build_labeled_foreground_copy_paste(
    labeled_image: torch.Tensor,
    labeled_mask: torch.Tensor,
    unlabeled_image: torch.Tensor,
    foreground_classes: list[int] | None = None,
    ignore_index: int = 255,
    min_foreground_ratio: float = 0.0,
    max_foreground_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Paste labeled foreground anatomy into unlabeled context.

    The returned target supervises only pasted foreground pixels.  Background and
    unlabeled context stay ignored, so this branch anchors rare foreground
    semantics without turning unlabeled background into hard negatives.
    """
    if labeled_image.shape != unlabeled_image.shape:
        raise ValueError(
            "labeled_image and unlabeled_image must share BCHW shape, "
            f"got {tuple(labeled_image.shape)} and {tuple(unlabeled_image.shape)}"
        )
    if labeled_mask.shape != labeled_image.shape[:1] + labeled_image.shape[-2:]:
        raise ValueError(
            "labeled_mask must share BHW with labeled_image, "
            f"got {tuple(labeled_mask.shape)} for image {tuple(labeled_image.shape)}"
        )

    if foreground_classes is None:
        valid_labels = torch.unique(labeled_mask[labeled_mask != int(ignore_index)])
        fg_classes = [int(label.item()) for label in valid_labels if int(label.item()) > 0]
    else:
        fg_classes = [int(cls) for cls in foreground_classes]
    paste_mask = _foreground_mask(labeled_mask, fg_classes, ignore_index)
    b, _, h, w = labeled_image.shape
    area = paste_mask.flatten(1).float().mean(dim=1)
    keep = (area >= float(min_foreground_ratio)) & (area <= float(max_foreground_ratio))
    paste_mask = paste_mask & keep.view(b, 1, 1)

    mixed = unlabeled_image.clone()
    mixed = torch.where(paste_mask.unsqueeze(1), labeled_image, mixed)
    target = labeled_mask.new_full((b, h, w), int(ignore_index))
    target = torch.where(paste_mask, labeled_mask, target)

    class_ratios = {}
    total_pixels = float(max(1, b * h * w))
    for cls in fg_classes:
        class_ratios[f"copy_paste_class{int(cls)}_ratio"] = float(((target == int(cls)).sum().detach()).item() / total_pixels)
    stats = {
        "copy_paste_active": 1.0 if bool(paste_mask.any()) else 0.0,
        "copy_paste_fg_ratio": float(paste_mask.float().mean().detach()),
        "copy_paste_kept_samples": float(keep.float().sum().detach()),
        **class_ratios,
    }
    return mixed, target, paste_mask, stats
