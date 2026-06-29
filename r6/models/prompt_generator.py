from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGenerator(nn.Module):
    """Trainable one-vs-rest mask-prompt generator for online SAM mentoring."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        mask_prompt_size: int = 256,
        min_component_area: int = 16,
        residual_scale: float = 0.15,
        box_threshold: float = 0.35,
        max_box_area_ratio: float = 0.12,
        fallback_box_half_size: float = 0.035,
        valid_min_peak: float = 0.20,
        max_components_per_class=2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_foreground = max(0, self.num_classes - 1)
        self.mask_prompt_size = int(mask_prompt_size)
        self.min_component_area = int(min_component_area)
        self.residual_scale = float(residual_scale)
        self.box_threshold = float(box_threshold)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.fallback_box_half_size = float(fallback_box_half_size)
        self.valid_min_peak = float(valid_min_peak)
        self.max_components_by_class = self._normalize_max_components(max_components_per_class)
        self.max_components_per_class = max(self.max_components_by_class[1:]) if self.num_foreground > 0 else 1
        refiner_in = 1 + 2 * self.num_foreground + 1
        hidden = max(16, min(64, refiner_in * 8))
        self.refiner = nn.Sequential(
            nn.Conv2d(refiner_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_foreground, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        teacher_prob: torch.Tensor | None = None,
        student_prob: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        mode: str = "unlabeled",
    ) -> dict[str, torch.Tensor]:
        if self.num_foreground == 0:
            raise ValueError("PromptGenerator requires at least one foreground class")
        if image.ndim != 4:
            raise ValueError("image must be BCHW")

        b, _, h, w = image.shape
        device = image.device
        if gt_mask is not None:
            target = gt_mask.clamp(0, self.num_classes - 1)
            base = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()[:, 1:]
            if teacher_prob is None:
                teacher_fg = base
                teacher_uncertainty = torch.zeros(b, 1, h, w, device=device)
            else:
                teacher_fg = teacher_prob.detach()[:, 1:]
                teacher_uncertainty = 1.0 - teacher_prob.detach().max(dim=1, keepdim=True).values
        else:
            if teacher_prob is None:
                raise ValueError("teacher_prob is required for unlabeled prompt generation")
            base = teacher_prob.detach()[:, 1:]
            teacher_fg = base
            teacher_uncertainty = 1.0 - teacher_prob.detach().max(dim=1, keepdim=True).values

        if student_prob is None:
            student_fg = torch.zeros_like(base)
        else:
            student_fg = student_prob.detach()[:, 1:]

        image_gray = image.detach().mean(dim=1, keepdim=True)
        refiner_in = torch.cat([image_gray, base, student_fg, teacher_uncertainty], dim=1)
        residual = torch.tanh(self.refiner(refiner_in))
        soft_prompt = (base + self.residual_scale * residual).clamp(0.0, 1.0)
        (
            component_prompts,
            boxes,
            point_coords,
            point_labels,
            negative_point_coords,
            prompt_valid_fg,
            prompt_area_fg,
            prompt_box_area_fg,
            prompt_valid_flat,
            prompt_component_area_flat,
            prompt_component_box_area_flat,
            image_index,
            class_ids,
            component_index,
            component_count,
        ) = self._metadata_from_masks(soft_prompt.detach())
        mask_prompt = F.interpolate(
            component_prompts,
            size=(self.mask_prompt_size, self.mask_prompt_size),
            mode="bilinear",
            align_corners=False,
        )
        prompt_quality_fg = self._quality(base, teacher_fg, gt_mask is not None) * prompt_valid_fg
        prompt_quality = image.new_ones((b, self.num_classes))
        prompt_quality[:, 1:] = prompt_quality_fg
        prompt_valid = image.new_ones((b, self.num_classes))
        prompt_valid[:, 1:] = prompt_valid_fg
        prompt_area_ratio = image.new_zeros((b, self.num_classes))
        prompt_area_ratio[:, 1:] = prompt_area_fg
        prompt_box_area_ratio = image.new_zeros((b, self.num_classes))
        prompt_box_area_ratio[:, 1:] = prompt_box_area_fg
        prompt_weight = prompt_valid_flat
        prompt_component_count = image.new_zeros((b, self.num_classes))
        prompt_component_count[:, 1:] = component_count
        return {
            "mask_prompt": mask_prompt,
            "soft_prompt": soft_prompt,
            "boxes_xyxy": boxes,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "negative_point_coords": negative_point_coords,
            "image_index": image_index,
            "class_ids": class_ids,
            "prompt_quality": prompt_quality,
            "prompt_valid": prompt_valid,
            "prompt_valid_flat": prompt_valid_flat,
            "prompt_weight": prompt_weight,
            "prompt_area_ratio": prompt_area_ratio,
            "prompt_box_area_ratio": prompt_box_area_ratio,
            "prompt_component_area_ratio": prompt_component_area_flat,
            "prompt_component_box_area_ratio": prompt_component_box_area_flat,
            "prompt_component_count": prompt_component_count,
            "component_index": component_index,
            "mode": mode,
        }

    def _quality(self, base: torch.Tensor, teacher_fg: torch.Tensor, labeled: bool):
        area = base.mean(dim=(2, 3))
        if labeled:
            return (area * base.shape[-1] * base.shape[-2] >= self.min_component_area).float()
        denom = base.sum(dim=(2, 3)).clamp_min(1e-6)
        conf = (base * teacher_fg).sum(dim=(2, 3)) / denom
        area_ok = (area * base.shape[-1] * base.shape[-2] >= self.min_component_area).float()
        return (0.7 * conf + 0.3 * area_ok).clamp(0.0, 1.0)

    def _metadata_from_masks(self, masks: torch.Tensor):
        b, c, h, w = masks.shape
        component_prompts = []
        boxes = []
        pos_points = []
        neg_points = []
        valid_rows = []
        area_rows = []
        box_area_rows = []
        flat_valid = []
        flat_area = []
        flat_box_area = []
        image_index = []
        class_ids = []
        component_index = []
        component_count_rows = []
        for bi in range(b):
            union_fg = masks[bi].max(dim=0).values
            valid_per_image = []
            area_per_image = []
            box_area_per_image = []
            count_per_image = []
            for ci in range(c):
                score = masks[bi, ci]
                peak = score.max()
                m = score > self.box_threshold
                pixel_count = int(m.sum())
                bg = union_fg < 0.1
                if bg.any():
                    yy_bg, xx_bg = torch.where(bg)
                    nx = xx_bg.float().mean() / max(1, w - 1)
                    ny = yy_bg.float().mean() / max(1, h - 1)
                else:
                    nx = ny = torch.tensor(0.0, device=masks.device)

                components = []
                class_id = ci + 1
                class_max_components = self._max_components_for_class(class_id)
                if pixel_count >= self.min_component_area and float(peak.detach()) >= self.valid_min_peak:
                    components = self._connected_components(m, score, class_max_components)
                valid_components = 0
                class_area = score.new_tensor(0.0)
                class_box_area = score.new_tensor(0.0)
                for ki in range(class_max_components):
                    if ki < len(components):
                        comp = components[ki]
                        yy = comp["yy"].to(device=masks.device)
                        xx = comp["xx"].to(device=masks.device)
                        comp_mask = torch.zeros_like(score, dtype=torch.bool)
                        comp_mask[yy, xx] = True
                        x0 = xx.min().float() / max(1, w - 1)
                        x1 = xx.max().float() / max(1, w - 1)
                        y0 = yy.min().float() / max(1, h - 1)
                        y1 = yy.max().float() / max(1, h - 1)
                        weight = score[comp_mask].float().clamp_min(1e-6)
                        px = (xx.float() * weight).sum() / weight.sum() / max(1, w - 1)
                        py = (yy.float() * weight).sum() / weight.sum() / max(1, h - 1)
                        box_area = ((xx.max().float() - xx.min().float() + 1.0) / max(1, w)) * (
                            (yy.max().float() - yy.min().float() + 1.0) / max(1, h)
                        )
                        valid = float(box_area.detach()) <= self.max_box_area_ratio
                        component_prompt = torch.where(comp_mask, score, torch.zeros_like(score))
                    else:
                        comp_mask = torch.zeros_like(score, dtype=torch.bool)
                        valid = False
                        box_area = score.new_tensor(0.0)
                        component_prompt = torch.zeros_like(score)
                    if not valid:
                        peak_source = torch.where(comp_mask, score, torch.full_like(score, -1.0)) if bool(comp_mask.any()) else score
                        peak_idx = peak_source.reshape(-1).argmax()
                        py_idx = torch.div(peak_idx, w, rounding_mode="floor").float()
                        px_idx = (peak_idx % w).float()
                        px = px_idx / max(1, w - 1)
                        py = py_idx / max(1, h - 1)
                        half = score.new_tensor(self.fallback_box_half_size)
                        x0 = (px - half).clamp(0.0, 1.0)
                        x1 = (px + half).clamp(0.0, 1.0)
                        y0 = (py - half).clamp(0.0, 1.0)
                        y1 = (py + half).clamp(0.0, 1.0)
                        component_prompt = torch.zeros_like(score)
                        box_area = ((x1 - x0).clamp_min(0.0) * (y1 - y0).clamp_min(0.0)).detach()
                    area_ratio = score.new_tensor(float(comp_mask.sum()) / float(max(1, h * w)))
                    valid_value = score.new_tensor(1.0 if valid else 0.0)
                    if valid:
                        valid_components += 1
                        class_area = class_area + area_ratio
                        class_box_area = torch.maximum(class_box_area, box_area.to(device=masks.device, dtype=masks.dtype))
                    component_prompts.append(component_prompt.view(1, h, w))
                    boxes.append(torch.stack([x0, y0, x1, y1]))
                    pos_points.append(torch.stack([px, py]).view(1, 2))
                    neg_points.append(torch.stack([nx, ny]).view(1, 2))
                    flat_valid.append(valid_value)
                    flat_area.append(area_ratio)
                    flat_box_area.append(box_area.to(device=masks.device, dtype=masks.dtype))
                    image_index.append(torch.tensor(bi, device=masks.device, dtype=torch.long))
                    class_ids.append(torch.tensor(class_id, device=masks.device, dtype=torch.long))
                    component_index.append(torch.tensor(ki, device=masks.device, dtype=torch.long))

                valid_per_image.append(score.new_tensor(1.0 if valid_components > 0 else 0.0))
                area_per_image.append(class_area.clamp(0.0, 1.0))
                box_area_per_image.append(class_box_area.clamp(0.0, 1.0))
                count_per_image.append(score.new_tensor(float(valid_components)))
            valid_rows.append(torch.stack(valid_per_image, dim=0))
            area_rows.append(torch.stack(area_per_image, dim=0))
            box_area_rows.append(torch.stack(box_area_per_image, dim=0))
            component_count_rows.append(torch.stack(count_per_image, dim=0))
        prompt_t = torch.stack(component_prompts, dim=0)
        boxes_t = torch.stack(boxes, dim=0)
        points_t = torch.stack(pos_points, dim=0)
        neg_t = torch.stack(neg_points, dim=0)
        labels_t = torch.ones(points_t.shape[:2], device=masks.device, dtype=torch.long)
        return (
            prompt_t,
            boxes_t,
            points_t,
            labels_t,
            neg_t,
            torch.stack(valid_rows, dim=0),
            torch.stack(area_rows, dim=0),
            torch.stack(box_area_rows, dim=0),
            torch.stack(flat_valid, dim=0),
            torch.stack(flat_area, dim=0),
            torch.stack(flat_box_area, dim=0),
            torch.stack(image_index, dim=0),
            torch.stack(class_ids, dim=0),
            torch.stack(component_index, dim=0),
            torch.stack(component_count_rows, dim=0),
        )

    def _normalize_max_components(self, value) -> list[int]:
        if isinstance(value, dict):
            out = [0 for _ in range(self.num_classes)]
            default = int(value.get("default", value.get("*", 1)))
            for cls in range(1, self.num_classes):
                out[cls] = max(1, int(value.get(cls, value.get(str(cls), default))))
            return out
        if isinstance(value, (list, tuple)):
            raw = [int(v) for v in value]
            if len(raw) == self.num_classes:
                return [0 if cls == 0 else max(1, raw[cls]) for cls in range(self.num_classes)]
            if len(raw) == self.num_foreground:
                return [0] + [max(1, v) for v in raw]
            raise ValueError(
                "max_components_per_class list must have length num_classes or num_classes - 1"
            )
        count = max(1, int(value))
        return [0] + [count for _ in range(self.num_foreground)]

    def _max_components_for_class(self, class_id: int) -> int:
        if class_id <= 0 or class_id >= len(self.max_components_by_class):
            return 1
        return max(1, int(self.max_components_by_class[class_id]))

    def _connected_components(self, mask: torch.Tensor, score: torch.Tensor, max_components: int) -> list[dict[str, torch.Tensor | float]]:
        mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        if not bool(mask_cpu.any()):
            return []
        score_cpu = score.detach().to(device="cpu", dtype=torch.float32)
        visited = torch.zeros_like(mask_cpu, dtype=torch.bool)
        height, width = mask_cpu.shape
        coords = torch.nonzero(mask_cpu, as_tuple=False).tolist()
        components: list[dict[str, torch.Tensor | float]] = []
        for y0, x0 in coords:
            if bool(visited[y0, x0]):
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            ys: list[int] = []
            xs: list[int] = []
            score_sum = 0.0
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                score_sum += float(score_cpu[y, x])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if bool(mask_cpu[ny, nx]) and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(ys) >= self.min_component_area:
                components.append(
                    {
                        "yy": torch.tensor(ys, dtype=torch.long),
                        "xx": torch.tensor(xs, dtype=torch.long),
                        "score": score_sum,
                        "area": float(len(ys)),
                    }
                )
        components.sort(key=lambda item: (float(item["score"]), float(item["area"])), reverse=True)
        return components[: max(1, int(max_components))]
