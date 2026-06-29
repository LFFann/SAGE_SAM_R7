from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_mask_png(mask, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask).astype(np.uint8)).save(path)


PALETTE = np.asarray(
    [
        [0, 0, 0],
        [230, 64, 64],
        [64, 160, 235],
        [90, 210, 120],
        [245, 190, 70],
        [175, 110, 230],
    ],
    dtype=np.uint8,
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_rgb(image) -> np.ndarray:
    arr = _to_numpy(image).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    lo = float(np.nanpercentile(arr, 1.0))
    hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def colorize_mask(mask, num_classes: int | None = None) -> np.ndarray:
    arr = _to_numpy(mask).astype(np.int64)
    if arr.ndim == 3:
        arr = arr[0]
    palette = PALETTE
    if num_classes and num_classes > len(palette):
        extra = np.random.default_rng(0).integers(0, 255, size=(num_classes - len(palette), 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    arr = np.clip(arr, 0, len(palette) - 1)
    return palette[arr]


def heatmap_gray(value) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def prompt_overlay(image, prompt_info, num_classes: int = 3) -> np.ndarray:
    """Render normalized SAM prompts on top of the source image."""

    base = Image.fromarray(_image_to_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(base)
    width, height = base.size
    if not isinstance(prompt_info, dict):
        return np.asarray(base)

    prompts = prompt_info.get("prompts", prompt_info)
    sample_index = int(prompt_info.get("sample_index", 0))
    boxes = prompts.get("boxes_xyxy") if isinstance(prompts, dict) else None
    if boxes is None:
        return np.asarray(base)
    boxes = _to_numpy(boxes).astype(np.float32)
    image_index = _to_numpy(prompts.get("image_index", np.zeros((boxes.shape[0],), dtype=np.int64))).astype(np.int64)
    class_ids = _to_numpy(prompts.get("class_ids", np.ones((boxes.shape[0],), dtype=np.int64))).astype(np.int64)
    component_ids = _to_numpy(prompts.get("component_index", np.zeros((boxes.shape[0],), dtype=np.int64))).astype(np.int64)
    points = prompts.get("point_coords")
    points = _to_numpy(points).astype(np.float32) if points is not None else None
    prompt_valid = prompts.get("prompt_valid")
    prompt_valid = _to_numpy(prompt_valid).astype(np.float32) if prompt_valid is not None else None
    prompt_valid_flat = prompts.get("prompt_valid_flat")
    prompt_valid_flat = _to_numpy(prompt_valid_flat).astype(np.float32) if prompt_valid_flat is not None else None

    for prompt_idx, box in enumerate(boxes):
        if prompt_idx < len(image_index) and int(image_index[prompt_idx]) != sample_index:
            continue
        cls = int(class_ids[prompt_idx]) if prompt_idx < len(class_ids) else 1
        cls = max(1, min(cls, max(1, num_classes - 1)))
        valid = True
        if prompt_valid_flat is not None and prompt_idx < prompt_valid_flat.shape[0]:
            valid = bool(prompt_valid_flat[prompt_idx] >= 0.5)
        elif prompt_valid is not None and sample_index < prompt_valid.shape[0] and cls < prompt_valid.shape[1]:
            valid = bool(prompt_valid[sample_index, cls] >= 0.5)
        color = tuple(int(x) for x in PALETTE[cls % len(PALETTE)]) if valid else (185, 185, 185)
        x0 = int(np.clip(box[0], 0.0, 1.0) * max(1, width - 1))
        y0 = int(np.clip(box[1], 0.0, 1.0) * max(1, height - 1))
        x1 = int(np.clip(box[2], 0.0, 1.0) * max(1, width - 1))
        y1 = int(np.clip(box[3], 0.0, 1.0) * max(1, height - 1))
        line_width = 2 if valid else 1
        for offset in range(line_width):
            draw.rectangle(
                [
                    max(0, x0 - offset),
                    max(0, y0 - offset),
                    min(width - 1, x1 + offset),
                    min(height - 1, y1 + offset),
                ],
                outline=color,
            )
        if points is not None and prompt_idx < points.shape[0]:
            px = int(np.clip(points[prompt_idx, 0, 0], 0.0, 1.0) * max(1, width - 1))
            py = int(np.clip(points[prompt_idx, 0, 1], 0.0, 1.0) * max(1, height - 1))
            r = 3 if valid else 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        comp = int(component_ids[prompt_idx]) if prompt_idx < len(component_ids) else 0
        label = f"c{cls}#{comp}" if valid else f"c{cls}#{comp} invalid"
        draw.text((max(0, x0), max(0, y0 - 12)), label, fill=color)
    return np.asarray(base)


def save_diagnostic_grid(image, panels, path, num_classes: int = 3, cols: int = 4):
    """Save a compact diagnostic grid without requiring matplotlib."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rendered = [("image", _image_to_rgb(image))]
    for title, value, kind in panels:
        if kind == "mask":
            arr = colorize_mask(value, num_classes=num_classes)
        elif kind == "heatmap":
            arr = heatmap_gray(value)
        elif kind == "prompt_overlay":
            arr = prompt_overlay(image, value, num_classes=num_classes)
        else:
            arr = _image_to_rgb(value)
        rendered.append((str(title), arr))

    height, width = rendered[0][1].shape[:2]
    label_h = 18
    cols = max(1, int(cols))
    rows = int(np.ceil(len(rendered) / cols))
    canvas = Image.new("RGB", (cols * width, rows * (height + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, arr) in enumerate(rendered):
        row = idx // cols
        col = idx % cols
        x = col * width
        y = row * (height + label_h)
        draw.rectangle([x, y, x + width, y + label_h], fill=(245, 245, 245))
        draw.text((x + 3, y + 2), title[:32], fill=(0, 0, 0))
        if arr.shape[:2] != (height, width):
            arr = np.asarray(Image.fromarray(arr).resize((width, height), Image.NEAREST))
        canvas.paste(Image.fromarray(arr.astype(np.uint8)), (x, y + label_h))
    canvas.save(path)
