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


def save_diagnostic_grid(image, panels, path, num_classes: int = 3, cols: int = 4):
    """Save a compact diagnostic grid without requiring matplotlib."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rendered = [("image", _image_to_rgb(image))]
    for title, value, kind in panels:
        if kind == "mask":
            arr = colorize_mask(value, num_classes=num_classes)
        elif kind == "heatmap":
            arr = heatmap_gray(value)
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
