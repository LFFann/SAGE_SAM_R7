from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from r6.utils.hd95 import per_class_hd95
from r6.utils.metrics import average_foreground, per_class_dice_iou
from r6.utils.visualization import save_mask_png


def _area_stats(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    valid = target != ignore_index
    valid_pixels = int(valid.sum().item())
    pred_counts = []
    gt_counts = []
    fp_counts = []
    fn_counts = []
    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid
        gt_cls = (target == cls) & valid
        pred_count = int(pred_cls.sum().item())
        gt_count = int(gt_cls.sum().item())
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        fp_counts.append(int((pred_cls & ~gt_cls).sum().item()))
        fn_counts.append(int((~pred_cls & gt_cls).sum().item()))
    return valid_pixels, pred_counts, gt_counts, fp_counts, fn_counts


def _normalize_component_caps(value, num_classes: int) -> list[int]:
    if value is None:
        return [0 for _ in range(num_classes)]
    if isinstance(value, int):
        return [0] + [max(0, int(value)) for _ in range(max(0, num_classes - 1))]
    if isinstance(value, (list, tuple)):
        caps = [max(0, int(v)) for v in value]
        if len(caps) == num_classes:
            return caps
        if len(caps) == num_classes - 1:
            return [0] + caps
    raise ValueError("max_components_per_class must be an int or a list with num_classes or num_classes - 1 entries")


def _connected_components(mask: np.ndarray, score: np.ndarray | None = None, min_area: int = 1):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    min_area = max(1, int(min_area))
    for y0 in range(height):
        for x0 in range(width):
            if visited[y0, x0] or not mask[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            ys = []
            xs = []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(ys)
            if area < min_area:
                continue
            if score is None:
                comp_score = float(area)
            else:
                comp_score = float(score[ys, xs].sum())
            components.append({"ys": np.asarray(ys), "xs": np.asarray(xs), "area": area, "score": comp_score})
    components.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    return components


def _apply_topology_postprocess(
    pred: torch.Tensor,
    prob: torch.Tensor | None,
    num_classes: int,
    cfg: dict | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return pred, {"topology_postprocess_active": 0.0}
    caps = _normalize_component_caps(cfg.get("max_components_per_class"), num_classes)
    min_area = int(cfg.get("min_component_area", 1))
    out = pred.detach().cpu().clone()
    prob_cpu = prob.detach().cpu() if prob is not None else None
    total_pixels = max(1, int(out.numel()))
    stats: dict[str, float] = {"topology_postprocess_active": 1.0}
    removed_total = 0
    kept_components = [0 for _ in range(num_classes)]
    dropped_components = [0 for _ in range(num_classes)]
    removed_pixels = [0 for _ in range(num_classes)]
    for b in range(out.shape[0]):
        pred_np = out[b].numpy()
        for cls in range(1, num_classes):
            cap = caps[cls] if cls < len(caps) else 0
            if cap <= 0:
                continue
            mask = pred_np == cls
            if not mask.any():
                continue
            score = prob_cpu[b, cls].numpy() if prob_cpu is not None else None
            components = _connected_components(mask, score=score, min_area=min_area)
            keep = components[:cap]
            drop = components[cap:]
            keep_mask = np.zeros_like(mask, dtype=bool)
            for comp in keep:
                keep_mask[comp["ys"], comp["xs"]] = True
            drop_mask = mask & ~keep_mask
            removed = int(drop_mask.sum())
            if removed > 0:
                pred_np[drop_mask] = 0
            kept_components[cls] += len(keep)
            dropped_components[cls] += len(drop)
            removed_pixels[cls] += removed
            removed_total += removed
        out[b] = torch.from_numpy(pred_np)
    stats["topology_removed_pixel_ratio"] = float(removed_total / total_pixels)
    batch_size = max(1, int(out.shape[0]))
    for cls in range(num_classes):
        stats[f"topology_removed_ratio_class{cls}"] = float(removed_pixels[cls] / total_pixels)
        stats[f"topology_kept_components_class{cls}"] = float(kept_components[cls] / batch_size)
        stats[f"topology_dropped_components_class{cls}"] = float(dropped_components[cls] / batch_size)
    return out.to(device=pred.device), stats


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    num_classes: int,
    device,
    compute_hd95: bool = True,
    save_dir=None,
    ignore_index: int = 255,
    topology_postprocess: dict | None = None,
):
    model.eval()
    device = torch.device(device)
    all_dice, all_iou, all_hd95 = [], [], []
    total_valid = 0
    pred_area = np.zeros(num_classes, dtype=np.float64)
    gt_area = np.zeros(num_classes, dtype=np.float64)
    fp_area = np.zeros(num_classes, dtype=np.float64)
    fn_area = np.zeros(num_classes, dtype=np.float64)
    topology_totals: dict[str, float] = {}
    topology_batches = 0
    rows = []
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for batch in dataloader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)
        prob = logits.softmax(dim=1) if topology_postprocess and bool(topology_postprocess.get("enabled", False)) else None
        pred, topology_stats = _apply_topology_postprocess(pred, prob, num_classes, topology_postprocess)
        topology_batches += 1
        for key, value in topology_stats.items():
            topology_totals[key] = topology_totals.get(key, 0.0) + float(value)
        for i in range(pred.shape[0]):
            dice, iou = per_class_dice_iou(pred[i], mask[i], num_classes, ignore_index)
            hd = per_class_hd95(pred[i].cpu().numpy(), mask[i].cpu().numpy(), num_classes, ignore_index) if compute_hd95 else [float("nan")] * num_classes
            valid_pixels, pred_counts, gt_counts, fp_counts, fn_counts = _area_stats(pred[i], mask[i], num_classes, ignore_index)
            total_valid += valid_pixels
            pred_area += np.asarray(pred_counts, dtype=np.float64)
            gt_area += np.asarray(gt_counts, dtype=np.float64)
            fp_area += np.asarray(fp_counts, dtype=np.float64)
            fn_area += np.asarray(fn_counts, dtype=np.float64)
            all_dice.append(dice)
            all_iou.append(iou)
            all_hd95.append(hd)
            sample_id = batch.get("id", [f"sample_{len(rows)}"])[i]
            case_row = {
                "id": sample_id,
                "avg_dice": average_foreground(dice),
                "avg_iou": average_foreground(iou),
                "avg_hd95": average_foreground(hd),
            }
            case_valid = max(1, valid_pixels)
            for cls in range(num_classes):
                case_row[f"class_{cls}_pred_ratio"] = pred_counts[cls] / case_valid
                case_row[f"class_{cls}_gt_ratio"] = gt_counts[cls] / case_valid
            rows.append(case_row)
            if save_dir:
                safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
                save_mask_png(pred[i].cpu().numpy(), save_dir / f"{safe_id}.png")
    class_dice = np.nanmean(np.asarray(all_dice, dtype=float), axis=0).tolist()
    class_iou = np.nanmean(np.asarray(all_iou, dtype=float), axis=0).tolist()
    if compute_hd95:
        hd_arr = np.asarray(all_hd95, dtype=float)
        class_hd95 = []
        for c in range(num_classes):
            finite = hd_arr[:, c][np.isfinite(hd_arr[:, c])]
            class_hd95.append(float(finite.mean()) if finite.size else float("nan"))
    else:
        class_hd95 = [float("nan")] * num_classes
    denom = max(1.0, float(total_valid))
    pred_ratio = (pred_area / denom).tolist()
    gt_ratio = (gt_area / denom).tolist()
    area_abs_error = np.abs(pred_area - gt_area) / denom
    pred_to_gt = pred_area / np.maximum(gt_area, 1.0)
    overseg = fp_area / np.maximum(gt_area, 1.0)
    underseg = fn_area / np.maximum(gt_area, 1.0)
    foreground_pred_ratio = float(pred_area[1:].sum() / denom) if num_classes > 1 else float(pred_area[0] / denom)
    foreground_gt_ratio = float(gt_area[1:].sum() / denom) if num_classes > 1 else float(gt_area[0] / denom)
    metrics = {
        "class_dice": class_dice,
        "class_iou": class_iou,
        "class_hd95": class_hd95,
        "avg_dice": average_foreground(class_dice),
        "avg_iou": average_foreground(class_iou),
        "avg_hd95": average_foreground(class_hd95),
        "class_pred_ratio": pred_ratio,
        "class_gt_ratio": gt_ratio,
        "class_area_abs_error": area_abs_error.tolist(),
        "class_pred_to_gt_ratio": pred_to_gt.tolist(),
        "class_overseg_ratio": overseg.tolist(),
        "class_underseg_ratio": underseg.tolist(),
        "foreground_pred_ratio": foreground_pred_ratio,
        "foreground_gt_ratio": foreground_gt_ratio,
        "foreground_area_abs_error": abs(foreground_pred_ratio - foreground_gt_ratio),
    }
    if topology_totals:
        denom_batches = max(1, topology_batches)
        metrics.update({key: value / denom_batches for key, value in topology_totals.items()})
    for cls in range(num_classes):
        metrics[f"class_{cls}_pred_ratio"] = pred_ratio[cls]
        metrics[f"class_{cls}_gt_ratio"] = gt_ratio[cls]
        metrics[f"class_{cls}_area_abs_error"] = float(area_abs_error[cls])
        metrics[f"class_{cls}_pred_to_gt_ratio"] = float(pred_to_gt[cls])
        metrics[f"class_{cls}_overseg_ratio"] = float(overseg[cls])
        metrics[f"class_{cls}_underseg_ratio"] = float(underseg[cls])
    if save_dir:
        with (save_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "avg_dice", "avg_iou", "avg_hd95"]
            for cls in range(num_classes):
                fieldnames.extend([f"class_{cls}_pred_ratio", f"class_{cls}_gt_ratio"])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return metrics
