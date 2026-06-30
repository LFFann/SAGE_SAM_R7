from __future__ import annotations

import math
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torch.amp import GradScaler as _AmpGradScaler
except ImportError:
    _AmpGradScaler = None

try:
    from torch.amp import autocast as _amp_autocast
except ImportError:
    _amp_autocast = None

try:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast
except ImportError:
    _CudaGradScaler = None
    _cuda_autocast = None

from r6.calibration import PromptReliabilityCalibrator, SAMUtilityScheduler
from r6.data.dataset_2d import SegmentationDataset2D, resolve_dataset_root
from r6.data.paired_sampler import paired_batches
from r6.data.split import create_train_calibration_split
from r6.engine.checkpoint import export_deploy_payload, safe_load, save_checkpoint
from r6.engine.evaluator import evaluate
from r6.engine.logger import OneLineProgress, append_jsonl, setup_logger
from r6.engine.model_factory import build_deploy_model
from r6.losses.boundary_losses import boundary_bce_loss
from r6.losses.foreground_safe_kd import (
    foreground_safe_sam_consistency_loss,
    foreground_safe_sam_kd_loss,
    sam_guided_extent_kd_loss,
    student_anchored_sam_agreement_loss,
)
from r6.losses.sam_adapter_losses import sam_ce_dice_loss
from r6.losses.tri_state_pseudo_loss import tri_state_pseudo_supervision_loss
from r6.losses.supervised import supervised_loss
from r6.models.dual_temporal_teacher import DualTemporalTeacher
from r6.models.promptable_sam_mentor import PromptableSAMMentor
from r6.models.real_sam_wrapper import RealSAMWrapper
from r6.ssl.adaptive_ultrasound_augmentation import make_weak_strong_views
from r6.ssl.foreground_correlation_locality import (
    build_foreground_structure_mask,
    build_masked_locality_view,
    expand_targets_with_correlation,
    foreground_correlation_loss,
    masked_locality_proxy_loss,
    propagate_foreground_correlation_targets,
)
from r6.ssl.online_sam_relation import online_sam_student_relation_loss
from r6.ssl.foreground_safe_target_builder import build_foreground_safe_targets
from r6.utils.visualization import save_diagnostic_grid


class _NoOpGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def make_grad_scaler(device_type: str, enabled: bool):
    enabled = bool(enabled) and str(device_type) == "cuda"
    if _AmpGradScaler is not None:
        try:
            return _AmpGradScaler(str(device_type), enabled=enabled)
        except TypeError:
            return _AmpGradScaler(enabled=enabled)
    if _CudaGradScaler is not None:
        return _CudaGradScaler(enabled=enabled)
    return _NoOpGradScaler()


def amp_autocast(device_type: str, enabled: bool):
    enabled = bool(enabled) and str(device_type) == "cuda"
    if not enabled:
        return nullcontext()
    if _amp_autocast is not None:
        try:
            return _amp_autocast(device_type=str(device_type), enabled=True)
        except TypeError:
            return _amp_autocast(enabled=True)
    if _cuda_autocast is not None:
        return _cuda_autocast(enabled=True)
    return nullcontext()


class SAGESAMR6Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["checkpoints", "predictions", "visualizations", "calibration"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.output_dir)
        train_cfg = config["train"]
        dev = train_cfg.get("device", "cpu")
        if dev == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but unavailable; falling back to CPU")
            dev = "cpu"
        self.device = torch.device(dev)
        data_cfg = config["data"]
        model_cfg = config["model"]
        self.num_classes = int(data_cfg["num_classes"])
        self.ignore_index = int(data_cfg.get("ignore_index", 255))
        self.student = build_deploy_model(config).to(self.device)
        self.logger.info("deploy_model=%s backbone=%s", self.student.__class__.__name__, model_cfg.get("deploy_backbone", "dual_fusion"))
        self.dual_teacher = DualTemporalTeacher(
            self.student,
            fast_decay=config["teacher"].get("fast_ema_decay", 0.99),
            slow_decay=config["teacher"].get("slow_ema_decay", 0.999),
            use_bn_eval=config["teacher"].get("use_bn_eval_for_teacher", True),
        ).to(self.device)

        sam_cfg = config.get("sam", {})
        self.use_sam = bool(sam_cfg.get("use_sam", False))
        self.mentor: PromptableSAMMentor | None = None
        if self.use_sam:
            wrapper = RealSAMWrapper(
                sam_cfg["model_type"],
                sam_cfg["checkpoint"],
                sam_cfg.get("device", str(self.device)),
                sam_cfg.get("image_size", 1024),
                in_channels=data_cfg.get("in_channels", 3),
                num_classes=data_cfg.get("num_classes", 3),
                train_peft=sam_cfg.get("train_peft", True),
                peft_type=sam_cfg.get("peft_type", "adapter"),
                train_mask_decoder=sam_cfg.get("train_mask_decoder", True),
                train_prompt_encoder=not sam_cfg.get("freeze_prompt_encoder", True),
                train_last_n_blocks=sam_cfg.get("train_last_n_blocks", 0),
                lora_rank=sam_cfg.get("lora_rank", 4),
                lora_alpha=sam_cfg.get("lora_alpha", 8),
                adapter_dim=sam_cfg.get("adapter_dim", 32),
                adapter_scale=sam_cfg.get("adapter_scale", 1.0),
                max_trainable_ratio=sam_cfg.get("max_trainable_ratio", 0.05),
                use_mask_prompt=sam_cfg.get("prompt", {}).get("use_mask_prompt", True),
                use_box_prompt=sam_cfg.get("prompt", {}).get("use_box_prompt", True),
                use_point_prompt=sam_cfg.get("prompt", {}).get("use_point_prompt", True),
                use_negative_points=sam_cfg.get("prompt", {}).get("use_negative_points", True),
            )
            if not wrapper.sam_is_real():
                raise RuntimeError("SAM did not load as a real model")
            self.mentor = PromptableSAMMentor(
                wrapper,
                num_classes=self.num_classes,
                config=sam_cfg,
                in_channels=data_cfg.get("in_channels", 3),
            ).to(self.device)

        cal_cfg = config.get("calibration", config.get("conformal", {}))
        calibrator_start_iter = cal_cfg.get("start_iter", cal_cfg.get("calibrator_start_iter", 500))
        if cal_cfg.get("update_only_after_warmup", False):
            calibrator_start_iter = max(int(calibrator_start_iter), int(train_cfg.get("warmup_iterations", 0)))
        self.calibrator = PromptReliabilityCalibrator(
            self.num_classes,
            start_iter=calibrator_start_iter,
            update_every=cal_cfg.get("update_every", 250),
            momentum=cal_cfg.get("momentum", 0.8),
            min_pixels_per_class=cal_cfg.get("min_pixels_per_class", 128),
            use_soft_gate=cal_cfg.get("use_soft_gate", True),
            min_participation_ratio=cal_cfg.get("min_participation_ratio", 0.0),
            max_quantile_clip=cal_cfg.get("max_quantile_clip", 0.97),
            temperature=cal_cfg.get("temperature", 0.05),
            coverage_target=cal_cfg.get("coverage_target"),
        )
        self.sam_utility = SAMUtilityScheduler(
            max_weight=sam_cfg.get("losses", {}).get("sam_student_kd_weight", sam_cfg.get("semantic_kd_max_weight", 0.15)),
            ema_decay=sam_cfg.get("utility_ema_decay", 0.9),
            disable_after_no_gain=sam_cfg.get("disable_semantic_kd_after_no_gain", 3),
        )
        self.optimizer = self._build_optimizer()
        self.base_lrs = [float(group.get("lr", train_cfg.get("lr", 1e-3))) for group in self.optimizer.param_groups]
        self.trainable_parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = make_grad_scaler(self.device.type, self.amp)
        self.grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation", 1)))
        self.best_metrics = {"avg_dice": -1.0, "avg_hd95": float("inf")}
        self.start_iteration = 0
        self.calibration_iter = None
        self.calibration_update_count = 0
        self.val_collapse_count = 0
        self.stop_requested = False
        diag_cfg = config.get("diagnostics", {})
        self.train_visualize_every = int(diag_cfg.get("train_visualize_every", 0))
        self.train_visualize_max_samples = max(1, int(diag_cfg.get("train_visualize_max_samples", 1)))
        self._log_trainability()
        self._build_data()

    def _build_optimizer(self):
        train_cfg = self.config["train"]
        base_lr = float(train_cfg.get("lr", 1e-3))
        groups = [{"params": list(self.student.parameters()), "lr": base_lr, "name": "student"}]
        sam_cfg = self.config.get("sam", {})
        if self.use_sam and self.mentor is not None:
            groups.extend(self.mentor.optimizer_param_groups(base_lr, sam_cfg))
            if sam_cfg.get("train_peft", True) and not any(str(g.get("name", "")).startswith("sam_") for g in groups):
                raise RuntimeError("sam.train_peft=true but optimizer has no SAM parameter group")
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=train_cfg.get("weight_decay", 1e-4))

    def _update_learning_rate(self, iteration: int) -> float:
        train_cfg = self.config.get("train", {})
        schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
        if schedule in ("constant", "none", ""):
            return 1.0
        max_iter = max(1, int(train_cfg.get("max_iterations", iteration)))
        decay_start = max(0, int(train_cfg.get("lr_decay_start_iteration", 0)))
        min_ratio = float(train_cfg.get("min_lr_ratio", 0.0))
        if "min_lr" in train_cfg and self.base_lrs:
            min_ratio = max(min_ratio, float(train_cfg["min_lr"]) / max(self.base_lrs[0], 1e-12))
        min_ratio = min(max(min_ratio, 0.0), 1.0)
        if iteration <= decay_start:
            scale = 1.0
        else:
            denom = max(1, max_iter - decay_start)
            progress = min(1.0, max(0.0, (iteration - decay_start) / denom))
            if schedule == "cosine":
                scale = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            elif schedule == "poly":
                power = float(train_cfg.get("lr_poly_power", 0.9))
                scale = min_ratio + (1.0 - min_ratio) * ((1.0 - progress) ** power)
            else:
                raise ValueError(f"Unsupported lr_schedule={schedule!r}")
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = float(base_lr) * float(scale)
        return float(scale)

    def _log_trainability(self):
        group_names = [str(g.get("name", f"group{idx}")) for idx, g in enumerate(self.optimizer.param_groups)]
        self.logger.info("optimizer_param_groups=%s", group_names)
        if self.mentor is None:
            return
        report = self.mentor.trainability_report()
        self.logger.info(
            "sam_trainability total=%s trainable=%s ratio=%.6f lora=%s adapter=%s mask_decoder=%s prompt_generator_trainable=%s prompt_generator_params=%s modules=%s",
            report.get("total_sam_params"),
            report.get("trainable_sam_params"),
            report.get("trainable_sam_ratio", 0.0),
            report.get("lora_param_count"),
            report.get("adapter_param_count"),
            report.get("mask_decoder_trainable"),
            report.get("prompt_generator_trainable"),
            report.get("trainable_prompt_generator_params"),
            report.get("trainable_sam_modules"),
        )
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "sam_trainability", **report})

    def _build_data(self):
        cfg = self.config["data"]
        root = resolve_dataset_root(
            cfg["root"],
            cfg.get("dataset_name"),
            cfg.get("labeled_subdir", "labeled"),
            cfg.get("image_dir_name", "image"),
        )
        self.config["data"]["resolved_root"] = str(root)
        self.logger.info("dataset_root=%s", root)
        common = dict(
            root=root,
            num_classes=cfg["num_classes"],
            image_size=cfg["image_size"],
            image_dir_name=cfg.get("image_dir_name", "image"),
            mask_dir_name=cfg.get("mask_dir_name", "mask"),
            ignore_index=cfg.get("ignore_index", 255),
        )
        labeled_all = SegmentationDataset2D(split=cfg.get("labeled_subdir", "labeled"), has_mask=True, **common)
        train_idx, cal_idx, shared = create_train_calibration_split(
            labeled_all.records,
            cfg.get("calibration_ratio", 0.15),
            cfg.get("calibration_min_images", 4),
            cfg.get("calibration_split_seed", 2026),
        )
        if shared:
            self.logger.warning("Calibration split shares labeled samples because labeled count is too small")
        self.labeled_ds = Subset(labeled_all, train_idx)
        self.calibration_ds = Subset(labeled_all, cal_idx)
        self._configure_labeled_foreground_prior(labeled_all)
        self.unlabeled_ds = SegmentationDataset2D(split=cfg.get("unlabeled_subdir", "unlabeled"), has_mask=False, **common)
        self.val_ds = SegmentationDataset2D(split=cfg.get("val_subdir", "val"), has_mask=True, **common)
        self.test_ds = SegmentationDataset2D(split=cfg.get("test_subdir", "test"), has_mask=True, **common)
        train_cfg = self.config["train"]
        self.labeled_loader = DataLoader(
            self.labeled_ds,
            batch_size=train_cfg.get("batch_size_labeled", 2),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            drop_last=False,
        )
        self.unlabeled_loader = DataLoader(
            self.unlabeled_ds,
            batch_size=train_cfg.get("batch_size_unlabeled", 2),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            drop_last=False,
        )
        self.val_loader = DataLoader(self.val_ds, batch_size=self.config.get("eval", {}).get("batch_size", 1), shuffle=False, num_workers=0)
        self.calibration_loader = DataLoader(self.calibration_ds, batch_size=self.config.get("eval", {}).get("batch_size", 1), shuffle=False, num_workers=0)
        self.calibration_iter = cycle(self.calibration_loader)

    def _configure_labeled_foreground_prior(self, labeled_all: SegmentationDataset2D):
        pseudo_cfg = self.config.setdefault("pseudo", {})
        if not bool(pseudo_cfg.get("use_labeled_foreground_prior", False)):
            return
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        for rec in labeled_all.records:
            mask = labeled_all._load_mask(rec["mask_path"])
            valid = mask != self.ignore_index
            if valid.any():
                counts += torch.bincount(mask[valid].clamp(0, self.num_classes - 1).reshape(-1), minlength=self.num_classes).double()
        total = counts.sum().clamp_min(1.0)
        priors = (counts / total).tolist()
        multiplier = float(pseudo_cfg.get("foreground_prior_cap_multiplier", 4.0))
        min_cap_cfg = pseudo_cfg.get("foreground_prior_min_cap", pseudo_cfg.get("min_fg_pixels_per_class_ratio", 0.0))
        max_cap_cfg = pseudo_cfg.get("foreground_prior_max_cap", 1.0)
        old_caps = pseudo_cfg.get("max_fg_candidate_ratio_per_class", [1.0 for _ in range(self.num_classes)])
        new_caps = []
        for cls in range(self.num_classes):
            old = self._class_trust_value({"max_fg_candidate_ratio_per_class": old_caps}, "max_fg_candidate_ratio_per_class", cls, 1.0)
            if cls == 0:
                new_caps.append(float(old))
                continue
            min_cap = self._class_trust_value({"foreground_prior_min_cap": min_cap_cfg}, "foreground_prior_min_cap", cls, 0.0)
            max_cap = self._class_trust_value({"foreground_prior_max_cap": max_cap_cfg}, "foreground_prior_max_cap", cls, 1.0)
            prior_cap = min(max_cap, max(min_cap, float(priors[cls]) * multiplier))
            new_caps.append(min(float(old), prior_cap))
        pseudo_cfg["labeled_class_prior"] = priors
        pseudo_cfg["max_fg_candidate_ratio_per_class"] = new_caps
        pseudo_prior_event = {}
        if bool(pseudo_cfg.get("prior_calibrated_foreground_budget", False)):
            old_min_fg = pseudo_cfg.get("min_fg_pixels_per_class_ratio", [0.0 for _ in range(self.num_classes)])
            old_collapse_min = pseudo_cfg.get("collapse_min_fg_ratio_per_class", 0.0)
            old_collapse_force = pseudo_cfg.get("collapse_force_fg_ratio_per_class", old_collapse_min)
            min_multiplier = float(pseudo_cfg.get("foreground_prior_min_ratio_multiplier", 0.55))
            min_floor = float(pseudo_cfg.get("foreground_prior_min_ratio_floor", 0.0015))
            collapse_min_multiplier = float(pseudo_cfg.get("foreground_prior_collapse_min_multiplier", 0.45))
            collapse_force_multiplier = float(pseudo_cfg.get("foreground_prior_collapse_force_multiplier", 0.70))
            collapse_floor = float(pseudo_cfg.get("foreground_prior_collapse_floor", 0.0010))
            new_min_fg = []
            new_collapse_min = []
            new_collapse_force = []
            for cls in range(self.num_classes):
                if cls == 0:
                    new_min_fg.append(0.0)
                    new_collapse_min.append(0.0)
                    new_collapse_force.append(0.0)
                    continue
                old_min_value = self._class_trust_value({"min_fg_pixels_per_class_ratio": old_min_fg}, "min_fg_pixels_per_class_ratio", cls, 0.0)
                calibrated_min = max(min_floor, float(priors[cls]) * min_multiplier)
                new_min_fg.append(min(old_min_value, calibrated_min) if old_min_value > 0.0 else calibrated_min)

                old_collapse_min_value = self._class_trust_value({"collapse_min_fg_ratio_per_class": old_collapse_min}, "collapse_min_fg_ratio_per_class", cls, 0.0)
                calibrated_collapse_min = max(collapse_floor, float(priors[cls]) * collapse_min_multiplier)
                new_collapse_min.append(
                    min(old_collapse_min_value, calibrated_collapse_min) if old_collapse_min_value > 0.0 else calibrated_collapse_min
                )

                old_collapse_force_value = self._class_trust_value({"collapse_force_fg_ratio_per_class": old_collapse_force}, "collapse_force_fg_ratio_per_class", cls, 0.0)
                calibrated_collapse_force = max(collapse_floor, float(priors[cls]) * collapse_force_multiplier)
                new_collapse_force.append(
                    min(old_collapse_force_value, calibrated_collapse_force) if old_collapse_force_value > 0.0 else calibrated_collapse_force
                )
            pseudo_cfg["min_fg_pixels_per_class_ratio"] = new_min_fg
            pseudo_cfg["collapse_min_fg_ratio_per_class"] = new_collapse_min
            pseudo_cfg["collapse_force_fg_ratio_per_class"] = new_collapse_force
            pseudo_prior_event = {
                "pseudo_prior_budget_calibrated": True,
                "old_min_fg_pixels_per_class_ratio": old_min_fg,
                "new_min_fg_pixels_per_class_ratio": new_min_fg,
                "old_collapse_min_fg_ratio_per_class": old_collapse_min,
                "new_collapse_min_fg_ratio_per_class": new_collapse_min,
                "old_collapse_force_fg_ratio_per_class": old_collapse_force,
                "new_collapse_force_fg_ratio_per_class": new_collapse_force,
            }
        trust_cfg = self.config.setdefault("trust", {})
        trust_prior_event = {}
        if bool(trust_cfg.get("prior_calibrated_min_foreground", False)):
            old_min_candidate = float(trust_cfg.get("min_candidate_foreground_ratio", 0.0))
            fg_prior = float(sum(priors[1:]))
            candidate_multiplier = float(trust_cfg.get("min_candidate_prior_multiplier", 0.85))
            candidate_floor = float(trust_cfg.get("min_candidate_foreground_floor", 0.0))
            candidate_ceiling = float(trust_cfg.get("min_candidate_foreground_ceiling", old_min_candidate or 1.0))
            calibrated_candidate = min(candidate_ceiling, max(candidate_floor, fg_prior * candidate_multiplier))
            trust_cfg["min_candidate_foreground_ratio"] = calibrated_candidate

            old_class_min = trust_cfg.get("min_class_foreground_ratio", [0.0 for _ in range(self.num_classes)])
            class_multiplier = float(trust_cfg.get("min_class_prior_multiplier", 0.55))
            class_floor = float(trust_cfg.get("min_class_foreground_floor", 0.0))
            new_class_min = []
            for cls in range(self.num_classes):
                old_value = self._class_trust_value({"min_class_foreground_ratio": old_class_min}, "min_class_foreground_ratio", cls, 0.0)
                if cls == 0:
                    new_class_min.append(float(old_value))
                    continue
                calibrated_class = max(class_floor, float(priors[cls]) * class_multiplier)
                new_class_min.append(min(float(old_value), calibrated_class) if old_value > 0.0 else calibrated_class)
            trust_cfg["min_class_foreground_ratio"] = new_class_min
            if bool(trust_cfg.get("prior_calibrated_max_foreground", False)):
                old_max_candidate = float(trust_cfg.get("max_candidate_foreground_ratio", 1.0))
                max_candidate_multiplier = float(trust_cfg.get("max_candidate_prior_multiplier", 1.65))
                max_candidate_floor = float(trust_cfg.get("max_candidate_foreground_floor", calibrated_candidate))
                max_candidate_ceiling = float(trust_cfg.get("max_candidate_foreground_ceiling", old_max_candidate))
                calibrated_max_candidate = min(max_candidate_ceiling, max(max_candidate_floor, fg_prior * max_candidate_multiplier))
                trust_cfg["max_candidate_foreground_ratio"] = calibrated_max_candidate

                old_class_max = trust_cfg.get("max_class_foreground_ratio", [1.0 for _ in range(self.num_classes)])
                max_class_multiplier = float(trust_cfg.get("max_class_prior_multiplier", 2.0))
                max_class_floor = float(trust_cfg.get("max_class_foreground_floor", 0.0))
                new_class_max = []
                for cls in range(self.num_classes):
                    old_max_value = self._class_trust_value({"max_class_foreground_ratio": old_class_max}, "max_class_foreground_ratio", cls, 1.0)
                    if cls == 0:
                        new_class_max.append(float(old_max_value))
                        continue
                    calibrated_class_max = max(max_class_floor, float(priors[cls]) * max_class_multiplier)
                    new_class_max.append(min(float(old_max_value), calibrated_class_max))
                trust_cfg["max_class_foreground_ratio"] = new_class_max
            else:
                old_max_candidate = trust_cfg.get("max_candidate_foreground_ratio")
                calibrated_max_candidate = old_max_candidate
                old_class_max = trust_cfg.get("max_class_foreground_ratio")
                new_class_max = old_class_max
            trust_prior_event = {
                "trust_prior_calibrated": True,
                "old_min_candidate_foreground_ratio": old_min_candidate,
                "new_min_candidate_foreground_ratio": calibrated_candidate,
                "old_min_class_foreground_ratio": old_class_min,
                "new_min_class_foreground_ratio": new_class_min,
                "old_max_candidate_foreground_ratio": old_max_candidate,
                "new_max_candidate_foreground_ratio": calibrated_max_candidate,
                "old_max_class_foreground_ratio": old_class_max,
                "new_max_class_foreground_ratio": new_class_max,
                "foreground_prior_sum": fg_prior,
            }
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {
                "event": "labeled_foreground_prior_caps",
                "class_pixel_counts": [float(x) for x in counts.tolist()],
                "class_prior": priors,
                "max_fg_candidate_ratio_per_class": new_caps,
                "foreground_prior_cap_multiplier": multiplier,
                **pseudo_prior_event,
                **trust_prior_event,
            },
        )
        if hasattr(self, "logger"):
            self.logger.info("labeled foreground priors=%s capped_fg_ratios=%s", priors, new_caps)

    def fit_calibrator(self):
        self.logger.info("Prompt reliability calibration is online; skipping random pre-training fit")
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "online_calibrator_waiting", "iteration": 0})

    def dry_run(self):
        batch_l = next(iter(self.labeled_loader))
        batch_u = next(iter(self.unlabeled_loader))
        result = self.train_one_iter(batch_l, batch_u, iteration=0, update=False)
        self.logger.info("dry-run ok: %s", result)
        return result

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        payload = safe_load(checkpoint_path, map_location="cpu")
        student_state = payload["student"]
        if hasattr(self.student, "branch_a") and student_state and not any(str(k).startswith("branch_") for k in student_state):
            student_state = {f"branch_a.{k}": v for k, v in student_state.items()}
            self.logger.warning("Loaded legacy single-branch student weights into dual-fusion branch_a only")
        report = self.student.load_state_dict(student_state, strict=False)
        if report.missing_keys or report.unexpected_keys:
            self.logger.warning("student_resume_report missing=%s unexpected=%s", report.missing_keys, report.unexpected_keys)
        if payload.get("fast_teacher") is not None:
            self.dual_teacher.fast.load_state_dict(payload["fast_teacher"], strict=False)
        if payload.get("slow_teacher") is not None:
            self.dual_teacher.slow.load_state_dict(payload["slow_teacher"], strict=False)
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
            self._move_optimizer_state_to_device()
        if payload.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(payload["scaler"])
            except Exception as exc:
                self.logger.warning("Skipping scaler state from %s: %s", checkpoint_path, exc)
        if payload.get("calibrator") is not None:
            self.calibrator.load_state_dict(payload["calibrator"])
        if payload.get("sam_utility") is not None:
            self.sam_utility.load_state_dict(payload["sam_utility"])
        if self.mentor is not None and payload.get("mentor") is not None:
            report = self.mentor.load_trainable_state_dict(payload["mentor"])
            if report.get("missing") or report.get("unexpected"):
                self.logger.warning("mentor_resume_report=%s", report)
        if payload.get("best_metrics") is not None:
            self.best_metrics = payload["best_metrics"]
        self.calibration_update_count = int(payload.get("calibration_update_count", self.calibration_update_count))
        self.start_iteration = int(payload.get("iteration", 0))
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "checkpoint_resumed", "iteration": self.start_iteration, "path": str(checkpoint_path)})
        self.logger.info("resumed checkpoint=%s iteration=%d", checkpoint_path, self.start_iteration)
        return payload

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def train(self, max_iterations: int | None = None):
        max_iter = int(max_iterations or self.config["train"].get("max_iterations", 1))
        pair_iter = paired_batches(self.labeled_loader, self.unlabeled_loader)
        progress = OneLineProgress(max_iter)
        last_val_metrics = {}
        self.logger.info("train_start max_iterations=%d output_dir=%s", max_iter, self.output_dir)
        if self.start_iteration >= max_iter:
            self.logger.warning("resume iteration %d already reaches max_iterations=%d", self.start_iteration, max_iter)
            return self.output_dir / "checkpoints" / "latest.pth"
        for iteration in range(self.start_iteration + 1, max_iter + 1):
            batch_l, batch_u = next(pair_iter)
            step_optimizer = iteration % self.grad_accum_steps == 0 or iteration == max_iter
            logs = self.train_one_iter(batch_l, batch_u, iteration=iteration, update=True, step_optimizer=step_optimizer)
            progress.update(
                iteration,
                loss=logs["loss_total"],
                sup=logs["loss_sup"],
                set=logs["loss_set"],
                lr=logs["lr"],
                sam=logs["sam_valid_ratio"],
                **last_val_metrics,
            )
            if iteration % int(self.config["train"].get("log_every", 20)) == 0 or iteration == 1 or iteration == max_iter:
                append_jsonl(self.output_dir / "metrics.jsonl", {"iteration": iteration, "phase": "train", **logs})
                self.logger.info(
                    "train iter=%d loss=%.6f sup=%.6f set=%.6f sam_sup=%.6f sam_kd=%.6f prompt_q=%.4f adapter_grad=%.6f lr=%.6g",
                    iteration,
                    logs["loss_total"],
                    logs["loss_sup"],
                    logs["loss_set"],
                    logs["loss_sam_sup"],
                    logs["loss_sam_kd"],
                    logs["prompt_quality"],
                    logs["sam_adapter_grad_norm"],
                    logs["lr"],
                )
            if iteration % int(self.config["train"].get("val_every", 250)) == 0 or iteration == max_iter:
                metrics = self.validate(iteration)
                last_val_metrics = {
                    "val_dice": metrics.get("avg_dice"),
                    "best_dice": metrics.get("best_dice", self.best_metrics.get("avg_dice")),
                    "val_iou": metrics.get("avg_iou"),
                    "val_hd95": metrics.get("avg_hd95"),
                }
                progress.update(
                    iteration,
                    loss=logs["loss_total"],
                    sup=logs["loss_sup"],
                    set=logs["loss_set"],
                    lr=logs["lr"],
                    sam=logs["sam_valid_ratio"],
                    **last_val_metrics,
                )
                if self.stop_requested:
                    self.logger.warning("early stop requested at iteration=%d after validation collapse guard", iteration)
                    break
        progress.close()
        latest = self.output_dir / "checkpoints" / "latest.pth"
        best = self.output_dir / "checkpoints" / "best_val_dice.pth"
        deploy_src = best if best.exists() and bool(self.config["train"].get("deploy_best_checkpoint", True)) else latest
        export_deploy_payload(deploy_src, self.output_dir / "checkpoints" / "deploy_student.pth")
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {"event": "deploy_exported", "source": str(deploy_src), "path": str(self.output_dir / "checkpoints" / "deploy_student.pth")},
        )
        self.logger.info("train_end latest=%s deploy_source=%s deploy=%s", latest, deploy_src, self.output_dir / "checkpoints" / "deploy_student.pth")
        return latest

    def _r6_stage_weights(self, iteration: int, fast_slow_agreement: float):
        cfg = self.config.get("r6", {})
        fg_start = int(cfg.get("foreground_grounding_start", 1200))
        corr_start = int(cfg.get("correlation_locality_start", 3000))
        self_reliance_start = int(cfg.get("self_reliance_start", self.config.get("sam", {}).get("self_reliance_start", 6000)))
        low_agreement = fast_slow_agreement < float(cfg.get("min_fast_slow_agreement", 0.10))
        agreement_scale = float(cfg.get("low_agreement_unsup_scale", 0.25)) if low_agreement else 1.0
        if iteration < fg_start:
            return {
                "stage": 0.0,
                "unsup": 0.0,
                "sam": 0.0,
                "correlation": 0.0,
                "locality": 0.0,
                "low_agreement": 1.0 if low_agreement else 0.0,
            }
        if iteration < corr_start:
            stage1_unsup = min(agreement_scale, float(cfg.get("stage1_unsup_max_scale", 1.0)))
            stage1_sam = min(agreement_scale, float(cfg.get("stage1_sam_max_scale", stage1_unsup)))
            return {
                "stage": 1.0,
                "unsup": stage1_unsup,
                "sam": stage1_sam,
                "correlation": 0.0,
                "locality": 0.0,
                "low_agreement": 1.0 if low_agreement else 0.0,
            }
        decay_scale = 1.0
        if iteration >= self_reliance_start:
            decay_scale = float(cfg.get("self_reliance_ssl_scale", 0.50))
        return {
            "stage": 2.0 if iteration < self_reliance_start else 3.0,
            "unsup": agreement_scale,
            "sam": agreement_scale * decay_scale,
            "correlation": agreement_scale,
            "locality": agreement_scale,
            "low_agreement": 1.0 if low_agreement else 0.0,
        }

    def _apply_dynamic_trust(self, iteration: int, targets: dict, stage_weights: dict):
        cfg = self.config.get("trust", {})
        if not bool(cfg.get("enabled", False)):
            return targets, stage_weights, {
                "trust_gate_active": 0.0,
                "trust_unsafe": 0.0,
                "trust_unsup_scale": 1.0,
                "trust_sam_scale": 1.0,
                "trust_negative_scale": 1.0,
            }
        stats = targets.get("stats", {})
        start_iter = int(cfg.get("start_iter", 0))
        if iteration < start_iter:
            return targets, stage_weights, {
                "trust_gate_active": 0.0,
                "trust_unsafe": 0.0,
                "trust_unsup_scale": 1.0,
                "trust_sam_scale": 1.0,
                "trust_negative_scale": 1.0,
            }

        candidate_fg = float(stats.get("candidate_foreground_ratio", 0.0))
        safe_neg = float(stats.get("safe_negative_pixel_ratio", 0.0))
        background_hard = float(stats.get("background_hard_ratio", 0.0))
        sam_support = float(stats.get("sam_foreground_support_ratio", 1.0))
        sam_gate = float(stats.get("sam_train_gate_ratio", 0.0))
        per_class_fg = stats.get("per_class_foreground_participation_ratio", [])
        per_class_neg = stats.get("per_class_safe_negative_ratio", [])
        fg_classes = [int(c) for c in self.config.get("pseudo", {}).get("foreground_classes", list(range(1, self.num_classes)))]
        min_candidate_fg = float(cfg.get("min_candidate_foreground_ratio", 0.03))
        min_class_fg_default = 0.005
        max_candidate_fg = float(cfg.get("max_candidate_foreground_ratio", 1.0))
        max_safe_neg = float(cfg.get("max_safe_negative_pixel_ratio", 0.65))
        max_class_neg = float(cfg.get("max_class_safe_negative_ratio", 0.50))
        max_bg = float(cfg.get("max_background_hard_ratio", 0.45))
        min_sam_support = float(cfg.get("min_sam_foreground_support_ratio", 0.0))
        max_sam_gate_without_support = float(cfg.get("max_sam_gate_without_support", 1.0))
        max_sam_gate_to_support = float(cfg.get("max_sam_gate_to_support_ratio", 0.0))
        sam_support_floor = float(cfg.get("sam_support_ratio_floor", 0.005))
        low_candidate = candidate_fg < min_candidate_fg
        high_candidate = candidate_fg > max_candidate_fg
        high_negative = safe_neg > max_safe_neg
        high_background = background_hard > max_bg
        low_sam_support = sam_support < min_sam_support
        support_den = max(sam_support, sam_support_floor)
        sam_gate_too_wide = max_sam_gate_to_support > 0.0 and sam_gate > max_sam_gate_to_support * support_den
        sam_overgate = low_sam_support and (sam_gate > max_sam_gate_without_support or sam_gate_too_wide)
        low_class = False
        high_class = False
        pre_ceiling_flood = False
        high_class_negative = False
        for cls in fg_classes:
            if 0 < cls < len(per_class_fg):
                cls_fg = float(per_class_fg[cls])
                min_class_fg = self._class_trust_value(cfg, "min_class_foreground_ratio", cls, min_class_fg_default)
                low_class = low_class or cls_fg < min_class_fg
                cls_max = self._class_trust_value(cfg, "max_class_foreground_ratio", cls, 1.0)
                high_class = high_class or cls_fg > cls_max
                before = float(stats.get(f"foreground_ceiling_before_ratio_class{cls}", cls_fg))
                max_before = self._class_trust_value(cfg, "max_pre_ceiling_foreground_ratio", cls, 1.0)
                pre_ceiling_flood = pre_ceiling_flood or before > max_before
            if 0 < cls < len(per_class_neg):
                high_class_negative = high_class_negative or float(per_class_neg[cls]) > max_class_neg

        unsafe = bool(
            low_candidate
            or high_candidate
            or low_class
            or high_class
            or high_negative
            or high_class_negative
            or high_background
            or pre_ceiling_flood
            or (sam_overgate and bool(cfg.get("sam_overgate_marks_unsafe", True)))
        )
        trust_unsup_scale = float(cfg.get("unsafe_unsup_scale", 0.25)) if unsafe else 1.0
        trust_sam_scale = float(cfg.get("unsafe_sam_scale", trust_unsup_scale)) if unsafe else 1.0
        if low_sam_support or sam_overgate:
            trust_sam_scale = min(trust_sam_scale, float(cfg.get("low_support_sam_scale", 0.0)))
        trust_negative_scale = float(cfg.get("unsafe_negative_scale", 0.0)) if unsafe else 1.0
        out_weights = dict(stage_weights)
        if unsafe or sam_overgate or low_sam_support:
            out_weights["sam"] = float(out_weights.get("sam", 0.0)) * trust_sam_scale
        if unsafe:
            out_weights["unsup"] = float(out_weights.get("unsup", 0.0)) * trust_unsup_scale
            if bool(cfg.get("disable_correlation_when_unsafe", True)):
                out_weights["correlation"] = 0.0
            if bool(cfg.get("disable_locality_when_unsafe", True)):
                out_weights["locality"] = 0.0

        out_targets = targets
        if trust_negative_scale < 1.0 and "safe_negative_weight" in targets:
            out_targets = dict(targets)
            out_targets["safe_negative_weight"] = (targets["safe_negative_weight"].float() * trust_negative_scale).detach()
            out_targets["negative_mask"] = (targets["negative_mask"].bool() & (out_targets["safe_negative_weight"] > 0)).detach()

        return out_targets, out_weights, {
            "trust_gate_active": 1.0,
            "trust_unsafe": 1.0 if unsafe else 0.0,
            "trust_low_candidate": 1.0 if low_candidate else 0.0,
            "trust_high_candidate": 1.0 if high_candidate else 0.0,
            "trust_low_class": 1.0 if low_class else 0.0,
            "trust_high_class": 1.0 if high_class else 0.0,
            "trust_high_negative": 1.0 if high_negative else 0.0,
            "trust_high_class_negative": 1.0 if high_class_negative else 0.0,
            "trust_high_background": 1.0 if high_background else 0.0,
            "trust_low_sam_support": 1.0 if low_sam_support else 0.0,
            "trust_sam_gate_too_wide": 1.0 if sam_gate_too_wide else 0.0,
            "trust_sam_overgate": 1.0 if sam_overgate else 0.0,
            "trust_pre_ceiling_flood": 1.0 if pre_ceiling_flood else 0.0,
            "trust_min_candidate_foreground_ratio": min_candidate_fg,
            "trust_min_sam_foreground_support_ratio": min_sam_support,
            "trust_max_sam_gate_without_support": max_sam_gate_without_support,
            "trust_max_sam_gate_to_support_ratio": max_sam_gate_to_support,
            "trust_unsup_scale": trust_unsup_scale,
            "trust_sam_scale": trust_sam_scale,
            "trust_negative_scale": trust_negative_scale,
        }

    @staticmethod
    def _class_trust_value(config: dict, key: str, cls: int, default: float) -> float:
        value = config.get(key, default)
        if isinstance(value, dict):
            return float(value.get(cls, value.get(str(cls), default)))
        if isinstance(value, (list, tuple)):
            if cls < len(value):
                return float(value[cls])
            return float(value[-1]) if value else float(default)
        return float(value)

    def _map_from_class_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            return tensor
        out = tensor.new_zeros(tensor.shape[0], tensor.shape[2], tensor.shape[3], dtype=torch.long)
        for cls in range(1, min(self.num_classes, tensor.shape[1])):
            out = torch.where(tensor[:, cls].bool(), torch.full_like(out, cls), out)
        return out

    def _prompt_stats(self, sam_out: dict) -> dict[str, float]:
        prompts = sam_out.get("prompts") if isinstance(sam_out, dict) else None
        if not isinstance(prompts, dict):
            return {}
        stats: dict[str, float] = {}
        for key in ("prompt_valid", "prompt_area_ratio", "prompt_box_area_ratio", "prompt_component_count"):
            tensor = prompts.get(key)
            if tensor is None or not hasattr(tensor, "detach"):
                continue
            values = tensor.detach().float()
            if values.numel() == 0:
                continue
            fg_values = values[:, 1:] if values.ndim == 2 and values.shape[1] > 1 else values
            stats[f"sam_{key}_mean"] = float(fg_values.mean())
            if values.ndim == 2:
                for cls in range(1, min(self.num_classes, values.shape[1])):
                    stats[f"sam_{key}_class{cls}"] = float(values[:, cls].mean())
        return stats

    def _maybe_save_training_visualization(
        self,
        iteration: int,
        image: torch.Tensor,
        teacher_prob: torch.Tensor,
        student_prob: torch.Tensor | None,
        sam_u: dict,
        targets: dict,
    ) -> str | None:
        if self.train_visualize_every <= 0:
            return None
        save_first = bool(self.config.get("diagnostics", {}).get("train_visualize_first", True))
        should_save = iteration % self.train_visualize_every == 0 or (save_first and iteration == 1)
        if not should_save:
            return None
        if image.ndim != 4 or image.shape[0] == 0:
            return None

        idx = 0
        teacher_pred = teacher_prob.argmax(dim=1)
        teacher_fg = teacher_prob[:, 1:].max(dim=1).values if teacher_prob.shape[1] > 1 else teacher_prob[:, 0]
        if student_prob is not None:
            student_pred = student_prob.argmax(dim=1)
            student_fg = student_prob[:, 1:].max(dim=1).values if student_prob.shape[1] > 1 else student_prob[:, 0]
        else:
            student_pred = torch.zeros_like(teacher_pred)
            student_fg = torch.zeros_like(teacher_fg)
        candidate_map = self._map_from_class_channels(targets["candidate_set"].bool())
        negative_map = self._map_from_class_channels(targets["safe_negative_set"].bool())
        panels = [
            ("teacher_pred", teacher_pred[idx], "mask"),
            ("teacher_fg_prob", teacher_fg[idx], "heatmap"),
            ("student_pred", student_pred[idx], "mask"),
            ("student_fg_prob", student_fg[idx], "heatmap"),
            ("candidate_set", candidate_map[idx], "mask"),
            ("safe_negative", negative_map[idx], "mask"),
            ("ambiguous", targets["ambiguous_mask"][idx].float(), "heatmap"),
            ("sam_train_gate", targets["sam_train_gate"][idx].float(), "heatmap"),
        ]
        if sam_u.get("valid") and sam_u.get("sam_prob") is not None:
            prompts = sam_u.get("prompts")
            if isinstance(prompts, dict):
                panels.append(("sam_prompt", {"prompts": prompts, "sample_index": idx}, "prompt_overlay"))
                soft_prompt = prompts.get("soft_prompt")
                if soft_prompt is not None and soft_prompt.ndim == 4 and idx < soft_prompt.shape[0]:
                    for fg_idx in range(min(self.num_classes - 1, soft_prompt.shape[1])):
                        panels.append((f"prompt_c{fg_idx + 1}", soft_prompt[idx, fg_idx], "heatmap"))
            sam_prob = sam_u["sam_prob"].detach()
            sam_pred = sam_prob.argmax(dim=1)
            sam_fg = sam_prob[:, 1:].max(dim=1).values if sam_prob.shape[1] > 1 else sam_prob[:, 0]
            panels.extend([("sam_pred", sam_pred[idx], "mask"), ("sam_fg_prob", sam_fg[idx], "heatmap")])
        if targets.get("sam_verifier_score") is not None:
            panels.append(("sam_verifier", targets["sam_verifier_score"][idx], "heatmap"))

        path = self.output_dir / "visualizations" / "train" / f"iter_{iteration:06d}.png"
        save_diagnostic_grid(image[idx].detach(), panels, path, num_classes=self.num_classes, cols=4)
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {"event": "train_visualization", "iteration": iteration, "path": str(path)},
        )
        return str(path)

    def _add_baseline_gaps(self, metrics: dict) -> dict:
        baseline_cfg = self.config.get("eval", {}).get("baseline", {})
        if not baseline_cfg:
            return metrics
        if baseline_cfg.get("avg_dice") is not None:
            metrics["baseline_gap_avg_dice"] = float(metrics.get("avg_dice", 0.0)) - float(baseline_cfg["avg_dice"])
        baseline_class_dice = baseline_cfg.get("class_dice")
        if isinstance(baseline_class_dice, (list, tuple)) and isinstance(metrics.get("class_dice"), list):
            gaps = []
            for idx, value in enumerate(metrics["class_dice"]):
                base = baseline_class_dice[idx] if idx < len(baseline_class_dice) else None
                gaps.append(float(value) - float(base) if base is not None else float("nan"))
            metrics["baseline_gap_class_dice"] = gaps
            for idx, gap in enumerate(gaps):
                metrics[f"baseline_gap_class{idx}_dice"] = gap
        return metrics

    def train_one_iter(self, batch_l, batch_u, iteration: int, update: bool = True, step_optimizer: bool = True):
        lr_scale = self._update_learning_rate(iteration)
        self.student.train()
        if self.mentor is not None:
            self.mentor.train()
        x_l = batch_l["image"].to(self.device)
        y_l = batch_l["mask"].to(self.device)
        x_u = batch_u["image"].to(self.device)
        aug_cfg = self.config.get("augmentation", {})
        x_u_w, x_u_s1, x_u_s2, _, _ = make_weak_strong_views(
            x_u,
            strong_kwargs=aug_cfg.get("strong", {}),
            weak_kwargs=aug_cfg.get("weak", {}),
        )
        with torch.no_grad():
            teacher_out = self.dual_teacher.predict_weak(x_u_w)
            student_w_prob = torch.softmax(self.student(x_u_w), dim=1) if self.use_sam else None
        stage_weights = self._r6_stage_weights(iteration, float(teacher_out["agreement"].detach()))

        pseudo_cfg = self.config.get("pseudo", {})
        pseudo_runtime_cfg = dict(self.config.get("r6", {}))
        pseudo_runtime_cfg.update(pseudo_cfg)
        pseudo_runtime_cfg["_iteration"] = iteration
        sam_loss_cfg = self.config.get("sam", {}).get("losses", {})
        structure_cfg = self.config.get("structure", {})
        sam_l = {"valid": False}
        sam_u = {"valid": False}
        with amp_autocast(self.device.type, self.amp):
            out_l = self.student(x_l, return_features=True)
            loss_sup_fusion, sup_logs = supervised_loss(out_l["logits"], y_l, self.num_classes, self.ignore_index)
            loss_sup_a = self._supervised_branch_loss(out_l, "logits_a", y_l)
            loss_sup_b = self._supervised_branch_loss(out_l, "logits_b", y_l)
            loss_cfg = self.config.get("losses", {})
            branch_sup_weight = float(loss_cfg.get("branch_sup_weight", 0.5))
            fusion_sup_weight = float(loss_cfg.get("fusion_sup_weight", 1.0))
            if "logits_a" in out_l and "logits_b" in out_l:
                loss_sup = fusion_sup_weight * loss_sup_fusion + branch_sup_weight * 0.5 * (loss_sup_a + loss_sup_b)
            else:
                loss_sup = loss_sup_fusion

            loss_sam_sup = x_l.new_tensor(0.0)
            loss_sam_unsup = x_l.new_tensor(0.0)
            loss_kd = x_l.new_tensor(0.0)
            loss_sam_extent = x_l.new_tensor(0.0)
            loss_sam_agreement = x_l.new_tensor(0.0)
            loss_relation = x_l.new_tensor(0.0)
            loss_boundary = x_l.new_tensor(0.0)
            loss_corr = x_l.new_tensor(0.0)
            loss_local = x_l.new_tensor(0.0)
            loss_conflict = x_l.new_tensor(0.0)
            sam_kd_gate_ratio = 0.0
            sam_kd_gate_weight_mean = 0.0
            sam_agreement_gate_ratio = 0.0
            sam_agreement_weight_mean = 0.0
            sam_agreement_effective_weight = 0.0
            sam_agreement_floor_active = False
            masked_locality_stats = {"masked_locality_ratio": 0.0, "foreground_masked_ratio": 0.0}
            if self.use_sam and self.mentor is not None:
                sam_l = self.mentor.forward_labeled(x_l, y_l)
                loss_sam_sup = sam_ce_dice_loss(sam_l["sam_prob"], y_l, self.num_classes, self.ignore_index)
                sam_u = self.mentor.forward_unlabeled(x_u_w, teacher_out["mean_prob"], student_w_prob)

            targets = build_foreground_safe_targets(teacher_out, sam_u, self.calibrator, pseudo_runtime_cfg)
            targets, stage_weights, trust_logs = self._apply_dynamic_trust(iteration, targets, stage_weights)
            out_s1 = self.student(x_u_s1, return_features=True)
            out_s2 = self.student(x_u_s2, return_features=True, feature_dropout="complementary")
            if stage_weights["correlation"] > 0.0 and float(pseudo_cfg.get("correlation_weight", 0.0)) > 0.0 and out_s1.get("fusion_feature") is not None:
                sam_shape = targets.get("sam_boundary")
                if sam_shape is None and sam_u.get("valid"):
                    sam_shape = sam_u.get("sam_boundary")
                propagated = propagate_foreground_correlation_targets(
                    out_s1["fusion_feature"],
                    torch.softmax(out_s1["logits"].detach(), dim=1),
                    targets,
                    sam_shape=sam_shape,
                    resolution=structure_cfg.get("correlation_resolution", structure_cfg.get("relation_resolution", 16)),
                    topk=structure_cfg.get("correlation_topk", structure_cfg.get("online_topk", 8)),
                    temperature=structure_cfg.get("correlation_temperature", structure_cfg.get("relation_temperature", 0.2)),
                    min_weight=structure_cfg.get("correlation_min_weight", 0.15),
                )
                targets = expand_targets_with_correlation(
                    targets,
                    propagated,
                    min_weight=structure_cfg.get("correlation_min_weight", 0.15),
                )
                loss_corr = foreground_correlation_loss(out_s1["logits"], propagated)
            ssl1 = tri_state_pseudo_supervision_loss(out_s1["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            ssl2 = tri_state_pseudo_supervision_loss(out_s2["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            branch_ssl = self._branch_ssl_loss(out_s1, out_s2, targets, pseudo_cfg)
            loss_conflict = 0.5 * (
                self._dual_consistency_loss(out_s1, targets) + self._dual_consistency_loss(out_s2, targets)
            )
            ramp = min(1.0, iteration / max(1, int(self.config["train"].get("unsup_ramp_iterations", 1))))
            loss_unsup = (
                pseudo_cfg.get("singleton_weight", 1.0) * (ssl1["loss_singleton"] + ssl2["loss_singleton"]) * 0.5
                + pseudo_cfg.get("set_weight", 0.5) * (ssl1["loss_set"] + ssl2["loss_set"]) * 0.5
                + pseudo_cfg.get("rank_weight", 0.1) * (ssl1["loss_rank"] + ssl2["loss_rank"]) * 0.5
                + pseudo_cfg.get("negative_weight", 0.1) * (ssl1["loss_negative"] + ssl2["loss_negative"]) * 0.5
                + pseudo_cfg.get("fuzzy_weight", 0.25) * (ssl1["loss_fuzzy"] + ssl2["loss_fuzzy"]) * 0.5
                + float(loss_cfg.get("branch_ssl_weight", 0.5)) * branch_ssl
            )
            if stage_weights["locality"] > 0.0 and float(pseudo_cfg.get("locality_weight", 0.0)) > 0.0:
                locality_cfg = self.config.get("locality", {})
                locality_seed = build_foreground_structure_mask(targets)
                x_u_local, masked_locality_stats = build_masked_locality_view(
                    x_u_w,
                    locality_seed,
                    mask_ratio=locality_cfg.get("mask_ratio", pseudo_cfg.get("locality_mask_ratio", 0.30)),
                    patch_size=locality_cfg.get("patch_size", pseudo_cfg.get("locality_patch_size", 16)),
                    fill=locality_cfg.get("fill", "mean"),
                )
                if masked_locality_stats["masked_locality_ratio"] > 0.0:
                    out_local = self.student(x_u_local, return_features=True, feature_dropout="complementary")
                    loss_local = masked_locality_proxy_loss(out_local["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))

            if self.use_sam and sam_u.get("valid"):
                foreground_mask = targets.get("sam_kd_gate")
                if foreground_mask is None:
                    foreground_mask = build_foreground_structure_mask(targets)
                if foreground_mask is None:
                    foreground_mask = targets.get("sam_train_gate", targets.get("sam_region_gate")).bool()
                foreground_mask = foreground_mask.to(device=out_s1["logits"].device).bool()
                sam_base_weight = targets.get(
                    "sam_kd_weight",
                    targets.get("structure_weight", targets["sam_weight"]),
                ).to(device=foreground_mask.device).float()
                sam_gate_weight = sam_base_weight * foreground_mask.float()
                sam_kd_gate_ratio = float(foreground_mask.float().mean().detach())
                sam_kd_gate_weight_mean = float(sam_gate_weight.mean().detach())
                sam_utility_value = float((sam_u["sam_prob"].detach()[:, 1:] * targets["teacher_only_soft_target"][:, 1:]).sum(dim=1).mean())
                self.sam_utility.update(sam_utility_value)
                loss_kd = foreground_safe_sam_kd_loss(
                    out_s1["logits"],
                    sam_u["sam_prob"],
                    foreground_mask=foreground_mask,
                    gate=sam_gate_weight,
                    temperature=float(self.config.get("sam", {}).get("kd_temperature", 1.0)),
                )
                loss_sam_unsup = foreground_safe_sam_consistency_loss(
                    sam_u["sam_prob"],
                    targets["teacher_only_soft_target"],
                    foreground_mask=foreground_mask,
                    gate=sam_gate_weight,
                )
                sam_guided_mask = targets.get("sam_guided_mask")
                sam_guided_weight = targets.get("sam_guided_weight")
                if sam_guided_mask is not None and sam_guided_weight is not None:
                    guided_gate = sam_guided_weight.to(device=foreground_mask.device).float() * sam_guided_mask.to(device=foreground_mask.device).float()
                    loss_sam_extent = sam_guided_extent_kd_loss(
                        out_s1["logits"],
                        sam_u["sam_prob"],
                        targets["teacher_only_soft_target"],
                        gate=guided_gate,
                        temperature=float(self.config.get("sam", {}).get("extent_kd_temperature", self.config.get("sam", {}).get("kd_temperature", 1.0))),
                        sam_mix=float(sam_loss_cfg.get("sam_extent_target_mix", 0.65)),
                    )
                if float(sam_loss_cfg.get("sam_agreement_weight", 0.0)) > 0.0:
                    loss_sam_agreement, sam_agreement_stats = student_anchored_sam_agreement_loss(
                        out_s1["logits"],
                        sam_u["sam_prob"],
                        targets["sam_support"],
                        targets["sam_verifier_score"],
                        min_support=float(sam_loss_cfg.get("sam_agreement_min_support", 0.06)),
                        min_verifier=float(sam_loss_cfg.get("sam_agreement_min_verifier", 0.45)),
                        uncertain_max_confidence=float(sam_loss_cfg.get("sam_agreement_uncertain_max_confidence", 0.72)),
                        temperature=float(
                            self.config.get("sam", {}).get(
                                "agreement_kd_temperature",
                                self.config.get("sam", {}).get("kd_temperature", 1.0),
                            )
                        ),
                    )
                    sam_agreement_gate_ratio = float(sam_agreement_stats["sam_agreement_gate_ratio"])
                    sam_agreement_weight_mean = float(sam_agreement_stats["sam_agreement_weight_mean"])
                if structure_cfg.get("use_online_relation", False):
                    relation_gate = targets["structure_gate"].to(device=foreground_mask.device).bool() & foreground_mask
                    loss_relation = online_sam_student_relation_loss(
                        out_s1["bottleneck"],
                        sam_u.get("sam_embedding"),
                        gate=relation_gate,
                        boundary=sam_u.get("sam_boundary"),
                        topk=structure_cfg.get("online_topk", 8),
                        resolution=structure_cfg.get("relation_resolution", 16),
                        temperature=structure_cfg.get("relation_temperature", 0.2),
                        rank_weight=structure_cfg.get("relation_rank_weight", 0.25),
                    )
                boundary_src = sam_u.get("sam_boundary", targets.get("sam_boundary"))
                if out_s1.get("boundary_logits") is not None and boundary_src is not None:
                    boundary_target = boundary_src.detach() * targets["structure_weight"].unsqueeze(1).float()
                    loss_boundary = boundary_bce_loss(out_s1["boundary_logits"], boundary_target)

            sam_scale = self._sam_self_reliance_scale(iteration)
            sam_sup_scale = 1.0 if self.config.get("sam", {}).get("keep_labeled_sam_sup", True) else sam_scale
            unsup_scale = ramp * float(stage_weights["unsup"])
            sam_ssl_scale = sam_scale * float(stage_weights["sam"])
            sam_kd_loss_weight = float(sam_loss_cfg.get("sam_student_kd_weight", self.sam_utility.semantic_weight(iteration)))
            sam_kd_raw_effective_weight = float(unsup_scale * sam_ssl_scale * sam_kd_loss_weight)
            sam_kd_effective_weight = sam_kd_raw_effective_weight
            sam_kd_floor_active = False
            sam_kd_floor = float(sam_loss_cfg.get("sam_kd_min_effective_weight", 0.0))
            sam_kd_floor_after = int(
                sam_loss_cfg.get(
                    "sam_kd_min_effective_after",
                    self.config.get("r6", {}).get("foreground_grounding_start", 0),
                )
            )
            sam_kd_floor_gate_ratio = float(sam_loss_cfg.get("sam_kd_min_effective_gate_ratio", 0.0))
            if (
                sam_kd_floor > 0.0
                and iteration >= sam_kd_floor_after
                and sam_u.get("valid")
                and sam_kd_gate_ratio >= sam_kd_floor_gate_ratio
            ):
                sam_kd_effective_weight = max(sam_kd_effective_weight, sam_kd_floor)
                sam_kd_floor_active = sam_kd_effective_weight > sam_kd_raw_effective_weight
            sam_agreement_loss_weight = float(sam_loss_cfg.get("sam_agreement_weight", 0.0))
            sam_agreement_raw_effective_weight = float(unsup_scale * sam_ssl_scale * sam_agreement_loss_weight)
            sam_agreement_effective_weight = sam_agreement_raw_effective_weight
            sam_agreement_floor = float(sam_loss_cfg.get("sam_agreement_min_effective_weight", 0.0))
            sam_agreement_floor_after = int(
                sam_loss_cfg.get(
                    "sam_agreement_min_effective_after",
                    self.config.get("r6", {}).get("foreground_grounding_start", 0),
                )
            )
            sam_agreement_floor_gate_ratio = float(sam_loss_cfg.get("sam_agreement_min_effective_gate_ratio", 0.0))
            if (
                sam_agreement_floor > 0.0
                and iteration >= sam_agreement_floor_after
                and sam_u.get("valid")
                and sam_agreement_gate_ratio >= sam_agreement_floor_gate_ratio
            ):
                sam_agreement_effective_weight = max(sam_agreement_effective_weight, sam_agreement_floor)
                sam_agreement_floor_active = sam_agreement_effective_weight > sam_agreement_raw_effective_weight
            sam_aux_effective_scale = float(unsup_scale * sam_ssl_scale)
            non_sam_unsup_loss = (
                loss_unsup
                + float(loss_cfg.get("conflict_review_weight", 0.2)) * loss_conflict
                + float(stage_weights["correlation"]) * pseudo_cfg.get("correlation_weight", 0.0) * loss_corr
                + float(stage_weights["locality"]) * pseudo_cfg.get("locality_weight", 0.0) * loss_local
            )
            sam_aux_loss = (
                float(sam_loss_cfg.get("sam_unsup_weight", 0.2)) * loss_sam_unsup
                + float(sam_loss_cfg.get("sam_extent_weight", 0.0)) * loss_sam_extent
                + float(sam_loss_cfg.get("sam_relation_weight", pseudo_cfg.get("relation_weight", 0.05))) * loss_relation
                + float(sam_loss_cfg.get("sam_boundary_weight", pseudo_cfg.get("boundary_weight", 0.05))) * loss_boundary
            )
            loss = (
                loss_sup
                + sam_sup_scale * sam_loss_cfg.get("sam_sup_weight", self.config.get("sam", {}).get("sam_sup_weight", 0.5)) * loss_sam_sup
                + unsup_scale * non_sam_unsup_loss
                + sam_aux_effective_scale * sam_aux_loss
                + sam_kd_effective_weight * loss_kd
                + sam_agreement_effective_weight * loss_sam_agreement
            )

        sam_grad_norm = 0.0
        if update:
            if (iteration - 1) % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss / self.grad_accum_steps).backward()
            if step_optimizer and self.config["train"].get("grad_clip_norm"):
                self.scaler.unscale_(self.optimizer)
                sam_grad_norm = self.mentor.sam_grad_norm() if self.mentor is not None else 0.0
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config["train"]["grad_clip_norm"])
            if step_optimizer:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.dual_teacher.update_fast(self.student)
                if iteration % int(self.config["teacher"].get("slow_refresh_every", 500)) == 0:
                    self.dual_teacher.refresh_slow(self.student)
                    append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "slow_teacher_refresh", "iteration": iteration})
                self._maybe_update_prompt_calibrator(iteration, fallback_out=out_l, fallback_y=y_l, fallback_sam=sam_l)

        prompt_quality = 0.0
        if sam_u.get("valid") and sam_u.get("prompt_quality") is not None:
            prompt_quality = float(sam_u["prompt_quality"].detach().mean())
        prompt_stats = self._prompt_stats(sam_u)
        logs = {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_sup_fusion": float(loss_sup_fusion.detach()),
            "loss_sup_a": float(loss_sup_a.detach()),
            "loss_sup_b": float(loss_sup_b.detach()),
            "loss_singleton": float(ssl1["loss_singleton"].detach()),
            "loss_set": float(ssl1["loss_set"].detach()),
            "loss_rank": float(ssl1["loss_rank"].detach()),
            "loss_negative": float(ssl1["loss_negative"].detach()),
            "loss_fuzzy": float(ssl1["loss_fuzzy"].detach()),
            "loss_branch_ssl": float(branch_ssl.detach()),
            "loss_conflict_review": float(loss_conflict.detach()),
            "loss_correlation": float(loss_corr.detach()),
            "loss_locality": float(loss_local.detach()),
            "masked_locality_ratio": float(masked_locality_stats["masked_locality_ratio"]),
            "foreground_masked_ratio": float(masked_locality_stats["foreground_masked_ratio"]),
            "loss_relation": float(loss_relation.detach()),
            "loss_boundary": float(loss_boundary.detach()),
            "loss_sam_sup": float(loss_sam_sup.detach()),
            "loss_sam_unsup": float(loss_sam_unsup.detach()),
            "loss_sam_kd": float(loss_kd.detach()),
            "loss_sam_extent": float(loss_sam_extent.detach()),
            "loss_sam_agreement": float(loss_sam_agreement.detach()),
            "loss_sam_sem": float(loss_kd.detach()),
            "unsup_weight": ramp,
            "r6_unsup_scale": float(unsup_scale),
            "r6_stage": float(stage_weights["stage"]),
            "r6_low_agreement": float(stage_weights["low_agreement"]),
            "sam_self_reliance_scale": sam_scale,
            "sam_semantic_weight": self.sam_utility.semantic_weight(iteration),
            "sam_ssl_scale": float(sam_ssl_scale),
            "sam_kd_loss_weight": float(sam_kd_loss_weight),
            "sam_kd_raw_effective_weight": float(sam_kd_raw_effective_weight),
            "sam_kd_effective_weight": float(sam_kd_effective_weight),
            "sam_kd_floor_active": 1.0 if sam_kd_floor_active else 0.0,
            "sam_agreement_gate_ratio": float(sam_agreement_gate_ratio),
            "sam_agreement_weight_mean": float(sam_agreement_weight_mean),
            "sam_agreement_effective_weight": float(sam_agreement_effective_weight),
            "sam_agreement_floor_active": 1.0 if sam_agreement_floor_active else 0.0,
            "sam_utility_ema": float(self.sam_utility.utility_ema),
            "sam_utility_disabled": 1.0 if self.sam_utility.disabled else 0.0,
            "fast_slow_agreement": float(teacher_out["agreement"].detach()),
            "sam_valid_ratio": 1.0 if sam_u.get("valid") else 0.0,
            "sam_kd_gate_ratio": sam_kd_gate_ratio,
            "sam_kd_gate_weight_mean": sam_kd_gate_weight_mean,
            "prompt_quality": prompt_quality,
            "sam_adapter_grad_norm": sam_grad_norm,
            "optimizer_step": 1.0 if (update and step_optimizer) else 0.0,
            "gradient_accumulation": float(self.grad_accum_steps),
            "lr": self.optimizer.param_groups[0]["lr"],
            "lr_scale": lr_scale,
            "gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if self.device.type == "cuda" else 0.0,
            **prompt_stats,
            **trust_logs,
            **targets["stats"],
            **sup_logs,
        }
        visual_path = self._maybe_save_training_visualization(
            iteration,
            x_u_w,
            teacher_out["mean_prob"],
            student_w_prob,
            sam_u,
            targets,
        )
        if visual_path:
            logs["train_visualization"] = visual_path
        return logs

    def _supervised_branch_loss(self, out: dict, key: str, target: torch.Tensor):
        if key not in out:
            return out["logits"].new_tensor(0.0)
        loss, _ = supervised_loss(out[key], target, self.num_classes, self.ignore_index)
        return loss

    def _branch_ssl_loss(self, out_s1: dict, out_s2: dict, targets: dict, pseudo_cfg: dict):
        losses = []
        for out in (out_s1, out_s2):
            for key in ("logits_a", "logits_b"):
                if key not in out:
                    continue
                ssl = tri_state_pseudo_supervision_loss(out[key], targets, pseudo_cfg.get("rank_margin", 0.5))
                losses.append(
                    pseudo_cfg.get("singleton_weight", 1.0) * ssl["loss_singleton"]
                    + pseudo_cfg.get("set_weight", 0.5) * ssl["loss_set"]
                    + pseudo_cfg.get("rank_weight", 0.1) * ssl["loss_rank"]
                    + pseudo_cfg.get("negative_weight", 0.1) * ssl["loss_negative"]
                    + pseudo_cfg.get("fuzzy_weight", 0.25) * ssl["loss_fuzzy"]
                )
        if not losses:
            return out_s1["logits"].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _dual_consistency_loss(self, out: dict, targets: dict):
        if "logits_a" not in out or "logits_b" not in out:
            return out["logits"].new_tensor(0.0)
        target_prob = torch.softmax(out["logits"].detach(), dim=1)
        mask = (targets["singleton_mask"] | targets["ambiguous_mask"] | targets["conflict_mask"]).to(out["logits"].device)
        if mask.sum() == 0:
            return out["logits"].new_tensor(0.0)
        weight = targets.get("candidate_weight", mask.float()).to(out["logits"].device).float() * mask.float()
        denom = weight.sum().clamp_min(1e-6)

        def _ce(logits):
            per_pixel = -(target_prob * F.log_softmax(logits, dim=1)).sum(dim=1)
            return (per_pixel * weight).sum() / denom

        return 0.5 * (_ce(out["logits_a"]) + _ce(out["logits_b"]))

    def _sam_self_reliance_scale(self, iteration: int):
        sam_cfg = self.config.get("sam", {})
        max_iter = int(self.config["train"].get("max_iterations", 10**9))
        default_start = int(0.7 * max_iter)
        start = int(sam_cfg.get("self_reliance_start", default_start))
        if iteration <= start:
            return 1.0
        decay = float(sam_cfg.get("self_reliance_decay", 1.0))
        floor = float(sam_cfg.get("self_reliance_min_weight", 0.05))
        return max(floor, decay ** max(0, iteration - start))

    @torch.no_grad()
    def _maybe_update_prompt_calibrator(
        self,
        iteration: int,
        fallback_out: dict | None = None,
        fallback_y: torch.Tensor | None = None,
        fallback_sam: dict | None = None,
    ):
        if not (self.use_sam and self.mentor is not None and self.calibrator.should_update(iteration)):
            return
        cal_cfg = self.config.get("calibration", {})
        event_name = "prompt_reliability_update"
        if self.calibration_iter is not None and cal_cfg.get("use_calibration_split", True):
            batch_c = next(self.calibration_iter)
            x_c = batch_c["image"].to(self.device)
            y_c = batch_c["mask"].to(self.device)
            student_was_training = getattr(self.student, "training", None)
            mentor_was_training = getattr(self.mentor, "training", None)
            if hasattr(self.student, "eval"):
                self.student.eval()
            if hasattr(self.mentor, "eval"):
                self.mentor.eval()
            try:
                out_c = self.student(x_c, return_features=True)
                sam_c = self.mentor.forward_labeled(x_c, y_c)
            finally:
                if student_was_training and hasattr(self.student, "train"):
                    self.student.train()
                if mentor_was_training and hasattr(self.mentor, "train"):
                    self.mentor.train()
            event_name = "prompt_reliability_update_calibration_split"
        else:
            out_c = fallback_out
            y_c = fallback_y
            sam_c = fallback_sam or {}
        if out_c is None or y_c is None or not sam_c.get("valid"):
            return
        student_prob_l = torch.softmax(out_c["logits"].detach(), dim=1)
        num_classes = int(getattr(self, "num_classes", student_prob_l.shape[1]))
        class_pixels = {
            f"calibration_pixels_class{cls}": int((y_c.detach() == cls).sum().item())
            for cls in range(num_classes)
        }
        self.calibrator.update_from_batch(
            teacher_prob=student_prob_l,
            sam_prob=sam_c["sam_prob"].detach(),
            sam_iou=sam_c.get("sam_iou"),
            prompt_quality=sam_c.get("prompt_quality"),
            gt=y_c.detach(),
        )
        if not hasattr(self, "calibration_update_count"):
            self.calibration_update_count = 0
        self.calibration_update_count += 1
        flat_thresholds = {}
        for cls in range(num_classes):
            flat_thresholds[f"teacher_q_class{cls}"] = float(self.calibrator.teacher_q[cls])
            flat_thresholds[f"sam_q_class{cls}"] = float(self.calibrator.sam_q[cls])
            flat_thresholds[f"sam_iou_q_class{cls}"] = float(self.calibrator.sam_iou_q[cls])
            flat_thresholds[f"prompt_stability_q_class{cls}"] = float(self.calibrator.prompt_stability_q[cls])
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {
                "event": event_name,
                "iteration": iteration,
                "calibration_batch_size": int(y_c.shape[0]),
                "calibration_update_count": int(self.calibration_update_count),
                **class_pixels,
                **flat_thresholds,
                "teacher_q": self.calibrator.teacher_q.tolist(),
                "sam_q": self.calibrator.sam_q.tolist(),
                "sam_iou_q": self.calibrator.sam_iou_q.tolist(),
                "prompt_stability_q": self.calibrator.prompt_stability_q.tolist(),
            },
        )

    def validate(self, iteration: int):
        metrics = evaluate(
            self.student,
            self.val_loader,
            self.num_classes,
            self.device,
            compute_hd95=self.config.get("eval", {}).get("compute_hd95", True),
            save_dir=None,
            ignore_index=self.ignore_index,
        )
        metrics = self._add_baseline_gaps(metrics)
        ckpt_dir = self.output_dir / "checkpoints"
        latest = save_checkpoint(
            ckpt_dir / "latest.pth",
            iteration=iteration,
            student=self.student,
            fast_teacher=self.dual_teacher.fast,
            slow_teacher=self.dual_teacher.slow,
            optimizer=self.optimizer,
            scaler=self.scaler,
            calibrator=self.calibrator,
            sam_utility=self.sam_utility,
            mentor=self.mentor,
            config=self.config,
            best_metrics=self.best_metrics,
            calibration_update_count=self.calibration_update_count,
        )
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "checkpoint_saved", "iteration": iteration, "path": str(latest)})
        previous_best = float(self.best_metrics.get("avg_dice", -1.0))
        is_best_dice = False
        if metrics["avg_dice"] >= previous_best:
            self.best_metrics["avg_dice"] = metrics["avg_dice"]
            is_best_dice = True
            self.val_collapse_count = 0
            save_checkpoint(
                ckpt_dir / "best_val_dice.pth",
                iteration=iteration,
                student=self.student,
                fast_teacher=self.dual_teacher.fast,
                slow_teacher=self.dual_teacher.slow,
                optimizer=self.optimizer,
                scaler=self.scaler,
                calibrator=self.calibrator,
                sam_utility=self.sam_utility,
                mentor=self.mentor,
                config=self.config,
                best_metrics=self.best_metrics,
                calibration_update_count=self.calibration_update_count,
            )
            append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "best_updated", "metric": "avg_dice", "iteration": iteration})
        else:
            train_cfg = self.config.get("train", {})
            collapse_enabled = bool(train_cfg.get("stop_on_val_collapse", False))
            collapse_delta = float(train_cfg.get("val_collapse_delta", 0.15))
            collapse_min_iter = int(train_cfg.get("val_collapse_min_iter", 0))
            if collapse_enabled and iteration >= collapse_min_iter and previous_best > 0.0 and previous_best - float(metrics["avg_dice"]) >= collapse_delta:
                self.val_collapse_count += 1
                append_jsonl(
                    self.output_dir / "diagnostics.jsonl",
                    {
                        "event": "val_collapse_guard",
                        "iteration": iteration,
                        "avg_dice": float(metrics["avg_dice"]),
                        "best_avg_dice": previous_best,
                        "count": int(self.val_collapse_count),
                    },
                )
                if self.val_collapse_count >= int(train_cfg.get("val_collapse_patience", 2)):
                    self.stop_requested = True
            else:
                self.val_collapse_count = 0
        hd = metrics.get("avg_hd95", float("inf"))
        hd_key = hd if not math.isnan(hd) else float("inf")
        if hd_key <= self.best_metrics.get("avg_hd95", float("inf")):
            self.best_metrics["avg_hd95"] = hd_key
            save_checkpoint(
                ckpt_dir / "best_val_hd95.pth",
                iteration=iteration,
                student=self.student,
                fast_teacher=self.dual_teacher.fast,
                slow_teacher=self.dual_teacher.slow,
                optimizer=self.optimizer,
                scaler=self.scaler,
                calibrator=self.calibrator,
                sam_utility=self.sam_utility,
                mentor=self.mentor,
                config=self.config,
                best_metrics=self.best_metrics,
                calibration_update_count=self.calibration_update_count,
            )
        best_dice = float(self.best_metrics.get("avg_dice", metrics["avg_dice"]))
        metrics["best_dice"] = best_dice
        metrics["best_avg_dice"] = best_dice
        metrics["is_best_dice"] = 1.0 if is_best_dice else 0.0
        row = {"iteration": iteration, "phase": "val", **metrics}
        append_jsonl(self.output_dir / "metrics.jsonl", row)
        self.logger.info(
            "val iter=%d avg_dice=%.4f best_dice=%.4f avg_iou=%.4f",
            iteration,
            metrics["avg_dice"],
            best_dice,
            metrics["avg_iou"],
        )
        return metrics
