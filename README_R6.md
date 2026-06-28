# SAGE-SAM R6

SAM-Calibrated Set-valued Self-training with Structure Propagation.

R6 keeps the single-loop dual-fusion deploy student, but changes SAM's role. SAM is not a hard foreground pseudo-label judge. It provides structure, boundary, region support, and calibration signals while class semantics remain controlled by the student/EMA teachers and the labeled medical data.

## Core Changes From R5

- `r6/ssl/foreground_safe_target_builder.py`: builds calibrated candidate label sets. Empty candidates are not silently converted to background; weak foreground evidence is recovered as top-k foreground candidates or left ignored.
- `r6/ssl/foreground_participation_controller.py`: enforces foreground participation, caps hard background supervision, and uses a collapse sentinel to force foreground candidates while disabling background hard CE when foreground participation collapses.
- `r6/losses/tri_state_pseudo_loss.py`: trains singleton sets, fuzzy candidate sets, rank separation, and U2PL-style probability-rank negative labels.
- `r6/ssl/foreground_correlation_locality.py`: uses a broad foreground structure mask (`foreground_seed | candidate_foreground | fuzzy_region | structure_gate`) as propagation seeds and writes propagated foreground labels back into the candidate set before SSL losses.
- `r6/engine/trainer.py`: SAM KD, SAM unsupervised consistency, relation, and locality use the same broad foreground structure mask and `structure_weight`, so SAM training no longer depends on hard foreground seeds being present.

## Training Schedule

```text
0 - 800 iter:
  supervised student + supervised SAM adapter/prompt only
  no background unsupervised hard CE
  no foreground conformal threshold updates

800 - 2000 iter:
  class-conditional conformal candidate sets
  fuzzy foreground supervision
  SAM shape/boundary support without hard veto
  background cap active

2000 - 5000 iter:
  SAM-anchored correlation propagation
  U2PL-style rank negative learning
  conflict and bias review

5000+ iter:
  self-reliance decay for SAM SSL/KD
  SAM boundary/shape regularization remains active
```

## Data Format

```text
<root>/<dataset_name>/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

Masks must contain integer ids in `0..num_classes-1` or `ignore_index`.

## Commands

CPU smoke:

```bash
python train_r6.py --config configs/r6_smoke_cpu.yaml --dry-run
python train_r6.py --config configs/r6_smoke_cpu.yaml
```

Server data/SAM checks:

```bash
python tools/validate_dataset.py --config configs/r6_3class_v100_tuned.yaml
python tools/verify_real_sam.py --config configs/r6_3class_v100_tuned.yaml
```

## V100 Server Settings

The server-tuned config is `configs/r6_3class_v100_tuned.yaml`.

```text
data.root: /root/autodl-tmp/echoData
data.dataset_name: 260513_data_labeled30pct
sam.checkpoint: /root/autodl-tmp/sam_vit_b_01ec64.pth
train.device / sam.device: cuda
data.image_size: 256
sam.image_size: 1024
batch_size_labeled / batch_size_unlabeled: 4 / 4
gradient_accumulation: 2
effective labeled / unlabeled batch: 8 / 8
num_workers: 8
amp: true
lr / weight_decay: 3e-4 / 1e-4
max_iterations: 8000
warmup / grounding / correlation / self-reliance: 1200 / 800 / 2000 / 5000
```

The first fallback for V100 memory pressure is to reduce
`train.batch_size_labeled` and `train.batch_size_unlabeled` from `4/4` to
`2/2` while keeping `gradient_accumulation: 2`. Keep `sam.image_size=1024`
unless real SAM verification or memory pressure proves it is necessary to
lower it.

V100 tuned training:

```bash
bash scripts/train_r6_v100_tuned.sh
```

Short diagnostic training before a full run:

```bash
bash scripts/diagnose_r6_short.sh
```

This runs dataset validation, real-SAM verification, a 1500-iteration diagnostic
run by default, and then checks `metrics.jsonl` with:

```bash
python tools/check_r6_diagnostics.py \
  --output-dir outputs/SAGE_SAM_R6_Diagnostic_1500 \
  --config outputs/SAGE_SAM_R6_Diagnostic_1500/resolved_config.yaml
```

Override the diagnostic length or output folder with environment variables:

```bash
MAX_ITERATIONS=1500 OUTPUT_DIR=outputs/SAGE_SAM_R6_Diagnostic_1500 bash scripts/diagnose_r6_short.sh
```

Full V100 training with explicit output:

```bash
CONFIG=configs/r6_3class_v100_tuned.yaml \
OUTPUT_DIR=outputs/SAGE_SAM_R6_3Class_V100_Tuned \
bash scripts/train_r6_v100_tuned.sh
```

Resume training:

```bash
RESUME=outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/latest.pth \
OUTPUT_DIR=outputs/SAGE_SAM_R6_3Class_V100_Tuned \
bash scripts/train_r6_v100_tuned.sh
```

Validation/test/export after training:

```bash
bash scripts/test_r6_v100_tuned.sh
```

Or with an explicit checkpoint:

```bash
OUTPUT_DIR=outputs/SAGE_SAM_R6_3Class_V100_Tuned \
CHECKPOINT=outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/best_val_dice.pth \
bash scripts/test_r6_v100_tuned.sh
```

Key diagnostics to watch in `metrics.jsonl`:

```text
per_class_sam_participation_ratio
hard_fg_ratio_class1 / hard_fg_ratio_class2
soft_fg_ratio_class1 / soft_fg_ratio_class2
background_hard_ratio
ambiguous_ratio
empty_candidate_ratio
candidate_foreground_ratio
foreground_propagated_ratio
empty_candidate_recovered_ratio
safe_negative_pixel_ratio
fast_slow_agreement
sam_train_gate_ratio
sam_kd_gate_ratio
sam_kd_gate_weight_mean
sam_foreground_support_ratio
masked_locality_ratio
foreground_masked_ratio
emergency_mode
collapse_sentinel_active
collapse_disabled_background
collapse_forced_fg_ratio
```

R6 is healthy only if foreground participation remains nonzero after the grounding stage, `sam_train_gate_ratio` does not collapse to zero, and `background_hard_ratio` stays capped instead of saturating the unsupervised loss.

For a 500-1500 iteration diagnostic run, `tools/check_r6_diagnostics.py` fails
the run if candidate foreground, safe negative supervision, SAM KD gate/weight,
or SAM KD loss remain zero when SAM is enabled. Correlation and locality checks
are automatically skipped before the correlation stage and enabled after the
configured iteration threshold.
