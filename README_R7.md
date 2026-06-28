# SAGE-SAM R7

SAGE-SAM R7 is an experimental successor to R6 for 3-class semi-supervised
medical image segmentation. It keeps the R6 dual-fusion deployment path but
changes the semi-supervised loop:

1. Class-balanced foreground-safe set supervision caps safe negatives by the
   available foreground budget instead of allowing dense all-image negative
   supervision for rare classes.
2. SAM is used as a structural verifier. Its prompt quality, SAM IoU and
   teacher/SAM agreement produce a verifier score; SAM no longer acts as a
   dense pixel teacher by default.
3. A dynamic trust curriculum disables correlation/locality propagation and
   suppresses negative SSL when the current pseudo-targets show foreground
   starvation, background takeover, or class-specific safe-negative saturation.
4. R7.1 adds dual-bound foreground safety: foreground candidates have both a
   lower participation floor and an upper flooding ceiling. Pixels removed by
   the ceiling may return as bounded high-confidence background candidates, so
   the unlabeled objective cannot collapse into all-background or all-foreground
   supervision.

V100 training:

```bash
bash scripts/train_r7_v100_tuned.sh
```

Resume or shorten:

```bash
MAX_ITERATIONS=1500 RESUME=outputs/SAGE_SAM_R7_3Class_V100_Tuned/checkpoints/latest.pth \
  bash scripts/train_r7_v100_tuned.sh
```

Validation and test:

```bash
bash scripts/test_r7_v100_tuned.sh
```

Key diagnostics to watch in `metrics.jsonl`:

- `per_class_foreground_participation_ratio`
- `per_class_safe_negative_ratio`
- `safe_negative_pixel_ratio`
- `foreground_ceiling_flood_class_count`
- `background_from_ceiling_ratio`
- `background_hard_ratio`
- `trust_unsafe`
- `trust_high_candidate`
- `trust_high_class`
- `sam_verifier_score_mean`
- `loss_sam_kd`

The default R7 config uses adapter-only SAM PEFT, freezes the prompt encoder,
freezes the SAM mask decoder, and keeps SAM primarily as a verifier for the
student/EMA pseudo-target loop.
