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
5. R7.2 adds prompt-audited AgreementSAM: invalid or over-large SAM prompts are
   marked unreliable instead of falling back to full-image boxes. Training
   visualizations now include SAM prompt overlays and prompt heatmaps, while
   metrics record prompt validity and prompt/box area ratios.
6. R7.3 adds component-aware multi-prompt SAM verification. A foreground class
   can produce multiple connected-component prompts, so bilateral class-1
   structures are decoded with separate SAM boxes/masks and merged back into
   the original class before supervision.
7. R7.4 adds prior-calibrated trust and late-stage LR stabilization. The
   dynamic trust gate now calibrates its minimum foreground thresholds from the
   labeled foreground prior, and SAM KD receives a small gated effective-weight
   floor so valid SAM prompts can actually influence the student without
   becoming hard pseudo-labels.
8. R7.5 adds class-specific prompt cardinality. Red class-1 structures may use
   one or two connected-component prompts, while blue class-2 structures use one
   fixed connected-component prompt.
9. R7.6 adds prior-calibrated pseudo-target budgeting. Foreground candidate
   caps, minimum participation, and collapse recovery ratios are calibrated from
   the labeled class prior so rare foreground classes are not systematically
   over-expanded by a shared fixed floor.

V100 training:

```bash
bash scripts/train_r7_v100_tuned.sh
```

Resume or shorten:

```bash
MAX_ITERATIONS=1500 RESUME=outputs/SAGE_SAM_R7_3Class_V100_Tuned_PriorBudget_ClassPrompt/checkpoints/latest.pth \
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
- `trust_min_candidate_foreground_ratio`
- `sam_verifier_score_mean`
- `sam_prompt_valid_mean`
- `sam_prompt_box_area_ratio_mean`
- `sam_prompt_component_count_class1`
- `sam_kd_raw_effective_weight`
- `sam_kd_effective_weight`
- `sam_kd_floor_active`
- `lr_scale`
- `loss_sam_kd`

The default R7 config uses adapter-only SAM PEFT, freezes the prompt encoder,
freezes the SAM mask decoder, and keeps SAM primarily as a verifier for the
student/EMA pseudo-target loop.
