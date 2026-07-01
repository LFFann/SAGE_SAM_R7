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
10. R7.7 adds SAM-guided pseudo refinement. SAM now promotes teacher-agreed,
    verifier-approved foreground pixels into the set-valued target under
    class-prior area budgets, and a gated full-channel extent KD loss lets SAM
    supervise foreground/background shape where its prompt is trusted.
11. R7.8 changes the default path from SAM-dominant foreground expansion to
    SAM-disagreement suppression. Low-teacher-confidence foreground candidates
    with weak SAM support and low verifier evidence are removed before the
    foreground budget stage, while high-confidence teacher regions and the
    class-prior foreground floor are preserved.
12. R7.9 adds student-anchored SAM agreement distillation. Reliable SAM regions
    can supervise the student even when the EMA teacher is empty or lagging, but
    only if SAM support is high and the student either agrees with SAM's
    foreground class or is still uncertain.
13. R7.10 adds student prior feedback. The unlabeled weak-view student
    foreground distribution is tracked with EMA and compared with the labeled
    anatomical class prior. When student foreground mass over-expands, the
    non-SAM unsupervised branch is automatically down-scaled and a light
    hinge-style prior feedback loss penalizes anatomically implausible area
    drift without using validation labels.
14. R7.11 restores earlier weak-to-strong learning under the prior-feedback
    guard. Foreground SSL starts at 1200 iterations, stage-1 unsupervised
    scaling is less conservative, and SAM agreement KD can become active in
    the same window where earlier R7 runs reached their best validation Dice.
15. R7.12 adds labeled-to-unlabeled anatomical anchoring inspired by BCP.
    Labeled foreground anatomy is pasted into unlabeled ultrasound context
    after SSL starts, and only the pasted foreground pixels are supervised.
    This keeps rare foreground semantics visible during unlabeled training
    without converting SAM or teacher mistakes into dense hard labels.
16. R7.13 adds labeled-prior class-balanced CE for supervised and anatomical
    copy-paste losses only. Background CE is down-weighted and foreground CE is
    mildly rebalanced from labeled class priors, improving the rarer class-2
    gradient without amplifying noisy pseudo labels.
17. R7.14 anchors the boundary head with labeled GT boundaries before using
    SAM boundary supervision on unlabeled images. This makes SAM contribute
    primarily as a structural/shape prior while labeled masks keep the boundary
    target stable.
18. R7.15 adds gated strong-view perturbation consistency. Two intensity-strong
    unlabeled views are required to agree only inside candidate or structural
    regions, borrowing the weak-to-strong/dual-view idea from UniMatch while
    preserving R7's SAM verifier and prior-feedback safety gates.
19. R7.16 adds anatomy-topology constrained pseudo-labeling and evaluation.
    Unlabeled pseudo-targets can keep at most two class-1 connected components
    and one class-2 connected component before SSL losses are applied, and
    validation/test use the same topology prior as a deploy-time postprocess.
    Both stages record how many extra predicted or pseudo-labeled components
    were removed, diagnosing whether late Dice loss is caused by anatomically
    invalid foreground over-expansion.
20. R7.17 adds low-cost SAM prompt consistency regularization. Inspired by
    CPC-SAM/CPAC-SAM prompt-consistency ideas, the trainable prompt generator
    is regularized against trusted SAM/teacher foreground support from the
    same SAM forward pass, avoiding a second expensive SAM call while reducing
    prompt drift in sparse ultrasound foreground regions.
21. R7.18 adds trust-conditioned SAM loss floors. The forced minimum weights
    for SAM KD and student-anchored SAM agreement now obey the dynamic trust
    gate, so low-support or over-wide SAM gates cannot bypass the safety
    controller and keep pushing the student after pseudo-target drift appears.

V100 training:

```bash
bash scripts/train_r7_v100_tuned.sh
```

Resume or shorten:

```bash
MAX_ITERATIONS=1500 RESUME=outputs/SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP/checkpoints/latest.pth \
  bash scripts/train_r7_v100_tuned.sh
```

Temporary config overrides:

```bash
python train_r7.py --config configs/r7_3class_v100_tuned.yaml \
  --output-dir outputs/pilot_no_copy_paste \
  --max-iterations 2000 \
  --opts copy_paste.enabled false
```

Validation and test:

```bash
bash scripts/test_r7_v100_tuned.sh
```

Core ablation suite:

```bash
# Run all default ablations. Override MAX_ITERATIONS for shorter pilot runs.
MAX_ITERATIONS=8000 bash scripts/ablate_r7_v100.sh

# Example pilot: only full vs no SAM vs no prior feedback.
MAX_ITERATIONS=2000 ABLATIONS="full no_sam no_prior_feedback" \
  bash scripts/ablate_r7_v100.sh
```

Compare the new run with prior R7 outputs:

```bash
python tools/compare_r7_runs.py \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned_SAMAgreeKD \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned
```

The comparison report includes `verdict`, `recommended_next_action`,
class-wise baseline gaps, `stable_within_drop_threshold`, and
`mechanism_coverage` flags so a new V100 run can be judged without manually
re-reading all metric curves.

Rank the training signals that are most associated with validation Dice drops:

```bash
python tools/diagnose_val_dice_drivers.py \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned_PriorFeedback_BCP \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned_SAMAgreeKD \
  outputs/SAGE_SAM_R7_3Class_V100_Tuned \
  --report outputs/r7_val_dice_driver_report.json
```

For mechanism attribution, run the default V100 ablation suite:

```bash
bash scripts/ablate_r7_v100.sh
```

It includes `no_sam`, `no_prior_feedback`, `no_copy_paste`,
`no_strong_consistency`, `no_trust_conditioned_floor`,
`no_topology_filter`, `no_prompt_consistency`, `no_eval_topology`,
`no_boundary`, and `no_class_balance`.

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
- `prior_feedback_drift`
- `prior_feedback_unsup_scale`
- `prior_feedback_sam_scale`
- `loss_prior_feedback`
- `prior_feedback_student_fg_ratio`
- `prior_feedback_fg_over`
- `loss_copy_paste`
- `copy_paste_effective_weight`
- `copy_paste_fg_ratio`
- `class_balanced_ce_weight_class1`
- `class_balanced_ce_weight_class2`
- `loss_sup_boundary`
- `loss_strong_consistency`
- `strong_view_consistency_mask_ratio`
- `topology_candidate_removed_ratio`
- `topology_candidate_removed_ratio_class1`
- `topology_candidate_removed_ratio_class2`
- `topology_candidate_dropped_components_class1`
- `topology_candidate_dropped_components_class2`
- `topology_removed_pixel_ratio`
- `topology_removed_ratio_class1`
- `topology_removed_ratio_class2`
- `topology_dropped_components_class1`
- `topology_dropped_components_class2`
- `sam_verifier_score_mean`
- `sam_prompt_valid_mean`
- `sam_prompt_box_area_ratio_mean`
- `sam_prompt_component_count_class1`
- `loss_prompt_consistency`
- `prompt_consistency_effective_weight`
- `prompt_consistency_mask_ratio`
- `prompt_consistency_abs_gap`
- `sam_kd_raw_effective_weight`
- `sam_kd_effective_weight`
- `sam_kd_floor_active`
- `sam_kd_floor_candidate`
- `sam_agreement_floor_candidate`
- `sam_floor_trust_scale`
- `sam_floor_blocked`
- `sam_floor_blocked_low_support`
- `sam_floor_blocked_overgate`
- `sam_guided_candidate_ratio`
- `sam_guided_weight_mean`
- `sam_disagreement_suppressed_ratio`
- `sam_disagreement_weight_mean`
- `loss_sam_agreement`
- `sam_agreement_gate_ratio`
- `sam_agreement_effective_weight`
- `loss_sam_extent`
- `lr_scale`
- `loss_sam_kd`

The default R7 config uses adapter-only SAM PEFT, freezes the prompt encoder,
freezes the SAM mask decoder, and lets SAM act as a budgeted structural mentor
or disagreement filter only where teacher agreement, prompt validity, SAM
support, and verifier score are jointly informative.
