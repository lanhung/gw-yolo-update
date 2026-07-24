# Blinded human mask annotation protocol

> **Superseded for primary experiments (2026-07-23).** This protocol is retained only for
> historical reproduction and optional disagreement analysis. Human masks are no longer training
> targets, publication gates or O4b/GWTC-5 unlock prerequisites. See
> `docs/AUTOMATIC_MASK_POLICY.md`.

## Purpose and boundary

This protocol measures whether the validation-selected GW-YOLO glitch masks agree with independent
human pixel segmentations. It is validation-only evidence. It does not authorize training on human
masks, threshold fitting on test data, a search claim, or access to O4b/GWTC-5 strain.

Each of the 93 frozen tasks contains only numeric time-frequency features and non-target IFO/Q
metadata. The annotator-facing manifest and web service reject internal source identifiers,
Gravity Spy labels, weak-mask paths and `mask`, `glitch_mask`, `chirp_mask` or `weak_mask` arrays.

## Independence rules

1. Assign three different people stable, non-identifying reviewer slugs.
2. Each reviewer uses only their assigned localhost service and must not see another reviewer's
   masks or discuss task boundaries before all three manifests are finalized.
3. Reviewers may inspect all nine H1/L1/V1 by Q=4/8/16 planes, but may not request the metadata weak
   mask or model prediction.
4. A reviewer may revise saved tasks until pressing **Finalize all**. Final manifests and masks are
   immutable.
5. Empty masks are allowed when no localized artifact is visible; they must not be replaced with a
   forced class-shaped mask.

## Annotator workflow

The three services bind to remote localhost only. Use separate SSH tunnels, for example remote
ports 18761, 18762 and 18763 mapped to a reviewer's local port. Open the local URL in a browser.

For every task:

1. cycle through all nine IFO/Q planes;
2. paint the visible glitch support with **Draw** and correct it with **Erase**;
3. use **Clear plane** only for the current plane;
4. press **Save task** before moving on;
5. after all 93 tasks are saved, press **Finalize all** once.

The display uses a per-plane frozen 1st--99th percentile stretch for visualization only. Saved masks
remain exact binary arrays with the original `(3, 3, 96, 96)` shape.

## Commands and outputs

The stable CLI is:

```bash
python -m gwyolo.cli gravityspy-mask-annotation-serve \
  --tasks gravityspy_mask_annotation_tasks.jsonl \
  --annotator-id reviewer-a \
  --output-dir reviewer-a \
  --host 127.0.0.1 \
  --port 18761
```

Repeat with distinct reviewer IDs, output directories and ports. Each finalized session writes
`annotations.<reviewer>.jsonl` plus a provenance report. The merge command requires exact 93-task
coverage from all three distinct IDs and rehashes all 279 mask artifacts:

```bash
python -m gwyolo.cli gravityspy-mask-annotation-merge \
  --tasks gravityspy_mask_annotation_tasks.jsonl \
  --annotation-manifest reviewer-a/annotations.reviewer-a.jsonl \
  --annotation-manifest reviewer-b/annotations.reviewer-b.jsonl \
  --annotation-manifest reviewer-c/annotations.reviewer-c.jsonl \
  --minimum-annotators 3 \
  --output completed_human_annotations.jsonl
```

The deployed merge waiter performs this command automatically. The existing publication waiter
then evaluates inter-annotator agreement, materializes majority consensus, runs the frozen model,
computes the predeclared 10,000-bootstrap endpoint and binds it to continuous-background evidence.

## Claim policy

All negative and null results are retained. A family with fewer than three audited tasks remains in
the overall table but cannot support a family-specific claim. Passing this audit does not by itself
show improved FAR or `<VT>`; failing it blocks an explicit mask-quality claim even if a functional
deglitch result is positive.
