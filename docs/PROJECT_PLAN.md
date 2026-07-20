# Project plan: GW-YOLO O4 Research Program

## 1. Product/research objective

Create a low-latency, interpretable gravitational-wave scene-understanding system that:

1. separates CBC chirps and detector glitches at instance level;
2. fuses H1/L1/V1 information coherently;
3. provides masks suitable for data-quality triage and deglitching;
4. improves a downstream search ranking statistic or recovered signal quality;
5. hands cleaned strain and fast source summaries to AMPLFI/DINGO-class parameter inference.

The primary scientific success criterion is not screenshot accuracy. It is a statistically significant improvement in injection recovery or sensitive spacetime volume at fixed false-alarm rate.

## 2. Work packages

The July 2026 audit and literature/GWTC-5 review change the priority order: a statistically useful evaluation corpus, real-noise transfer and physics-coherent detector-set fusion precede capacity scaling. The project no longer treats a 200k corpus or a giant backbone as a scheduled milestone. See `PHYSICS_COHERENT_STRATEGY.md` for the controlling scientific question and `DATA_SCALING_PLAN.md` for conditional promotion gates.

### WP0 — Reproducibility and governance (weeks 1–2)

Deliverables:

- populated Git repository and tagged baseline;
- locked Python/CUDA environment;
- model/config/data manifest hashes;
- deterministic CLI and CI tests;
- legacy audit and corrected model report.

Exit gate: a clean machine can reproduce the legacy audit and a one-epoch smoke run.

### WP1 — Leakage-safe legacy baseline (weeks 1–3)

Experiments:

- YOLOv8n/m-seg versus YOLO26n/m-seg;
- original random split versus physical group split;
- legacy image augmentation versus physically conservative augmentation;
- five seeds for the final comparison.

Metrics: class-wise box/mask mAP50 and mAP50-95, recall, calibration, inference latency. These are representation metrics, not search metrics.

Exit gate: zero group overlap and an explained reproduction gap relative to arXiv:2508.17399.

Status: initial gate completed for one seed. YOLO26m reached validation mask mAP50 0.747 and locked-test 0.765, but the evaluation contains only 25/24 independent validation/test groups and is not a paper endpoint.

### WP1.5 — Data scaling decision gate (weeks 1–5)

Deliverables:

- frozen 5k–10k validation and 20k–50k injection test corpora;
- manifest fields for waveform, injection, glitch, GPS block, IFO, run, SNR, source family, duration, Q, and overlap severity;
- group-safe learning curve at 2k/5k/10k, followed by 25k/50k only when frozen transfer endpoints justify it;
- at least three seeds per scaling point;
- in-domain and O4-transfer curves reported separately.

Exit gate: evidence establishes whether performance is data-limited, domain-limited, or representation-limited. Major architecture selection is not frozen before this gate.

### WP2 — Numeric multi-Q data factory (weeks 2–8)

Build strain-to-tensor generation with:

- whitened strain and PSD metadata;
- multiple Q planes instead of only the maximum-energy plane;
- 1/4/16/64 s windows;
- detector-specific validity masks;
- physical time-domain signal and glitch composition;
- deterministic provenance IDs.

Conditional scale targets:

- 10k independent scenes for the first credible representation baseline;
- 25k scenes only after a controlled 10k scale gain;
- 50k scenes only after a controlled 25k gain;
- 200k or online generation only if the compact model remains demonstrably data-limited after GPS/run/glitch diversity is expanded.

Use O1–O3 for training, O4a for development/calibration, and keep O4b locked.

Exit gate: numeric input materially improves O3→O4 transfer relative to rendered plots, and scaling gains are measured using physical groups rather than augmented image counts.

### WP3 — Multi-detector scene model (weeks 5–9)

Architecture tracks, in promotion order:

- fixed-channel early fusion as a baseline only;
- shared per-IFO encoder plus variable-detector set fusion as the primary track;
- coherence features/head with predeclared time-delay limits and timing uncertainty;
- instance segmentation heads for chirp and glitch;
- OOD/abstention head evaluated by held-out glitch family and observing run;
- optional bounded auxiliary-evidence branch, separately ablated from strain-only.

Exit gate: at fixed validation FAR, multi-IFO fusion beats the best single-IFO model with paired confidence intervals excluding zero.

### WP4 — Search and background benchmark (weeks 6–11)

Required data:

- software injections spanning BBH/NSBH/BNS populations;
- clean and glitch-overlap strata;
- continuous analysis-ready background with known events removed;
- multi-detector time slides for long equivalent background;
- public event-validation/retraction hard negatives where licensing permits.

Required metrics:

- efficiency versus network and per-IFO SNR;
- FAR/IFAR versus ranking statistic;
- sensitive distance and `<VT>`;
- latency and throughput;
- subgroup results over mass, mass ratio, spin, IFO availability, glitch class, and overlap severity.

Exit gate: background exposure is sufficient for the claimed FAR. With zero false positives, the 90% upper limit is approximately `2.3/T`.

### WP5 — Mask-informed downstream system (weeks 9–14)

Track A: add GW-YOLO mask/coherence/OOD outputs to a search reranker.
Track B: mask-informed gating, inpainting, or BayesWave guidance, followed by matched filtering.
Track C: paired raw/contaminated/mask-conditioned data passed into AMPLFI/DINGO/Bilby; GW-YOLO is evaluated as a robust front end, not a replacement posterior estimator.

Metrics:

- `<VT>` at common FAR;
- recovery SNR before/after deglitching;
- parameter bias and posterior coverage;
- false-veto rate on clean injections;
- end-to-end alert latency;
- searched sky area/volume when a compatible PE backend is available.

Exit gate: ≥10 percentage-point gain on overlap recovery or ≥5%–10% `<VT>` gain at fixed FAR, with <1 percentage-point loss on clean injections.

## 3. Automated experiment ladder

The existing legacy pipeline stops at the first candidate passing the configured validation quality gate, then evaluates the selected checkpoint exactly once on test:

1. audit source data;
2. build physical groups;
3. optimize a balanced 70/15/15 group split;
4. materialize a new dataset without modifying legacy files;
5. train YOLO26n;
6. if the gate fails, train YOLO26m;
7. select on validation only;
8. evaluate frozen test;
9. run mask-preserving GWTC-4 catalog inference;
10. join GWOSC metadata and produce SNR diagnostics.

The revised research ladder adds a prerequisite data program:

1. freeze a large, group-disjoint validation/test corpus;
2. audit physical provenance and two-axis mixture leakage;
3. run a compact-model learning curve over independent group counts;
4. diagnose data, domain, label, and representation bottlenecks;
5. build the 10k baseline and run fixed-epoch/fixed-update scale controls;
6. shortlist compact numeric YOLO, detector-set fusion, coherence, small masked-Q pretraining and OOD arms;
7. expand to 25k/50k only after a frozen O4a promotion gate;
8. freeze the primary multi-IFO architecture;
9. replace mAP gating with fixed-FAR efficiency and `<VT>`.

The initial gate `mask mAP50 ≥ 0.72` is deliberately attainable on a harder, leakage-safe split. It is a baseline gate, not a paper success criterion. Later configs must replace it with fixed-FAR efficiency and `<VT>` gates.

Each epoch also writes `best_target.pt` when the configured primary metric improves. This avoids silently selecting a checkpoint with a framework-default fitness function that differs from the paper endpoint.

## 4. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| 414-image dataset is too small | high variance, shortcut learning | generate numeric injection corpus; report multiple seeds |
| source ID semantics unavailable | residual leakage | reconstruct generator metadata and group conservatively |
| Q-transform favors loud glitches | low-SNR chirp loss | multi-Q and learned/adaptive plane selection |
| single-IFO Q-scan misses network events | non-monotonic SNR response | explicit H1/L1/V1 fusion and validity masks |
| catalog positives have no background denominator | inflated claims | continuous background and time slides |
| AMPLFI/DINGO task mismatch | invalid comparison | compare end-to-end PE latency/coverage; position masks as complementary |
| closed-set Gravity Spy labels miss new O4 glitches | confident false classifications | OOD abstention, held-out-family and O3→O4 audits |
| fixed H1/L1/V1 channels learn missing-IFO shortcuts | brittle O4b transfer | shared encoder, detector availability mask, set fusion and detector dropout |
| scale consumes schedule without transfer gain | long experiments with no paper evidence | fixed-update control and stop at the first frozen endpoint plateau |
| O4b becomes contaminated during tuning | invalid locked test | access log, config freeze, one-time evaluation |
| long GPU runs become irreproducible | lost results | atomic state, checkpoint hashes, resumable run directories |

## 5. Staffing-style backlog

Revised priority order:

1. data/provenance lead: large evaluation set, scaling curve, strain generation, grouping, manifests;
2. data-quality lead: O3/O4 glitch anchors, quiet/hard negatives, label audits;
3. search-statistics lead: injections, time slides, FAR/`<VT>`;
4. model lead: multi-Q/multi-IFO architecture after the scaling gate;
5. inference lead: calibration and AMPLFI/DINGO interface;
6. reproducibility lead: CI, containers, artifact registry, paper tables.
