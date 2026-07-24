# GW-YOLO related-work and collision audit — 2026-07-22

## Decision

The literature now rules out three broad novelty claims for this project:

1. GW-YOLO cannot be presented as the first YOLO-style multi-transient segmentation system for
   LIGO data. That contribution belongs to the published GW-YOLO work.
2. It should not compete for the claim of a general low-latency CBC search that matches established
   search sensitivity. Aframe has searched O3, was deployed during O4, and now reports BNS
   sensitivity comparable to matched-filter pipelines.
3. It should not claim that generic glitch subtraction or generic signal--glitch joint inference is
   new. BayesWave, production O3 subtraction, adaptive methods, neural denoisers and a
   Gravity-Spy-informed joint Bayesian glitch model already occupy that space.

The defensible paper question is narrower and stronger:

> Does an explicit, human-audited instance-mask front end, with variable-detector support and
> predeclared network coherence, improve fixed-FAR sensitive spacetime volume or downstream
> posterior reliability specifically for signal--glitch overlap and nonstationary detector data,
> without degrading clean signals?

This is a collision audit, not proof of novelty. Searches were performed on 2026-07-22 across arXiv,
GWOSC/LIGO material and primary project papers. A final manuscript still needs a conventional
reference-chain review and citation search immediately before submission.

## Highest-risk collisions

| Existing work | What it already establishes | Collision risk | Required project response |
|---|---|---:|---|
| [GW-YOLO](https://arxiv.org/abs/2508.17399) | Simultaneous signal/noise scene identification with pixel masks and quantitative overlap efficiency | **Critical** | Treat its image protocol as the legacy baseline. Never claim first segmentation, first masks, or first realistic-overlap study. Novelty must come from physical splits, numeric tensors, network coherence, continuous background and measured downstream mask utility. |
| [Aframe O3 search](https://arxiv.org/abs/2505.21261) and [Aframe BNS search](https://arxiv.org/abs/2607.01372) | An ML search recovered a catalog-scale O3 candidate set; Aframe was deployed in O4 and the BNS extension reports sensitivity comparable to matched filtering | **Critical** | Retire “general fast CBC detector” as a primary goal. Use search only to test the incremental value of masks/coherence on contaminated or missing-IFO strata at a common FAR. |
| [AMPLFI](https://arxiv.org/abs/2407.19048) | Aframe plus accelerated likelihood-free PE is already an integrated low-latency detection-to-inference workflow | **High** | Do not sell GW-YOLO + AMPLFI as a new end-to-end concept. Measure whether mask-conditioned input improves AMPLFI within the same backend, prior, events and noise realization. |
| [Joint data-informed glitch inference](https://arxiv.org/abs/2505.00657) | A normalizing-flow prior trained on Gravity Spy glitches is incorporated into Bilby for simultaneous signal/glitch inference and bias mitigation | **Critical** | Make this, not a no-cleaning run, the strongest learned-glitch PE comparator. A paper-level PE claim needs a matched subset against this approach or a clearly justified reproducible approximation. |
| [BayesWave overlap modeling](https://arxiv.org/abs/2205.13580), [O3 glitch subtraction](https://arxiv.org/abs/2207.03429) and [PE mitigation study](https://arxiv.org/abs/2311.09159) | Joint wavelet/template modeling and operational glitch subtraction already separate overlaps and mitigate posterior bias | **Critical** | Promote the planned BayesWave/Bilby subset from optional illustration to mandatory gold-standard comparison. Report accuracy and latency trade-offs, not generic superiority. |
| [WaveFormer](https://arxiv.org/abs/2212.14283) and [adaptive spline subtraction](https://arxiv.org/abs/2301.02398) | Learned and nonparametric strain denoising already target glitch/noise removal while preserving GW signals | **High** | Add at least one fast strain-domain deglitch baseline. The mask method must win on a declared combination of clean non-inferiority, overlap recovery, interpretability and cost. |
| [GSpyNetTree](https://arxiv.org/abs/2304.09977) | Signal-versus-glitch classification under O4-era, overlapping and new-noise scenarios | **High** | Do not use image accuracy against Gravity Spy as the main result. Compare selective error/abstention and overlap behavior, while emphasizing localization and all-instance masks. |
| [Unphysical-template glitch veto](https://arxiv.org/abs/2401.15237) | A strong search-specific glitch veto using unphysical inspiral-template sectors | **High** | Include it, or an exact reproducible implementation, in the search-baseline table. Mask reranking must demonstrate incremental fixed-FAR value rather than compare only with an unvetoed score. |
| [GWAK](https://arxiv.org/abs/2309.11537) | Semi-supervised anomaly embeddings cover CBCs, glitches and unmodeled transient classes | **Medium/high** | Frame OOD as calibrated abstention for safety, not anomaly detection novelty. Compare risk--coverage and unknown false-acceptance rates. |
| [Frozen-DINOv2 O4a morphology study](https://arxiv.org/abs/2605.28572) | A large O4a zero-shot morphology analysis reports no statistically supported new glitch family in its corpus | **High for the current OOD wording** | Remove any assumption that O4 necessarily contains new families. Use run transfer, rare tails and leave-one-family-out as controlled distribution shifts; retain a null O4 novelty result if reproduced. Add frozen DINOv2 features as a serious non-generative baseline. |
| [GWSkyNet II](https://arxiv.org/abs/2408.06491) | O4-ready alert validation from skymaps and candidate metadata, including documented overlap failure modes | **Medium** | Keep alert classification out of the primary novelty claim. Use sky-area improvement after mask conditioning as a downstream endpoint and discuss complementary roles. |
| [DeepClean](https://arxiv.org/abs/2005.06534) and [Coherence DeepClean](https://arxiv.org/abs/2501.04883) | Auxiliary-channel regression and increasingly autonomous online noise subtraction | **Medium/high for auxiliary-assisted claims** | Auxiliary channels may provide attribution/evidence only. Report strain-only and auxiliary-assisted arms separately, and never let auxiliary evidence silently veto a strain-coherent candidate. |
| [DINGO-IS](https://arxiv.org/abs/2210.05686) | Fast neural PE with likelihood correction, evidence and sample-efficiency diagnostics that flag some glitch/OOD failures | **High** | Do not claim a replacement PE system. Use paired DINGO raw/contaminated/mask-conditioned changes in coverage, bias, width and importance-sampling efficiency. |
| [ML-enhanced cWB](https://arxiv.org/abs/2105.04739) and [robust real-noise DL search](https://arxiv.org/abs/2306.11797) | ML reranking and real-data searches already report sensitivity gains and expose robustness failures | **Medium/high** | Continuous-background FAR/IFAR and injection sensitivity are entry requirements, not novelty by themselves. Add adversarial/nonstationary stress tests and retain negative results. |

## What still appears differentiable

The search did not identify one prior system that combines all of the following in one locked,
reproducible evaluation:

- all-instance chirp and glitch masks rather than a window label or one winning object;
- numeric multi-Q inputs, not rendered plots as the primary scientific representation;
- one shared model for H1/L1, H1/V1, L1/V1 and H1/L1/V1 with an explicit detector-validity mask;
- network coherence constrained by light-travel time plus empirically calibrated timing uncertainty;
- conservative OOD abstention that cannot veto a strain-coherent candidate;
- common-FAR raw-versus-mask continuous-background evaluation and paired `<VT>`;
- clean-signal non-inferiority before any deglitch gain is promoted;
- three-person blinded mask consensus;
- paired raw/contaminated/mask-conditioned DINGO and AMPLFI reliability evaluation;
- a one-time locked O4b/GWTC-5 transfer test with all selection choices frozen on earlier data.

The combination, not YOLO architecture alone, is therefore the candidate contribution. This claim
must remain conditional until the baseline implementations and locked results exist.

## Immediate strategy changes

### 1. Change the primary claim now

Use “mask- and coherence-conditioned robustness front end” throughout the paper plan. Avoid these
phrases unless explicitly attributed to prior work:

- first multi-transient segmentation;
- first realistic signal--glitch overlap assessment;
- first AI CBC search;
- matched-filter replacement;
- first neural deglitcher;
- first joint signal--glitch inference;
- discovery of new O4 glitch families.

### 2. Strengthen the baseline ladder

The minimum publication ladder should now be:

1. original rendered-image GW-YOLO reproduction;
2. numeric single-IFO and variable-detector GW-YOLO arms;
3. morphology-only versus predeclared physical-coherence reranking;
4. GSpyNetTree-style signal/glitch classification or a faithful candidate-level comparator;
5. unphysical-template veto for the supported CBC domain;
6. fast strain-domain subtraction: adaptive spline or WaveFormer-class comparator;
7. BayesWave joint signal/glitch inference on a stratified expensive subset;
8. Gravity-Spy-informed joint glitch inference on the same subset if code/runtime can be reproduced;
9. frozen DINOv2 and GWAK-style embeddings for OOD/abstention;
10. DINGO and AMPLFI, each compared only within backend on identical raw, contaminated and
    mask-conditioned events.

If a comparator cannot be reproduced, the manuscript must state that limitation and must not turn
published numbers from a different population into a head-to-head result.

### 3. Redirect the data-scaling plan

The O4a DINOv2 work analyzes more than 188,000 spectrograms, so a few thousand labeled glitches are
not a competitive representation-learning corpus. The response should not be to fabricate more
rendered augmentations. Instead:

- retain the group-safe labeled aligned strain bank for supervised masks;
- add a much larger unlabeled, GPS-disjoint O1--O4a numeric multi-Q bank for masked/self-supervised
  pretraining;
- compare compact masked-Q pretraining with frozen DINOv2 features under fixed updates;
- promote scale only when the frozen O4a hard subset improves under both fixed-epoch and
  fixed-update controls;
- report physical source files, GPS blocks, glitches, injections, IFO sets and live time separately.

### 4. Make PE comparisons scientifically stronger

The existing DINGO/AMPLFI triplets remain useful, but they do not by themselves establish a
state-of-the-art deglitch result. Add a predeclared 25--50-event expensive subset spanning broadband,
narrowband, signal-like and overlapping artifacts. On identical strain and waveform assumptions,
compare:

- raw likelihood analysis;
- GW-YOLO mask conditioning;
- BayesWave joint reconstruction/subtraction;
- data-informed glitch-prior joint inference where reproducible.

Report posterior coverage, paired absolute bias, interval width, sky area, evidence or
importance-sampling efficiency, residual matched-filter SNR and end-to-end latency. A useful outcome
may be that GW-YOLO is less accurate than joint BayesWave but much faster and supplies interpretable
instance masks; that is a publishable trade-off if quantified honestly.

### 5. Reframe OOD and O4 transfer

The primary OOD endpoint should be selective safety, not novel-class discovery:

- known-family false abstention;
- held-family true abstention and false acceptance with Wilson intervals;
- risk--coverage curves and calibration under run transfer;
- rare-family and low-SNR strata;
- H1/L1/V1 detector-set changes;
- a frozen DINOv2-feature baseline;
- explicit retention of a null “no new O4 morphology” outcome.

### 6. Preserve the locked GWTC boundary

[GWTC-4.0](https://arxiv.org/abs/2508.18082) reports 128 new O4a candidates and
[GWTC-5.0](https://arxiv.org/abs/2605.27225) makes O4b a large, relevant transfer set. Public
availability does not make it valid model-selection data. Continue to freeze architecture,
thresholds, veto/calibration policy, OOD rule and subgroup definitions before the one-time O4b
strain evaluation. Catalog hit rate remains descriptive; the search claim requires continuous
background, injections, FAR/IFAR and `<VT>`.

## Revised paper hierarchy

1. **PRD:** primary target only if mask/coherence conditioning improves paired `<VT>` at common FAR
   or meaningfully improves posterior reliability under overlaps, with clean non-inferiority and
   strong deglitch baselines.
2. **ApJS:** primary fallback if the main durable product is the group-safe numeric multi-IFO
   benchmark, human mask consensus, locked protocols and comprehensive transfer tables.
3. **JCAP:** pursue only if the robust front end materially changes a population, selection
   function, standard-siren or other astrophysical inference. Segmentation or latency alone is not
   sufficient.

## Go/no-go gates created by this audit

| Gate | Go condition | No-go response |
|---|---|---|
| General detection | Incremental mask/coherence gain at common FAR on contaminated strata | Do not continue a general Aframe/matched-filter sensitivity race |
| Deglitch novelty | Better speed--reliability--interpretability trade-off than at least one fast baseline and BayesWave subset | Publish benchmark/null result or pivot to ApJS |
| OOD novelty | Calibrated abstention improves held-family/run risk with controlled known false abstention | Retain DINOv2/null transfer result; remove novelty language |
| PE utility | Paired raw/contaminated/mask change with uncertainty and clean non-inferiority | Do not infer utility from mask IoU or visual examples |
| Data scale | O4a hard endpoint improves under fixed-update and fixed-epoch controls | Increase domain/GPS diversity or change representation, not schedule alone |
| Locked transfer | All validation gates pass before one-time O4b access | Keep O4b unopened; never use it to rescue a weak method |

## Monitoring list before submission

Repeat this collision audit monthly and immediately before abstract freeze for:

- new GW-YOLO versions and code releases;
- Aframe/GWAK/AMPLFI O4 production papers;
- DINGO-T1 and likelihood-corrected PE updates;
- O4 Gravity Spy morphology releases;
- signal--glitch joint-inference or neural BayesWave successors;
- GWTC-5 methods/data-quality papers and any GWTC-6 announcement;
- new learned search challenge results reporting common-FAR `<VT>`.

Any new work that simultaneously delivers instance masks, variable-IFO coherence, common-FAR
`<VT>` and paired PE robustness should trigger an immediate claim rewrite before more compute is
spent.
