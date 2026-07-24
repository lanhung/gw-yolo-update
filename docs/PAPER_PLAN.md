# Publication plan: PRD / JCAP / ApJS

## 1. Recommended paper claim

**A variable-detector, physics-coherent, multi-Q instance-segmentation front end provides interpretable chirp/glitch masks that improve fixed-FAR search sensitivity or downstream posterior reliability under nonstationary and overlapping artifacts, while satisfying clean-signal non-inferiority.**

This claim is stronger and more defensible than “YOLO classifies GWTC screenshots accurately.”

## 2. Relationship to AMPLFI and DINGO

AMPLFI is likelihood-free BBH parameter estimation based on normalizing flows. Its accelerated Aframe+AMPLFI workflow reports an end-to-end latency around six seconds in a mock stream. DINGO performs neural posterior estimation, with importance sampling available for likelihood correction and failure diagnosis; Dingo-T1 supports flexible detector/frequency configurations. Both answer “what are the source parameters?” after a candidate exists.

GW-YOLO answers complementary questions:

- is the time-frequency scene chirp, glitch, both, multiple instances, or OOD?
- where is each component?
- can the glitch component be removed or down-ranked without erasing the chirp?
- is a candidate coherent across the detectors that were actually operating?

Fair common comparisons:

| Dimension | AMPLFI/DINGO-compatible metric | GW-YOLO contribution |
|---|---|---|
| Latency | end-to-end seconds/event | segmentation and cleaning overhead |
| Calibration | PP plots, coverage, SBC | confidence/OOD calibration and downstream PE coverage |
| Localization | searched sky area/volume | whether deglitching improves localization |
| Intrinsic parameters | bias, credible interval coverage | whether masks reduce overlap-induced bias |
| Robustness | failure under nonstationary noise | explicit glitch mask and hard-negative behavior |
| Interpretability | posterior samples | time-frequency instance masks |

Do not claim that detection mAP “beats” a posterior estimator.

Primary references:

- AMPLFI: https://arxiv.org/abs/2407.19048
- AMPLFI public-alert validation: https://arxiv.org/abs/2509.22561
- DINGO: https://github.com/dingo-gw/dingo
- DINGO importance sampling: https://arxiv.org/abs/2210.05686
- Dingo-T1: https://arxiv.org/abs/2512.02968
- GW-YOLO: https://arxiv.org/abs/2508.17399
- GWTC-5.0 results: https://arxiv.org/abs/2605.27225

### Frozen joint-comparison contract

Current primary sources sharpen the comparison. AMPLFI's public-alert validation uses BBH injections
in O3 noise and compares searched sky area/volume with BAYESTAR; DINGO-IS exposes posterior coverage,
importance-sampling efficiency and evidence uncertainty as reliability diagnostics. The shared study
will therefore use an identical paired BBH subset under raw and GW-YOLO-cleaned conditions and report:

- AMPLFI: searched sky area/volume, coarse chirp-mass coverage and end-to-end latency;
- DINGO/DINGO-IS: parameter coverage, importance-sampling efficiency, evidence uncertainty and
  proposal/inference latency;
- GW-YOLO overhead: mask/deglitch latency, signal retention, glitch residual and false veto rate;
- paired changes with bootstrap intervals, split by clean/overlap and glitch morphology.

The same waveform prior, detector set, PSD, calibration assumptions and injections are mandatory.
Published numbers from different populations are context only, never a head-to-head result. AMPLFI
and DINGO remain PE systems; no detection-mAP comparison is permitted.

### Distributed paired-PE evidence transport

DINGO and AMPLFI may run on different assigned GPUs or machines, but their outputs may be combined
only through the content-addressed validation bundle:

```bash
python -m gwyolo.cli pe-within-backend-bundle-export \
  --summary /artifacts/backend/within_backend_paired_smoke_summary.json \
  --output-dir /artifacts/backend-portable-bundle

python -m gwyolo.cli pe-within-backend-bundle-import \
  --bundle-receipt \
    /transferred/backend-portable-bundle/within_backend_pe_evidence_bundle.json \
  --output-dir /artifacts/backend-import
```

The export contains the original summary, batch report, robustness report and manifest plus every
unique posterior, analysis input, base manifest, native-conditioning object, contamination
manifest, mask, mask model and mask policy referenced by the validation rows. Objects are
deduplicated by SHA-256. Import replays every byte count and hash, then creates an explicit local
path projection; it never edits the original scientific reports. The projected summary can enter
`run_paired_pe_portfolio_validation.sh`. That evaluator reopens every posterior and provenance
file, requires the same injections and input hashes across backends and still forbids absolute
DINGO-versus-AMPLFI ranking. Test-split rows are rejected by both export and import.

## 3. Venue-specific framing

### Physical Review D

Best fit if the main result is quantitative GW data-analysis performance:

- fixed-FAR injection sensitivity and `<VT>`;
- overlap recovery and parameter-bias reduction;
- rigorous comparison to search/deglitch baselines;
- physics-driven population and waveform coverage.

Minimum evidence target: paired confidence interval demonstrating a robust sensitivity improvement, with complete background methodology.

### JCAP

Best fit if the method enables an astrophysical/cosmological inference that changes materially after robust deglitching:

- recovered low-SNR population near glitches;
- selection-function changes and population inference;
- lensing/overlapping-event science;
- impact on standard sirens or merger-rate inference.

A methods-only image segmentation paper is less naturally aligned unless connected to a clear astrophysics outcome.

### The Astrophysical Journal Supplement Series

Best fit if the principal contribution is a reusable catalog/data/software resource:

- O4a/O4b multi-detector Q-map and mask benchmark;
- documented event/glitch annotations;
- reproducible pipeline and extensive validation tables;
- public trained weights, manifests, and tutorials.

This is the most realistic first target if the team releases a high-quality benchmark but the `<VT>` gain is still modest.

## 4. Core figures and tables

1. End-to-end workflow: strain → multi-Q/multi-IFO model → masks → rerank/deglitch → PE.
2. Leakage audit: random split versus physical group split.
3. Efficiency-versus-SNR curves for clean and overlap BBH/NSBH/BNS.
4. FAR/IFAR versus efficiency and `<VT>` comparison.
5. Performance versus overlap severity and glitch class.
6. Single-Q/single-IFO versus multi-Q/multi-IFO ablation.
7. Fixed-channel versus variable-detector set fusion, with calibration and missing-IFO strata.
8. Before/after deglitch Q-maps and matched-filter recovery.
9. Known versus held-out/O4 glitch OOD and abstention results.
10. Posterior coverage/PP plots before and after mask cleaning, alongside AMPLFI/DINGO-compatible results.
11. Automatic-mask reproducibility and annotator-disagreement boundary: deterministic component
    pseudo-labels are evaluated through functional search/PE outcomes, not presented as human
    pixel ground truth.
12. Latency and compute table.
13. GWTC-4/O4a development and locked GWTC-5/O4b results with explicit selection criteria.

## 5. Required baselines

- published GW-YOLO/YOLOv8 protocol;
- current YOLO26 legacy-image model;
- numeric single-Q model;
- multi-Q single-IFO model;
- multi-Q multi-IFO model;
- fixed-channel versus shared-encoder detector-set model;
- morphology-only versus physical-coherence ranking;
- supervised-only versus small masked-Q pretraining;
- closed-set versus OOD/abstention model;
- model without glitch mask/deglitching;
- conventional gating or a documented deglitch baseline;
- matched-filter/search statistic before and after reranking;
- AMPLFI/DINGO or published reference posteriors for a shared event subset, if environment and waveform priors can be matched.

## 6. Statistical protocol

- predefine populations, splits, thresholds, and primary endpoint;
- preserve O4b as locked test until all choices are frozen;
- use importance weights for astrophysical injection populations;
- paired bootstrap over injections/background blocks;
- report confidence intervals and effective sample size;
- test calibration with SBC/PP plots and expected calibration error;
- correct for multiple comparisons in large ablation grids;
- publish all seeds, including failed or negative runs.

## 7. Minimum publishable package

- public code with tagged release;
- environment/container specification;
- dataset/provenance manifest and group split;
- at least one reusable O4 benchmark product;
- machine-readable tables;
- five-seed model comparison;
- continuous-background analysis;
- fixed-FAR sensitivity result;
- documented limitations and failure gallery;
- model cards and inference latency benchmark.

## 8. Data-scale evidence required

The current 414-image/300-group release cannot support the primary paper claim by itself. Before submission, include:

- a group-safe learning curve over at least 250→10k independent scenes;
- a table separating rendered scenes from unique injections, glitches, GPS blocks, IFOs, and observing runs;
- at least 10k independent training scenes for an image-method claim, or a learning-curve justification for less;
- a 10k physical baseline followed by evidence-triggered 25k/50k expansion; 200k is not required if controlled transfer curves plateau;
- 5k–10k validation and 20k–50k locked-test injection scenes;
- continuous/time-slide background exposure commensurate with the claimed FAR;
- BBH/BNS/NSBH, low-SNR, overlap-severity, glitch-morphology, detector, and run strata;
- an explicit demonstration that adding data improves O4 transfer rather than only O3-like mAP.

Offline augmentations are reported separately and never counted as independent physical samples.

For the chirp-frozen residual adapter, scale promotion is explicitly conditional:

1. The frozen one-seed adapter must pass the absolute validation gates without test access.
2. Exactly four additional seeds run only after that pass; at least 4/5 seeds and all frozen
   median stability gates must pass.
3. Only then may 250/500/1,000/full physical-group scales run under both fixed-epoch and
   fixed-optimizer-update controls.
4. The scaling configs must preserve the one-seed adapter architecture, optimizer and sampling
   policy. A family-balanced adapter is a separate predeclared ablation, not an implicit change
   inside the data-volume curve.
5. A larger hard endpoint is authorized only when both controls show the frozen material
   improvement while preserving clean-chirp non-inferiority.

These gates prevent a training schedule or a more aggressive sampler from being mislabeled as
evidence that physical sample count caused the gain.
