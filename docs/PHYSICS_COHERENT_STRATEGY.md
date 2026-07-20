# Physics-coherent GW-YOLO strategy — 2026-07-20

## Decision

The project will not attempt to replace DINGO or AMPLFI as a general clean-data posterior estimator.
Its primary contribution is a lightweight, interpretable front end for the regime in which current
search and inference systems are least reliable: nonstationary detector noise, unseen glitches,
variable detector availability, and signal--glitch overlap.

The primary scientific question is:

> Can physics-coherent, multi-instance time--frequency segmentation improve fixed-FAR search
> sensitivity and downstream posterior reliability under real detector artifacts without harming
> clean signals?

The target system remains GW-YOLO: numeric multi-Q planes are treated as scientific image tensors,
and a YOLO-style multiscale encoder/instance decoder returns boxes, identities and masks. Rendered
PNG spectrograms remain a legacy baseline. The new system differs from the original GW-YOLO by
using physical group splits, numeric tensors, variable detector sets, continuous background,
network coherence and a measured downstream use for every promoted mask.

## Evidence motivating the change

- Original GW-YOLO reports strong image metrics and overlap injection efficiency, but continuous
  fixed-FAR search, mask-informed deglitching and event validation were future work:
  <https://arxiv.org/abs/2508.17399>.
- In the first ML gravitational-wave search challenge, the leading ML search reached about 70% of
  the traditional search sensitive distance in real O3 noise, despite much stronger performance in
  Gaussian noise. Real-noise false alarms and transfer are therefore the relevant bottleneck:
  <https://doi.org/10.1103/PhysRevD.107.023021>.
- DINGO-IS already supplies likelihood correction, evidence and sample-efficiency failure
  diagnostics: <https://arxiv.org/abs/2210.05686>. Dingo-T1 adds flexible detector/frequency
  configurations: <https://arxiv.org/abs/2512.02968>.
- AMPLFI targets low-latency BBH posterior inference and has public-alert validation:
  <https://arxiv.org/abs/2407.19048> and <https://arxiv.org/abs/2509.22561>.
- GWTC-5.0 adds 161 O4b candidates, all compatible with BBH, including five with network SNR above
  30. Virgo participates in O4b and detector subsets vary across events:
  <https://arxiv.org/abs/2605.27225>.
- O4 provides selected auxiliary channels and alternate cleaned strain products, enabling bounded
  detector-evidence and calibration/cleaning robustness studies: <https://gwosc.org/O4/O4b/>.
- New glitch morphologies continue to appear and challenge closed-set classifiers:
  <https://arxiv.org/abs/2508.13923>.

O4b is public but remains a locked test in this repository. Its catalog metadata may define strata;
its strain, auxiliary channels and alternate cleaning products may not select a model or threshold.

## Primary model contract

The promoted model is a 10--20 million parameter system, not a vision foundation model:

1. a shared lightweight encoder processes each available IFO's numeric Q=4/8/16 planes;
2. detector availability is explicit and the same weights accept H1/L1, H1/V1, L1/V1 and
   H1/L1/V1 without retraining;
3. set/attention fusion is permutation aware and never confuses a missing detector with zero strain;
4. pairwise time-lag/cross-correlation features use predeclared physical light-travel limits;
5. the instance decoder preserves every chirp and glitch mask and exposes an unknown-artifact score;
6. a candidate ROI may invoke a local high-resolution or conventional correlation timing refiner;
   full-window neural millisecond timing is not a required contribution;
7. optional auxiliary summaries affect glitch/OOD evidence only and are ablated against strain-only.

SAM 2 streaming memory motivates lightweight cross-window object memory, and Audio-MAE motivates
masked pretraining on unlabeled Q-maps. The project will not fine-tune full SAM/DINO-scale models
unless a small-model pilot demonstrates a clear fixed-FAR deficit that capacity can plausibly solve.

## Three paper contributions

### C1 — Variable-detector, physics-coherent instance segmentation

Show that shared per-IFO encoding plus detector-set fusion and physical coherence improves the
frozen O4a endpoint relative to single-IFO, fixed-channel and morphology-only baselines. Report every
detector subset separately and include calibration amplitude/phase perturbations.

### C2 — Open-set glitch masks with conservative abstention

Use small masked-Q pretraining and a calibrated embedding/OOD score. Evaluate leave-one-glitch-family
out, O1--O3 to O4a transfer and newly observed morphologies. Unknown artifacts are review candidates,
not forced members of a known class. Weak Gravity Spy masks remain weak labels until the blinded
human audit is complete.

`ood-abstention-evaluate` enforces the statistical half of this protocol independently of the
eventual embedding choice. Its threshold uses only known validation scores and a predeclared maximum
known-artifact abstention rate; ties are handled conservatively. Glitch and GPS groups must be
disjoint from evaluation. Held-out families and later observing runs are then evaluated once with
Wilson intervals for known false abstention, unknown true abstention and unknown false acceptance,
plus diagnostic AUROC and family/run strata. Unknown scores are explicitly prohibited from threshold
selection. `glitch-ood-train` supplies a compact shared single-IFO Q encoder, known-family classifier
and cosine-distance-to-known-prototype score. Checkpoint selection uses known calibration accuracy;
prototypes use known training embeddings; held-out embeddings are first consumed after both are
frozen. This head is auxiliary attribution/review evidence and is prohibited from vetoing a
strain-coherent candidate. It remains an unpromoted candidate until the verified Gravity Spy bank
completes and leave-one-family-out/O3→O4a results are available.

### C3 — Mask-conditioned search and inference robustness

Measure raw, contaminated and mask-conditioned versions of the same injection. Primary search
metrics are efficiency and `<VT>` at common FAR. Downstream PE metrics are paired changes in bias,
credible-interval coverage/width, sky area, DINGO-IS sample efficiency and latency. Clean injections
must pass non-inferiority before an overlap gain is accepted.

## Bounded experiment ladder

### Stage A — seven-day discriminator

Use the frozen 10k train/3k validation corpus, one seed and a bounded update budget:

| Arm | Question |
|---|---|
| compact numeric baseline | Current physical reference |
| numeric YOLO | Does a YOLO-style multiscale decoder add value? |
| + detector-set fusion | Does one model handle variable IFO sets? |
| + physical coherence | Do real-background false alarms fall? |
| + small masked-Q pretraining | Does unlabeled real noise improve transfer? |
| + OOD score | Can held-out glitch families abstain safely? |

No arm is promoted on IoU alone. It must improve weighted efficiency at the frozen validation
background operating point or a predeclared overlap/OOD endpoint, while retaining at least 98% of
the clean baseline efficiency.

The first hand-designed coherence arm is now rejected: multiplying morphology by the square root
of mean absolute lag correlation reduced validation weighted efficiency from 0.08745 to 0.02109 at
the same background count, with paired recovered-`VT` change -75.88% and a wholly negative 95%
interval. Future coherence work must expose timing/correlation as separately calibrated features or
a learned reranker and must preserve a morphology-only fallback; the multiplicative formula is not
a candidate for the locked search.

### Stage B — shortlisted evidence

Run only the best two arms for three seeds at 10k. Compare 10k with 25k under fixed-epoch and
fixed-update protocols. Continue to 50k only if the primary O4a endpoint improves by at least one
absolute percentage point, or a predeclared low-SNR/overlap endpoint improves materially with a
paired interval excluding zero.

### Stage C — publication search

Freeze the model, ranking statistic and thresholds on O4a. Stream at least 30 coincident
detector-days, cluster triggers, generate time-slide background, and report FAR/IFAR and `<VT>`.
The original 800-pair O4a acquisition plan is sufficient in raw duration but must be processed by
download--verify--score--evict streaming because local disk cannot retain it.

### Stage D — bounded downstream PE

Use 100--300 stratified paired cases rather than a catalog-scale PE campaign. Analyze clean,
contaminated and mask-conditioned inputs with identical priors, PSDs and waveform assumptions. Use
a small expensive BayesWave/Bilby subset only as a reference. A published number from another
population is not a head-to-head comparison.

`pe-robustness-evaluate` freezes the executable table schema for this stage. Every backend/injection
must provide a complete clean, contaminated and mask-conditioned triplet. Publication mode rejects
changes in backend/model hash, prior, waveform approximant, detector set, calibration version,
source-event identity, hardware or latency scope across the triplet. It reports parameter coverage
with Wilson intervals; paired changes in absolute bias and credible width; effective samples per
second; 90% sky area; and mask-conditioning latency. DINGO and AMPLFI are reported as separate
backends, and the joint comparison gate stays false until both are present. This is a downstream
robustness protocol, not a detection-mAP competition.

### Stage E — one-time locked O4b/GWTC-5 evaluation

After architecture, thresholds, calibration perturbations, auxiliary policy, OOD rule and all
subgroup definitions are frozen, evaluate O4b once. Primary evidence is locked injections plus
continuous background. Catalog recovery is descriptive and never called search recall.

## Stop and promotion rules

- Stop independent full-window timing development if validation p90 is not useful for the declared
  coincidence window; retain it as a negative result and use ROI/correlation refinement.
- Stop scale expansion when two successive controls plateau; add GPS/run/glitch diversity instead.
- Stop auxiliary-channel expansion unless a 5--20 channel pilot reduces hard-negative false alarms
  without violating clean-signal non-inferiority.
- Stop a large visual-backbone experiment unless the compact model is demonstrably capacity-limited.
- Do not open O4b to rescue a weak O4a result.

The completed exact-GPS high-resolution timing refiner activates the first stop rule. Its
validation-selected epoch 9 has median absolute error 1.519 s, p90 4.209 s and only 23/500 (4.6%)
predictions within 10 ms, despite a 7.8125 ms output grid. The representation-resolution gate passes
but the empirical accuracy gate fails. Report SHA256 is
`b958906308ac66ada621a0f1922254d2e49f5f29b2badf59fc1d2f491b744f44`. No additional independent
full-window timing run is scheduled; ROI/correlation refinement is the promoted track.

The first correlation-contract smoke at code commit `7006423` read 100 frozen O4a validation
signals without opening test data. Every H1--L1 lag remained inside the strict 10 ms window; the
largest discrete lag was 20/2048 s (9.766 ms), and median absolute signal-only correlation was
0.983. Report SHA256 is
`7634830534ef6d95249d3990a0473619467fa54fb2c7c896aadb9b1c1b8daf6b`. This establishes numerical
feasibility only. It is explicitly not a real-background, false-alarm or ranking result; the next
valid comparison must score identical continuous O4a background with morphology-only and
coherence-assisted frozen candidates.

That comparison path is now executable. With `--coherence-config`, both background and injection
scorers retain the morphology score and additionally crop the same one-second whitened strain ROI
around the network mask peak. For every available detector pair they maximize absolute normalized
correlation only inside the predeclared light-travel limit plus the 1 ms timing allowance. The
coherence-assisted ranking is the morphology score times the square root of mean pairwise absolute
correlation. Coarse mask-bin arrivals are not mislabeled as a timing gate. The command
`coherence-validation-compare` independently calibrates morphology and coherence thresholds on the
same validation background count and performs an injection-paired `<VT>` bootstrap. It remains a
short-exposure development comparison until continuous clustered background and time slides exist.
The scorer also records an 8 ms-smoothed analytic-envelope peak at native 1/1024 s resolution; its
network-median absolute timing error and 10 ms success rate are audited against injection GPS. This
refinement may feed candidate timing only if the empirical p90 gate passes—sample resolution alone
is not treated as timing accuracy.

The first detector-set architecture boundary is now implemented separately from the running
fixed-channel control. `DetectorSetQNet` applies one shared Q-plane encoder to every configured IFO,
uses an explicit binary availability tensor to mask set-attention, fuses only available detectors,
and applies a shared per-IFO decoder/head. Missing IFO logits are forced inactive rather than inferred
from zero-valued strain. A deterministic warm-start maps the fixed-channel encoder/head by averaging
only the IFO axis, copies shape-compatible bottleneck/decoder weights, and initializes attention
uniformly. This is a candidate arm, not a promoted model.

The frozen 10k/3k physical corpus contains H1/L1 only; V1 is an explicit missing slot. It can test
that missing-detector masking works, but cannot establish H1/V1, L1/V1 or H1/L1/V1 generalization.
Detector-set training therefore remains blocked on a group-safe O1--O4a corpus with real detector-set
coverage. No fixed-channel result is relabeled as variable-detector evidence.

Commit `6bc15e9` makes that evidence gate executable. Both continuous-background and injection
scorers accept an explicit enabled-IFO subset while retaining the checkpoint's immutable H1/L1/V1
slot order. Detector-set checkpoints receive the corresponding validity tensor; disabled strain is
not mistaken for an observed zero. `detector-subset-summarize` requires the same checkpoint,
background manifest, injection IDs, waveform IDs and weights across H1/L1, H1/V1, L1/V1 and
H1/L1/V1. Each subset receives its own validation-background-only threshold, followed by paired
`<VT>` bootstrap comparisons against the full network and a predeclared one-sided non-inferiority
margin. The report cannot pass its completeness gate when any network subset is absent.

A provenance-complete GWOSC inventory resolves that data blocker without opening O4b. O4a contains
no aligned H1/L1/V1 4-kHz file triples. O3b contains 2,029 such triples; a seed-20260720 stratified
200-triple development plan spans GPS 1256697856--1269338112 and has SHA256
`f7df61fb720369fbe8eaec4306fc3baea04a922f9d991220151b4c27100cce2a`. Its 9.48 raw detector-days
are an acquisition upper bound, not analyzed live time. Because the raw HDF set exceeds local disk,
the promoted data path is a verified numeric background bank with immediate source eviction, not a
bulk download. O3b supplies variable-detector training; frozen O4a H1/L1 remains the run-transfer
development endpoint.

## Paper success hierarchy

1. **PRD target:** paired fixed-FAR `<VT>` or overlap-efficiency improvement with clean-signal
   non-inferiority and complete background methodology.
2. **ApJS target:** a reusable O4 numeric multi-IFO/mask benchmark, manifests, software and extensive
   domain-shift tables even if the sensitivity gain is modest.
3. **JCAP target:** only if robust deglitching materially changes a population, selection-function,
   standard-siren or other astrophysical inference. A segmentation-only paper is not sufficient.
