# Data scaling and domain-coverage plan

## Decision

Data scale and physical coverage remain important, but the 2026-07-20 strategy review removes blind scale-up as the default path. The first decision is whether a compact physics-coherent detector-set model is limited by waveform count, GPS/run diversity, glitch/OOD coverage or representation. Architecture and data increments are promoted only by frozen O4a endpoints under fixed-epoch and fixed-update controls.

The short-exposure development endpoint is now executable as `physical-validation-endpoint`. For the
current 824-window O4a validation background, each checkpoint is calibrated on background only with
at most eight surviving windows and is then evaluated on the same frozen 3,000 physical validation
injections. This is an exposure-limited model-selection metric, not a FAR claim. A scale increment is
not promoted from mask IoU alone and still requires consistent improvement under both fixed-update
and fixed-epoch controls; publication sensitivity remains conditional on clustered time slides,
adequate continuous exposure, and the untouched locked test corpus.

`physical-scale-epoch-series` executes the complementary equal-epoch control over the same nested
manifests and seeds. It requires validation-only best-checkpoint selection, forbids an optimizer-step
cap, verifies all manifest hashes, and writes a three-seed summary. Different scales are expected to
consume different optimizer examples in this control; the held constant is 30 complete epochs.

The program distinguishes three quantities:

1. **Rendered images** — may include augmentations and are not an independence count.
2. **Physical groups** — unique waveform/injection, glitch, GPS block, IFO/network state, and observing-run provenance.
3. **Search exposure** — injection population size and continuous/time-slide background live time.

Only the latter two support generalization and search claims.

## Current baseline

| Quantity | Current value |
|---|---:|
| Rendered images | 414 |
| Independent physical groups | 300 |
| Training images | 295 |
| Training physical groups | 251 |
| Training chirp-only groups | 117 |
| Training noise-only groups | 48 |
| Training chirp+noise groups | 83 |
| Training quiet groups | 3 |
| Validation physical groups | 25 |
| Locked-test physical groups | 24 |

There are 180 images in multi-image groups, so offline augmentation is already a material fraction of the corpus. The core overlap task has only 83 independent training groups, and false-alarm behavior has only three quiet training groups. Neither quantity is adequate for an O4 search claim.

The implemented `gwyolo scale-plan` command was executed against the formal remote manifest. It reports:

| 10k baseline component | Current train groups | Additional groups needed |
|---|---:|---:|
| chirp-only | 117 | 2,383 |
| glitch-only | 48 | 2,452 |
| chirp+glitch | 83 | 3,917 |
| quiet/hard-negative | 3 | 997 |
| **Total** | **251** | **9,749** |

The 10k target is a 39.84× expansion in independent training groups; the 200k target is 796.81×. The audit also confirms that all eleven expected provenance fields are absent from the legacy manifest. Its machine-readable output is stored remotely at `/root/GW-YOLO-v2-artifacts/data/scale_plan.json`.

All current images are 640×640. The median chirp polygon occupies about 0.67% of the image and the median noise polygon about 0.65%, making low-SNR segmentation a small-object problem. The manifest also lacks explicit source family, SNR, masses, spins, distance, waveform, glitch class, GPS, IFO, observing run, Q plane, duration, and overlap severity.

## Revised experiment order

### DS0 — Freeze a meaningful evaluation corpus

Before using a larger training set to choose architectures, create:

- 5,000–10,000 independent validation scenes;
- 20,000–50,000 independent locked-test injection scenes;
- at least 200–500 examples in each primary source/SNR/glitch/overlap stratum;
- an O4a development/calibration partition split by GPS blocks;
- an untouched O4b/GWTC-5 test partition;
- continuous or time-slide background measured in years, not images.

No event window placed in a catalog test may be used as a training noise window.

### DS1 — Group-safe learning curve

Train the same compact model with the same optimizer and fixed validation set at:

`250, 500, 1k, 2k, 5k, 10k, 25k, 50k` independent groups.

Use at least three seeds per point and five seeds for the final selected scale. Report:

- mask mAP50 and mAP50-95 as representation diagnostics;
- low-SNR and overlap recall;
- clean-injection loss and false-veto rate;
- O3→O4 transfer;
- calibration error;
- fixed-FAR efficiency once search triggers are available.

Fit a scaling curve such as `M(N) = M_inf - a N^(-alpha)`. Continue scaling while a data doubling improves the primary O4 endpoint by at least 1 percentage point or improves a pre-registered low-SNR/overlap endpoint materially. If in-domain mAP rises while O4 performance stays flat, prioritize domain coverage. If both plateau, prioritize multi-Q/multi-IFO representation rather than duplicating images.

### DS2 — 10k independent-scene baseline

Minimum credible image/scene corpus:

| Scene type | Target |
|---|---:|
| chirp-only | 2,500 |
| glitch-only | 2,500 |
| chirp+glitch | 4,000 |
| quiet/hard-negative | 1,000 |
| **Total** | **10,000** |

This level is an engineering and representation baseline, not a search benchmark.

### DS3 — 200k research corpus

Recommended first publication-scale generator target:

| Scene type | Target |
|---|---:|
| chirp-only | 50,000 |
| glitch-only | 40,000 |
| chirp+glitch | 80,000 |
| quiet/hard-negative/OOD | 30,000 |
| **Total** | **200,000** |

Requirements:

- BBH, BNS, and NSBH source families;
- SNR 4–50, with deliberate emphasis on SNR 4–15;
- mass ratio, aligned/precessing spin, distance, sky position, inclination, and waveform-family coverage;
- H1/L1/V1 and missing-detector network states;
- O1/O2/O3/O4a real noise domains;
- major Gravity Spy morphologies plus rare/unknown O4 glitches;
- clean, nearby, partial, and severe time-frequency overlap;
- 1/4/16/64-second numeric windows and multiple Q planes.

### DS4 — Online 0.5M–2M scene generation

Do not materialize millions of RGB plots. Maintain versioned pools of real strain/GPS blocks and glitch triggers, sample physical injections, combine them in the time domain, and generate numeric multi-Q tensors during training. Cache deterministic validation/test tensors only.

This policy is now executable. `configs/data_factory_research.yaml` defines a 200k physical-recipe
corpus with 160k/10k/30k train/validation/test scenes and uses `recipe_only` materialization. The
implemented pilot measured about 0.41 MB per full-debug scene, implying roughly 82 GB for 200k scenes,
well above the current server's approximately 14 GB free space. Online generation is therefore a
hard infrastructure requirement, not just an optimization.

## Leakage rules for generated mixtures

A mixture depends on at least two identities: injection/waveform ID and glitch/GPS ID. A random split of mixture rows is invalid. Use a disjoint bipartite split:

- no waveform/injection ID crosses train/validation/test;
- no glitch ID or GPS block crosses train/validation/test;
- all rendered durations/Q planes/augmentations of one physical scene remain together;
- nearby windows from one continuous background block remain together;
- O4b and catalog-event exclusion windows remain locked.

## Source distribution

The 200k target should be adjusted by importance weighting, but a starting allocation is:

- source family: 40% BBH, 30% BNS, 25% NSBH, 5% stress/OOD;
- SNR: at least 60% below 15;
- overlap scenes: at least 40% of all training scenes;
- quiet/hard negatives: at least 15%;
- each major glitch morphology: at least 1,000 independent anchors or a documented long-tail sampler;
- each IFO and observing domain: enough independent GPS blocks to prevent one run/site from dominating.

Repeatedly mixing one glitch with many injections does not create the same amount of information as new glitch anchors. Report both scene count and unique waveform/glitch/GPS counts.

## Search exposure targets

The data factory must ultimately produce trigger-level outputs. At a frozen threshold report FAR, IFAR, efficiency, sensitive distance, and `<VT>`. If zero background events survive, the 90% Poisson FAR upper limit is approximately `2.3/T`:

- a 1/year FAR claim requires at least 2.3 years of equivalent background;
- a 0.01/year FAR claim (IFAR 100 years) requires about 230 years;
- stronger claims require proportionally more time slides or background.

`candidate-exposure-plan` now computes the exact detector-duty-cycle-correct exposure before any
GPU scoring. On the frozen 824-window O4a validation manifest, 4,096 sequential non-cyclic 8-second
offsets give only 475 nonempty shifts and 388,000 seconds (`0.012295 yr`) of equivalent exposure.
With zero surviving candidates, the 90% FAR upper limit is still `187.28/yr`. Even an ideal schedule
using every one of the 339,076 available positive-lag window pairs only reaches `0.08596 yr`, whose
zero-count upper limit is `26.79/yr`. Thus the current corpus cannot measure an astrophysical FAR,
regardless of model score.

For a zero-count 90% upper limit at FAR `0.1/yr` (IFAR 10 yr), the all-pairs best case needs at least
13,479 valid 8-second dual-IFO windows, or 1.248 zero-lag detector-days in each independently frozen
partition. That is 16.36 times the current validation window count before allowing for gaps,
segment-boundary restrictions, unavailable IFOs or reserved test data. The corresponding best-case
window requirements are:

| Target IFAR | Target FAR | Equivalent background for 90% zero-count bound | Minimum 8 s windows | Minimum zero-lag time |
|---:|---:|---:|---:|---:|
| 1 yr | 1/yr | 2.303 yr | 4,263 | 9.47 h |
| 10 yr | 0.1/yr | 23.026 yr | 13,479 | 1.248 d |
| 100 yr | 0.01/yr | 230.259 yr | 42,623 | 3.946 d |

These are lower bounds, not acquisition quotas. The operational target remains at least 30
coincident detector-days across runs because run transfer, glitch diversity, calibration drift,
independent validation/test partitions and bounded shift schedules all consume additional live time.

## 2026-07-20 physical-domain checkpoint

The five-seed 10k analytic curve is now statistically stable enough to reject a simple “only add
more of the same synthetic images” strategy. Mean validation IoU is `0.88087 ± 0.00678`, while a
validation-only real-O4a experiment using 300 cosmological PyCBC/LAL injections recovered only
22.3% unweighted and 16.8% volume-weighted at the permissive 7,953/year measured FAR point. At the
threshold nominally requested for 1/year it recovered 9.0% unweighted and 6.1% weighted, but the
available 3.3 hours of time-slide exposure supports only a 90% zero-count upper limit of 6,104/year.
The “1/year” number is therefore a threshold stress test, not a measured FAR claim.

This combination indicates a domain-and-exposure bottleneck, not merely insufficient row count:

- 10k analytic scenes are nearing their in-domain scaling asymptote;
- the physical pilot has only 300 injections and six GPS blocks;
- the expanded O4a validation background has 824 windows in 26 GPS blocks, but even all positive
  window pairs provide only 0.086 equivalent years;
- the waveform backend still lacks an external equivalence certificate;
- the network time grid is 83 ms, too coarse for publication coincidence.

The next scaling matrix must therefore increase independent information along separate axes:

| Axis | Nested development points | Promotion target |
|---|---|---|
| Physical training waveforms | 10k, 25k, 50k, 100k, 200k | 200k unique waveform IDs |
| Real background | 100, 300, 1k, 3k, 10k GPS blocks | at least 30 coincident detector-days across runs |
| Validation injections | 1k, 3k, 10k | at least 10k, family/SNR/overlap stratified |
| Locked test injections | never sampled during development | 50k–200k after pre-registration |
| Glitch anchors | 1k, 5k, 20k, 50k unique IDs | coverage-driven, not remix-count driven |
| Time-slide exposure | 2.3, 23, 100, 230 equivalent years | chosen from the preregistered FAR/IFAR endpoint |

At each doubling, hold the validation injections, GPS blocks, waveform population, seed set and FAR
definition fixed. Continue adding data if the primary O4a weighted efficiency or low-SNR/overlap
endpoint improves by at least one percentage point. If analytic IoU rises but O4a efficiency does
not, add run/IFO/glitch diversity; if both plateau, change representation (higher temporal output,
long/multi-rate context and coherent fusion) before generating more correlated mixtures.

The acquisition side is executable through `gwosc-run-plan`, which paginates the official GWOSC v2
run strain-file endpoint, intersects exact GPS starts across requested IFOs and samples time strata
deterministically. `gwosc-batch-download` then resumes each HDF5 transfer and requires a complete
Fletcher32/statistics/DQ scan before recording it. `gwosc-event-exclusions` snapshots every catalog
event in the run with a declared padding, and `background-batch-plan` applies those vetoes before one
global GPS-block split across all files. Repeat `--batch-report` for every verified acquisition
batch when using the legacy balanced split; planning those batches separately is forbidden because
adding files could otherwise change the validation/test allocation. All acquisition commands reject
O4b development access.

Large continuous acquisitions use `--split-strategy hash_threshold_v1`. It maps each complete
256-second GPS block independently through the frozen seed and validation/test fractions, so adding
later files cannot move an earlier block between splits. This is the only allowed mode for streaming
shards: download and fully verify one bounded file batch, plan it, score all selected windows, retain
hash-linked candidates, then release the recoverable GWOSC cache before the next batch. The legacy
`balanced_rank_v1` strategy remains the default solely to reproduce already frozen manifests; it
must still see every batch at once and is prohibited for incremental acquisition.

Streaming eviction is executable and deliberately narrow. `candidate-probability-evict` requires a
candidate report that hash-binds a complete probability-saving score report, validates every saved
probability/strain artifact inside an explicit cache root, writes an immutable intent record, and
only then removes those exact files. `background-source-evict` additionally requires a passing
single-batch GWOSC report, a `hash_threshold_v1` background plan, and complete candidate extraction
for every non-empty validation/test split before releasing the hash-verified public HDF sources.
Training rows in a search-only shard are counted as intentionally unscored, never mistaken for
evaluation exposure. Both reports retain public recovery instructions and byte-level hashes; model
reports, manifests and candidates are never evicted.

`gwosc-plan-shard` slices the frozen parent acquisition plan by pair index and records the parent
hash, exact half-open index range and selected pair-ID hash. `background-stream-shard` consumes one
such non-overlapping slice end to end: resumable verified download, event-vetoed stable block split,
complete validation/test scoring, all-instance candidate extraction, application of the exact
checkpoint/config/commit-matched timing calibration, probability eviction, and finally source-HDF
eviction. A completed shard is immutable under its full run identity and can be resumed after cache
release without attempting to redownload a deleted source. Shards containing only training blocks
are explicitly counted and safely released for this search-only corpus.

After any acquisition tranche, `background-stream-merge` verifies a single parent/split/model/
timing identity, non-overlapping parent pair-index ranges, unique windows and candidates, and one
split per GPS block across every shard. It writes globally ordered background plus val/test
calibrated-candidate manifests and records whether the supplied ranges cover the complete parent
plan. Partial merges are useful for preregistered exposure audits but remain explicitly incomplete.

## Storage and compute strategy

At the current JPEG size, 100k plots would be only several GB, but RGB plots discard physical information. A float32 tensor with several Q planes and three IFOs can consume hundreds of GB per 100k scenes. Prefer:

- float16 or compressed chunked arrays for locked sets;
- on-the-fly generation for training;
- cached PSDs and transforms;
- deterministic recipe manifests instead of duplicated tensors;
- streaming dataloaders and resumable shards.

For full 64-second waveform contexts, `signal_scaled_float16` stores each IFO divided by its
float64 peak and restores the physical scale on load. This avoids direct float16 underflow while
halving the payload relative to actual float32 storage. Every sample must pass relative L2 <=1e-3
and normalized overlap >=0.999999; the materialization report records the worst reconstruction.

For injection sensitivity, count and storage targets must be expressed separately. The current 25k
H1 recipe artifact is only a provenance pilot and is too correlated (192 windows, six GPS blocks).
The publication corpus should target at least 10k independently seeded validation injections and
50k--200k locked test injections across BBH/BNS/NSBH strata, but the decisive increase is independent
GPS background blocks and source-population coverage, not repeated injection rows on the same few
seconds. Run a group-safe scaling matrix over injection count, unique GPS live time, glitch overlap,
SNR/distance, source family, IFO network and observing run. Stop increasing count only when bootstrap
uncertainty on paired `<VT>` improvement is below the predeclared effect size and performance has
plateaued across at least two successive scale points.

The 200k/0.5M--2M targets are now contingency ceilings, not scheduled deliverables. After the frozen
2k/5k/10k controls, run 25k only if the primary O4a endpoint or a predeclared low-SNR/overlap
endpoint improves materially. Run 50k under the same rule. If transfer is flat, spend the next unit
of compute on new GPS blocks, observing runs, real/held-out glitches, detector subsets, calibration
perturbations or hard negatives rather than duplicated waveform scenes. See
`PHYSICS_COHERENT_STRATEGY.md`.

On the current RTX 4090 D, a 100k-scene YOLO26m experiment is expected to be an order of roughly one day rather than minutes, before multi-Q generation overhead. Measure generator throughput before committing to a 1M-scene run, and do not start it without the preceding promotion evidence.

## Promotion gates

Architecture work is promoted from exploratory to primary only after:

1. DS0 evaluation data are frozen;
2. at least the 250→10k learning curve is complete;
3. the manifest reports independent waveform, glitch, GPS, IFO, and run counts;
4. O4 transfer is measured separately from in-domain mAP;
5. label audits cover mask consistency and hard-negative contamination.

A paper claim requires DS3-scale coverage or an evidence-based learning-curve argument for a smaller corpus, plus continuous-background search exposure. Dataset size alone never substitutes for FAR/`<VT>`.

## 2026-07-20 physical training and real-glitch update

The first context-safe O4a physical training batch contains 2,000 train and 500 validation
injections over 38 and 13 disjoint GPS blocks. Injection ID, waveform ID and GPS-block overlap are
all zero. A 30-case family-stratified comparison covering `IMRPhenomXAS`,
`IMRPhenomXAS_NRTidalv3` and `IMRPhenomNSBH` passed the direct-LALSimulation waveform gate; the
passing report SHA256 is `0498c7ee60c8adbc567582e06d44f8c3ab5c24893e4ab359531c36b2012dfe2b`.
Population and detector-projection validation remain separate gates.

Two negative controls materially change the scale plan. Before fixing physical-power float32
underflow, validation mask occupancy reached 55.8% and the best IoU was 0.1034; this run is invalid
as a target-construction benchmark but is retained. After scaling only the signal-only target before
the transform, occupancy became 0.15–0.67% for BBH, 0.74–2.12% for BNS and 0.42–1.83% for NSBH.
The corrected all-row run reached only 0.0473 validation IoU. Filtering train to SNR>=4 retained just
1,065 rows and reached 0.0431, so deletion did not solve the domain problem.

The empirical-noise optimal-SNR audit explains why nominal row count overstates useful training
scale:

> Correction: the table below is the first full-context diagnostic. It integrated signal power over
> 64 seconds while the network sees eight seconds, so it overstates long-waveform SNR. It motivated
> the correct analysis-window audit but is not final SNR evidence; corrected counts supersede it.

| Split | Rows | SNR <4 | SNR 4–8 | SNR 8–15 | SNR 15–30 | SNR >=30 | Median SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2,000 | 935 | 669 | 288 | 88 | 20 | 4.25 |
| validation | 500 | 236 | 159 | 73 | 26 | 6 | 4.27 |

The corrected analysis-window annotation supersedes that table:

| Split | Rows | SNR <4 | SNR 4–8 | SNR 8–15 | SNR 15–30 | SNR >=30 | Median SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2,000 | 1,027 | 596 | 271 | 85 | 21 | 3.89 |
| validation | 500 | 256 | 143 | 71 | 24 | 6 | 3.91 |

Corrected manifest SHA256 values are
`66d84048891ca1d61b03dba149f7166b166d49c501f8b2216ab79067b828dafc` (train) and
`bf6a7222cb9f6538e3dc3d5a54bf613c14738c21d65b524aebea3568e905d40f` (validation).

Thus the corrected audit finds just over 51% of the volume-drawn pilot below the training floor.
Future training generation
must deliberately cover SNR 4–50 (especially 4–15), rather than drawing distance only from the
astrophysical evaluation proposal. Validation/test must retain the population proposal and weights.
A train-only curriculum may rescale the existing 1,027 sub-floor signals into SNR 4–8, but it still
counts as exactly 2,000 unique waveforms and 38 GPS blocks—not 2,935 samples and not new evidence.
The first curriculum is itself imbalanced after the corrected audit: the 1,027 sub-floor rows join
596 native rows in SNR 4–8, placing about 81% of training in one band. The promoted scale protocol
therefore uses `physical-snr-quota` to assign exact train-only quotas of 40% at SNR 4–8, 35% at
8–15, 20% at 15–30 and 5% at 30–50. Every assignment is deterministic by injection ID; it changes
neither physical sample counts nor validation/test proposals and `<VT>` weights.
`configs/physical_finetune_snr_quota.yaml` freezes the corresponding 20-epoch 2k ablation. It must
use the same validation manifest and cannot be interpreted as a data-scaling point because it adds
no waveform or GPS identity.

The official Gravity Spy expansion now contains 80,496 unique high-confidence O1–O3b H1/L1 glitch
IDs. IFO-independent network-GPS splitting gives zero train/validation/test overlap. Of 64,284 train
anchors, 59,933 (93.23%) map to a single official GWOSC HDF5 file containing a full 64-second
context, spanning 16,297 unique source files. Downloading all source files would require hundreds of
GB, far beyond the current roughly 8 GB free disk. Real-glitch training therefore requires a bounded
download/extract/verify/evict cache or external artifact storage; repeatedly downloading a small
subset is not an acceptable substitute for unique glitch and GPS coverage.

The promotion sequence is now:

1. finish the 2k SNR-curriculum and per-IFO visibility ablation, retaining all negative results;
2. run the same checkpoint at 1,024 time bins (7.8125 ms) and pass the <=10 ms timing gate;
3. generate nested 10k/25k/50k physical train sets with explicit SNR quotas and increasing GPS/run
   diversity, using at least three seeds per scale and five at the promoted endpoint;
4. implement the bounded real-glitch strain cache and numeric mask pipeline before claiming a
   deglitch advantage;
5. expand validation to >=10k injections and background to >=30 coincident detector-days, then
   accumulate pre-registered time-slide exposure in years;
6. freeze architecture, calibration and thresholds before any O4b/locked-test evaluation.

The first 1,024-bin run has now closed step 2 as a negative result. Its nominal 7.8125-ms grid passes
the representation gate, but validation endpoint error is 3.63 s median and 6.17 s at the 90th
percentile. The next representation change therefore needs an explicit coalescence-time/local-peak
objective and suppression of window-spanning false activations; further time-bin increases alone are
not promoted.

The subsequent independent 1,024-bin exact-GPS refiner likewise failed promotion: its selected
epoch reached 1.519 s median and 4.209 s p90 error, with only 4.6% of 500 validation injections
within 10 ms. The grid was sufficiently fine but the global eight-second classification target was
not learnable at the required precision from 2k examples. Timing work therefore moves to a
coarse-to-fine local crop conditioned on a clustered chirp trigger; global-bin tuning is stopped.

The Gravity Spy acquisition plan is now partitioned into 510 deterministic shards of at most 32
whole source files. All 59,933 planned anchors and all 16,297 files are preserved exactly once; the
sharded manifest SHA256 is `5fcc63ae5e0e3dc8d5504317f92be19d2cc703c149fe4bbebb8808708959e718`.

High-yield source files alone are not an acceptable scaling rule because recurrent narrowband
families can dominate a file. `gravityspy-strain-select` treats official HDF5 files as atomic
download units and greedily fills frozen per-label deficits, accounting for already verified
numeric rows and excluding their source files. Its report exposes every underfilled class, run/IFO
coverage, and source/output hash. This controls bounded acquisition; it does not turn
metadata-derived weak masks into human segmentation labels.

The O4a waveform axis is now frozen as a strictly nested 10k/25k/50k plan. The existing 10k core
is preserved exactly; the next levels add 15k and 25k unique waveforms, with BBH/BNS/NSBH counts
of 4,500/3,000/2,500, 11,250/7,500/6,250 and 22,500/15,000/12,500. Every scale has zero injection,
waveform and GPS-block overlap with the shared 3k validation proposal. The machine report SHA256 is
`77b8ff6ae2ae7e9ddb84235bf229527d2c2aeb711f832771e2533113e1323ad9`; the 50k manifest SHA256 is
`2b20664721f668f8f0f52507ba5a3e55af136011cc36a7d5711d209612a23139`. Thirty supplemental-only
direct-LAL cases passed (report SHA256
`1c6fddd1bf717ce89d09eec061340749258b4a45ec9ab9cf4e812604f48d4a9a`).

All three levels currently reuse 76 O4a training GPS blocks. This is intentionally reported as
`gps_diversity_saturated=true`: it isolates waveform-count scaling but does not satisfy the domain
diversity endpoint. More globally split O1--O4a blocks must be added as a separate scaling axis
before interpreting a plateau as waveform-data saturation.

The scale matrix now has two compute protocols. The primary protocol holds epochs, optimizer and
batch size fixed, so it measures the practical benefit of adding data with proportionally more
compute. A paired control sets `max_optimizer_updates: 3750`, equal to 30 passes over the frozen 2k
reference at batch size 16. This control drops only the randomized incomplete training batch, stops
exactly at that many updates, validates the final partial epoch, and records
`steps_per_full_epoch`, `optimizer_updates`, `optimizer_examples` and the budget gate in its machine
report. Thus every scale sees exactly 60,000 examples. A scale gain is attributed to independent
data only if it survives this fixed-update/fixed-seen-example control; disagreement between
protocols is reported as a compute-data interaction, not hidden.
Because different scales complete different numbers of epochs within 3,750 updates, this control
selects only the final-update checkpoint. Per-epoch validation is retained as a diagnostic but
cannot grant smaller datasets more checkpoint-selection opportunities. The primary fixed-epoch
protocol continues to select its checkpoint only by the shared validation metric.
The two executable configurations are
`configs/physical_finetune_scale_fixed_updates.yaml` and
`configs/physical_finetune_scale_fixed_epochs.yaml`; their reports must never be pooled into one
seed distribution.

Repeated validation transforms may use `--validation-feature-cache-dir`. Cache keys bind the
materialized-signal hash, injection ID, per-IFO SNR visibility metadata, tensor configuration,
IFO/Q order and sample rate. Features remain float32 and binary plane targets are restored exactly;
atomic writes and key/shape/content checks make a bad cache a hard failure. Training tensors are not
disk-cached, avoiding a multi-gigabyte duplicate of every scale and preserving fresh online
construction as the primary path.

`physical-scale-series` is the publication runner for the fixed-update control. It hash-verifies the
frozen subset and validation manifests, requires `final_update`, runs every requested scale/seed
from the declared pretrained checkpoint, resumes only identity-matching outputs, and then invokes
the strict summary gate. Summary pooling requires identical code commit, config, pretrained hash,
optimizer updates/examples and validation-cache version; three seeds per scale remain the minimum.

The completed fixed-update endpoint does not meet the predeclared promotion margin. Across three
seeds, frozen O4a weighted efficiency changes from 0.08340 at 2k to 0.08543 at 5k and 0.08698 at
10k. The successive absolute gains (+0.00202 and +0.00155) are below one percentage point, and all
three paired 5k→10k bootstrap intervals include zero. Therefore 25k/50k waveform-count expansion is
not promoted from this control. The fixed-epoch result remains required, but the next independent
information axis is already identified: O3b H1/L1/V1 detector-set coverage, additional globally
split GPS blocks, real overlapping glitches and later-run OOD families. Training-data growth remains
necessary; repeating the same 76 GPS blocks with more waveforms is not the needed growth.

### Updated data decision: scale independent domains, not rendered rows

The fixed-update result and completed equal-epoch control change the acquisition plan materially.
Equal epochs improve weighted efficiency from 0.08000 to 0.12452 to 0.14897, but also increase seen
examples from 60k to 150k to 300k; fixed 60k examples change only 0.08340 to 0.08543 to 0.08698.
The defensible interpretation is compute/data interaction, not permission for blind waveform
generation. Retain the 10k/30-epoch candidate, increase independent physical axes, and adjudicate
them at fixed updates and fixed epochs. The next increment is a matrix rather than one row-count
target:

| Axis | Current evidence | Next promotion unit | Gate |
|---|---|---|---|
| waveform/injection identity | fixed-update plateau but strong update-coupled equal-epoch gain | no 25k promotion yet; retain 10k/30-epoch arm | fixed-epoch and fixed-update endpoint agreement |
| background GPS/run | only 76 train blocks in the scale curve | new globally split O1--O4a blocks | frozen O4a efficiency or hard-subset interval improves |
| detector set | primary corpus is H1/L1 | O3 H1/L1/V1 and missing-IFO subsets | each subset independently calibrated; clean non-inferiority |
| real glitch overlap | analytic overlap is not adequate | unique Gravity Spy glitch and GPS groups mixed in strain | weak-mask audit plus paired contaminated gain |
| open-set morphology | O1--O3 labels are closed set | held-family and later-run unknowns | known false abstention and unknown false acceptance intervals |
| background exposure | 824 validation windows; all-pairs upper exposure 0.086 yr | >=13,479 windows per split for IFAR 10 yr lower bound, operationally >=30 coincident days across runs | FAR/IFAR and `<VT>` at a common frozen threshold |

`physical-overlap-materialize` now pairs unique waveform/injection identities with unique real
Gravity Spy glitches without reuse, adds them in the time domain, performs a fresh whitening and
multi-Q transform, retains both chirp and weak glitch masks, and writes an explicit detector
availability vector. `physical-overlap-audit` jointly rejects waveform, injection, glitch, injection
GPS and glitch-GPS leakage across generated split manifests. Rendered image count is reported as
zero; mixtures, waveforms, injections, glitches and GPS blocks are counted separately.

`physical-overlap-finetune` is the bounded training arm. It uses detector-set fusion, supervises
both masks only on available planes, interleaves clean physical injections, distills the pretrained
clean glitch response, and selects a checkpoint only when clean-chirp IoU retention clears the
predeclared validation gate. It is resumable and calibrates chirp/glitch thresholds on validation
only. Even a successful validation run remains ineligible for a search claim until the paired
clean/contaminated fixed-FAR protocol passes.

Single-IFO Gravity Spy strain is useful for learning local glitch morphology but is not network
evidence. `gravityspy-network-strain-plan` therefore matches each event GPS independently to official
H1/L1/V1 GWOSC files and records the actually available subset. The first O2 validation shard has
five events, all with full H1/L1 64-second contexts, no V1 coverage, and only two unique source
files. `gravityspy-network-strain-materialize` verifies those whole files and DQ vectors, stores
aligned raw strain and numeric planes for every available IFO, and preserves the glitch mask only on
the catalog event IFO. Network-aware overlap generation then injects the physically coherent signal
into every available detector; it refuses an injection that lacks any required IFO.

The first recovered aligned shard is complete with five H1L1 events over four network GPS blocks.
All five were successfully converted to network-aware physical overlaps and paired clean/contaminated
validation overrides. This proves the data path but not effect size; five correlated rows cannot
promote a deglitch or search claim. The Gravity Spy train/validation expansion and later aligned
shards remain the source of the statistically useful overlap bank.

The practical conclusion is that the corpus is still far too small in independent domain coverage,
but increasing 10k to 50k with the same 76 GPS blocks is unlikely to create a qualitative change.
A qualitative gain is plausible only after real overlap, new GPS/run, detector-subset and OOD axes
grow together and pass frozen hard-subset endpoints. Scale promotion remains an empirical decision,
not a calendar milestone.

The first five completed aligned-network train shards sharpen the real-glitch part of this decision.
They account for 1,082 planned glitches, but only 859 pass finite-strain/DQ materialization; 182 are
rejected for non-finite L1 context and 41 for non-finite V1 context. Among the 859 usable rows,
`1400Ripples` alone contributes 352, while several families have only one or two examples. Reports
must therefore use 859—not 1,082—as the current usable physical count. The primary remedy remains
new independent glitch/GPS/run coverage. As a bounded optimization ablation,
`physical_overlap_finetune_family_balanced.yaml` draws the same number of rows per epoch using
square-root inverse family frequency capped at 4×; families with fewer than five physical examples
are not boosted. The report explicitly records that replacement sampling adds zero independent
physical examples. Compare this arm with uniform sampling at the same updates/seeds; it cannot
authorize a data-scale or paper claim by itself.

The next GPS-domain experiment is now explicitly paired rather than merely equal-sized.
`injection-background-remap` preserves every training injection/waveform identity and all intrinsic
and extrinsic source parameters, distance and `<VT>` weight while deterministically moving the
population onto new train-only GPS blocks and detector sets. It excludes every GPS block present
in the baseline arm or shared validation manifest. `injection-domain-pair-audit` then verifies the
materialized arms have identical source populations, identical row and identity counts, zero
cross-arm GPS overlap and zero train/validation identity or GPS overlap.

The predeclared gate is `configs/physical_data_domain_promotion.yaml`, executed with
`physical-data-domain-compare`. Both a 3,750-update final-update control and a 30-epoch
validation-selected control must pass on the same validation manifest with at least three paired
seeds. Independent GPS coverage must increase by at least 1.5x, overall chirp IoU must improve by
at least 0.005 with its paired seed-bootstrap lower bound above zero, and no BBH/BNS/NSBH family
may regress by more than 0.005. Passing permits a larger independent-domain experiment; it remains
a validation-only promotion decision and cannot support a FAR, sensitivity or paper claim.

The historical 2k pilot cannot be reused as the 2k point of this curve. Its 2,000 waveform and
injection IDs are contained in the new 10k core, but four of its older GPS blocks overlap the new
frozen 3k validation split. `physical-scale-subsets` therefore constructs fresh, strictly nested
2k/5k/10k prefixes from the SNR-quota 10k manifest, balanced jointly by source family and assigned
SNR bin, and reruns the injection/waveform/GPS split audit against the shared validation manifest at
every scale. All scale points start from the identical analytic checkpoint
`61730b9734a90fd01e4678470026cacc8c3e78cdf008e68cbcaf88ebd3ae8e72`; using a physical 2k
checkpoint as the 10k initializer is prohibited because it confounds initialization with prior
physical exposure.

The first executable curve is now frozen. Native 10k empirical SNR has median 3.847 with 51.9%
below 4; the train-only quota manifest assigns exactly 4,000/3,500/2,000/500 rows to
4--8/8--15/15--30/30--50 and has SHA256
`bf6575b14a5817f7d0b916b18f5f89e3f125479a85c73343c4b627357c0c0590`. Nested 2k/5k/10k
manifest hashes are `e9027c25bfa252e323727725a7af9801b30717af48c7cf9d6bdc570f60f1d62c`,
`445398681546bcdf3511b3ad9c76cbdba174318a32885400ffb899e39a39277e` and the full-quota hash.
All three pass zero-overlap injection/waveform/GPS audits against the frozen 3k validation corpus.
All three already saturate the current 76 train GPS blocks, so this curve isolates waveform scale;
it cannot answer the separate run/GPS-domain scale question.

The physical tensor target explicitly retains every IFO and Q-plane component mask. `MultiIFOQNet`
predicts two classes for each input plane, so chirp supervision has shape
`[batch, 1, IFO*Q, frequency, time]`; it must not be collapsed before the plane-wise loss. A
pixelwise IFO/Q union is a derived network-level mask for trigger clustering and catalog display,
not a replacement for the preserved component masks. Temporal localization alone collapses the
IFO/Q and frequency axes while retaining time.

An official O4a acquisition plan was also frozen with seed 20260720. The API exposed 3,309 aligned
H1/L1 4-kHz files; 800 disjoint 4,096-second pairs were selected over GPS
1368268800--1389408256. This is 3,276,800 seconds (37.93 raw coincident detector-days) before DQ,
known-event, context and category exclusions, leaving margin for the pre-registered >=30-day usable
target. Plan SHA256 is `d9043337438db689b581bade1922c1191ed52fde94ce056d460c4c9e74316d04`.
It is a development acquisition plan, not measured live time or a search result. O4b remains locked.

The weak-mask gate is executable rather than rhetorical. `gravityspy-mask-audit-plan` samples only
the frozen numeric validation split, deterministically stratified by morphology. Each task requires
an odd panel of at least three independent annotators blinded to the metadata-derived mask, so
pixelwise majority consensus has no ties. `gravityspy-mask-audit-evaluate`
hash-verifies NPZ masks and reports inter-annotator IoU, weak-versus-consensus IoU, per-label
agreement and Wilson intervals. Only that narrow weak-mask agreement claim is enabled by a completed
audit; segmentation, deglitch and search claims remain separate locked evaluations.
