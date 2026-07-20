# Data scaling and domain-coverage plan

## Decision

Data scale and physical coverage are now the program's highest-priority bottleneck. The current corpus is sufficient only for a legacy image-segmentation proof of concept. Architecture work beyond a compact baseline must not be treated as the main path until the project has measured a group-safe learning curve and built a substantially larger evaluation set.

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
- the continuous background is one 4,096-second O4a file and 31 short nonzero shifts;
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
Fletcher32/statistics/DQ scan before recording it. Both commands reject O4b development access.

## Storage and compute strategy

At the current JPEG size, 100k plots would be only several GB, but RGB plots discard physical information. A float32 tensor with several Q planes and three IFOs can consume hundreds of GB per 100k scenes. Prefer:

- float16 or compressed chunked arrays for locked sets;
- on-the-fly generation for training;
- cached PSDs and transforms;
- deterministic recipe manifests instead of duplicated tensors;
- streaming dataloaders and resumable shards.

For injection sensitivity, count and storage targets must be expressed separately. The current 25k
H1 recipe artifact is only a provenance pilot and is too correlated (192 windows, six GPS blocks).
The publication corpus should target at least 10k independently seeded validation injections and
50k--200k locked test injections across BBH/BNS/NSBH strata, but the decisive increase is independent
GPS background blocks and source-population coverage, not repeated injection rows on the same few
seconds. Run a group-safe scaling matrix over injection count, unique GPS live time, glitch overlap,
SNR/distance, source family, IFO network and observing run. Stop increasing count only when bootstrap
uncertainty on paired `<VT>` improvement is below the predeclared effect size and performance has
plateaued across at least two successive scale points.

On the current RTX 4090 D, a 100k-scene YOLO26m experiment is expected to be an order of roughly one day rather than minutes, before multi-Q generation overhead. Measure generator throughput before committing to a 1M-scene run.

## Promotion gates

Architecture work is promoted from exploratory to primary only after:

1. DS0 evaluation data are frozen;
2. at least the 250→10k learning curve is complete;
3. the manifest reports independent waveform, glitch, GPS, IFO, and run counts;
4. O4 transfer is measured separately from in-domain mAP;
5. label audits cover mask consistency and hard-negative contamination.

A paper claim requires DS3-scale coverage or an evidence-based learning-curve argument for a smaller corpus, plus continuous-background search exposure. Dataset size alone never substitutes for FAR/`<VT>`.
