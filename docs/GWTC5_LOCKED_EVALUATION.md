# GWTC-5 locked evaluation protocol

## Current public-data state

GWTC-5.0 and O4b strain became public in May 2026. The official catalog paper reports 150 new O4b
candidates with `p_astro >= 0.5`; the cumulative population analysis uses 267 mergers. The public
candidate release contains a 1.3 MB search summary and a 1.5 GB archived-search product, while the
event-level PE release is tens of GB. The population release reports BBH support from 2.5 to 200
solar masses, features near 10 and 35 solar masses, unequal-mass structure above 40 solar masses,
and a rapidly spinning subpopulation near dimensionless spin 0.7.

Primary sources:

- [GWTC-5 observations paper](https://arxiv.org/abs/2605.27225)
- [GWTC-5 population paper](https://arxiv.org/abs/2605.27226)
- [GWTC-5 candidate data release](https://doi.org/10.5281/zenodo.20348004)
- [GWTC-5 population data release](https://doi.org/10.5281/zenodo.20292639)
- [GWTC-5 GWOSC documentation](https://gwosc.org/GWTC-5.0/)

Public availability does not turn O4b into development data. This repository keeps event-level O4b
strain, candidate tables and PE samples locked until the following items are frozen on O1--O3/O4a:

1. code commit, environment and preprocessing hashes;
2. primary architecture, checkpoint selection rule and at least five seeds;
3. raw and cleaned ranking definitions and validation-only thresholds;
4. target FAR, coincidence/cluster rule, veto categories and time-slide schedule;
5. injection population, waveform-systematics set and `<VT>` weighting convention;
6. all primary/secondary endpoints, uncertainty method and failure criteria.

Aggregate published population facts may be used now to design coverage strata. Event names,
event-level parameters, candidate ranks and O4b strain may not be used to choose the model.

The unopened inventory contract must be produced by `gwtc5-locked-corpus-freeze`, not the generic
evaluation-corpus freezer. It binds the exact `locked_suite_v2` configuration, GWTC-5.0/O4b/test
identity, at least 3,000 unique injections and waveforms, every required detector subset, source
family and stress stratum, while rejecting score/result fields. The freeze reads score-blind
manifest metadata only and records zero strain rows read. The validation ledger also checks that
the predeclared access-log path is still absent at audit time; a stale unopened report cannot pass
after the one-time opening.

The first upstream artifact is the exhaustive score-blind detector-availability inventory. It is
created by `gwtc5-locked-availability-plan`, a deliberately separate command from the development
`gwosc-run-plan`. It queries only the official GWOSC O4b strain-file listing, groups aligned file
metadata by GPS start, records which of H1/L1/V1 are available and proves that all four predeclared
detector subsets can be scheduled. It never requests an HDF5/GWF file, a catalog endpoint, an event
record or a score. The report records zero downloaded strain files, zero strain bytes and zero
strain rows read. Both the command and its replay fail once the exclusive access-log path exists.

```bash
scripts/run_gwtc5_locked_availability_plan.sh \
  configs/locked_evaluation_suite_gwtc5.yaml \
  /artifacts/gwtc5-locked-access.json \
  /artifacts/gwtc5-score-blind-availability
```

This availability inventory is necessary but not sufficient for `locked_corpus_unopened`.
`gwtc5-locked-injection-plan` deterministically binds the frozen proposal population and physical
stress scenarios to every GPS/IFO block exactly once. The machine-readable population contract is
[`configs/gwtc5_locked_injection_population.yaml`](../configs/gwtc5_locked_injection_population.yaml).
It fixes the family fractions and minimum counts, source-frame mass/spin/distance proposals,
detector-subset minima, coalescence context, calibration scenarios, waveform alternatives and the
post-access strain-only glitch assignment rule.

Stress labels are recomputed, not trusted. `missing_detector` requires the exact available IFO count
to be below three; `high_mass_unequal_mass` requires its mass and mass-ratio inequalities;
`high_spin_precessing` requires a precessing approximant plus a sufficiently large three-dimensional
in-plane spin. Calibration and waveform-systematics rows must replay an exact configured scenario or
primary/alternative pair. A glitch-overlap row freezes a hash assignment key, uses no auxiliary veto
and must retain an explicit unavailable result without replacement if no post-access strain-only
candidate exists. Untagged rows are explicitly `nominal` rather than being decorated with all stress
names.

The current policy schedules every frozen availability block once and requires at least 4,000
pre-access attempts and 3,000 usable post-DQ injections. Pre-access `<VT>` weights are forbidden:
they are computed only after opening from the analyzed post-DQ live time and the stored comoving-
volume/source-frame-time proposal density. DQ-invalid attempts are retained and cannot be replaced.
Availability counts alone are therefore never reported as injection count or analyzed live time.

The frozen waveform-systematics alternatives use preceding Phenom generations. This is a
within-family generation-shift audit, not an EOB-versus-Phenom claim. LALSuite's official
SEOBNRv5/SEOBNRv5HM ROM files require more than 2.6 GB of external assets; they are deliberately
excluded until a separately hashed `LAL_DATA_PATH` asset contract is added, rather than being
silently assumed present.

Before the unopened report can pass, all primary and alternative approximants are sampled by
family/approximant stratum and compared between PyCBC and the matching direct LALSimulation FD/TD
API, including
the three-dimensional spins used by the precessing stratum. The isolated runtime is pinned by
[`requirements-waveforms.txt`](../requirements-waveforms.txt); its setup receipt records the exact
requirements hash, PyCBC/LALSuite versions and complete `pip freeze` hash.

```bash
scripts/setup_waveform_runtime.sh /path/to/base/python /artifacts/waveform-runtime

TASK_PYTHON=/path/to/control/python \
WAVEFORM_PYTHON=/artifacts/waveform-runtime/venv/bin/python \
WAVEFORM_RUNTIME_RECEIPT=/artifacts/waveform-runtime/waveform_runtime_receipt.json \
scripts/run_gwtc5_locked_injection_plan.sh \
  /artifacts/gwtc5-score-blind-availability \
  configs/locked_evaluation_suite_gwtc5.yaml \
  configs/gwtc5_locked_injection_population.yaml \
  /artifacts/gwtc5-locked-access.json \
  /artifacts/gwtc5-locked-injection-contract
```

The final freeze replays eight artifacts: injection manifest, its producer report, waveform runtime
validation, the exact isolated-runtime receipt, availability manifest/report, population config and
suite config. Any file mutation, fabricated stress label, premature `<VT>` weight, post-DQ
replacement policy or live access-log path makes `locked_corpus_unopened` fail.

## Predeclared GWTC-5 endpoints

The primary search claim is not catalog-image hit rate. It is paired raw-versus-mask-cleaned
injection recovery at common frozen FAR, reported as `<VT>` and distance/SNR/source-family efficiency
with paired bootstrap intervals. Continuous O4b background and predeclared nonzero time slides supply
FAR/IFAR exposure.

Secondary locked endpoints are:

- recovery of public candidates at the already frozen threshold, with own-search FAR attached;
- missed-event and false-trigger morphology, retaining every instance and mask;
- BBH strata spanning 2.5--200 solar masses, 10/35-solar-mass features, unequal high-mass systems,
  high-spin/precessing stress cases, BNS and NSBH transfer;
- latency, GPU/CPU throughput, calibration and OOD/failure flags;
- AMPLFI/DINGO within-backend posterior coverage, width, bias and end-to-end latency changes on
  identical raw/cleaned events, without an absolute backend ranking when native priors differ;
- glitch-overlap recovery, removed glitch energy and injected-chirp projection retention.

Candidate recovery is descriptive. It cannot replace background exposure or injection sensitivity,
and a miss is retained rather than removed from the event set.

The machine-readable contract is
[`configs/locked_evaluation_suite_gwtc5.yaml`](../configs/locked_evaluation_suite_gwtc5.yaml).
It fixes the primary metric, 0.1/year target FAR, at least 23.02585093 years of equivalent test
background, at least 3,000 paired injections, at least 100 PE injections, 10,000 paired bootstrap
replicates and seed 20260722. It also names every output path before any locked score is read.
The contract separately freezes fourteen post-access intermediate paths: raw/mask time-slide and
injection-ranking reports, the OOD source/score manifest and score report, DINGO/AMPLFI source
batches, and the catalog numeric source, candidate manifest/report plus normalized prediction
manifest/report. The
exclusive access receipt must additionally hash the two validation calibrations, raw/mask
comparison, OOD report, PE promotion and catalog metadata. Thus a final report cannot be made valid
by swapping an intermediate test artifact after opening the corpus.

The sequence is fail-closed:

```bash
python -m gwyolo.cli locked-evaluation-suite-freeze \
  --validation-evidence-report /artifacts/validation-evidence-ready.json \
  --config configs/locked_evaluation_suite_gwtc5.yaml \
  --output-root /artifacts/gwtc5-locked-final \
  --code-commit "$(git rev-parse HEAD)" \
  --output /artifacts/gwtc5-locked-suite-plan.json

# evaluation-corpus-open-once must include
# --artifact locked_suite_plan=/artifacts/gwtc5-locked-suite-plan.json and must
# predeclare the plan's suite_receipt output. It may run only after all validation gates pass.
```

After the exclusive access receipt exists,
[`scripts/run_locked_search_endpoints.sh`](../scripts/run_locked_search_endpoints.sh) applies the two
validation-frozen thresholds and writes the paired raw/mask `<VT>` endpoint. The script obtains
all endpoint settings and output paths from the frozen plan; it accepts no replacement test
threshold. Once OOD, PE and catalog endpoints are also complete,
`locked-evaluation-suite-finalize` hashes all eight outputs into the predeclared suite receipt.
The locked-final publication ledger must then pass all nine requirements before interpretation.

`ood-abstention-evaluate-locked` similarly accepts only a detector-set, logit-energy validation
report whose known-only threshold, checkpoint and score manifests replay by hash. It rejects any
O4b glitch/GPS overlap with either OOD validation role, requires explicit detector availability,
and reports known false abstention and unknown false acceptance overall and by family, run and
detector subset without refitting the threshold.
`glitch-ood-score-frozen` is the only locked score producer: it reloads the exact validation
checkpoint/config, preserves explicit H1/L1/V1 availability, writes no threshold, and records that
test scores were not used for model, threshold or score-method selection. The source manifest,
score manifest and score report are all separately predeclared and automated by
[`scripts/run_locked_ood_endpoint.sh`](../scripts/run_locked_ood_endpoint.sh).

The catalog endpoint has an equally explicit producer boundary. `catalog-predict-locked` accepts
only the predeclared numeric source and unthresholded candidate manifest/scoring report, replays the
frozen model, config, source tensor and catalog metadata identities, and groups every candidate,
instance and mask into one row per event without applying a catalog threshold.
`catalog-eval-locked` then replays that producer report before applying the already frozen own-search
threshold. The resumable wrapper is
[`scripts/run_locked_catalog_endpoint.sh`](../scripts/run_locked_catalog_endpoint.sh).

For PE, `pe-backend-bind-locked` accepts completed DINGO or AMPLFI test batches only after the
matched-event portfolio promotion report passes. It requires each locked batch to preserve its own
validation-fixed backend version, model, prior, waveform, detector set, calibration, hardware,
latency scope and sky-area estimator. `pe-robustness-portfolio-evaluate-locked` then requires
identical test injection triplets and common source bytes/truth across both backends, while computing
coverage, bias, posterior width, sky area, effective-sample rate and latency changes separately
inside each fixed backend. Its output permanently forbids an absolute DINGO/AMPLFI ranking. The
resumable wrapper is
[`scripts/run_locked_pe_endpoints.sh`](../scripts/run_locked_pe_endpoints.sh).

## Promotion decision

The current model is not ready to unlock O4b. Its first three real-noise physical injections produced
high ranking for one BBH and one NSBH but near-zero ranking for one BNS. The continuous dual-IFO
background exposure is only 768 seconds per validation/test split, and multi-seed validation is still
running. The correct next action is to enlarge independent O4a background, validate the population
and waveform systematics, complete fixed-FAR learned-mask comparisons, and write the immutable freeze
record before any event-level GWTC-5 evaluation.
