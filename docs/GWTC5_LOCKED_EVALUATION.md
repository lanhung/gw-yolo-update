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
- AMPLFI/DINGO posterior coverage, width, bias and end-to-end latency on identical raw/cleaned events;
- glitch-overlap recovery, removed glitch energy and injected-chirp projection retention.

Candidate recovery is descriptive. It cannot replace background exposure or injection sensitivity,
and a miss is retained rather than removed from the event set.

The machine-readable contract is
[`configs/locked_evaluation_suite_gwtc5.yaml`](../configs/locked_evaluation_suite_gwtc5.yaml).
It fixes the primary metric, 0.1/year target FAR, at least 23.02585093 years of equivalent test
background, at least 3,000 paired injections, at least 100 PE injections, 10,000 paired bootstrap
replicates and seed 20260722. It also names every output path before any locked score is read.
The contract separately freezes eight post-access intermediate paths: raw/mask time-slide and
injection-ranking reports, OOD scores, DINGO/AMPLFI source batches and catalog predictions. The
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

For PE, `pe-backend-bind-locked` accepts completed DINGO or AMPLFI test batches only after the
validation promotion report passes. It requires the locked batch to preserve the validation-fixed
backend version, model, prior, waveform, detector set, calibration, hardware, latency scope and
sky-area estimator. `pe-robustness-joint-evaluate-locked` then requires identical test injection
triplets in both backends and evaluates coverage, bias, posterior width, sky area, effective sample
rate and latency with the suite's frozen credible level and bootstrap seed. The resumable wrapper is
[`scripts/run_locked_pe_endpoints.sh`](../scripts/run_locked_pe_endpoints.sh).

## Promotion decision

The current model is not ready to unlock O4b. Its first three real-noise physical injections produced
high ranking for one BBH and one NSBH but near-zero ranking for one BNS. The continuous dual-IFO
background exposure is only 768 seconds per validation/test split, and multi-seed validation is still
running. The correct next action is to enlarge independent O4a background, validate the population
and waveform systematics, complete fixed-FAR learned-mask comparisons, and write the immutable freeze
record before any event-level GWTC-5 evaluation.
