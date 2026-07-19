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

## Promotion decision

The current model is not ready to unlock O4b. Its first three real-noise physical injections produced
high ranking for one BBH and one NSBH but near-zero ranking for one BNS. The continuous dual-IFO
background exposure is only 768 seconds per validation/test split, and multi-seed validation is still
running. The correct next action is to enlarge independent O4a background, validate the population
and waveform systematics, complete fixed-FAR learned-mask comparisons, and write the immutable freeze
record before any event-level GWTC-5 evaluation.
