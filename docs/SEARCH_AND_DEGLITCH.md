# Search-statistics and mask-deglitch evidence

## Continuous-background protocol

`gwyolo background-plan` first requires a passing `gwosc-verify` report and checks every detector
file SHA256 against it. It then intersects per-second DQ masks across detector files, excludes event or
hardware-injection intervals, forms numeric windows, and assigns entire coarse GPS blocks to
train/validation/test. Live time is the union of intervals, not the sum of possibly overlapping
windows. It also removes edge windows that cannot supply the declared full whitening context
(64 seconds by default), so an accepted row cannot later fail solely because preprocessing extends
beyond the source file. The resulting manifest is the input contract for trigger generation and time slides.
DQ and hardware-injection safety are checked over that entire whitening context, not merely the
central 8-second analysis crop. This prevents a nominally valid crop from importing NaNs or an
invalid detector state through its PSD context. Non-finite whitening inputs, features or model
probabilities are hard failures, and JSON provenance refuses NaN serialization.

The first real O4a H1 pilot used the 4096-second file surrounding GW231123_135430, 8-second
non-overlapping windows, 256-second split blocks, and excluded event GPS ±16 seconds. It produced:

| Quantity | Value |
|---|---:|
| Valid windows | 507 |
| Independent GPS blocks | 16 |
| Train live time | 2,520 s |
| Validation live time | 768 s |
| Test live time | 768 s |
| Cross-split block overlap | 0 |
| Manifest SHA256 | `ee332c704c1dcb8b35ec6f24e70061eb1185930f2e5d815d4f14e49c425de03c` |

This exposure is only an integration pilot. Even with zero surviving test triggers, 768 seconds gives
a 90% Poisson FAR upper limit of roughly 94,615/year. It cannot support an astrophysical FAR or IFAR
claim. Publication evaluation needs continuous O4 segments and time slides totaling years to hundreds
of years, depending on the target IFAR.

The next real-data step is to repeat the plan on the aligned H1+L1 intersection, then generate and
cluster triggers. O4b remains locked; O4a provides development backgrounds only.

`gwyolo trigger-score` now stores each IFO's maximum chirp/glitch probability and the time-bin/GPS
location of each peak. `gwyolo time-slide-background` can then build nonzero, non-cyclic H1/L1
slides independently for validation or test, optionally requiring the shifted peaks to fall inside a
fixed coincidence window. Equivalent exposure is the union of valid reference intervals within each
slide and the sum across slides; rejected peak coincidences reduce trigger count, not exposure. A
hand-calculated test covers both ranking and exposure.

Background scoring checkpoints atomically every five windows under a manifest/checkpoint/config and
preprocessing identity. Restarts reuse only matching window IDs. Any unreadable accepted window is
reported and forces a nonzero exit, rather than quietly shrinking the declared live time.
With `--save-probabilities`, the run identity covers a versioned artifact containing hash-checked
float16 masks plus float32 whitened analysis strain and its sample rate. `candidate-extract` keeps
every contiguous per-IFO chirp cluster. The mask supplies a broad region proposal; the saved strain
then refines every cluster from its own local envelope at sample resolution. A 96-bin mask is thus no
longer misrepresented as an 83 ms timing measurement. Sample resolution alone is still not an error
bar, so every candidate remains uncalibrated until the exact local-envelope method is measured on
validation injections.
`candidate-time-slides` pairs every retained H1/L1 cluster after a non-cyclic shift, applies an exact
peak-time coincidence, clusters nearby network events by loudest ranking statistic, and computes
exposure from all paired DQ-safe windows—including windows with no candidate. This fixes the earlier
one-maximum-per-window counting contract. Exposure additionally requires the contributing detector
to be available on each side of a pair, so missing-IFO windows cannot inflate live time. The timing
gate requires a predeclared light-travel limit, a validation-calibrated timing allowance, <=10 ms
candidate resolution and one consistent calibration hash.

Resolution is not accuracy. The timing calibration now also requires its frozen empirical error
quantile to be at most the predeclared `--maximum-empirical-timing-uncertainty-seconds` (10 ms by
default). The first fixed-update candidate diagnostic had 0.9766 ms numeric resolution but a 99th
percentile error of 245.8 ms and only 368/6000 arrivals matched within ±250 ms; its implied 501.6 ms
H1–L1 coincidence and 3/3000 recovered injections are therefore rejected by this new hard gate.
The 10k/30-epoch checkpoint improves the within-250-ms association count to 464/6000 but still has
246.7 ms p99 error; it is likewise stopped before slides. More epochs improve window sensitivity,
not the missing temporal representation.

The independent 2k time-domain detector-arrival head also fails the unconditional gate: its selected
checkpoint has 1.769 s median and 4.626 s p90 error over all 6,000 validation arrivals. Its high-SNR
tail is informative but not sufficient (network-SNR >=30 p90 is 12.68 ms). Because more than half
of the physical validation injections have network SNR below 4, `detector-arrival-timing-validation-stratify`
now reports per-IFO detectability, worst-IFO error and pairwise-delay error without replacing the
all-injection metric. Candidate-level coverage and timing at a frozen morphology threshold remain
the only route to a search timing claim.

The paired-visibility audit makes the remaining failure concrete: among 275 injections with both
H1 and L1 optimal SNR >=8, only 56.73% have both predictions within 10 ms and worst-IFO p90 is
0.875 s. At both-IFO SNR >=10, worst-IFO p90 is still 26.77 ms. These results forbid promotion based
on the sub-10-ms conditional median; the tail, pairwise delay and retained-candidate coverage must
all pass together.

If the frozen v3 spectrogram arm fails, the next timing unit is a candidate instance, not a whole
window. The fallback design is predeclared as follows:

1. retain every connected chirp/glitch proposal from the frozen mask model, including overlapping
   and low-score proposals needed to measure proposal recall;
2. for each `(window, IFO, proposal)` construct a numeric high-resolution spectrogram crop plus an
   explicit proposal-support channel and detector-availability mask;
3. predict an arrival heatmap and an abstention probability per proposal, so multiple arrivals and
   unknown glitches are not collapsed to one window-level argmax;
4. train positives from group-disjoint physical injections/real-glitch overlaps and negatives from
   held-family/OOD glitch proposals, with no auxiliary-channel veto of strain-coherent events;
5. report proposal coverage, conditional timing, worst-IFO and pairwise-delay tails together, then
   propagate every miss into the frozen-threshold `<VT>` denominator.

A good conditional timing number cannot repair poor proposal coverage, and an abstaining proposal
cannot be silently counted as a recovered injection.

`candidate-proposal-audit` is the executable gate for step 1. It joins the locked physical-injection
manifest to every retained per-IFO candidate, keeps zero-proposal arrivals in the denominator, and
reports raw interval coverage plus a separately declared padding coverage by family, SNR stratum and
IFO. To prevent a full-window proposal from manufacturing perfect coverage, it also reports proposal
interval-union fraction and the narrowest truth-containing interval width. It deliberately labels the
result proposal support rather than search recall; it consumes no background threshold and makes no
FAR/`<VT>` claim.

The proposal-threshold Pareto gate is frozen in
`configs/candidate_proposal_threshold_selection.yaml`. A threshold qualifies only with >=95% padded
coverage overall, >=90% in every BBH/BNS/NSBH and SNR>=4 stratum, median/p90 proposal-union fractions
<=0.50/0.80, and median truth-containing width <=2 s. `candidate-proposal-sweep-select` verifies that
all audits share one injection manifest, scoring checkpoint, preprocessing config and trigger
manifest. If no threshold passes every check, no proposal threshold is selected.

The frozen 0.3--0.9 sweep selects none. The decisive crossing is between 0.5 (98.48% coverage but
0.781 median union fraction and 3.17 s median containing width) and 0.6 (1.08 s median containing
width but 92.32% overall coverage, 0.531 median union fraction and failed BNS coverage). Since both
coverage and active support are monotone in a common threshold, a finer scalar threshold cannot
repair this conflict. Candidate timing remains blocked while an independent dense endpoint proposal
objective is trained; the chirp/glitch instance masks remain required outputs and are not replaced by
that proposal map.

`detector-endpoint-proposal-train` implements the first bounded repair arm. It warm-starts the
validated v3 numeric-spectrogram encoder, replaces the single categorical interpretation with one
sigmoid endpoint heatmap per IFO, and retains every disconnected component above a calibrated
threshold. Its dense target builder accepts multiple endpoints per IFO even though the initial
group-safe 2k corpus contains one injection per window. The source segmentation checkpoint is never
loaded or modified, so chirp/glitch masks remain an independent mandatory product. Checkpoint
selection uses validation dense focal-BCE loss; only afterward does the fixed 0.05--0.9 threshold
grid face the unchanged proposal coverage--compactness gate. The command consumes no test rows and
cannot make a search claim even if that validation gate passes.

The executable timing path is now ordered and leakage-safe:

1. `injection-arrival-annotate` adds PyCBC geometric Earth-center-to-detector delays to an existing,
   hash-verified physical-injection manifest;
2. validation injections and validation background are scored with `--save-probabilities`;
3. `candidate-timing-calibrate` freezes the predeclared error quantile on validation injections only,
   selecting at most the nearest candidate per `(injection, IFO, method)` and requiring the error
   quantile—not just array resolution—to pass the predeclared accuracy limit;
4. `candidate-timing-apply` hash-links the calibration to background and injection candidates;
5. time slides use exactly `physical delay + 2 × per-detector uncertainty`, while
   `injection-candidate-rank` retains misses with score zero in the `<VT>` denominator.

The resulting report is still non-claimable until validation/test provenance is frozen, GPS blocks
are disjoint, and enough nonzero shifts support the requested IFAR.

For the multi-day acquisition, temporary arrays and public HDF files have separate certified
lifetime gates. After `candidate-extract`, run `candidate-probability-evict` with its exact score and
candidate reports. After all non-empty validation/test splits from one stable-hash shard have
candidate reports, run `background-source-evict` with the shard's batch and background reports.
Each command first writes a hash-bound intent file and refuses an existing output path. This permits
bounded streaming without weakening the retained candidate, detector-duty, GPS-block or source
provenance needed by time slides and locked evaluation.

The resumable wrapper is:

```bash
python -m gwyolo.cli gwosc-plan-shard \
  --plan o4a-parent-plan.json --shard-index 0 --pairs-per-shard 1 \
  --output shard-000/acquisition-plan.json
python -m gwyolo.cli background-stream-shard \
  --parent-plan o4a-parent-plan.json --shard-index 0 --pairs-per-shard 1 \
  --event-exclusions o4a-event-exclusions.json \
  --timing-calibration-report frozen-validation-timing.json \
  --checkpoint model.pt --config experiment.yaml \
  --coherence-config configs/physics_coherent_yolo_pilot.yaml \
  --cache-root artifacts/bounded-gwosc-cache --output-dir shard-000
```

The wrapper itself creates and verifies the child plan, so the first command is useful for audit or
distributed scheduling but is not required before the second. The timing calibration must match the
scoring checkpoint, config and code commit exactly; a calibration from an older code snapshot is a
hard failure rather than an implicitly reused error bar.

The validation-only candidate pipeline applies the same bounded-storage rule internally. It scores
and extracts background candidates first, certifies and releases those probability arrays, then
scores injections, freezes the timing calibration, certifies and releases the injection arrays, and
only afterward builds calibrated candidates, slides and the frozen threshold. Both eviction stages
are resumable identity checkpoints, so a later failure cannot cause a deleted array to be silently
treated as a reusable scorer output.

Use `background-stream-merge --shard-report ... --output-dir ...` after a tranche. The merge refuses
overlapping acquisition ranges, repeated GPS windows/candidate IDs, cross-split GPS blocks, mixed
checkpoint/config/commit/timing identities, or a calibrated-candidate hash mismatch. Its report
distinguishes a complete parent plan from a partial exposure tranche before any time-slide command
is allowed to consume the combined manifests.

The provenance path is transitive rather than name-based. Candidate extraction verifies the adjacent
score report and carries checkpoint/config/commit hashes. Timing application succeeds only when the
validation calibration and target candidates came from that exact scoring identity. Time-slide and
injection-ranking reports require one common calibration, checkpoint, config and commit. Finally,
`candidate-search-calibrate` reads validation reports only, and
`candidate-search-evaluate-frozen` has no threshold argument, rejects any validation/test GPS,
injection or waveform overlap, refuses to overwrite an existing locked result, and reports FAR,
IFAR and weighted `<VT>` with bootstrap uncertainty. An empty background-candidate list freezes a
threshold above probability support; it can never turn score-zero injection misses into detections.

The intended H1/L1 sequence is:

```bash
python -m gwyolo.cli injection-arrival-annotate --manifest val.jsonl --output-dir val-arrivals
python -m gwyolo.cli injection-score --manifest val-arrivals/materialized_injections_arrivals.jsonl \
  --checkpoint model.pt --config experiment.yaml --output-dir val-score \
  --required-split val --save-probabilities --coherence-config configs/physics_coherent_yolo_pilot.yaml
python -m gwyolo.cli candidate-timing-calibrate \
  --injection-triggers val-score/injection_triggers.jsonl --output timing-calibration.json
python -m gwyolo.cli candidate-search-calibrate \
  --validation-time-slide-report val-slides/val_candidate_time_slide_report.json \
  --validation-injection-ranking-report val-rank/val_injection_candidate_ranking_report.json \
  --target-far-per-year 1 --output frozen-candidate-threshold.json
python -m gwyolo.cli candidate-search-evaluate-frozen \
  --calibration-report frozen-candidate-threshold.json \
  --test-time-slide-report test-slides/test_candidate_time_slide_report.json \
  --test-injection-ranking-report test-rank/test_injection_candidate_ranking_report.json \
  --minimum-test-live-time-years 10 --minimum-test-injections 20000 \
  --output locked-candidate-search.json
```

Intermediate extraction, calibration application, slide and ranking commands are deliberately kept
separate so their JSON reports can be audited before the locked command receives access to test data.
For routine O4a development, `candidate-search-validation-pipeline` executes those validation-only
stages in order, resumes the two expensive scorers by run identity, selects only the calibrated local
per-cluster strain method, derives the coincidence width from the configured detector-pair light
travel time and measured uncertainty, and writes a frozen threshold artifact with
`test_evaluation: null`. It has no test-manifest argument. The separate locked command remains the
only route to test FAR/IFAR/`<VT>`.

The initially downloaded H1 file subsequently failed a complete HDF5 Fletcher32 scan, while L1
matched all 16,777,216 samples, official strain statistics and DQ/injection bit sums. Consequently,
all H1 and aligned H1+L1 background, trigger and time-slide numbers below are retained only as
corruption-detection integration evidence and are excluded from scientific comparisons. The source
gate now prevents such a file from entering a new background manifest.

The invalidated aligned H1+L1 run scored all 96 validation and all 96 test windows; failures were confined
to 8 edge windows whose 64-second context exceeded the file and 11 training-window HDF reads. Thirty
one nonzero 8-second slides per split produced 11,904 seconds (`3.77e-4 yr`) of equivalent exposure,
with 50 validation and 47 test peak coincidences at a 0.1-second integration window. This is only
3.3 hours, so even zero events would have a 90% FAR upper bound of roughly 6,100/year. It verifies
bookkeeping, not an astrophysical IFAR.

## Injection and `<VT>` recipe contract

`gwyolo injection-plan` samples BBH/BNS/NSBH families, sky/orientation, component masses and spins,
while inheriting the background window's split and GPS block. The current planner uses a flat
Lambda-CDM grid with Planck 2018 `H0=67.66 km/s/Mpc` and `Omega_m=0.3111`, samples uniformly in
comoving volume, transforms source-frame masses to detector-frame masses, and applies the
`1/(1+z)` source-time factor to each `Mpc^3 yr` weight. Cosmology and the proposal measure are stored
in the report. This replaces the earlier Euclidean-distance pilot, which is retained only as a
historical integration artifact. The values follow the
[Planck 2018 cosmological-parameter result](https://doi.org/10.1051/0004-6361/201833910).

The earlier H1 integration pilot generated 5,000 validation and 20,000 test recipes:

- 11,250 BBH, 7,500 BNS, and 6,250 NSBH;
- 25,000 unique injection and waveform IDs;
- zero validation/test injection-ID overlap;
- manifest SHA256 `6a7a280f77c1b949a99250f9ba34f0227a56afdd3545ce758ee5916c2887084f`.

This is not a valid sensitivity corpus and must be regenerated with the cosmological planner. It
reuses only 192 background windows from six GPS blocks, and every old row deliberately says
`waveform_backend=unassigned_requires_lal_or_validated_equivalent`. The historical recipe validates
cardinality and split provenance only.

`gwyolo injection-materialize` is now the physical next stage. It reads the real detector strain for
the referenced GPS window, asks PyCBC/LALSimulation for a time-domain approximant, projects plus and
cross polarizations into every IFO with sky-dependent antenna response and arrival delay, aligns the
result to the detector sample grid, and atomically stores noise, signal and mixture numeric arrays.
It stores a 64-second context by default together with exact indices for the central 8-second analysis
window; this prevents the injected signal from defining a PSD from only its own short window.
The default `signal_only` mode stores the projected waveform and a hash-addressed reference to each
GWOSC source instead of duplicating noise and mixture for every injection. `injection-score` verifies
each source hash and reconstructs the mixture before whitening. A `full` mode remains available for
small immutable debugging sets; it is not the scalable default.
Signal-only materialization stores strain as float32 (with dtype in run identity and reports), then
promotes it to float64 for reconstruction and whitening. This preserves the physical strain dynamic
range while roughly halving the dominant per-injection storage for 10k/50k scaling.
Materialization writes a run-identity state and an atomic partial manifest every ten injections. A
restart verifies every indexed artifact hash, rejects a changed manifest/config/backend, and resumes
without regenerating completed rows. Waveform-domain failures remain nonzero exits and are never
silently skipped.
The output records PyCBC/LALSuite versions, approximant, source-file hashes and waveform summaries.
The implementation follows the official PyCBC interfaces for
[waveform generation](https://pycbc.org/pycbc/latest/html/waveform.html) and
[detector projection](https://pycbc.org/pycbc/latest/html/pycbc.detector.html).

The materializer fails if PyCBC/LALSuite is absent. Without a passing external-reference
waveform-equivalence report it labels output `integration_only_unvalidated_backend`; an internal
smoke report is explicitly rejected as claim evidence. Even with an external report it does not
authorize a sensitivity claim. Before freezing the corpus, the broad pilot mass/spin and provisional
tidal proposal must be replaced or reweighted to documented GWTC population models, each selected
approximant must pass match/epoch/amplitude checks, and substantially more independent real
background must be available.

`gwyolo waveform-validate` makes that wrapper-level gate executable. It selects a deterministic,
family-stratified sample and compares PyCBC frequency-domain output against the direct
`LALSimulation.SimInspiralChooseFDWaveform` API for both polarizations. The gate checks array length,
phase-invariant complex overlap, relative L2 error, amplitude ratio and epoch, records both package
versions and full provenance, and exits nonzero on any failed case. This validates code-path and
parameter equivalence; it does not validate the provisional population model or detector projection.
The acceptance thresholds are overlap at least `0.999999`, relative L2 error at most `0.001`,
amplitude-ratio error at most `0.001`, and epoch error at most one nanosecond. The L2 tolerance
allows the documented last-bit solar-mass-constant difference to accumulate phase over long BNS
signals while the independent overlap, amplitude and epoch gates remain strict.

The first 300-row materialization attempt exposed a useful hard failure at row 114: a source-frame
NS mass drawn up to 2.5 solar masses can exceed the `IMRPhenomNSBH` detector-frame 3-solar-mass domain
after cosmological redshift. No failed row was skipped. The planner now limits the NSBH source-frame
NS mass to 2.2 solar masses, records the maximum detector-frame value, and fails the entire plan if
any NSBH row exceeds the approximant domain. The failed v2 attempt is retained as a negative result;
the corrected recipe is regenerated rather than patched row by row.

`gwyolo injection-score` verifies every materialized-array hash, downsamples and whitens the full
context, crops only the recorded central window, applies the frozen multi-IFO/multi-Q checkpoint, and
emits trigger rows carrying the original `vt_weight`. Thus the same validation-only threshold and
search-statistics code can later evaluate physical injections. These rows remain domain-transfer
diagnostics until the waveform equivalence, background exposure, population model and locked-test
gates all pass.

Scoring is resumable: every five completed injections it atomically checkpoints trigger rows and a
run identity covering manifest/checkpoint/config hashes, IFO/Q layout and probability-storage mode.
A restart verifies every saved probability hash before reuse. Any failed input produces a report and
a nonzero exit instead of silently accepting a partial corpus. When invoked with
`--save-probabilities`, the scorer stores hash-locked float16 masks plus float32 whitened analysis
strain so injection and background candidates use the same local timing algorithm. Existing strain
arrays need not be regenerated: `injection-arrival-annotate` verifies them and writes a new manifest
with geometric detector-arrival targets. Waveform-array end time is explicitly not used because the
post-coalescence ringdown makes it a biased timing proxy.
`gwyolo learned-deglitch` applies those frozen soft masks to the raw central strain and reports
per-IFO/network injected-signal projection retention, waveform change and post-clean signal error.
This closes the learned-mask execution path without inventing a clean real-noise counterfactual: its
report explicitly withholds a deglitch-benefit claim until targeted chirp+known-glitch mixtures and a
paired fixed-FAR search demonstrate both glitch removal and signal preservation.

## Fair raw-versus-cleaned comparison

`gwyolo search-compare` calibrates a separate validation-background threshold for raw and
mask-cleaned methods at the same target FAR. It then freezes both thresholds and evaluates the same
test background and the same weighted injections. Reports include Wilson efficiency intervals,
weighted-efficiency bootstrap intervals, paired recovered-`<VT>` change, and clean/overlap strata.

This interface is designed for the later AMPLFI/DINGO experiment: raw strain and mask-cleaned strain
must share events, injection weights, live time, waveform population, and FAR definition. A mAP
comparison is explicitly not part of the protocol.

For the primary locked result, `search-calibrate` reads only validation background and writes the
threshold plus its source hash. `search-evaluate-frozen` accepts that artifact, test background and
test injections, has no validation-data or threshold argument, and refuses to overwrite an existing
result. This makes the no-test-tuning boundary executable rather than merely documentary.
`search-validation-injections` separately measures family-stratified, weighted injection efficiency
and rejects every non-validation row, so development diagnostics cannot accidentally consume the
locked test injection corpus.

`physical-validation-endpoint` is the scale-selection bridge for the present short O4a development
background. Both scorers must be invoked with `--required-split val`; their reports now bind code
commit, exact command, environment, manifest/checkpoint/config hashes, and the split contract into
the resumable run identity. The endpoint hash-verifies both trigger artifacts, requires the same
checkpoint/config/commit, rejects duplicate physical injection or waveform IDs, derives exposure
from the union of scored GPS intervals, and freezes a threshold using only a predeclared maximum
number of surviving validation-background windows. It then reports overall and source-family
weighted injection efficiency with Wilson and bootstrap intervals. The nominal window FAR is marked
diagnostic-only: this bounded operating point is suitable for comparing 2k/5k/10k development
checkpoints, but it is not an astrophysical FAR/IFAR and cannot unlock O4b.
It also requires the originating physical-finetune report, re-hashes its checkpoint and config,
requires the scorer's injection manifest to equal the training validation manifest, and rejects any
training report whose `test_evaluation` is non-null.
When a background planner emits a combined train/validation/test JSONL, `manifest-select-split`
first writes a hash-addressed explicit split manifest and report. The scorer never silently filters a
mixed manifest, so its recorded input hash always denotes exactly the rows it evaluated.

For the frozen batch-8 O4a scale study the predeclared value is eight surviving validation windows
(approximately one percent of the 824-window validation background, subject to score ties). This
choice was fixed before examining any validation-injection endpoint. A typical final aggregation is:

```bash
python -m gwyolo.cli physical-validation-endpoint \
  --training-report artifacts/scale/physical_finetune_report.json \
  --background-score-report artifacts/endpoint/background/trigger_score_report.json \
  --injection-score-report artifacts/endpoint/injections/injection_score_report.json \
  --maximum-validation-false-alarms 8 \
  --bootstrap-replicates 10000 \
  --seed 20260720 \
  --output artifacts/endpoint/physical_validation_endpoint.json
```

After all seeds finish, `physical-validation-summarize` re-verifies every endpoint and underlying
trigger artifact, enforces identical scoring/training controls, reports seed mean/spread including
null or negative deltas, and computes paired injection-level recovered-`<VT>` bootstrap changes for
each adjacent scale and seed. Its promotion flag remains false until the equal-epoch checkpoints are
scored on the same endpoint and adequate clustered background exposure exists.

`mask-search-validation` freezes the deglitch promotion table before overlap results are examined.
Raw and mask-conditioned rankings receive separate thresholds from identical validation background
windows and the same allowed false-alarm count. Those thresholds are applied unchanged to
waveform-matched clean and contaminated arms. Development success requires both a paired clean
`<VT>` lower bound no worse than the predeclared one-percentage-point efficiency margin and a paired
contaminated gain of at least five percentage points with a positive bootstrap lower bound. The
report remains non-claimable until continuous clustered background, time slides and the locked
injection corpus are complete.

The full validation path is now executable as `mask-search-validation-pipeline`. It runs six score
arms from one frozen detector-set checkpoint: raw/mask-conditioned background, clean
raw/mask-conditioned injections, and real-glitch-contaminated raw/mask-conditioned injections.
Raw arms save every IFO/Q chirp and glitch probability. Cleaning suppresses
`glitch × (1−chirp)` only in the central analysis crop, stores a hash-locked numeric override, then
reinserts that crop into the original 64-second context before a fresh full-context whitening and
rescore. Thus raw and cleaned rankings do not accidentally use different PSD contexts.

`physical-overlap-contamination` converts network-aware real-glitch overlaps into this scorer
contract and emits the exactly waveform-matched clean manifest at the same time. Validation/test
overrides reject any train-only signal rescaling. `learned-background-deglitch` provides the
corresponding background overrides while hash-verifying every original source file. All
intermediate score, probability, override and report hashes are retained; each stage is resumable,
and the aggregate output still has `test_evaluation: null` and `promotion_allowed: false`.

A successful five-row engineering pilot is not a sensitivity result. The paper gate still needs a
statistically useful, group-independent contaminated corpus, clean non-inferiority, at least five
learned seeds, clustered continuous background/time slides and the one-time locked evaluation.

`gwyolo pe-evaluate` now provides the corresponding posterior-side contract. Its JSONL manifest
contains one `raw` and one `cleaned` row for every `(backend, injection_id)` pair, with an NPZ
posterior, the common truth dictionary, and measured end-to-end latency. The command rejects missing
pairs, duplicate conditions, truth mismatches, empty samples and non-finite samples. It records file
hashes, per-parameter bias/absolute bias, central credible width, mean absolute distance to truth,
90% coverage with Wilson intervals, and cleaning latency overhead. A typical invocation is:

```bash
python -m gwyolo.cli pe-evaluate \
  --manifest artifacts/pe/paired_manifest.jsonl \
  --output artifacts/pe/paired_report.json \
  --credible-level 0.9 \
  --bootstrap-replicates 10000 \
  --require-publication-provenance
```

The publication protocol uses `pe-robustness-evaluate` instead of the legacy two-arm diagnostic. It
requires clean, contaminated and mask-conditioned triplets and adds effective sample size, ESS/s,
90% sky area and strict cross-condition backend/prior/waveform provenance checks.

This adapter is backend-neutral: it does not pretend that a synthetic NPZ is an AMPLFI or DINGO
result. Publication comparisons still require both actual backends to use matched priors, waveform
conventions, detector data, injection truth and hardware, with the backend version and model hash
added to each manifest row. The paired adapter is an executable evaluation boundary, not evidence
that either external inference run has already been completed.

The publication gate requires matching `backend_version`, `backend_model_hash`, `prior_hash`,
`waveform_approximant`, `detector_set`, `calibration_version`, `source_event_hash`, `hardware`, and
`latency_scope` on each publication triplet (or each legacy raw/cleaned pair). The report adds
deterministic paired-bootstrap intervals
for absolute-bias changes, credible-width ratios and cleaning latency, plus paired coverage
transitions. Development manifests may omit the gate, but no paper comparison may do so.

`gravityspy-glitch-finetune` is the bounded real-glitch training boundary. It accepts only a frozen
train/validation pair with disjoint glitch and network-GPS-block identities, hash-verifies every
numeric sample, and samples train labels with inverse-frequency weights. A checkpoint is eligible
only if its fixed-threshold physical-chirp validation IoU retains a configured fraction (0.95 by
default) of the incoming checkpoint; among eligible epochs, validation glitch IoU is primary.
Threshold calibration happens only after checkpoint selection on Gravity Spy validation. These are
metadata-derived weak masks, so the command always withholds a segmentation claim until an
independent human pixel-mask audit and mixture/search experiment exist.

## Oracle mask-deglitch upper bound

The invertible cleaning baseline applies a Hamming-window complex STFT, suppresses
`glitch_probability × (1 − chirp_probability)`, and reconstructs strain by normalized overlap-add.
The Hamming window fixes edge reconstruction: a zero mask now changes all 26 chirp-only pilot scenes
by exactly zero within stored float precision.

Using ground-truth masks on all 41 analytic overlap scenes at suppression strength 0.9 gives:

| Metric | Mean | Median | 5th–95th percentile |
|---|---:|---:|---:|
| MSE reduction vs clean reference | 0.710 | 0.757 | 0.356–0.980 |
| Chirp projection retention | 0.997 | 1.000 | 0.969–1.003 |

The factory report SHA256 is
`0634f01751ed1f00610772f0dde9f9f30bb71b6c0c21138313f2e5cbdede8255`.
These are analytic oracle upper bounds, not learned-model results and not O4 evidence. Their role is to
show that protected mask cleaning can materially remove synthetic glitches without an inherent clean
false-veto. The required next comparison replaces oracle masks with frozen model probabilities and
measures fixed-FAR efficiency, `<VT>`, recovery SNR, and posterior coverage on real-noise injections.
