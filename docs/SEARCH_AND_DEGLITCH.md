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

The systematic validation refinement selects threshold 0.39 and passes every frozen proposal gate:
95.83% padded coverage, 0.212/0.257 median/p90 union fraction and 0.281 s median containing width.
This promotes candidate-conditioned refinement, not search recall. There are still 127,102 retained
proposals (median/p90 22/30 per detector arrival), and nearest-peak p90 is 380 ms. Current proposal
score ordering is inadequate: top-1/top-16/top-24 padded coverage is 30.65%/92.45%/95.42%.
Accordingly no top-k pruning is allowed; the next head must score every candidate, learn local timing
and expose abstention before a frozen continuous-background threshold is considered.

`candidate-refiner-train` implements that all-candidate local arm on aligned 2.5 s numeric strain.
Parent injection hashes, rather than candidate rows, split validation into checkpoint-selection and
threshold-calibration roles. The initial categorical baseline (`77f6880`) is a recorded failure:
calibration AP 0.2275, timing median/p90 0.492/1.151 s and 6.87% within 10 ms. The corrected v2 target
(`d4d0330`) asks whether the physical arrival lies inside the peak-centred crop; it does not misuse
connected-component interval support as a feature-identifiability label. The original support label
remains preserved for proposal auditing. `candidate-refiner-validation` scores every calibration
candidate, freezes the highest predeclared presence threshold retaining at least 95% of detector
arrivals, and writes every refined prediction. Even a passed timing gate cannot promote search use
until signal-free continuous background supplies false acceptance and FAR/IFAR evidence.

The corrected from-scratch arm does not pass that gate. It reaches calibration AP 0.4601 and
median/p90 continuous timing errors 0.461/1.126 s; at the selected 0.5 threshold, top-score 20 ms
accuracy is 13.12% despite 99.83% arrival acceptance. Search promotion therefore remains false.
The endpoint-warm arm is allowed to answer only whether reuse of the dense proposal representation
repairs local optimization. Since the warm checkpoint was selected on all current validation
parents, that arm disables calibration metrics and cannot freeze a candidate/search threshold until
a newly acquired group-disjoint O4a calibration set exists.

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

`scripts/run_background_morphology_range.sh` is the resumable range orchestrator for a large frozen
acquisition plan. All machine paths and the half-open shard range are explicit environment inputs.
For each shard it checks free cache space and GPU occupancy, runs the validation-only morphology
stream, requires the immutable completion report and relies on the stream command to remove source
HDF5 files only after score/candidate hashes have been verified. Re-running the same range verifies
and reuses completed shards. Merge and validation candidate-rate calibration run only after every
requested shard succeeds; merely freezing an 800-pair parent plan is never reported as processed
live time.

`scripts/run_candidate_background_range.sh` is the stricter calibrated successor. It refuses to
start unless the hash-bound paired validation comparison explicitly sets
`scale_continuous_background=true`, reuses the promoted scorer commit and timing calibration, and
requires a complete zero-based parent-plan range. It streams validation blocks only
(`test_fraction=0`), merges calibrated all-instance candidates, freezes and executes the score-blind
GPS-block permutation schedule, and freezes the 0.1/year validation threshold. A restart reuses
identity-matched shard and block-background reports; any scorer, schedule, input or timing drift is
a hard failure. The wrapper exits nonzero if the full corpus still cannot reach the predeclared FAR
exposure, preserving that negative result without opening locked test data.

```bash
export TASK_PYTHON=/path/to/python
export PARENT_PLAN=artifacts/o4a/gwosc_run_plan.json
export EVENT_EXCLUSIONS=artifacts/o4a/event_exclusions.json
export CHECKPOINT=artifacts/model/best.pt
export CONFIG=configs/physical_finetune_scale_fixed_updates.yaml
export COHERENCE_CONFIG=configs/physics_coherent_yolo_pilot.yaml
export CACHE_ROOT=/large-cache/o4a-morphology
export OUTPUT_ROOT=artifacts/o4a/morphology-shards
export SHARD_START=0 SHARD_STOP_EXCLUSIVE=200 PAIRS_PER_SHARD=4
export GWYOLO_CODE_COMMIT=$(git rev-parse HEAD)
bash scripts/run_background_morphology_range.sh
```

Large time-slide schedules are also resumable. `candidate-time-slides --slide-start-index S
--slide-count N` evaluates the absolute half-open offset range `[S, S+N)`; candidate IDs and offsets
retain those absolute indices. `candidate-time-slide-merge` verifies identical candidate/background
hashes, detector/timing/model provenance and physics settings, rejects repeated offsets or candidate
IDs, and combines exposure only after every shard manifest hash and row count passes. A gapped
absolute offset range without a schedule is retained as partial engineering evidence and cannot pass
the merged publication timing gate.

Discontinuous observing segments should not be bridged by scanning millions of zero-exposure
offsets. `candidate-time-slide-schedule-freeze` accepts explicit positive absolute indices, uses only
background GPS and detector availability, rejects every zero-exposure offset, and freezes the
background hash, detector pair, step, ordered indices and target FAR before candidate scores are
read. Scheduled runner shards select by `--schedule-offset`; the merge passes its execution-complete
gate only when the union of shard indices exactly equals the frozen schedule. Thus a deliberately
sparse schedule is distinct from an accidentally missing contiguous shard.

For a large newly merged background, `candidate-time-slide-range-schedule-freeze` removes the
manual offset-selection step. Within one predeclared positive half-open range it evaluates detector
availability only, discards zero-exposure offsets and freezes the shortest absolute-index prefix
whose summed exposure reaches the requested zero-count FAR upper limit. If the entire range is
insufficient it freezes every nonzero offset and records the shortfall rather than silently changing
the FAR target. The scanned range, available exposure, required exposure, selection rule and the
fact that candidate scores were not inspected are covered by the schema-v2 schedule ID.

Absolute offsets remain inefficient when accepted observing time is split into many distant
256-second GPS blocks. `candidate-block-permutation-schedule-freeze` instead orders the frozen GPS
blocks and circularly pairs block `i` on the reference IFO with block `i+s` on the shifted IFO.
Only equal relative eight-second slots are paired, and a slot contributes exposure only when the
required detector is explicitly available on each side. The nonzero shifts enumerate every ordered
cross-block pair once when the full `1..N-1` range is used. The planner reads no candidate scores and
freezes the shortest shift prefix reaching the target exposure, or truthfully records that all
available permutations are insufficient. This is a standard background-resampling exposure, not
additional independent zero-lag strain.

Before committing the expensive full parent bank,
`candidate-block-permutation-capacity-forecast` combines a DQ-verified pilot schedule, its exact
background report and the proposed parent acquisition plan. It projects the maximum circular-shift
capacity from observed validation blocks per source pair and seconds per block-shift, solves the
quadratic block count required by the zero-count FAR exposure, and applies a predeclared safety
factor (1.5 by default). It reads no candidates or scores and is explicitly a planning forecast,
not achieved live time. A failed safety margin is nonzero by default but still writes an atomic
report with the recommended minimum source-pair count; `--allow-insufficient` is only for retaining
diagnostics and cannot make the later exact schedule pass.

`candidate-block-permutations` re-hashes that immutable schedule and the background manifest,
requires validation-calibrated candidate timing at <=10 ms resolution, applies the predeclared
light-travel limit plus twice the empirical timing allowance to relative within-window peaks, and
checks every executed shift's block/window exposure exactly against the score-blind plan. Its report
uses the existing candidate-search calibration contract, while recording the distinct pairing
method so an absolute slide cannot be substituted after threshold selection.

```bash
python -m gwyolo.cli candidate-time-slides \
  --candidates val-candidates.jsonl --background-manifest val-background.jsonl \
  --output-dir val-slides-0001 --split val --slide-start-index 1 --slide-count 512 \
  --step-seconds 8 --coincidence-window-seconds 0.012
python -m gwyolo.cli candidate-time-slides \
  --candidates val-candidates.jsonl --background-manifest val-background.jsonl \
  --output-dir val-slides-0513 --split val --slide-start-index 513 --slide-count 512 \
  --step-seconds 8 --coincidence-window-seconds 0.012
python -m gwyolo.cli candidate-time-slide-merge \
  --report val-slides-0001/val_candidate_time_slide_report.json \
  --report val-slides-0513/val_candidate_time_slide_report.json \
  --output-dir val-slides-merged --split val
```

For a predeclared sparse schedule:

```bash
python -m gwyolo.cli candidate-time-slide-schedule-freeze \
  --background-manifest val-background.jsonl --output val-slide-schedule.json \
  --split val --step-seconds 8 --slide-index 1 3 5 7 \
  --target-far-per-year 0.1
python -m gwyolo.cli candidate-time-slides \
  --candidates val-candidates.jsonl --background-manifest val-background.jsonl \
  --slide-schedule val-slide-schedule.json --schedule-offset 0 --slide-count 4 \
  --output-dir val-scheduled-slides --split val --step-seconds 8 \
  --coincidence-window-seconds 0.012
```

Or derive the nonzero prefix automatically from a frozen range:

```bash
python -m gwyolo.cli candidate-time-slide-range-schedule-freeze \
  --background-manifest val-background.jsonl --output val-slide-schedule.json \
  --split val --step-seconds 8 --slide-start-index 1 \
  --slide-stop-index-exclusive 100001 --target-far-per-year 0.1
```

For fragmented run-scale background:

```bash
python -m gwyolo.cli candidate-block-permutation-capacity-forecast \
  --pilot-schedule pilot-block-schedule.json \
  --pilot-background-report pilot-background/background_plan_report.json \
  --planned-parent-plan gwosc-o4a-parent-plan.json \
  --safety-factor 1.5 --output block-capacity-forecast.json
python -m gwyolo.cli candidate-block-permutation-schedule-freeze \
  --background-manifest val-background.jsonl --output val-block-schedule.json \
  --split val --reference-ifo H1 --shifted-ifo L1 --target-far-per-year 0.1
python -m gwyolo.cli candidate-block-permutations \
  --candidates val-candidates.jsonl --background-manifest val-background.jsonl \
  --schedule val-block-schedule.json --output-dir val-block-background --split val \
  --reference-ifo H1 --shifted-ifo L1 --physical-delay-limit-seconds 0.010 \
  --empirical-timing-uncertainty-seconds 0.001 \
  --coincidence-window-seconds 0.012
```

The provenance path is transitive rather than name-based. Candidate extraction verifies the adjacent
score report and carries checkpoint/config/commit hashes. Timing application succeeds only when the
validation calibration and target candidates came from that exact scoring identity. Time-slide and
injection-ranking reports require one common calibration, checkpoint, config and commit. Finally,
`candidate-search-calibrate` reads validation reports only, and
`candidate-search-evaluate-frozen` has no threshold argument, rejects any validation/test GPS,
injection or waveform overlap, refuses to overwrite an existing locked result, and reports FAR,
IFAR and weighted `<VT>` with bootstrap uncertainty. An empty background-candidate list freezes a
threshold above probability support; it can never turn score-zero injection misses into detections.
Calibration from an unscheduled or exposure-insufficient slide report remains available as explicit
engineering output, but is marked `publication_calibration_eligible=false`. The locked command now
fails closed unless both validation and test reports re-hash a score-blind frozen schedule, execute
every scheduled absolute offset or block permutation, match the requested FAR, reproduce the
schedule's equivalent live time and reach its predeclared zero-count exposure. A nominal threshold can therefore no longer
cross the paper boundary merely because its empirical surviving count is zero.

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

The validation-only morphology stream also has a deliberately narrower rate diagnostic.
`background-morphology-calibrate` computes exposure as the union of valid intervals separately for
each IFO and calibrates a single-IFO candidate rate per detector-year above the immutable extraction
floor. It records per-IFO rates and zero-count Poisson upper limits, refuses any test windows and
sets `network_far_claim_allowed=false`. This diagnostic can expose an unusably noisy mask front end
before timing work completes, but it is not a substitute for calibrated H1/L1/V1 coincidence,
time-slide FAR/IFAR or injection `<VT>`.

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
`latency_scope` on each backend triplet (or each legacy raw/cleaned pair). It additionally requires
both actual DINGO and AMPLFI results on the identical injection set. For every condition, the two
backends must consume byte-identical strain/input artifacts and share the same prior, waveform,
detector, calibration, hardware and latency scope. The evaluator verifies the input, base-injection,
contamination, mask, mask-model and mask-policy files against their declared SHA-256 values. A
cleaned-strain arm must differ from the contaminated strain; an auxiliary-mask arm must preserve
the contaminated strain hash and identify the separate mask artifact. The report adds
deterministic paired-bootstrap intervals
for absolute-bias changes, credible-width ratios and cleaning latency, plus paired coverage
transitions. Development manifests may omit the gate, but no paper comparison may do so.

Before either backend can enter that evaluation, `pe-backend-lock-audit` enforces a separate
readiness boundary. `configs/pe_backend_environment_lock.yaml` pins the upstream source tag and
commit while machine paths are supplied through environment variables. The strict command verifies
clean Git sources, distinct Python interpreters, compatible Python and installed distribution
versions, CUDA visibility, and the SHA-256 of both the selected checkpoint and its training/model
metadata. It also freezes a shared BBH, H1/L1, 4,096 Hz, 16-second source-input contract with byte-identical
clean, contaminated and mask-conditioned source artifacts across backends. Backend-native
conditioning may differ only when its derived artifact and settings are separately recorded and
hashed. The committed checkpoint hashes intentionally remain `UNRESOLVED`; therefore the strict
gate fails until actual domain-compatible DINGO and AMPLFI models have been selected. A readiness
diagnostic can be written without promotion using:

```bash
python -m gwyolo.cli pe-backend-lock-audit \
  --config configs/pe_backend_environment_lock.yaml \
  --output artifacts/pe/backend_lock_audit.json \
  --allow-incomplete
```

Removing `--allow-incomplete` is the publication gate and must return non-zero while any environment,
model, metadata or hash is missing. Installing the packages alone is not evidence that an actual
backend comparison has run.

`scripts/setup_pe_backends.sh` is the reproducible environment bootstrap. It requires explicit
source, commit, tag, interpreter and output variables; refuses a shared DINGO/AMPLFI environment;
holds a nonblocking atomic installation lock before inspecting package-manager state; refuses to run
beside another active package installer; and writes sorted package plus CUDA runtime snapshots.
Hermetic virtual environments are the default. A system-site-package overlay is available
only as an explicit engineering option and its full observed package-set hash must still be frozen
before publication.

Checkpoint readiness uses a standardized sidecar created by `pe-backend-model-freeze`. The command
will only freeze a checkpoint when a separate selection report has status
`validation_selected_checkpoint`, says `selection_split: validation`, is explicitly
`publication_eligible`, names the selection metric and contains the checkpoint's exact SHA-256. It
also hashes the training
configuration, training-data manifest, common analysis prior, selection report and backend-native
conditioning configuration. Both backends additionally require their exact native prior settings
and a machine-readable semantic projection report produced by `dingo-common-prior-audit` or
`amplfi-common-prior-audit`. The report must have passed and its canonical-prior, native-prior and
training-configuration hashes must match the other frozen artifacts; copying a passed report from a
different training run fails closed. The sidecar records the native and common analysis waveform
approximants, common source contract and inference parameters. The environment audit reloads and
verifies every referenced artifact, then requires DINGO and AMPLFI to use the same analysis prior,
analysis waveform and explicitly mapped common parameter set. Native output spaces may differ, but
every canonical paper parameter must map to a real native posterior field. This prevents an arbitrary downloaded checkpoint or
a test-selected model from entering the paper table.

For Lightning/AMPLFI runs, create the checkpoint index inside the pinned backend environment. The
indexer deserializes trusted local checkpoints, records their epoch/global-step identities and
hashes, and never selects a model itself. `pe-lightning-checkpoint-select` then reads the CSV
validation trajectory, rejects any populated test-metric column, excludes `last.ckpt`, and matches
the best validation row to exactly one indexed checkpoint. The configured training budget and
observed validation trajectory must meet the predeclared publication minima; otherwise it writes a
useful engineering selection report with `publication_eligible: false`, which cannot pass model
freeze. In particular, the one-epoch AMPLFI smoke is a load/compatibility test only.

```bash
artifacts/pe/envs/amplfi/bin/python scripts/index_lightning_checkpoints.py \
  --checkpoint-dir artifacts/pe/amplfi/checkpoints \
  --output artifacts/pe/amplfi/checkpoint_index.json

python -m gwyolo.cli pe-lightning-checkpoint-select \
  --training-config configs/amplfi_common_bbh_publication.yaml \
  --training-data-manifest artifacts/pe/amplfi/training_manifest.jsonl \
  --metrics-csv artifacts/pe/amplfi/csv_logs/metrics.csv \
  --checkpoint-index artifacts/pe/amplfi/checkpoint_index.json \
  --selection-metric valid_loss --selection-metric-mode min \
  --minimum-publication-epochs 100 --minimum-validation-points 50 \
  --output artifacts/pe/amplfi/selection.json
```

```bash
python -m gwyolo.cli dingo-common-prior-audit \
  --canonical-prior configs/pe_common_bbh_analysis_prior.yaml \
  --dingo-prior-config artifacts/pe/dingo/model_settings.txt \
  --training-config artifacts/pe/dingo/model_settings.txt \
  --output artifacts/pe/dingo/prior_projection.json

python -m gwyolo.cli pe-backend-model-freeze \
  --backend DINGO \
  --model artifacts/pe/dingo/model.pt \
  --initialization-model artifacts/pe/dingo/time_model.pt \
  --training-config artifacts/pe/dingo/train.yaml \
  --training-data-manifest artifacts/pe/dingo/train.jsonl \
  --analysis-prior artifacts/pe/common/analysis_prior.yaml \
  --native-prior artifacts/pe/dingo/model_settings.txt \
  --prior-projection-report artifacts/pe/dingo/prior_projection.json \
  --selection-report artifacts/pe/dingo/selection.json \
  --native-conditioning-config artifacts/pe/dingo/conditioning.yaml \
  --source-sample-rate-hz 4096 \
  --source-duration-seconds 16 \
  --source-post-trigger-seconds 2 \
  --analysis-waveform-approximant IMRPhenomXPHM \
  --native-model-waveform-approximant IMRPhenomXPHM \
  --model-training-backend-version 0.5.8 \
  --native-inference-parameters chirp_mass mass_ratio luminosity_distance theta_jn ra dec psi \
  --reported-parameter-mapping chirp_mass=chirp_mass mass_ratio=mass_ratio \
    luminosity_distance=luminosity_distance theta_jn=theta_jn ra=ra dec=dec psi=psi \
  --output artifacts/pe/dingo/model_metadata.json
```

The AMPLFI invocation uses the same common fields and must also bind its native-prior projection:

```bash
python -m gwyolo.cli amplfi-common-prior-audit \
  --canonical-prior configs/pe_common_bbh_analysis_prior.yaml \
  --amplfi-prior configs/amplfi_common_bbh_training_prior.yaml \
  --training-config configs/amplfi_common_bbh_publication.yaml \
  --output artifacts/pe/amplfi/prior_projection.json

python -m gwyolo.cli pe-backend-model-freeze \
  --backend AMPLFI \
  --model artifacts/pe/amplfi/model.ckpt \
  --training-config configs/amplfi_common_bbh_publication.yaml \
  --training-data-manifest artifacts/pe/amplfi/training_manifest.jsonl \
  --analysis-prior configs/pe_common_bbh_analysis_prior.yaml \
  --native-prior configs/amplfi_common_bbh_training_prior.yaml \
  --prior-projection-report artifacts/pe/amplfi/prior_projection.json \
  --selection-report artifacts/pe/amplfi/selection.json \
  --native-conditioning-config configs/amplfi_common_native_conditioning.yaml \
  --source-sample-rate-hz 4096 \
  --source-duration-seconds 16 \
  --source-post-trigger-seconds 2 \
  --analysis-waveform-approximant IMRPhenomXPHM \
  --native-model-waveform-approximant ml4gw.waveforms.IMRPhenomPv2 \
  --model-training-backend-version 0.6.0 \
  --native-inference-parameters chirp_mass mass_ratio distance phic inclination dec psi phi \
  --reported-parameter-mapping chirp_mass=chirp_mass mass_ratio=mass_ratio \
    luminosity_distance=distance theta_jn=inclination ra=phi dec=dec psi=psi \
  --output artifacts/pe/amplfi/model_metadata.json
```

The shared source artifact is deliberately a superset of backend-native conditioning. The selected
official O4a DINGO precessing HL model uses 16 seconds at 4,096 Hz and analyzes 20--1,024 Hz, while
the AMPLFI v0.6 CBC default uses a 3-second, 2,048 Hz native window. Both must derive their native
input from the same hashed 16-second source artifact, and each derived tensor plus conditioning
configuration must also be hashed. Native parameter names are mapped to a common paper space; for
example AMPLFI `distance`, `inclination` and `phi` map to canonical `luminosity_distance`, `theta_jn`
and `ra`, respectively.

`pe-input-materialize` now constructs that source artifact from three exactly paired manifests:
clean injection strain, the same injection with a real Gravity Spy glitch, and the corresponding
learned-mask-conditioned strain. It rejects train rows, unequal injection sets, waveform/GPS/truth
identity changes, mismatched glitch lineage, masks not derived from the contaminated override,
missing H1/L1, truth outside the canonical prior support and a contaminated series numerically
identical to clean. Eligible BBHs are selected by an ID hash and frozen seed before posterior
results exist. Every output is a numeric NPZ containing 16 seconds of H1/L1 strain at 4,096 Hz, with
one SHA-256 that both backends must consume. The event is placed exactly two seconds before the
source-window end rather than centered on an arbitrary eight-second background window.

```bash
python -m gwyolo.cli pe-input-materialize \
  --clean-manifest artifacts/pe/paired_clean_val.jsonl \
  --contaminated-manifest artifacts/pe/contaminated_val.jsonl \
  --mask-conditioned-manifest artifacts/pe/mask_conditioned_val.jsonl \
  --common-prior configs/pe_common_bbh_analysis_prior.yaml \
  --mask-model artifacts/models/best_validation.pt \
  --mask-policy configs/pe_mask_conditioning_policy.yaml \
  --output-dir artifacts/pe/common_sources_val \
  --required-split val \
  --required-ifos H1 L1 \
  --source-sample-rate-hz 4096 \
  --source-duration-seconds 16 \
  --source-post-trigger-seconds 2 \
  --analysis-high-frequency-hz 1024 \
  --limit 100 \
  --selection-seed 20260721
```

The current physical injection bank is native 2,048 Hz. Its conversion to the common 4,096 Hz
container is explicitly recorded as band-limited FFT interpolation: it adds no information above
the original 1,024 Hz Nyquist, and the command refuses an analysis band beyond that limit. The FFT
implementation splits an even-length source Nyquist coefficient when it becomes a paired target
frequency; a unit test verifies both the original samples and half-sample interpolation. This is a
valid shared 20--1,024 Hz input for the present robustness study, but future waveform-systematics
work should regenerate primary PE injections and background directly at 4,096 Hz rather than claim
that interpolation created high-frequency detector information. The report also states that the
selected detection-injection population lies inside common-prior support but was not sampled from
that prior distribution.

The common source includes a condition-invariant ASD as well as strain. It is estimated once from
the clean 64-second materialized noise context using off-source Hann-windowed periodograms; every
segment touching the central analysis interval plus a two-second guard is excluded. The three
conditions must contain an identical ASD and frequency-grid hash. This prevents backend- or
condition-specific PSD estimation from masquerading as a mask-cleaning effect.

`pe-native-condition` creates backend-native, hash-locked artifacts. For DINGO it applies the
official 0.4-second Tukey roll-off, `rfft × delta_t`, two-second frequency-domain time translation
and 20--1,024 Hz domain, writing an `EventDataset`-schema HDF5. For AMPLFI it anti-alias downsamples
the same source to 2,048 Hz and stores the common ASD beside strain; its runtime contract forbids
PSD re-estimation. Both native manifests retain the common-source, ASD and conditioning-config
hashes.

```bash
python -m gwyolo.cli pe-native-condition \
  --source-manifest artifacts/pe/common_sources_val/common_pe_inputs.jsonl \
  --config configs/dingo_o4a_native_conditioning.yaml \
  --output-dir artifacts/pe/dingo_native_val \
  --required-split val
```

Real DINGO GNPE sampling is executed by the pinned backend interpreter through
`dingo-common-batch` and `scripts/run_dingo_common_event.py`. The runner verifies the event, main
model and time-initialization model hashes, loads upstream `EventDataset`, `GWSampler` and
`GWSamplerGNPE`, retains the native result HDF5, and writes numeric posterior NPZ plus measured
latency. Backend import or model compatibility failures are explicit and non-zero; no synthetic
posterior fallback exists. A resumed batch revalidates every posterior and native-result hash.
The standardized metadata requires the time-initialization network as a DINGO-specific artifact;
the batch executor also reopens every training-config, training-manifest, analysis-prior, native
prior, projection-report, selection-report and conditioning-config artifact. It requires native rows
to use the frozen conditioning and common-prior hashes, requires `--native-prior` to match metadata,
and refuses a runtime initialization network whose bytes differ from metadata.

Real AMPLFI sampling follows the same fail-closed contract through `amplfi-common-batch` and
`scripts/run_amplfi_common_event.py`. The pinned interpreter reconstructs the exact v0.6 NSF and
`MultiModalPsd` architecture from the hashed training config, strictly loads a validation-selected
Lightning checkpoint and its fitted parameter scaler, crops the common 16-second source so that the
event is 0.5 seconds from the right edge of the final three-second model input, and applies the
upstream `ml4gw.transforms.Whiten` implementation with the shared ASD. It does not invoke AMPLFI's
runtime PSD estimator. Samples outside the hashed native training-prior support are removed before
`phi` is converted to physical right ascension. Both native and canonical posterior fields are
retained, together with model-load, preprocessing, sampling and end-to-end latency. AMPLFI v0.6's
`MultiModalPsd` scales its ASD argument in place, so the runner supplies a fresh clone to every
embedding call; otherwise the sampling context would be multiplied again during log-probability
evaluation and subsequent sample chunks.

Both batch adapters compute the required 90% sky-area field directly from the retained RA/Dec
posterior samples. The dependency-free primary implementation uses a frozen 360 by 180 grid in
right ascension and `sin(dec)`, so all 64,800 pixels have equal solid angle. It reports the greedy
credible-pixel count, pixel area, sample count and exact method alongside `sky_area_90_deg2`; that
estimator record must match across conditions and backends. This fixed-grid statistic is an
identical paired robustness metric, not a BAYESTAR or adaptive-HEALPix sky map. A future
waveform-systematics table may add a common `ligo.skymap` estimator as a separately frozen stratum,
but cannot replace one backend's estimator after posterior results are opened.

The batch boundary independently reopens and hashes the metadata-bound canonical prior, native
prior and semantic projection report before launching any subprocess. The runtime `--native-prior`
must match the metadata artifact byte-for-byte, and the passed projection must bind that prior, the
canonical prior and the exact training config. Thus a caller cannot swap the sampling support after
checkpoint freeze even if the runner command itself is otherwise well formed.

```bash
python -m gwyolo.cli amplfi-common-batch \
  --native-manifest artifacts/pe/amplfi_native_val/amplfi_native_conditioning.jsonl \
  --model-metadata artifacts/pe/amplfi/model_metadata.json \
  --native-prior configs/amplfi_common_bbh_training_prior.yaml \
  --python-executable artifacts/pe/envs/amplfi/bin/python \
  --runner-script scripts/run_amplfi_common_event.py \
  --output-dir artifacts/pe/amplfi_posteriors_val \
  --required-split val
```

The AMPLFI runner is now unit-tested at the orchestration, hash, resume, architecture and prior
contract boundaries. A real checkpoint-load smoke remains mandatory; a fake subprocess used by the
CPU orchestration test is not posterior evidence and cannot enable a scientific claim.

After both batch reports exist, `pe-robustness-joint-evaluate` is the only supported publication
join path. It verifies the two batch-report and manifest hashes, requires identical injection and
condition sets with complete clean/contaminated/mask-conditioned triplets, then executes the full
cross-backend input-lineage, prior, waveform, detector, hardware, sky-area and latency-scope gates
before atomically writing a combined manifest. `scripts/run_joint_paired_pe_validation.sh` wires the
validation-only native-input smoke to both pinned backend interpreters and this strict joint gate;
it remains fail-closed until validation-selected model sidecars for both backends are present.
The subsequent `pe-robustness-promote` command applies the pre-result thresholds in
`configs/pe_robustness_promotion.yaml`. Bias deltas are normalized by each event's clean-posterior
credible width, while coverage non-inferiority uses paired bootstrap differences. Both backends must
pass sample-size, coverage, bias, width, sky-area, effective-sample-rate and latency gates before
`promote_to_locked_test` can become true; the validation decision itself never enables a paper claim.

`scripts/run_paired_pe_smoke.sh` closes the preceding validation-data gap without touching a locked
test corpus. After a detector-set overlap run writes its validation-selected checkpoint, the script
builds paired clean and real-glitch-contaminated overrides, scores every contaminated instance with
saved numeric chirp/glitch probabilities, applies the frozen mask policy, selects a bounded BBH
subset before posterior results exist, and materializes both DINGO and AMPLFI native inputs. Each
stage is resumable by its atomic report and all machine paths are explicit environment variables.
`GWYOLO_MODEL_CONFIG` must be the exact selected overlap configuration: the script re-hashes it
against `config_file_sha256` in the model report before scoring, so a family-balanced champion cannot
silently be evaluated under the uniform-arm provenance. The smoke defaults to three validation
injections and remains ineligible for a scientific claim.

`scripts/run_promoted_paired_pe_smoke.sh` is the five-seed handoff. It resolves the selected
checkpoint back to exactly one hash-listed finetune report, chooses the matching uniform or
family-balanced configuration, and re-hashes the selected checkpoint, configuration, overlap
validation manifest and clean validation manifest before invoking the paired smoke. It rejects a
non-validation summary or an ambiguous report rather than guessing a champion path.

Before an event is admitted, `scripts/run_pe_model_load_smoke.py` verifies checkpoint/config hashes
inside the pinned interpreter and loads both DINGO GNPE networks or the AMPLFI Lightning model,
architecture and fitted scaler. It records backend version, parameter counts, GPU runtime and load
latency and refuses to overwrite its report. This distinguishes a real 5.9 GB DINGO or trained
AMPLFI checkpoint from a source-import smoke, while still withholding all posterior claims.

Official external weights are acquired through `pe-model-sources-acquire`, not an unrecorded browser
download. `configs/pe_official_model_sources.yaml` freezes the Zenodo record, exact filenames, byte
sizes and published MD5 values for the O4a DINGO manifest, settings, posterior model and time
initialization model. The command defaults to verify-only. `--download` enables resumable `.part`
downloads, verifies size and MD5 before atomic promotion, records a local SHA-256, enforces a free
space reserve and refuses to overwrite a corrupt existing target.

```bash
python -m gwyolo.cli pe-model-sources-acquire \
  --config configs/pe_official_model_sources.yaml \
  --output-dir artifacts/pe/official_models/dingo-o4a \
  --report artifacts/pe/official_models/dingo-o4a-acquisition.json \
  --minimum-free-bytes 16106127360 \
  --download
```

AMPLFI is absent from this source manifest until a real reusable checkpoint is identified or a
validation-selected common-domain model is trained. A 20 MB paper-figure archive is not silently
substituted for executable weights.

The common-domain AMPLFI training path is now explicit. `amplfi-background-export` consumes the
frozen GW-YOLO background manifest, verifies every GWOSC source hash, rejects any GPS block assigned
to more than one split, coalesces only contiguous valid windows, and writes native HDF5 segments to
separate `train/background`, `validation/background` and `test/background` directories. A 4,096 to
2,048 Hz conversion uses `scipy.signal.resample_poly` with a recorded Kaiser window; no plot image or
unfiltered decimation enters PE training. `GroupSafeFlowDataset` overrides AMPLFI's default
file-order validation split and rechecks HDF5 GPS-block identities plus validation live time before
training.

```bash
python -m gwyolo.cli amplfi-background-export \
  --manifest artifacts/o4a/background_windows.jsonl \
  --output-dir artifacts/pe/amplfi-data \
  --report artifacts/pe/amplfi-background-export.json \
  --target-sample-rate 2048 \
  --minimum-segment-seconds 16
```

`configs/amplfi_common_bbh_publication.yaml` retains the published AMPLFI v0.6 CBC architecture and
full 800-epoch/800-batch schedule, but replaces the unsafe implicit split with the group-safe data
module and uses local CSV logging. `configs/pe_common_bbh_analysis_prior.yaml` is the canonical
backend-neutral prior; `configs/amplfi_common_bbh_training_prior.yaml` is its native AMPLFI
projection. `amplfi-common-prior-audit` checks all fourteen intrinsic, extrinsic and nuisance
distributions, their bounds, H1/L1 identity and the native 2,048 Hz/3-second contract. This is a
semantic gate in addition to file hashes.

The source-safe one-seed overlap-sampling decision is frozen in
`configs/physical_overlap_sampling_promotion.yaml`. `physical-overlap-sampling-promote` accepts
only uniform and family-balanced reports trained on byte-identical overlap and clean manifests with
the same seed and pretrained checkpoint. Every overlap row must bind the passed network-corpus
audit. Selection uses validation-only clean chirp retention, overall glitch IoU, family-median and
worst-family IoU, zero-IoU family count and bounded family regressions. The command either names one
arm for five-seed expansion or records `scale_to_five_seeds=false`; it never opens test data.
`scripts/run_overlap_five_seed_promotion.sh` exits without training when that flag is false. When it
is true, it retains the promoted seed, runs four additional declared seeds and requires exactly five
unique validation-selected reports with identical data, config and pretrained-model hashes.
`physical-overlap-five-seed-summarize` records mean, sample standard deviation, extrema and
per-family IoU while continuing to withhold any test or search claim.
The summary also selects the downstream checkpoint by maximum validation overlap mean IoU with a
seed tie-break and re-hashes the checkpoint. `scripts/run_promoted_candidate_validation.sh` binds
that five-seed summary and the exact promoted config before it can score O4a validation data. It
then creates a validation-only background manifest, annotates independent validation injections
with PyCBC detector arrivals, and runs timing calibration, calibrated time slides and threshold
freezing. Its 40-pair `100/year` setting is an engineering gate; it is not the final 0.1/year FAR
claim or a substitute for the frozen exposure target.
Before any 800-pair continuous-background scale-up,
`candidate-search-validation-compare` evaluates the old fixed-channel baseline and promoted
detector-set model on the same independent-GPS validation injections and background, using the same
scorer commit, slide schedule and target FAR. Each model freezes its own validation threshold. The
promotion gate uses paired injection outcomes and VT weights, requires a positive paired-bootstrap
lower bound, bounds timing degradation and limits source-family/SNR-stratum regressions. Only a
report with `scale_continuous_background=true` can authorize the expensive background run.
The original 40-pair pipelines used an unscheduled absolute-slide diagnostic, so they cannot satisfy
the frozen-exposure component merely by having zero surviving events.
`candidate-search-validation-block-recalibrate` reuses each pipeline's exact calibrated candidates
and injection rankings, freezes the same score-blind GPS-block schedule, and replaces only its
background calibration. `scripts/run_candidate_validation_block_comparison.sh` applies that
operation to both arms before repeating the paired promotion gate. Thus the 800-pair decision can
become eligible from measured 40-pair block exposure without relaxing the FAR contract or rerunning
the candidate scorer.

`gravityspy-glitch-finetune` is the bounded real-glitch training boundary. It accepts only a frozen
train/validation pair with disjoint glitch and network-GPS-block identities, hash-verifies every
numeric sample, and samples train labels with inverse-frequency weights. A checkpoint is eligible
only if its fixed-threshold physical-chirp validation IoU retains a configured fraction (0.95 by
default) of the incoming checkpoint; among eligible epochs, validation glitch IoU is primary.
Threshold calibration happens only after checkpoint selection on Gravity Spy validation. These are
metadata-derived weak masks, so the command always withholds a segmentation claim until an
independent human pixel-mask audit and mixture/search experiment exist.

## Detector-set OOD abstention

The original `glitch-ood-train` baseline consumed only the event detector even when the numeric
sample contained aligned H1/L1/V1 context. `architecture: detector_set` now requires the aligned
network tensor, verifies its fixed detector order and Q values, matches the row and array validity
masks, and rejects nonzero unavailable planes. A shared per-IFO encoder pools the declared detector
set with masked attention. Fixed H1/L1/V1 one-hot identities enter the fusion, so an unavailable
detector cannot be confused with a valid zero tensor and detector identity is not discarded.

`gravityspy-ood-family-freeze` selects the next held family using only family labels, row counts and
independent GPS-block counts. Previously opened families must be listed with `--exclude-family`.
The command rejects train/validation overlap in glitch, GPS block and, when available, official
network strain source. It records that no model or unknown score existed at selection time. The
result then feeds the existing group-safe `gravityspy-ood-split`; checkpoint selection and the
abstention threshold still use known-family validation rows only. The held-family scores remain an
evaluation, never a tuning set.

The precommitted detector-set arm is
`configs/glitch_ood_network_contrastive_energy.yaml`. It uses supervised contrastive training and
logit energy because those choices predate the new held-family result. A positive OOD result remains
auxiliary attribution/abstention evidence and may not silently veto a strain-coherent candidate.

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
