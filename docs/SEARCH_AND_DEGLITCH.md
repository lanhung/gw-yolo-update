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
With `--save-probabilities`, the run identity also covers probability storage and every float16
per-window mask is hash-checked on resume; these maps feed multi-candidate temporal clustering.
`candidate-extract` keeps every contiguous per-IFO chirp cluster instead of only one window maximum,
adds three-bin parabolic peak refinement and states the underlying half-bin timing floor. The present
96-bin/8-second model therefore still fails the <=10 ms publication timing gate; interpolation is not
misrepresented as new information.
`candidate-time-slides` pairs every retained H1/L1 cluster after a non-cyclic shift, applies an exact
peak-time coincidence, clusters nearby network events by loudest ranking statistic, and computes
exposure from all paired DQ-safe windows—including windows with no candidate. This fixes the earlier
one-maximum-per-window counting contract, while retaining an explicit timing-gate failure until the
network output grid itself reaches <=10 ms.

The present scorer has only 96 time bins over an 8-second window, so its time resolution is about
83 ms. Every time-slide report therefore says `window_level_time_slide_integration_only` and forbids
a scientific claim. Publication FAR needs sub-window candidate extraction and clustering at a
predeclared millisecond coincidence window, veto/category policy, many independent continuous O4a
segments, and enough nonzero shifts to support the requested IFAR. The current implementation proves
the split/exposure plumbing and will be replaced at that boundary rather than silently overstated.

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
`--save-probabilities`, the scorer stores float16 chirp/glitch masks with hashes.
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

This adapter is backend-neutral: it does not pretend that a synthetic NPZ is an AMPLFI or DINGO
result. Publication comparisons still require both actual backends to use matched priors, waveform
conventions, detector data, injection truth and hardware, with the backend version and model hash
added to each manifest row. The paired adapter is an executable evaluation boundary, not evidence
that either external inference run has already been completed.

The publication gate requires matching `backend_version`, `backend_model_hash`, `prior_hash`,
`waveform_approximant`, `detector_set`, `calibration_version`, `source_event_hash`, `hardware`, and
`latency_scope` on each raw/cleaned pair. The report adds deterministic paired-bootstrap intervals
for absolute-bias changes, credible-width ratios and cleaning latency, plus paired coverage
transitions. Development manifests may omit the gate, but no paper comparison may do so.

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
