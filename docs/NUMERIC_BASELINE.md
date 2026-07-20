# Multi-IFO/multi-Q numeric baseline

## Purpose and claim boundary

`MultiIFOQNet` is a compact U-Net-style engineering baseline that consumes flattened
IFO × Q numeric planes and predicts separate chirp/glitch masks for every plane. It validates the
complete recipe → online transform → validation-selected checkpoint → frozen-threshold test path.

All results in this document use analytic chirps and glitches. They are not estimates of O4 search
sensitivity and must not be compared with AMPLFI, DINGO, PyCBC, GstLAL, MBTA, cWB, or search FAR.

## Sparse-mask failure and correction

In the 72-scene pilot training split, chirp pixels occupy 1.223% of target planes and glitch pixels
only 0.170%. With unweighted BCE plus Dice, the first model reached chirp test IoU 0.648 but glitch
test IoU only 0.009, a clear class-collapse failure.

The selected loss uses positive weights `[10, 40]` for chirp/glitch and Dice class weights `[1, 4]`.
Thresholds are scanned only on validation and then frozen. A medium candidate with `[5, 20]` and
`[1, 2]` reached validation mean IoU 0.409; the heavier candidate reached 0.419 and was selected.
Its independent 16-scene engineering test reached mean IoU 0.399, with chirp 0.579 and glitch 0.219.

## Nested data-scale experiment

To isolate training-data scale, two nested manifests were created from the 200k recipe pool. They
share exactly the same 64 validation and 64 test scenes and use identical model, optimizer, seed,
loss, and training length. Both passed the four-axis provenance audit.

| Train scenes | Validation chirp IoU | Validation glitch IoU | Validation mean IoU | Selected thresholds |
|---:|---:|---:|---:|---:|
| 72 | 0.605 | 0.217 | 0.411 | 0.7 / 0.9 |
| 250 | 0.788 | 0.561 | 0.675 | 0.7 / 0.7 |
| 500 | 0.815 | 0.624 | 0.719 | 0.8 / 0.8 |
| 1,000 | 0.866 | 0.708 | 0.787 | 0.7 / 0.6 |
| 2,000 | 0.875 | 0.756 | 0.816 | 0.7 / 0.8 |
| 5,000 | 0.915 | 0.831 | 0.873 | 0.6 / 0.7 |
| 10,000 | 0.918 | 0.856 | 0.887 | 0.7 / 0.7 |

The 72→250 expansion improved mean IoU by 26.35 percentage points on the same validation set. This
is strong evidence that the current regime is data-limited, even in the simplified synthetic domain.
It supports increasing independent physical scenes before investing primarily in a larger model.

The later points confirm that conclusion through 5,000 scenes. Gains remain positive but are not
smooth: 250→500 adds 4.47 points, 500→1k adds 6.75, 1k→2k adds 2.89, and 2k→5k adds 5.72. A
constrained pre-10k fit of `M(N)=M_inf-aN^-alpha` forecast 0.877 at 10k. Refitting all seven points
gives `alpha=0.513`, asymptote 0.920, and R² 0.994, with exploratory forecasts 0.892 at 20k and 0.903
at 50k. The completed 10k run selected epoch 20 with pre-calibration
validation mean IoU 0.883 and validation-selected 0.7/0.7 thresholds; calibrated validation mean IoU
is 0.887 (chirp 0.918, glitch 0.856), 1.05 points above 5k. This confirms continued improvement but
also a marked reduction from the 2k→5k gain.

The required five-seed repeat is complete. Pre-calibration best validation mean IoU is 0.88087,
with sample standard deviation 0.00678 and Student-t 95% interval `[0.87245, 0.88928]`; individual
best values span 0.86975–0.88810. Seed 20260721 is the validation-selected checkpoint (epoch 17,
SHA256 `61730b9734a90fd01e4678470026cacc8c3e78cdf008e68cbcaf88ebd3ae8e72`). The aggregate report
SHA256 is `39f3407bfe8abdbdf8a753532ab23b12a101b017dfb2f967710daa027b09245f`.
No test split was evaluated during this sweep.

After selecting the 250-scene point, the frozen 0.7/0.7 thresholds were applied once to the shared
64-scene test:

| Class | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|
| chirp | 0.889 | 0.852 | 0.770 | 0.870 |
| glitch | 0.710 | 0.722 | 0.558 | 0.716 |
| mean | — | — | 0.664 | — |

Reproducibility identifiers:

- 72-scene manifest SHA256:
  `47b5c3faa7ac0cb0af5eaceebde3582ad8eae8f420bc9a671b3d67de29713a1c`;
- 250-scene manifest SHA256:
  `cdbf3cff7678669be72efec281740bffc6fdc73028d5501a0d9b40f5b2ce4233`;
- selected 250-scene checkpoint SHA256:
  `268cb89b3ccb3cfd707aa45abc5465e8f23a3187e5a6a407a674de0655997427`;
- numeric config hash: `5b012c436dbbfc6d`;
- seed: `20260719`.

The 10k manifest SHA256 is
`b7fabb1c89ac07573487f386bc426ccf48a62f6055eca2335a18d520dbd55f44`; its checkpoint SHA256 is
`0974da172ed13b6cdcff99e522f7130b7700369a7526c8bdbbee88d299b4bcb0`, report SHA256 is
`825d0b6e68d420d7b470ae115eedf8795d096962515d35159f9247a04f6b0b37`, config hash is
`52d78273f647eba6`, and elapsed training time was 2,477.6 seconds on the RTX 4090 D. Test evaluation
was disabled and remains null.

## Immediate implications

1. Treat the completed five-seed 10k experiment as the analytic-domain reference, not a paper result.
   `gwyolo numeric-multiseed` resumes finished seed directories, validates manifest/config identity,
   and retains every seed including the lower 0.86975 run. Test data remained unevaluated.
2. The optional in-memory cache produces exactly identical metrics and reduced the 72-scene run from
   157.5 to 27.6 seconds (5.7×). Use it for fixed scaling scenes, but not as a substitute for online
   recipe diversity in publication training.
3. Repeat the scaling curve after replacing analytic signals with validated waveforms and real
   O1–O4a background anchors. Synthetic ease may exaggerate the slope and absolute IoU.
4. Keep O4b and the final test corpus locked until the real-data architecture, thresholds, and loss
   are frozen.

`gwyolo physical-finetune` is the next executable bridge. It accepts separate train/validation
materialized manifests, rejects waveform/injection/GPS-block overlap, constructs real-noise features
and relative physical-waveform masks, selects checkpoints only on physical validation chirp IoU, and
never accepts a test manifest. A frozen teacher distills the pretrained glitch head while the chirp
path adapts; this is a preservation constraint, not a substitute for real glitch labels.
Remote publication runs set `GWYOLO_CODE_COMMIT` to the exact deployed commit; the report also
records the literal command, environment, config hash, both manifest hashes, pretrained-model hash,
selected-model hash, and seed.

The first 2,000/500 physical fine-tune run is retained as a negative numerical-control result. Its
best validation IoU was about `0.1034`; audit then showed BNS target-mask occupancy as high as
`0.5579`. Physical strain power near `1e-48` had been cast to float32 inside the shared transform
before the relative mask threshold, causing underflow and quantization. Physical target construction
now scales each signal-only IFO to unit peak before power construction. This scaling affects only the
amplitude-invariant target morphology; the real-noise mixture input retains its physical amplitude.

The `physical_finetune_highres.yaml` promotion configuration uses 1,024 time bins over eight seconds
(7.8125 ms nominal bins). Its STFT path uses batched FFTs and vectorized frequency/time interpolation,
with a hand-checked equivalence test against `numpy.interp`; this makes the <=10 ms timing experiment
computationally executable without changing physical split identities.

`gwyolo injection-snr-annotate` computes per-IFO and network optimal SNR against a median-Welch PSD
from each injection's own full real-noise context, but integrates signal power only over the recorded
eight-second analysis window seen by the model. It is resumable and writes explicit `<4`, `4–8`,
`8–15`, `15–30`, and `>=30` strata. Physical training may require an annotated manifest and select
only train injections above a configured SNR floor; validation remains unfiltered so sensitivity
loss at low SNR stays visible rather than being removed from evaluation.
`gwyolo physical-snr-curriculum` can retain sub-floor train waveforms by deterministically rescaling
their signal-only component into an SNR 4–8 curriculum band. It never accepts validation/test rows,
does not alter their population weights, and explicitly reports that rescaling adds no independent
waveforms or GPS blocks.
Mask supervision is also visibility-aware per detector: an IFO plane below the configured optimal-SNR
floor is target-negative even when the network injection is retained, while the physically scaled
signal remains in the mixture input. This prevents the loss from demanding a resolved mask in an
antenna-suppressed detector and preserves the louder detector's instance mask.
Background and injection scorers resolve tensors from either `numeric_training` or
`physical_training`, so a promoted physical checkpoint can enter the same resumable probability,
candidate, FAR and sensitivity interfaces without a parallel ad-hoc runner.
The long physical ablation exposes focal BCE as an explicit configuration (`gamma=2`) while retaining
Dice and frozen-teacher glitch distillation. The ordinary BCE path is exactly `gamma=0` and has a
unit equivalence test, so any improvement can be attributed to a declared loss/config change.
Every physical epoch also writes an atomic last-state checkpoint containing model, optimizer,
data-loader RNG, complete history, best-validation state and immutable run identity. A restart
rejects changed code/config/data/model identities and resumes at the next epoch; a completed report
is idempotently returned rather than overwritten.

An earlier diagnostic annotation integrated the full 64-second signal and therefore overstated the
available input SNR for long BNS/NSBH waveforms. Its manifests and curriculum runs are retained as
negative controls but must not be used for final SNR-stratified evidence.
Physical fine-tuning now estimates the whitening PSD from the known real-noise context and applies
the same linear filter to mixture and signal-only target. Injection scoring uses the same stored
noise reference; continuous background is the special case where reference and input are identical.
This removes the earlier unwhitened-target versus whitened-input frequency mismatch.
The reference-whitened run then produced only 0.0124 validation IoU: filtering the target itself
introduced a label definition that the compact model did not learn. Whitening and target construction
are therefore explicit tensor-config axes. The promoted baseline returns to self-whitened inputs and
amplitude-normalized morphology masks; the failed noise-reference setting remains reproducible in
`physical_finetune_reference_whitened.yaml`.
