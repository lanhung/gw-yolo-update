# Search-statistics and mask-deglitch evidence

## Continuous-background protocol

`gwyolo background-plan` intersects per-second DQ masks across detector files, excludes event or
hardware-injection intervals, forms numeric windows, and assigns entire coarse GPS blocks to
train/validation/test. Live time is the union of intervals, not the sum of possibly overlapping
windows. The resulting manifest is the input contract for trigger generation and time slides.

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

## Fair raw-versus-cleaned comparison

`gwyolo search-compare` calibrates a separate validation-background threshold for raw and
mask-cleaned methods at the same target FAR. It then freezes both thresholds and evaluates the same
test background and the same weighted injections. Reports include Wilson efficiency intervals,
weighted-efficiency bootstrap intervals, paired recovered-`<VT>` change, and clean/overlap strata.

This interface is designed for the later AMPLFI/DINGO experiment: raw strain and mask-cleaned strain
must share events, injection weights, live time, waveform population, and FAR definition. A mAP
comparison is explicitly not part of the protocol.

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
