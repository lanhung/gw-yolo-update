# Verified O4a validation checkpoint — 2026-07-20

## Added physical-domain controls

- Physical manifests: 2,000 train / 500 validation, 38 / 13 disjoint GPS blocks.
- Direct LALSimulation waveform-equivalence gate: 30/30 passed across BBH/BNS/NSBH; report SHA256
  `0498c7ee60c8adbc567582e06d44f8c3ab5c24893e4ab359531c36b2012dfe2b`.
- Corrected-mask all-row validation IoU: 0.0473; SNR>=4 filtered validation IoU: 0.0431. Both are
  negative validation-only results, not search sensitivity.
- Empirical optimal-SNR manifests: train SHA256
  `4399ce654d0781f08d06de9e2fdc2396dfce6b113767a6e0844c8d82c5b3f940`, validation SHA256
  `573c901db4b2b9ce3e7dde44d9e3037a5d893252d22586997b256b70a50e0c9b`.
  These hashes are retained full-context diagnostics and were superseded after finding that their
  SNR integration included signal outside the model's eight-second analysis window.
- Corrected eight-second SNR manifests: train SHA256
  `66d84048891ca1d61b03dba149f7166b166d49c501f8b2216ab79067b828dafc`, validation SHA256
  `bf6a7222cb9f6538e3dc3d5a54bf613c14738c21d65b524aebea3568e905d40f`. Corrected median network
  SNR is 3.89/3.91 and 51.35%/51.2% of train/validation injections are below SNR 4.
- Gravity Spy: 80,496 unique anchors; network-block-safe split has zero overlap. The train strain
  plan maps 59,933 anchors to 16,297 official GWOSC files at 93.23% coverage.
  The bounded 32-file shard plan contains 510 shards, preserves all anchors/files once, and has
  manifest SHA256 `5fcc63ae5e0e3dc8d5504317f92be19d2cc703c149fe4bbebb8808708959e718`.

## Corrected-SNR training ablations

The 30-epoch focal-loss run on the corrected eight-second SNR curriculum selected epoch 29 and
reached calibrated validation IoU 0.03615 (checkpoint SHA256
`3219b677173550dc0bf7eaaa4600c49e89711ae4ece662b0fb7c1f3635ba1648`). It is a negative result:
focal loss did not recover the earlier 0.04725 validation value. Under the current per-IFO visibility
target, however, the earlier checkpoint scores only 0.03317, so the focal checkpoint is the fair
same-target promotion source.

The first curriculum over-concentrated about 81% of rows at SNR 4--8. The train-only exact-quota
manifest now contains 800/700/400/100 rows at SNR 4--8/8--15/15--30/30--50, respectively, with
SHA256 `b0c46d40f540bb667a6d567eaa1e5ef69987d8bf86e2af3de243c11dc1a5e7a9`. It still represents
exactly 2,000 waveforms and 38 GPS blocks and is not a data-scaling point.
The resulting 20-epoch quota ablation selected epoch 12 and reached calibrated validation IoU
0.03946, a 9.1% relative increase over the same-target focal result (checkpoint SHA256
`72b6b6fafdd5050ef6b6d1bad14d1e41a3e3c4c52bf92a8cc247fa22a30e511c`). This is evidence that
training-proposal coverage matters, but the absolute result remains inadequate. Its 96-bin endpoint
audit still fails badly: 3.17 s median and 5.83 s 90th-percentile absolute error.

The first one-file Gravity Spy execution completed on an O3a H1 `Extremely_Loud` anchor. Official
full-file statistics and DQ bit sums passed; the finite 3-IFO x 3-Q x 96 x 96 tensor manifest has
SHA256 `88fb6bb414088faccf6cdd3c60eb6ee341bdc592f3131c65d439eeb31a7e5659`. Its mask is explicitly
weak metadata supervision (`human_pixel_masks=0`), so it validates acquisition and preprocessing,
not segmentation quality.

The 1,024-time-bin promotion selected epoch 4 and reached only 0.029385 validation IoU (checkpoint
SHA256 `d840310ea9b94eab69d8250d653ce7d8f087ffe286d1e89c46e7e7e499b8bc6b`). Its 7.8125-ms bin
width passes the representation gate, but it does not pass timing accuracy: the last-active-mask-bin
endpoint error is 3.63 s median and 6.17 s at the 90th percentile on 367 validation injections with
non-empty visibility-gated targets. Thus higher resolution alone does not validate millisecond
coincidence; explicit localization supervision and false-activation control are required.

## Evidence boundary

This checkpoint uses only analytic training data and O4a development/validation data. It is not a
GWTC-5 result, does not open the locked O4b test corpus, and does not support an astrophysical FAR,
IFAR or `<VT>` claim. Its purpose is to decide what the next data and representation investments
must be.

## Source integrity and continuous background

The first H1 download was rejected after a Fletcher32 failure. A fresh download was scanned in
1,048,576-sample chunks and both H1 and L1 matched all 16,777,216 samples, official mean/standard
deviation/extrema, file sizes, NaN fraction and DQ/injection bit sums.

| Artifact | SHA256 / value |
|---|---|
| Verified H1 source | `2150091da920c99d21717fc40cce11fbd62103d879d13209184929e3cbf12e92` |
| Verified L1 source | `9bb67f4014ebe724e9aa31d5a20065b90303c6322f498c3fc87ced188c28f238` |
| Verification report | `40ef49868ecc2fe623dbb0535257a9cb68599e968e20044840f065c41f84a4dd` |
| Background report | `01f55f445a21197cf538d136c1ee694f16bcebf7a9d57be88797f39d95d8d06d` |
| Background manifest | `b5f3b5e762d238527e922f1ecd4230a4bb2b9e65454db2003b7886b136e3bf52` |
| Background windows / blocks | 499 / 16 |
| Validation / test live time | 768 s / 768 s |
| Cross-split GPS-block overlap | 0 |

The validation-only 31-shift integration produced 11,904 seconds (`3.77215e-4 yr`) exposure and 70
window-level coincident rows. Its manifest SHA256 is
`99b3a648a8f550b29a346d553b528bd18c83a6d24970d4cbdfe6262248cc58db`.
The 83-ms output grid and tiny exposure prohibit a search claim.

## Model selection and physical injections

Five 10k analytic-data seeds completed without test evaluation. Validation-selected seed 20260721
has mean IoU 0.88810; the five-seed mean is 0.88087 with sample standard deviation 0.00678.

Exactly 300 validation injections were rematerialized against the verified sources: 124 BBH,
95 BNS and 81 NSBH. All 300 scored successfully with full-context whitening and saved float16 masks.
The materialized manifest SHA256 is
`4d307b3c9c763e323f1e61cfffd00cb9063e19fac777b690a2d7516a63467256`;
the trigger manifest SHA256 is
`2982a9fcb6b790c34ef82909e3c6f4aa13d8db4878111e0ea59185e2db18fc45`.

| Validation threshold | Measured/limited FAR | Unweighted efficiency | Weighted efficiency (95% bootstrap) |
|---|---:|---:|---:|
| 0.89005 | 7,953/year (3 events) | 22.3% | 16.8% `[10.8%, 23.4%]` |
| 0.93830 | zero observed; 90% UL 6,104/year | 9.0% | 6.1% `[2.5%, 10.3%]` |

The second row's target input was 1/year, but the exposure cannot measure that FAR. It is explicitly
reported as a stress-test threshold. At the first threshold, weighted family efficiencies are 16.3%
BBH, 28.3% BNS and 25.1% NSBH. The BBH population dominates total volume weight, so its weak result
drives the 16.8% network value.

Learned cleaning preserved injected-signal projection: mean retention is 1.00001 for BBH, 1.00113
for BNS and 1.00004 for NSBH. The learned-deglitch manifest SHA256 is
`c6ae9475332699d13f7d5249b8bde5e7cea4c57a572d96bd0d9ccf3beba85976`.
This proves signal preservation on these injections, not glitch removal or fixed-FAR improvement.

## Decision

Training data are far below publication needs, but the required increase is not a blind multiplication
of rendered or analytic scenes. The observed gap is dominated by physical-domain diversity,
independent GPS/glitch coverage, background exposure and temporal representation. The next promoted
experiment must use validated physical waveforms on real O1–O4a noise, scale unique waveform and GPS
groups independently, extract sub-window clustered candidates, and choose thresholds only from the
expanded validation background. O4b/GWTC-5 remains locked until those choices are frozen.
