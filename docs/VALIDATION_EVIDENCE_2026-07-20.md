# Verified O4a validation checkpoint — 2026-07-20

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
