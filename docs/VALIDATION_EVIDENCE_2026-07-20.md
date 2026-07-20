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

Two low-resolution temporal-profile losses were also negative. Reusing positive weight 10 reached
0.04104 IoU only after validation calibration to threshold 0.7, but endpoint error remained 3.08 s
median/5.70 s at 90%. Decoupling temporal positive weight to 1 reduced endpoint error to 2.75 s
median/5.42 s at 90% but collapsed calibrated IoU to 0.02909. Neither is promoted; the next timing
model requires an explicit peak/coalescence-time head rather than max-profile BCE.

The first explicit temporal-peak competition run (weight 1.0) selected epoch 15 and reached only
0.02598 validation IoU (checkpoint SHA256
`099b251de2ea9bc8da61263f2ec662b387649943ae0be9bc03696d8a670e74ca`). Against the exact-SNR-quota
checkpoint on the identical 500 validation injections, however, its threshold-independent
peak-to-target-endpoint error improved from 3.50 s median/6.25 s p90 to 1.33 s/3.53 s. For SNR
15--30, p90 improved from 4.14 s to 0.142 s. Thus the localization signal is informative but weight
1.0 damages the mask objective. Both checkpoints still fail the timing gate: their 83.3-ms bins
cannot establish <=10-ms accuracy, and their measured errors are far larger. A frozen weight-0.1
Pareto ablation is the final low-resolution check before replacing the shared mask maximum with a
separate high-resolution timing head.

The weight-0.1 Pareto run reached 0.03340 IoU (checkpoint SHA256
`71eb1aab375d4f79e48114f43a358ee343b4a81cdfb4696fc8b71fa9409feaf6`) and 1.42 s median/3.75 s
p90 peak error. It retains more mask quality than weight 1.0 but is not competitive with the
no-peak-loss mask checkpoint, while providing nearly the same timing behavior as weight 1.0. Shared
mask-maximum timing tuning therefore stops here. `physical-timing-train` now defines an independent
1,024-bin candidate refiner trained against exact geocentric injection GPS rather than a quantized
mask endpoint; it selects checkpoints by validation p90 absolute timing error and leaves the mask
checkpoint as a separate product.

That exact-GPS refiner is also a retained negative result. On the same 2,000/500 train/validation
manifests it selected epoch 9, but validation error was 1.519 s median and 4.209 s p90; only 23/500
(4.6%, Wilson 95% interval 3.08%--6.81%) landed within 10 ms. This is worse at p90 than the
low-resolution weight-1 peak model despite the 7.8125-ms output grid, so it is rejected rather than
multiseeded. The report SHA256 is
`b958906308ac66ada621a0f1922254d2e49f5f29b2badf59fc1d2f491b744f44`; checkpoint SHA256 is
`a554c4d28c01632b50c42159e826fad0f8b012bc9d54864a0a33a5982f538762`.

Five fully verified Gravity Spy pilot shards now merge to five unique train glitches/blocks across
O2/O3a/O3b and four classes (manifest SHA256
`9f992a0dd7b726e2ff5025d9414736a024af0f23384e8be13505aa1d127549c8`). Their five official source
files were evicted only after output revalidation, recovering 648,372,382 bytes while preserving
URL/SHA256 tombstones. A bounded five-file scale run covering O1--O3b is in progress.

The five-file scale run is now complete and, together with the pilot, hash-merges to 709 unique
train glitches over 162 network GPS blocks (manifest SHA256
`0ca4bdb68a21a6176e08b2f77aefaae2cb4bb45b0038412753e3bd709df0bd8f`). All five source files were
evicted only after numeric-output verification. One O2 row and 111 O3b rows were explicitly rejected
for non-finite strain context; requested rows are otherwise fully accounted. The high-yield-file
sample is strongly imbalanced (`1400Ripples=352`, `Whistle=166`), so it is acquisition evidence,
not a promoted training distribution.

A whole-source-file deficit selector now adds 1,075 proposed train glitches across 36 files and 407
blocks, bringing every one of 21 labels to at least 50 when combined with the 709 verified rows
(selection manifest SHA256
`0eb8272a498770e6b4eab7ffcebf3b68bdbd7397d59a82680c49a5cd9a429cd0`). The independently frozen
validation split maps 7,643/8,257 anchors to 4,851 official files (plan SHA256
`e00a56714c4b5eeaa85b8d240fcbb799acfdc387492ad2044b4b5165d1e63bf0`). Its bounded selection has
434 glitches, 158 blocks and 52 files (SHA256
`3d25dcffe9ea9e5d2d8a7a19a6590051fe8209f0b09afcdddb905d441389f9b1`); 20 labels reach at least
20 examples, while only eight complete-context `Wandering_Line` examples exist. Validation
materialization is running before any real-glitch model selection. All masks remain weak metadata
supervision.

The train deficit selection was reproduced under the standardized provenance contract at commit
`6b88e9b`; its manifest remained bit-identical at
`0eb8272a498770e6b4eab7ffcebf3b68bdbd7397d59a82680c49a5cd9a429cd0`. The selection report
SHA256 is `9ec446317e33b4cc36e2727a9806e775e2167f774042461c6092fca8821b179c`. Its 1,075 rows and 36
official files are now assigned to 36 single-file resumable shards with manifest SHA256
`0d65cf37e420f56fb56178dcca97e1a72c37feeba68ae6466a3426ebbcdec918` and report SHA256
`544a2c4968d08693bf205ee5a16a11bb3bf35c5c9f77cd48e0a3f84fbc6fac00`. Train acquisition is
queued only after validation acquisition releases the bounded source cache; every source will again
be evicted only after numeric-output verification.
supervision and cannot support a segmentation claim without a frozen human pixel-mask audit.

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

The second verified O4a acquisition batch adds four disjoint H1/L1 pairs (batch-report SHA256
`a5e0e6bcbf7715b73e9af230b841dd400b633136b2d3610620b4bb5dc282abb3`). A single global split over
both batches yields 4,009 context-safe windows and 128 blocks with zero overlap: 2,385/824/800
train/validation/test windows and 19,080/6,592/6,400 seconds. The global manifest SHA256 is
`8f2285bd2dfaaed3d6be06d8302f12981caf7e6669d83e2cd2da1601e3e28f61`.

The next continuous-background acquisition is frozen rather than opportunistically downloaded. The
official O4a API contains 3,309 aligned H1/L1 4-kHz files; seed 20260720 selected 800 4,096-second
pairs spanning GPS 1368268800--1389408256. Their 37.93 raw coincident detector-days are only an
upper bound before DQ, event, context and category exclusions. The acquisition-plan SHA256 is
`d9043337438db689b581bade1922c1191ed52fde94ce056d460c4c9e74316d04`; its declared status is
`development_acquisition_plan`, and it contains no O4b data.

On that background, a 10k-train/3k-validation recipe plan contains 13,000 unique waveform and
injection IDs over 102 blocks (manifest SHA256
`90d258fdeeec19955f718bd5e565176fbb5893140b4b3da3ded9703b01b29cd5`). A fresh direct-LAL 30-case
gate passed. A 100-case scaled-float16 storage pilot used 12,409,144 bytes and had worst relative L2
reconstruction error 2.14e-4; full 10k train materialization is therefore running resumably under
the <=1e-3 storage gate.

The full 10k train materialization is now complete: 10,000 unique selected train recipes occupy
1,207,608,352 bytes with maximum signal reconstruction relative L2 `2.1813e-4`, passing the
pre-registered `<=1e-3` gate. The manifest SHA256 is
`6a9a491f513c83579374c51431a09064340a4bf652c7bbd01ff28d97053f0b79`; its waveform certificate,
recipe and background hashes remain `c768b6a65b...`, `90d258fdeeec...` and `8f2285bd2dfa...` as
recorded in the machine report. Eight-second empirical-SNR annotation is running on train, while the
unchanged 3k validation corpus has completed separately. Neither process consumed test recipes.

The frozen 3k validation corpus and its eight-second empirical-SNR annotation are now complete. It
contains 1,350 BBH, 900 BNS and 750 NSBH; median network optimal SNR is 3.864 and 1,559/3,000
(52.0%) lie below SNR 4. The unmodified validation manifest SHA256 is
`76e5e248ec70c24eb4cc74b39d152fd86524577573de1257eba847245f3d128d`; annotation report SHA256
is `1e2983abd128ca11b3f65792bd4d9c58a5ce18f717d4fa75bedc74014d5dbbb1`. This distribution is
frozen for selection and calibration; it is not a locked test and is never SNR-rescaled.

An identity audit also prevents a misleading reuse of the old 2k pilot in the new scale curve. All
2,000 old waveform/injection IDs occur in the 10k core, but four old training GPS blocks overlap the
new 3k validation GPS blocks. The old result remains an ablation only. Fresh nested 2k/5k/10k
subsets will be generated from the new split-safe 10k SNR-quota manifest, and each starts from the
same analytic checkpoint SHA256
`61730b9734a90fd01e4678470026cacc8c3e78cdf008e68cbcaf88ebd3ae8e72`.

The 10k empirical-SNR annotation is complete: median network optimal SNR is 3.847 and 5,190/10,000
(51.9%) native injections are below SNR 4. Its output-manifest SHA256 is
`29b4a918323e3909f0e51486d60e61504829fda0b8c53778bdd9b6425ecabca4`; report SHA256 is
`93a5c253344afa0fba68eae168e157a577edb122c2393e1cfe83e01e73d21125`. The deterministic
train-only proposal contains exactly 4,000/3,500/2,000/500 rows in SNR 4--8/8--15/15--30/30--50,
while retaining 10,000 unique waveform and injection IDs and 76 GPS blocks. Quota-manifest SHA256
is `bf6575b14a5817f7d0b916b18f5f89e3f125479a85c73343c4b627357c0c0590`; report SHA256 is
`e1d8088a653d8a58d5b66711951395942a92e95fda3e46b99c9a01347a1f5aab`.

Fresh nested scale subsets now pass the frozen validation audit. Their 2k/5k/10k manifests have
SHA256 `e9027c25bfa252e323727725a7af9801b30717af48c7cf9d6bdc570f60f1d62c`,
`445398681546bcdf3511b3ad9c76cbdba174318a32885400ffb899e39a39277e` and
`bf6575b14a5817f7d0b916b18f5f89e3f125479a85c73343c4b627357c0c0590`. Each has the exact
requested number of unique injections/waveforms, all already cover the 76 train GPS blocks, and all
have zero injection, waveform and GPS overlap with the shared 3k validation corpus. The nesting and
joint family/SNR-stratum report SHA256 is
`13f943ec9ee87dc6f78cd88012c4d48d13cfa68fb314d3907117d914198d73ec`.

The first scale-launch preflight exposed and fixed a tensor-contract regression before any optimizer
update. Physical targets are nine IFO/Q planes and `MultiIFOQNet` outputs two classes for each of
those nine planes; an intermediate implementation had incorrectly treated the output as a single
network-level mask. Commit `eb463a4` restores plane-wise chirp supervision and collapses only
IFO/Q plus frequency for the auxiliary temporal profile. A real O4a sample passed an actual
forward/loss/backward smoke with feature and target shapes `(9,96,96)`. The failed launch artifacts
are not reused; the fixed scaling series starts in a new commit-tagged output directory.

The corrected exploratory fixed-update n=2,000 run then completed exactly 3,750 updates and 60,000
seen examples. Final-update epoch 30 (not the better-looking epoch 29) was selected by protocol; its
validation-calibrated IoU is 0.03232 at threshold 0.6. The split audit again has zero injection,
waveform and GPS overlap and `test_evaluation` is null. Report SHA256 is
`f7a2b1974f72646ef443264729a4c98ff24a2b9150559bef812ca47952d0c61b`; checkpoint SHA256 is
`50287eb0da92db3078a7aed51ca8d52669ba079555802fa668e92656f9d83471`. This is an exploratory
single seed at commit `eb463a4`; the strict summary gate will not pool it with the final-code
three-seed series.

The matching exploratory n=5,000 control also reached exactly 3,750 updates/60,000 examples and
selected final-update epoch 13. Its validation-calibrated IoU is 0.03377 at threshold 0.5, only
0.00145 absolute above the exploratory n=2,000 value and far below the predeclared 0.01 promotion
margin. Report SHA256 is `ebff219beafce6029e6e80352ecd6da7a6d9b32be71901e52bcdadae5a9b1e9d`;
checkpoint SHA256 is `ed19e175e3d3ccb032bc510e0548ed1e03348a0802fb1821a9a60a410fe31ef5`.
This single-seed difference is not a scaling conclusion; formal same-commit seeds and the
fixed-epoch protocol remain required.

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
