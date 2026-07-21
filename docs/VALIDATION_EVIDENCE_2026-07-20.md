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

The exploratory n=10,000 point completed the same 3,750 updates/60,000 examples in six epochs and
selected the required final update. Its validation-calibrated IoU is 0.03420 at threshold 0.5:
only 0.00043 above n=5,000 and 0.00188 above n=2,000, again far below the 0.01 promotion margin.
Report SHA256 is `e0adf35f2c31ef6cc20813f26b15a13a27be7b58fac94e39d1dbaa923fefe5ef`;
checkpoint SHA256 is `e94da142fa30464b73265db299bef84c116810e6d0bc1dbf4979169c097ae3e8`.
The one-seed curve therefore shows diminishing mask-IoU returns under a fixed example budget, but
does not establish a data-scaling law. The formal same-commit 2k/5k/10k × three-seed fixed-update
series has started, followed automatically by the frozen background/injection endpoint and the
independent 30-epoch control; `test_evaluation` remains null throughout.

For that endpoint the mixed 4,009-window O4a manifest was explicitly materialized to 824 validation
rows rather than silently filtered in the scorer. The split manifest SHA256 is
`a6dd57b9a1b3829cac534844f096e7151d0ada1934a40cb916ede86422429e2b`; its parent remains
`8f2285bd2dfaaed3d6be06d8302f12981caf7e6669d83e2cd2da1601e3e28f61`. The predeclared
development work point permits at most eight surviving background windows before applying the
frozen threshold to all 3,000 validation injections. Its nominal window FAR is diagnostic-only.

The formal fixed-update series is now complete at commit `b428896`: all nine 2k/5k/10k × three-seed
runs reached exactly 3,750 optimizer updates and 60,000 seen examples, passed zero-overlap audits,
hash-verified their checkpoints and retained `test_evaluation: null`. The frozen 824-window O4a
validation background and all 3,000 validation injections were scored for every checkpoint. Mean
weighted efficiencies are 0.08340, 0.08543 and 0.08698 for 2k, 5k and 10k, with seed sample standard
deviations 0.00407, 0.00164 and 0.00041. The mean gains are only +0.00202 and +0.00155. Only the
5k→10k direction is positive in all three seeds, and every injection-paired 95% bootstrap interval
for that step crosses zero. The 2k→5k comparison is positive with a nonzero interval in only one of
three seeds. This is evidence of a waveform-count plateau under fixed examples, not evidence that
the total physical/OOD corpus is large enough. It rejects blind 25k/50k waveform duplication while
strengthening the case for new GPS/run, V1 and glitch-family coverage. The equal-epoch control is
now complete and scale promotion remains false.

The independent 30-epoch control completed all nine endpoints at training commit `b1071a2` and
scoring commit `f33e1b3`. Mean weighted efficiencies are 0.08000, 0.12452 and 0.14897 at 2k, 5k and
10k, with seed sample standard deviations 0.00035, 0.00913 and 0.00408. Every seed improves at both
steps. The mean gains are +0.04452 and +0.02446, and all six injection-paired 95% bootstrap
intervals exclude zero. Summary SHA256 is
`9902a38df90c8b6ba46dfc6b1ad90768d62d6c7d1efbeb17c510f3f9c217438a`. This does not overturn
the fixed-update plateau: equal epochs couple data scale to 3,750/9,390/18,750 optimizer updates and
60k/150k/300k seen examples. Jointly, the controls say the present 10k corpus is under-trained at
3,750 updates, while additional unique waveforms at a fixed 60k-example budget have not shown a
material gain. The next action is therefore to retain the trained 10k equal-epoch arm as a
validation candidate and scale independent run/IFO/glitch coverage; 25k/50k remains blocked.

The first validation-only physical-coherence reranker is also a retained negative result. At eight
surviving background windows, morphology alone reaches weighted efficiency 0.08745, while
`morphology × sqrt(mean absolute lag correlation)` reaches 0.02109. The paired recovered-`VT`
change is -397,711.28, or -75.88%, with 95% bootstrap interval
[-494,680.72, -305,603.05]. This rejects that hand-designed multiplicative score. Coherence remains
useful as a separately calibrated physical coincidence feature or learned validation-only reranker;
it must not suppress morphology by construction.

When the all-instance candidate pipeline first enabled the same coherence summary, its resumable
JSONL checkpoint correctly failed after five windows because the nested arrival gate contained a
NumPy boolean scalar. Commit `a15e682` converts that boundary value explicitly to a JSON-native
boolean and adds end-to-end serialization regressions. The failed output is non-claimable and a new
same-commit validation run was started in a separate directory; no threshold or test data were
consumed.

That rerun completed as an integration diagnostic. The local per-cluster envelope has 0.9766 ms
sample resolution, but only 368/6000 detector arrivals match any candidate inside ±250 ms; its
median/90th/99th-percentile absolute errors are 133.4/223.2/245.8 ms. The resulting 501.6 ms H1–L1
window yields 9,221 background coincidences over 0.012295 equivalent years and only 11/3000
injections with any candidate pair. At the validation-only 100/year target, three injections survive
(weighted efficiency 0.00285). This demonstrates that the current segmentation masks do not supply
publication timing even though their saved strain gives sub-ms numerical resolution. The timing
gate now additionally caps the empirical uncertainty at a predeclared 10 ms, so this method fails
instead of converting a truncated ±250 ms association window into a misleading physical allowance.

Commit following that diagnosis adds a separate time-domain detector-arrival head. Unlike the prior
1,024-bin multi-Q classifier, it preserves IFO identity, incorporates an explicit availability-masked
network context, and supervises H1/L1/V1 against their geometric detector arrival times. Its 7.8125
ms grid only passes representation, not accuracy; validation p90 <=10 ms is still mandatory. The
bounded scale order is 2k then 5k/10k only after measured p90 improvement, so this architecture does
not justify blind data expansion by itself.

The first 2k detector-arrival run at commit `8826228` completed all 12 epochs in 303.4 seconds and
selected epoch 8 using validation p90. Its split audit has zero waveform, injection or GPS-block
overlap (2,000/3,000 unique train/validation waveforms over 76/26 GPS blocks). The 7.8125 ms output
grid passes the representation gate, but the frozen all-validation accuracy gate fails: median/p90
per-IFO error is 1.769/4.626 seconds and only 9.48% of 6,000 detector arrivals are within 10 ms.
BBH reaches 20.48% within 10 ms, while BNS and NSBH reach only 0.33% and 0.67%. The checkpoint SHA256
is `0a7392240ea58b60d1dbab9a87d230eca0ffe8c7f8031506fdeca60ddf7cb4ac`; the train and validation
arrival-manifest SHA256 values are `e4e8d6f3e0580b6df84f5921915e3ca3a319dde014c950a11e18763fd97dfbb4`
and `422fa08528bdc57c8d65e6123fb58209282d0ce755d404832d4951abfcfd09cd`.

That aggregate is deliberately not interpreted as timing conditional on detection. The physical
validation population contains 1,559/3,000 injections below network optimal SNR 4 and only 585 at
SNR >=8. At network SNR >=30, 81.25% of arrivals are within 10 ms and p90 is 12.68 ms; at SNR 15--30,
the corresponding values are 68.84% and 144.7 ms. This bimodality shows real learned timing at high
SNR but does not pass the search gate. It also exposes a needed separation between injection
coverage and conditional timing accuracy. The validation stratifier therefore keeps the all-sample
gate unchanged while adding per-IFO-SNR groups, worst-IFO error and pairwise relative-delay error.
Those conditional strata are diagnostic only until combined with candidate coverage at a frozen
search threshold. No 5k/10k timing scale is promoted from this result.

The explicit detectability stratification confirms that this is a bimodal miss problem rather than
a uniformly coarse estimator. For individual IFO arrivals with optimal SNR >=8, the median is
7.23 ms but p90 is 2.093 s and only 60.21% lie within 10 ms. Requiring both IFOs to have SNR >=8
leaves 275 injections: worst-IFO median/p90 is 8.82 ms/0.875 s, both-IFO 10 ms coverage is 56.73%,
and pairwise-delay p90 is 16.34 ms. Even the 149 injections with both IFOs at SNR >=10 have
worst-IFO p90 26.77 ms and pairwise-delay p90 10.55 ms. Reporting the good conditional medians alone
would therefore hide catastrophic outliers. Stratification-report SHA256 is
`4dc5ef0a29dffc64ec450656581047859571ba582214e780a4bd5d78f01464c3`.

The same-budget full-context v2 arm then completed exactly 1,500 updates/24,000 examples. Validation
p90 selected epoch 1: all-arrival p90 falls to 3.512 s, but the estimator has not learned usable
chirp timing. All-arrival 10 ms coverage collapses from 9.48% to 0.67%; among 275 both-IFO-SNR >=8
injections, joint 10 ms coverage falls from 56.73% to 0.73% and worst-IFO p90 rises from 0.875 s to
3.877 s. At both-IFO SNR >=10, worst-IFO and pairwise-delay p90 are 3.850 s and 2.489 s. Thus the
aggregate all-sample p90 improvement is a positional-prior artifact, not a timing improvement. The
checkpoint SHA256 is `24869d2312bc2bfaac62f76749c676625377da67ad0dbf998c15b4d7ee2cca76` and
report SHA256 is `b43dfa4c9ab8fa1ccea74dc556f7f163c51d865ac6f75a468745cd3b16155530`.
The precommitted paired gate remains the authoritative adjudicator, but these endpoint values already
exclude 5k/10k promotion under its conditional requirements.

The 3,000-injection paired comparison at commit `382612d` formally sets `promotion_allowed=false`:
only the two unconditional-p90 checks pass. All-sample p90 improves 21.66%, with paired p90-delta
interval [-1.219, -0.945] s, but joint 10 ms coverage falls by 7.03 percentage points. In the 275
both-IFO-SNR >=8 injections, worst-IFO p90 worsens by 3.002 s and 10 ms coverage falls by 56.0
percentage points, with paired interval [-61.82, -50.18] points. All five conditional/high-SNR
checks fail. Comparison-report SHA256 is
`537888eb940678491b01841aaf15b92af3f47cc77c198698b13040aefc16c30b`; v1/v2 prediction-manifest
SHA256 values are `f2ef74cfaea88143d4a8a4508fb3c61a855f90640263422eeb3664bb30af14c2` and
`9020a07a96dacc5781b481d0e74494d82af5e827fb3d05a28b42c781835e7605`.

The final same-budget v3 representation control replaces raw-strain timing features with a numeric
256-sample Hann STFT at an eight-sample hop. It still uses the identical 2k/3k split and exactly
1,500 optimizer updates. Its selected epoch-12 checkpoint has SHA256
`b9397f1db0ba6f38604db7bd50b1941114d4308cd2b1ca7f72a9203582211fc0`. Among per-IFO arrivals with
SNR >=8, p90 improves from 2.093 s to 51.59 ms, and for the 275 injections with both IFOs at SNR >=8
worst-IFO p90 improves from 0.875 s to 27.31 ms. This useful representation gain nevertheless fails
the frozen promotion gate: all-arrival p90 improves by only 3.57%, both-IFO-SNR >=8 joint 10 ms
coverage rises by only 0.73 percentage points with interval [-5.46, 7.27], and at both-IFO SNR >=10
worst-IFO/pairwise-delay p90 remain 21.70/10.48 ms. Only 2/7 checks pass, so
`promotion_allowed=false` and standalone full-window timing scaling is retired. The paired report
SHA256 is `62e9649374e512cc9a65946a7b21db5deb44bccfd082b7f18863e6b3e705c7f1`; the v3 prediction and
stratification SHA256 values are `1c66e90dd9ef2ad348171c1062f2ccb3f5d9ec9aa54601d2d17443b67d0dbe16`
and `ac9d6e28d7f1a48cc9c71c9d20acdf4e429677ace8086498a4f0b0dbbe5bf1d2`.

The first frozen-threshold candidate-support audit then shows why conditional timing training cannot
start at threshold 0.3. Although raw/padded arrival coverage is 99.75%/99.93%, the median/p90 union
fractions are 0.927/0.938 of the eight-second analysis window and the median truth-containing
proposal is 7.42 s wide. This is nearly full-window support rather than localized proposal recall.
The width-aware audit SHA256 is
`916a30bb2bb62fc987681f80eef6792a8ce5a7f594563d396b583d2b3e651de4`. A precommitted 0.3--0.9
threshold sweep therefore decides whether any validation-only coverage--compactness operating point
exists; if none passes, the next experiment must repair the proposal objective rather than train a
timing refiner on these broad intervals.

That seven-point sweep is now complete and its machine decision is
`promotion_allowed=false`. Thresholds 0.3/0.4/0.5 retain 99.93%/99.70%/98.48% padded coverage, but
their median union fractions are 0.927/0.906/0.781 and their median truth-containing widths are
7.42/7.08/3.17 s. At 0.6, median width finally falls to 1.08 s and p90 union fraction to 0.781, but
overall coverage has already fallen to 92.32%, BNS fails the 90% group gate, and median union
fraction is still 0.531. Thresholds 0.7--0.9 are compact but cover only 74.63%, 44.48% and 18.55%.
Because coverage is non-increasing and support width non-increasing with threshold, no untested
intermediate threshold can satisfy both the already-failed 0.6 coverage gate and the not-yet-passed
0.6 median-union gate. Selection-report SHA256 is
`763100f057afaf0d356df689c1dd70cc014e3a5403cff2853372835c58cd4cea`.

After every candidate and audit report was hash-bound, the reusable probability cache was released:
3,000 files/848,497,304 bytes. The trigger JSONL, seven candidate manifests, seven audits, checkpoint
and selection decision remain; exact recovery is the frozen scorer followed by extraction. The
eviction-report SHA256 is `a6fb3597f7ca6441b3affe4d29eaac1efb6acb312f7f2cb6a2600cd4c12824de`.

The first independent dense endpoint-proposal arm at commits `376f297`/`fcb6f66` then completed the
same group-safe 2k/3k split and exactly 1,500 optimizer updates. It warm-starts the v3 numeric
spectrogram encoder but uses one sigmoid endpoint map per IFO, leaving the chirp/glitch segmentation
checkpoint unchanged and retaining every connected proposal. The validation-loss-selected epoch 7
checkpoint SHA256 is `b683b4442dd51799d21cfdca9406924f238d641f15ec54d05a23ca93494f7539`;
the training report SHA256 is `c0c2d783c8bc238cedbc468a652986ebd110fdb640e0cf7f6744c53b55284786`.
An invalid preliminary process exposed `BCEWithLogits(-inf, 0)=NaN` for missing V1 slots and was
stopped before any epoch/checkpoint existed; `fcb6f66` indexes unavailable IFOs out of the loss and
adds a regression test. The accepted rerun is finite and reports zero waveform, injection and GPS
block overlap.

The coarse 0.3/0.4 results bracketed a genuine validation Pareto region, so commit `078b761` evaluated
the full systematic 0.31--0.39 grid with the same fixed checkpoint and no retraining/test access.
Thresholds 0.35--0.39 all pass the predeclared gate. The machine selector chooses 0.39: padded
coverage is 95.83% over 6,000 detector arrivals, every BBH/BNS/NSBH and SNR>=4 group exceeds 90%,
median/p90 union fractions are 0.212/0.257, and median truth-containing width is 0.281 s. The
selection report SHA256 is `160933842b4068a58a4e56eceef17837862d4cc82af2c5cc0091030a0754b62b`;
the selected audit/candidate-manifest SHA256 values are
`006b2d88d032dea542c987e7d5f5b600b7a0c476a68057f0175d0cbaa3594b9e` and
`db1ed6338f8c2d8ef6800db1fc1ad1a6981bf4092e81e7f7e0f2f302a20a9bcc`.

Passing proposal support does not imply useful timing or ranking. The selected threshold emits
127,102 candidates: median/p90 22/30 per detector arrival. Nearest-peak median/p90 errors are
72/380 ms, and taking only the highest-score proposal preserves padded truth support for just 30.65%
of arrivals. Even top 16 reaches only 92.45%; top 24 reaches 95.42%. Thus the scalar proposal score
cannot be used as a timing estimate or silent top-k filter. Every proposal remains retained while a
candidate-conditioned local refiner/abstention head is trained and calibrated.

Commit `0c35bad` applied the frozen 0.39 proposal rule to the 2,000 training parents and produced
74,126 candidates without top-k pruning. Its group-safe refiner plan has 596 validation-selection
parents and 2,404 disjoint validation-calibration parents; train/validation waveform, injection and
GPS-block overlap is zero. The training, selection and calibration manifest SHA256 values are
`0bbc365e2dff8949454dccc76e289a79c0b4bd3ce2772fbb722251bd7c87f81e`,
`2d0370d345209c87131034fd87fa9b709e8e08df8fbea4f6ee10d8b40d9a95a2` and
`725ff8f44120311e8ee8d93794606405719a7dbfd723ada03ff01cc8a020df27`.

The first local-refiner baseline at `77f6880` completed 1,500 updates and failed cleanly. Its selected
epoch 3 has calibration candidate AP 0.2275, positive-candidate timing errors 0.492/1.151 s at the
median/p90 and only 6.87% within 10 ms. Report/checkpoint SHA256 values are
`14db2be133fb1e82166261eecd12f06f81040af5a4e8b53c6f725ee99d270774` and
`53941d75bdc75374535604fe521e29d7f08e3be7ec90d4d0ed78a369b27cf970`. This is a
validation-only negative result, not a search metric. A supervision audit then found that 24,279
training crops physically contain the arrival, while the connected-interval support label marks only
11,202; 13,077 valid timing crops were therefore being treated as negatives although interval
geometry is absent from the local strain input. Commit `d4d0330` corrects the learning target to
physical arrival-in-crop, keeps the original interval label as a proposal audit field, and adds a
Gaussian timing distribution plus continuous coordinate loss. Its result must still pass the frozen
`8f3554f` validation gate before background calibration.

The corrected from-scratch v2 arm completed 3,000 updates at `d4d0330`. On the 596-parent
checkpoint-selection partition, candidate AP rises to 0.4498, but remains below the frozen 0.50
gate. Its training report/checkpoint SHA256 values are
`7de6cd1ba465f089f2d5555a7e17b9e3862ff95494620419e753d7460adfbabb` and
`1d011387b5e49d4309e15aec010e481bd2ec7899f0d51d7370dfd35d581e3ca5`.
Commit `b318fdf` froze probability expectation—not post-hoc argmax switching—as the continuous
timing estimator before scoring the 2,404 calibration parents. The completed `26cc330` report has
candidate AP 0.4601 and localizable-candidate median/p90 timing errors 0.461/1.126 s. Threshold 0.5
retains 99.83% of 4,808 physical detector arrivals, but retains a median 10 candidates per arrival;
the top-score refined time is within 20 ms for only 13.12% of all arrivals. AP, p90 timing and top
score timing gates all fail, while search promotion remains explicitly false. Validation-report and
prediction-manifest SHA256 values are
`900d31239835e3836337b8bba81e7890f2ac5f92bd79cbaa03e7bda1f950b41f` and
`2c0e86885cf1b89fde5234ebbf0a282936af06edd1fb0c465f8be9af09e1c6f9`.

This failure does not authorize blind physical-scale growth: the local network was trained from
scratch even though the frozen dense endpoint arm already learned an informative representation on
the same 2,000 physical train parents. Commit `6491461` therefore defines a bounded transfer
diagnostic that loads the exact dense endpoint backbone and sharpens it with the same 3,000-update
budget. Because that endpoint checkpoint was originally selected on all 3,000 current validation
parents, the warm arm sets `calibration_evaluation_allowed: false`; even a strong selection result
requires a new group-disjoint O4a calibration corpus before any threshold can be frozen.

The endpoint-warm diagnostic completed exactly 3,000 updates at `6491461`. Selection AP is 0.4566,
only 0.0068 above the from-scratch arm; timing median/p90 is 0.494/1.114 s and 10 ms coverage is
3.59%. Calibration metrics are `null` by construction. Report/checkpoint SHA256 values are
`fe3475fb253dd58bcdeead0aa50278f6cc0b44d169fd30dc5dd92fc005af1248` and
`be0aa66cabad9b1d4b49db2f3c85cb6bb8d499fcfb6b460e7fc0e8c12c531b2b`.
Representation transfer therefore produces no qualitative repair and does not justify more updates
on the same 2,000 physical parents. The promoted diagnostic direction is an all-candidate set-level
network ranker with explicit pairwise light-travel constraints, followed by calibration on newly
acquired GPS blocks.

The first truth-free set heuristic at `36ce7f7` evaluates every H1/L1 interval pair compatible with
the predeclared 10 ms light-travel limit. All 596 selection parents have at least one compatible
pair, and the compatible-set oracle contains a padded truth pair for 94.13% (Wilson 95% interval
91.94--95.75%). Yet the validation-selected score/center/width rule recovers only 29.70%, with
selected-pair peak-error p90 4.61 s. Report SHA256 is
`f2726238d4d1f247005a434acff7ed472d951b97e6477352be6366302ecc5110`.
Thus proposal-set support is adequate but linear hand ranking is not. The next bounded arm is a
group-safe learned pair/set score that evaluates every compatible pair; the oracle is a ceiling
diagnostic and may not be reported as operational recall.

The group-safe learned geometry baseline at `266d2dc` then trained on 35,320 compatible train pairs
and scored all 12,205 selection pairs. It completed the fixed 900-update budget with zero train/
selection waveform, injection or GPS-block overlap. Overall top-1 padded truth is only 31.21% and
peak p90 is 4.60 s, so both gates fail; report SHA256 is
`40c804ed712ea095081c6184f244bbbc2cd7e9c0333acd40ee21ab4f232c7a2b`.
The failure is strongly SNR-stratified: 27 SNR 15--30 and five SNR>=30 parents both reach 100%
padded top-1 with 35.5/29.3 ms p90, SNR 8--15 reaches 69.86%, and SNR<8 reaches 20.98%.
This motivates strain-content/coherence features at the intermediate-SNR boundary, not a larger MLP
over the same proposal geometry.

The matched-budget strain-feature arm at `d9aef12` adds absolute/signed H1/L1 correlation within
the 10 ms physical lag, local RMS/peak amplitudes and interval duration for every compatible pair.
It also fails: best-epoch padded top-1 is 31.71%, peak p90 4.58 s and SNR 8--15 coverage 68.49%,
statistically indistinguishable from or slightly worse than the geometry-only 31.21%/69.86% arm.
Report/checkpoint SHA256 values are
`de7c67531b5452d02048b55360143fed4b7171a6d865e9f048dbfa5b945824a0` and
`794e5e2bd5f1a47afc0223703062127d2cdc3546bf7ebe918d84197478f02bc6`.
Simple local correlation is therefore retired. The next data-scaling experiment must combine richer
time-frequency candidate embeddings with 2k/5k/10k independent physical parents under fixed-update
control; expanding the same 16--23 scalar pair features is not justified.

The stronger 10k/30-epoch mask checkpoint was then evaluated under the corrected gate at commit
`a83eadd`. It increases arrivals associated inside ±250 ms from 368 to 464/6000, but median/p90/p99
errors remain 125.1/221.3/246.7 ms. The 10 ms empirical gate is false, so execution stops before
time slides and threshold calibration. This clean negative result separates window-level sensitivity
(which improved strongly with 30 epochs) from unusable sub-window localization. The staged pipeline
then hash-verified and released 824 background plus 3,000 injection probability files (1,082,764,926
bytes total); candidate/calibration reports and immutable eviction intents remain.

## Real-glitch physical overlap and aligned network contexts

Commit `bcb7c8e` introduced a time-domain real-glitch overlap factory. A remote two-row validation
smoke used verified Gravity Spy O2/O3 strain artifacts and O4a physical injections. Both outputs are
finite `3 IFO × 3 Q × 96 × 96` numeric tensors, unavailable detector planes are exactly zero, and
mixture strain reconstructs as real glitch plus physical injection. One low-per-IFO-SNR signal has
an intentionally empty visible chirp mask while its physical strain remains present in the mixture;
the target gate therefore does not delete subthreshold signals. The manifest SHA256 is
`5b218a86cf20f9d651ab5d787e527b0d4e82c9ee8904f4982a9e3136acf78d70`; report SHA256 is
`2c48526c3405714e9af833082323e357b5da7099545ba68a6a6634aba647ec2a`.

This smoke is engineering evidence only because its Gravity Spy artifact contains physical strain
for the event IFO alone. The scale runner is queued behind the verified Gravity Spy train/validation
bank and will create 300/100 group-unique overlap rows, then run the joint waveform/injection/glitch/
GPS split audit. The follow-on detector-set fine-tune waits for that audit, active package installs
and GPU jobs to finish, and a final disk preflight. It retains a clean-injection validation gate and
does not consume test data.

Commit `c391a10` added an explicit companion-IFO acquisition plan. The first official O2 shard
contains five unique glitches over four network GPS blocks. All five have full H1/L1 64-second
contexts and none has V1 at those GPS times, so the only legal detector subset is H1L1. The two
official 4-kHz source files total 256,022,424 bytes. Plan manifest SHA256 is
`8484ffc320ad9136db38bd02600ed91a0d88421c19165c978ceed7f9e12f3974`; report SHA256 is
`45e863c519a5864ced18232e0a2537eb7ff01fbf6b86887588c4037e86f88ed1`. Commit `1f21eeb`
adds resumable full-file verification, per-IFO DQ checks, aligned raw-strain storage, detector
availability, weak-mask provenance and verified source eviction. The recovered shard is now
complete: five H1L1 rows over
four network GPS blocks, with report SHA256
`d78b79315158e583aba2c212f66da42d26c8b756544d79a0d455416c14276866`.

The first network-aware physical-overlap smoke then paired those five glitches with five group-unique
O4a validation injections. Every row contains aligned H1L1 strain, and the generated manifest/report
SHA256 values are `98f629ee1f3e8ced98a707bae64ea2669619342d1a2c472e84a25388f02c10fb`
and `56ad53d7e32abf6e2816dcfeba8e5905d9e97f838b7f36e9203834db1c32a8da`.
Commit `8614e99` converted exactly those rows into paired clean/contaminated injection overrides;
the report SHA256 is `acdaa623fe204433d72edc904d2df69b3c8e034c63525931c412de433e9c0fa0`.
This is an executable five-row smoke only. Mask-conditioned gain remains blocked on the scaled
real-overlap checkpoint, clean non-inferiority, continuous background and a much larger aligned
glitch corpus.

Commit `dbfe4de` makes overlap construction network-aware. A network Gravity Spy row may be paired
only with an injection supporting every available IFO; the coherent physical signal is added to all
available detector strains before fresh transforms. The event-local weak glitch mask remains local,
all chirp and glitch instances remain stored, and the report distinguishes aligned-network from
single-IFO rows. Physical-lag coherence, human weak-mask audit and continuous-background FAR remain
hard gates.

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

## Candidate content and physical-parent scaling — 2026-07-21

The nested 10,000-parent training corpus now has geometric H1/L1 detector-arrival annotations for
every row. The annotated manifest contains 10,000 unique injections and waveforms and preserves the
train-only SNR scaling fields; its SHA256 is
`67cd81b48c91488afacf0af7ee393e6bb4a24f763008b4c520f42f55ea816ba9`, and the annotation-report
SHA256 is `46f26886093f828dad3a941de7e995b63c22bb4179af6080502a0dc19f13864f`.

Commit `69696e8` made frozen endpoint application resumable in immutable 256-parent shards. Applying
the unchanged endpoint checkpoint `b683b444...` and validation-selected threshold 0.39 completed
40/40 shards and produced 367,749 candidates: 177,909 H1 and 189,840 L1. All connected components
are retained and `top_k_pruning` is null. The final candidate-manifest SHA256 is
`dbed7554f2fbb9664bf710c7bf4da1c1785d00e354a63637b707c9d361af9914`; the application-report
SHA256 is `4985bfc853b44e19fdd0c094c589b7e5bc552869d701e73792ad10136278aeca`.
A same-identity rerun returned the existing result in about three seconds and reproduced the same
manifest hash.

Commit `866abd6` then created strict nested 2k/5k/10k physical-parent views containing
74,126/184,187/367,749 candidates. The 2k parent manifest exactly reproduces the earlier group-safe
training manifest (`e4e8d6...`), while every view keeps all candidates and counts physical samples
as unique waveform/injection parents. The scaling-plan report SHA256 is
`f738faa33d8cce8da84999b64b90a38aa2e0d64877c57d2d8d3a025220f9f196`.

The bounded v3 ranker replaces the failed 16--23 scalar MLPs with a trainable log-STFT CNN shared
between H1 and L1. Each compatible pair receives two whitened 1.5-second crops on one common GPS
axis plus proposal geometry; no truth time is used to center the crop. The fixed-update and
fixed-epoch curves use the identical 596-parent validation-selection set:

| Budget | Physical parents | Updates | Top-1 padded pair | Peak p90 | SNR 8--15 top-1 |
|---|---:|---:|---:|---:|---:|
| fixed updates | 2,000 | 900 | 0.3322 | 4.477 s | 0.6849 |
| fixed updates | 5,000 | 900 | 0.3238 | 4.364 s | 0.7534 |
| fixed updates | 10,000 | 900 | 0.3322 | 4.342 s | 0.7534 |
| fixed epochs | 2,000 | 1,104 | 0.3339 | 4.342 s | 0.7260 |
| fixed epochs | 5,000 | 2,736 | 0.3289 | 4.368 s | 0.7397 |
| fixed epochs | 10,000 | 5,472 | 0.3389 | 4.342 s | 0.7260 |

The predeclared scale evaluator fails the required overall top-1 gain, reproducible SNR 8--15 gain,
and p90 reduction checks; only the tolerance-aware monotonicity check passes. Its machine diagnosis
is `waveform_scale_plateau_with_fixed_gps_support`, and its report SHA256 is
`e860b99b4603429f3c18fb3fabecb0b769b536264975ddde4f3868fac9c42470`. This wording is important:
all three scales still use the same 76 training GPS blocks, so the result rejects further waveform
multiplication on fixed noise support, not additional independent detector-noise or glitch data.
Scaling to 25k/50k remains disallowed. The promoted data axis is independent GPS/run/glitch
diversity; the next representation must expose actual multi-Q candidate structure rather than add
epochs or scalar summaries.

## Decision

Training data are far below publication needs, but the required increase is not a blind multiplication
of rendered or analytic scenes. The observed gap is dominated by physical-domain diversity,
independent GPS/glitch coverage, background exposure and temporal representation. The next promoted
experiment must use validated physical waveforms on real O1–O4a noise, scale unique waveform and GPS
groups independently, extract sub-window clustered candidates, and choose thresholds only from the
expanded validation background. O4b/GWTC-5 remains locked until those choices are frozen.

## Real-glitch, OOD and exposure update — 2026-07-21

The split-safe physical-overlap bank is complete at 300 train and 100 validation mixtures. A joint
audit reports zero overlap for glitch ID, glitch GPS block, injection GPS block, injection ID,
mixture ID, network GPS block and waveform ID. The first single-IFO weak-mask fine-tune completed
20 epochs but no checkpoint passed the frozen clean-retention requirement. At epoch 20, clean chirp
IoU retention is 0.5522, overlap chirp IoU is 0.0164 and overlap glitch IoU is zero. This arm is a
negative control; it is not eligible for checkpoint promotion or a five-seed claim.

The first held-family OOD baseline also fails. Its threshold was selected on 307 known-family
validation rows only and fixes known abstention at 15/307 = 4.89%. On a disjoint 19-row evaluation
set, all 17 held-out Blips are falsely accepted: true abstention 0/17 and unknown false acceptance
17/17 (Wilson 95% interval 81.57%--100%). Diagnostic AUROC is 0.5294. The report SHA256 is
`8ae3bea24ba1653e9cdf1417985042c75561ebbfbb13ac3a1a8d6e421fa8be47`; the selected checkpoint
SHA256 is `82b9574ce61c892823cac24efb55414c158cfb3b4d32060684a8765bf9ade94c`. These results reject
closed-set softmax-style embedding confidence as the O4 OOD policy.

Aligned network-context planning now covers 1,345 train rows over 448 network GPS blocks and 91
source files, plus 306 validation rows over 123 network GPS blocks and 95 source files. Train
detector subsets are H1L1=572, H1L1V1=721, H1V1=51 and L1V1=1; validation subsets are
H1L1=85, H1L1V1=189, H1V1=24 and L1V1=8. Both plans are assigned to seven bounded,
source-connected shards, ensuring that each official source file is materialized and evicted as a
unit. The validation plan/report SHA256 values are
`a552e4db3ab2e0bdd57319d769d02273bbdc02d6115df86beb3f143c30e53976` and
`4cf0e0c7d5b15a2f8da0f3acc355acb08e2e17ae865ad151481fc5b6915147e3`; its shard manifest
SHA256 is `3396267edfb0062d82bd4090ca166ac4f9ad30520d27d0c4d5596acf1ac38641`.

The fresh O4a background shard contains 1,393 eight-second windows over 45 GPS blocks. Its frozen
validation partition contains 301 windows over ten blocks. Three hundred non-cyclic positive-lag
shifts yield only 361,200 seconds = 0.01145 years of observed exposure. Even with zero surviving
events, the 90% FAR upper limit is 201.17/year, just 0.0497% of the exposure needed for a
0.1/year claim. The optimistic all-unordered-pairs lower bound is 13,479 valid dual-IFO windows.
The exposure report SHA256 is
`b55d36a14ddee6cd7d948bf20f349130953d0ef575a298448a09b34dc8414b85`. Therefore the current
background remains an engineering shard and cannot support FAR/IFAR or `<VT>` publication claims.

Tomte was then frozen as the next held family by validation sample count before any Tomte model
score was opened. Its split contains 1,427 known train rows, 273 known calibration rows and a
53-row evaluation set with 27 unknown Tomtes plus 26 group-coincident known artifacts; all three
roles have disjoint GPS blocks. The split-report SHA256 is
`cd7d4fbe8ed3f4641766643ca190d5f9b887c6d84549c2ce575c7f86022630e8`.
Commit `7f3b566` precommits supervised contrastive training and logit-energy scoring. The known-only
threshold abstains on 13/273 calibration rows. Frozen Tomte evaluation reaches diagnostic AUROC
0.7735 and true abstention 5/27, but unknown false acceptance remains 22/27 (81.48%; Wilson 95%
63.30%--91.82%) and known false abstention is 3/26. Its report SHA256 is
`54aa87d7ef7490c70c5fb5813e2500eb9429ffd3458dc21e28de3a202f7cb5dc`. This is an improvement
over the first embedding diagnostic but fails operational promotion; further OOD development waits
for aligned detector contexts and a precommitted outlier-exposure or self-supervised representation.

To address the exposure deficit, parent-plan pair indices 4--39 are frozen as nine additional
four-pair O4a shards. They are queued after aligned Gravity Spy acquisition to avoid competing for
the bounded cache. A global `hash_threshold_v1` 40-pair background plan and exposure audit are
queued after all source files pass full-file verification. Raw pair count is not a success gate:
only DQ-valid validation windows and their realized non-cyclic positive-lag exposure determine
whether the 0.1/year target becomes measurable.
