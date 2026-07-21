# Numeric data factory and real-strain acquisition

## Implemented status

The repository now has a deterministic numeric data path independent of the legacy PNG export.
Each `SceneRecipe` records split, scene type, run, GPS block, detector network, Q values,
waveform/injection ID, glitch ID and detector, source family, target SNR, and seed. The audit fails
if a waveform, injection, glitch, or GPS block crosses train/validation/test.

The first remote pilot generated 104 scenes on the research server:

| Item | Result |
|---|---:|
| Tensor shape per scene | 3 IFO × 3 Q × 96 frequency × 96 time |
| chirp-only | 26 |
| glitch-only | 26 |
| overlap | 41 |
| quiet | 11 |
| Scenes with non-empty chirp masks | 67/67 expected |
| Scenes with non-empty glitch masks | 67/67 expected |
| Cross-split physical-ID overlaps | 0 |
| Manifest SHA256 | `140a18513693ea12eeba59869e1476c574f7aac3ff467803eddd5f8d136aa45d` |
| Materialized size | 42 MB |

This is a generator integration test, not an astrophysical benchmark. Its chirps and glitches are
analytic stress signals used to test provenance, tensor shapes, masks, and I/O. Publication training
must replace the analytic source backend with validated waveform models and real noise/glitch anchors.

## Materialization policy

Three modes are supported:

- `full`: features, masks, mixture, clean strain, chirp strain, and glitch strain; for debugging only.
- `tensor`: float16 features and masks; for frozen validation/test shards.
- `recipe_only`: physical recipes only; the production training default.

The full pilot uses approximately 0.41 MB per scene. Naively extending it to 200,000 scenes would
consume roughly 82 GB, while the current server has only about 14 GB free. Therefore
`configs/data_factory_research.yaml` defines 160,000 train, 10,000 validation, and 30,000 test recipes
but materializes none of them. Training scenes must be synthesized deterministically online. The
locked evaluation set may be materialized in bounded float16 shards after its source population is
frozen.

The complete 200,000-recipe manifest has now been generated and audited remotely. It is 93,093,283
bytes (about 89 MiB), has SHA256
`ac36fc3732fc8583b1903b78cccb50048b8f2680d36d1c483ee576569c5b9505`, and has zero cross-split
overlap. Its exact composition is:

| Split | chirp-only | glitch-only | overlap | quiet | Total |
|---|---:|---:|---:|---:|---:|
| train | 40,000 | 40,000 | 64,000 | 16,000 | 160,000 |
| validation | 2,500 | 2,500 | 4,000 | 1,000 | 10,000 |
| locked test | 7,500 | 7,500 | 12,000 | 3,000 | 30,000 |

This proves the provenance and storage path at target cardinality; it does not mean that 200,000
validated astrophysical waveforms and real noise anchors have already been populated. Replacing the
analytic recipe backend is still required before training a paper model.

## GWOSC O4 ingestion

`gwyolo gwosc-pilot` uses the official GWOSC API v2, records every source URL and file SHA256, reads
GPS-aligned HDF5 strain and DQ/injection masks, downsamples with an FFT anti-alias filter, whitens on a
longer context, and produces the same IFO × Q × frequency × time interface as the synthetic backend.

O4a is development/calibration data. O4b is rejected by default and can only be opened with the
explicit `--allow-locked-evaluation-data` flag after thresholds and architecture are frozen. This
matches the project rule that GWTC-5/O4b cannot influence model selection.

The initial real-data target is `GW231123_135430`, because official O4a data include aligned H1 and L1
4 kHz files. The acquisition is resumable because each 4096-second file is about 129 MB and the current
remote route to GWOSC is bandwidth-limited.

The H1 half of that target has now passed end-to-end validation:

- official HDF5 size: 129,057,388 bytes;
- source SHA256: `da6eb3cc48e2a0abb41ec31f70a2ecc3e425bc3c907c77e6ccc6d99794a59671`;
- strain: 16,777,216 samples at 4,096 Hz;
- event: GPS 1384782888.6, using a 64-second whitening context and 8-second output;
- output: H1 × 3 Q × 96 × 96 finite numeric tensor;
- tensor SHA256: `cbb7e12c11d6d914509a8f10d9bfe5c502cfe49f9f9c3f1f0bf4e8ab8cbf4aeb`;
- all 64 context seconds have `DQmask=511` and `Injmask=23`.

This validates real O4a HDF5 access and preprocessing for one detector. It is not yet a network result;
the corresponding L1 file must be validated and aligned before any multi-detector claim.

## Gravity Spy anchor index

`gwyolo gravityspy-split` derives an IFO-independent `network_gps_block` from observing run and
64-second GPS epoch, then hashes that group into train/validation/test. All H1/L1 records and all
glitch IDs in the same network interval therefore remain together; the report explicitly audits
glitch-ID and network-block overlap.
`gwyolo gravityspy-strain-plan` then maps each already-split anchor to one official GWOSC HDF5 file
that contains the entire requested whitening context. Boundary and unavailable cases remain explicit
rejections. The plan is metadata only: no real-glitch learning claim is allowed until every selected
source is downloaded, hash/DQ verified and converted to numeric arrays and masks.
`gwyolo gravityspy-strain-shard` assigns every source file and all of its anchors to exactly one
bounded shard. This enables resumable download/verify/extract/evict execution without splitting one
file's glitches across cache jobs or pretending that the bounded cache reduces the full corpus.
`gwyolo gravityspy-strain-materialize` executes one such shard resumably: every official source is
fully checked against its GWOSC statistics and DQ bit sums before a 64-second whitening context is
converted to an 8-second numeric multi-Q tensor. The accompanying masks are conservative weak labels
derived from Gravity Spy duration, peak frequency and Q metadata. They are explicitly marked as
non-human masks and cannot support a segmentation-accuracy claim until a pixel-mask audit is frozen.
The command does not evict verified source files automatically.
Individual anchors whose 64-second context contains publisher-recorded non-finite strain or lacks
the DATA bit are explicit row-level rejections. They do not abort other anchors in the same verified
file; completed plus rejected IDs must exactly equal requested IDs in the final report.
`gwyolo gravityspy-numeric-merge` then verifies every shard report, manifest and numeric-sample hash
before merging one declared split. Duplicate glitch IDs and mixed train/validation/test rows are
fatal; weak and human pixel-mask counts remain separate in the merged report.
`gwyolo gravityspy-strain-evict` is the only supported cache-reclamation path. It rechecks the
completed report, numeric manifest, every sample hash, source hash and cache-root containment before
unlinking an exact HDF5 path. Its atomic tombstone retains URL, SHA256 and byte count so the source
is reproducible from GWOSC; numeric samples are never evicted by this command.

The aligned-network path additionally treats detector validity as observed data.
`gravityspy-network-strain-materialize` evaluates every planned H1/L1/V1 context independently. A
bad companion IFO is zeroed and removed from the effective validity mask only when the event IFO
and at least one other IFO remain usable; the planned subset is retained separately for audit. An
invalid event IFO or a resulting one-detector row is rejected. This is detector-set downgrade, not
silent zero-filling, and the report counts every downgrade and unusable-detector reason.

Expanded aligned-network corpora must not train directly from separately selected historical
train/validation roles. `gravityspy-network-corpus-resplit` freezes the score-blind
`source_component_balanced_v1` assignment over all verified merge reports, keeping every connected
official HDF5 source component and GPS block in one role. `gravityspy-network-corpus-audit` then
hash-verifies every numeric sample and requires zero train/validation overlap in glitch ID, network
GPS block, official source URL and numeric sample hash before overlap materialization or training.

After a stricter detector-set implementation is introduced,
`gravityspy-network-recovery-plan` can freeze only rows rejected by earlier completed shards. It
re-hashes every source report/state/partial/manifest/sample, proves full shard accounting and emits
the original rejected glitch IDs with their prior reasons. Running the normal network materializer
on this immutable plan avoids reacquiring already accepted rows. Recovery rows are existing
physical identities and are never counted as newly generated glitches.

`scripts/run_gravityspy_network_recovery.sh` performs plan → selective materialization → verified
source eviction → old-plus-recovered merge for train and validation. The separate
`scripts/run_recovered_overlap_ablation.sh` then rebuilds the paired overlap bank, re-runs the joint
leakage audit and trains equal-budget uniform/family-balanced arms. Both scripts are resumable at
immutable report boundaries and require explicit paths; neither contains a machine-specific default.

For the next independent scale, run `gravityspy-network-strain-plan` on the full group-safe train or
validation anchor manifest, then use `gravityspy-network-strain-select`. The selector treats every
connected H1/L1/V1 source-file component as one acquisition unit, excludes existing glitch IDs,
GPS blocks and source files, and greedily fills per-family deficits under an explicit source-file
budget. Its report separates label, run, event-IFO, detector-subset and new-GPS coverage. It does
not claim that a balanced draw creates new physical examples; every selected row remains one
original Gravity Spy ID in one frozen split.

The official H1 O1 metadata CSV from Zenodo record 5649212 has also been downloaded and verified
against the publisher MD5 `91963313b1574e083bc58915e0aa8ca1`. Of 15,305 rows, 10,988 pass a 0.9
ML-confidence threshold after excluding Chirp/No_Glitch/None_of_the_Above. Stratifying at up to 100
per class yields:

- 1,391 unique Gravity Spy glitch IDs;
- 1,138 unique 64-second GPS blocks;
- 19 glitch classes;
- anchor-manifest SHA256
  `2a08d499f003c4d6181dc983f5c5eee1084cb6ba3ed1de2e39f02e577b71cd7c`.

The four Omega-scan URLs are retained as views of one physical glitch, never counted as four samples.
GPS-block identity is also retained because distinct triggers in the same background block cannot be
split independently. O3a/O3b indexing is the next expansion because it adds newer detector domains
and O3-specific morphologies.

H1 O3a indexing is now complete as well. The 90,238,691-byte publisher file passes MD5
`29aea278b622cd97496971f7c07f7d6a`; 80,763 raw rows contain 59,857 eligible high-confidence
non-chirp glitches. Stratification at up to 1,000 per class yields 10,450 unique glitch IDs in 7,917
independent 64-second GPS blocks across 21 classes, including O3-specific Blip_Low_Frequency and
Fast_Scattering anchors. The manifest SHA256 is
`f6cc0a84f7f5ca894b9897cc9ec5033f566d4e2bff11b7c06bedbc05216c231e`.

Combined with the O1 pilot, the indexed real-glitch pool now has 11,841 selected rows before
cross-run de-duplication, versus only 48 glitch-only physical groups in the legacy training split.

Primary data references:

- [GWOSC API v2](https://gwosc.org/api/)
- [GWOSC O4 technical details](https://gwosc.org/O4/o4_details/)
- [GWOSC O4 discovery-event data documentation](https://gwosc.org/o4_eventdata_docs/)
- [Gravity Spy O1–O3 machine classifications and metadata](https://zenodo.org/records/5649212)

## Production replacement work

The next backend must preserve the same recipes and split identities while replacing approximations:

1. generate BBH/BNS/NSBH waveforms with PyCBC/LALSimulation or an equivalent validated engine;
2. sample sky position and coherent H1/L1/V1 antenna responses and light-travel delays;
3. use real O1–O4a GPS noise blocks and Gravity Spy IDs as background anchors;
4. use a validated constant-Q transform (GWPy) alongside the dependency-free Q-conditioned STFT
   baseline;
5. produce paired mixture/clean targets for mask gating and posterior-bias experiments;
6. freeze real O4b background and injection manifests before evaluating the selected system.

## Streaming numeric background bank

Multi-detector run-scale acquisition cannot retain every 4,096-second HDF source on the experiment
disk. `background-bank-materialize` therefore hash-verifies every exact source file referenced by a
bounded background manifest, extracts each unique full-context numeric noise window once, stores it
as float32 with detector/GPS/index metadata, and writes a resumable enriched manifest. Subsequent
signal-only or scaled-float16 injection artifacts reference that bank; loading verifies both hashes
and refuses detector, window, sample-rate or context mismatches.

`background-bank-evict-sources` is the only authorized source-removal boundary for this workflow. It
re-hashes the bank report and manifest, opens and validates every bank artifact, re-hashes every
source, and requires every exact source path to lie below an explicit bounded cache root before
unlinking it. Its report records each removed path/size/hash and the public-GWOSC recovery route.
This enables download → verify → extract → verify → evict streaming without weakening provenance or
duplicating the same 64-second noise context for every injection.

No result from the analytic backend may be reported as O4 sensitivity, FAR, IFAR, or `<VT>`.
