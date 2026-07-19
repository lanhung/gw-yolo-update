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

No result from the analytic backend may be reported as O4 sensitivity, FAR, IFAR, or `<VT>`.
