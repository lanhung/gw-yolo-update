# GW-YOLO Research Pipeline

This repository turns the legacy GW-YOLO image experiment into a reproducible research pipeline. It provides:

- provenance-aware dataset auditing;
- group-safe train/validation/test splitting;
- configurable Ultralytics segmentation training;
- validation-only model selection and test evaluation;
- mask-preserving catalog inference;
- GWTC metadata joins and SNR-stratified hit-rate diagnostics;
- machine-readable experiment and quality-gate reports;
- physical-group data-scale audits and learning-curve target plans;
- deterministic multi-IFO, multi-resolution numeric scene generation;
- GWOSC O4 HDF5 acquisition with an O4b evaluation lock;

The project does **not** treat mAP on rendered Q-scans as a gravitational-wave search metric. Publication experiments must add continuous background, software injections, FAR/IFAR, and `<VT>` evaluation.

## Quick start

```bash
python -m pip install -e '.[dev]'
make test
PYTHONPATH=src python -m gwyolo.cli audit --config configs/legacy_remote.yaml
PYTHONPATH=src python -m gwyolo.cli pipeline --config configs/legacy_remote.yaml
PYTHONPATH=src python -m gwyolo.cli scale-plan \
  --manifest /root/GW-YOLO-v2-artifacts/data/manifest.csv \
  --output /root/GW-YOLO-v2-artifacts/data/scale_plan.json
PYTHONPATH=src python -m gwyolo.cli data-factory \
  --config configs/data_factory_pilot.yaml \
  --output-dir artifacts/data_factory_pilot
PYTHONPATH=src python -m gwyolo.cli data-factory \
  --config configs/data_factory_research.yaml \
  --output-dir artifacts/data_factory_research
```

The remote legacy config expects the original project at `/root/GW-YOLO` and writes only to `/root/GW-YOLO-v2-artifacts`.

## Stable CLI

```text
gwyolo audit        audit source data and provenance groups
gwyolo split        build a leakage-safe dataset
gwyolo train        train configured model candidates
gwyolo evaluate     evaluate one checkpoint on a frozen split
gwyolo predict      preserve every box and segmentation polygon
gwyolo catalog-eval join predictions to GWOSC metadata
gwyolo search-eval  freeze a validation FAR threshold and report test FAR/IFAR/weighted VT
gwyolo scale-plan   audit a manifest and generate independent-data scaling targets
gwyolo data-factory create leak-safe physical recipes and optional numeric tensors
gwyolo gwosc-pilot  acquire O4a strain and validate real multi-IFO preprocessing
gwyolo numeric-train train the validation-selected multi-IFO/multi-Q numeric baseline
gwyolo numeric-multiseed resume and aggregate validation-only runs across at least five seeds
gwyolo numeric-evaluate evaluate one selected numeric checkpoint with frozen thresholds
gwyolo recipe-subset build nested, leak-audited manifests for learning curves
gwyolo gravityspy-index download and stratify official real-glitch metadata anchors
gwyolo fit-curve fit an exploratory power-law learning curve to controlled scale points
gwyolo background-plan build DQ-safe continuous multi-IFO windows and true live-time exposure
gwyolo search-compare compare raw/mask-cleaned methods at a common validation-calibrated FAR
gwyolo oracle-deglitch establish the chirp-protected mask-cleaning upper bound in the time domain
gwyolo oracle-deglitch-benchmark measure the oracle upper bound over overlap and clean scenes
gwyolo learned-deglitch apply frozen soft masks and measure injected-signal retention
gwyolo trigger-score convert continuous DQ-safe windows into multi-IFO ranking triggers
gwyolo time-slide-background build split-safe nonzero time-slide background exposure
gwyolo injection-plan create volume-weighted, split-safe CBC injection recipes for `<VT>`
gwyolo injection-materialize project validated PyCBC/LAL waveforms into real detector strain
gwyolo injection-score score full-context-whitened physical injections with the frozen model
gwyolo pe-evaluate compare paired raw/cleaned AMPLFI/DINGO-compatible posteriors
gwyolo pipeline     run audit → split → train → test → catalog diagnostics
```

## Scientific positioning

AMPLFI and DINGO are parameter-estimation systems. The defensible comparison is therefore a joint low-latency workflow:

1. GW-YOLO: multi-detector chirp/glitch scene understanding and masks;
2. mask-informed reranking or deglitching;
3. AMPLFI/DINGO-compatible parameter inference on cleaned data.

Shared metrics are end-to-end latency, posterior coverage, searched sky area/volume, and robustness under glitch overlap. GW-YOLO's distinct contribution is interpretable time-frequency localization and background reduction.

See [DATA_FACTORY.md](docs/DATA_FACTORY.md), [PROJECT_PLAN.md](docs/PROJECT_PLAN.md),
[DATA_SCALING_PLAN.md](docs/DATA_SCALING_PLAN.md),
[NUMERIC_BASELINE.md](docs/NUMERIC_BASELINE.md),
[SEARCH_AND_DEGLITCH.md](docs/SEARCH_AND_DEGLITCH.md), and [PAPER_PLAN.md](docs/PAPER_PLAN.md).
