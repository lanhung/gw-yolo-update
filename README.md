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
gwyolo pipeline     run audit → split → train → test → catalog diagnostics
```

## Scientific positioning

AMPLFI and DINGO are parameter-estimation systems. The defensible comparison is therefore a joint low-latency workflow:

1. GW-YOLO: multi-detector chirp/glitch scene understanding and masks;
2. mask-informed reranking or deglitching;
3. AMPLFI/DINGO-compatible parameter inference on cleaned data.

Shared metrics are end-to-end latency, posterior coverage, searched sky area/volume, and robustness under glitch overlap. GW-YOLO's distinct contribution is interpretable time-frequency localization and background reduction.

See [PROJECT_PLAN.md](docs/PROJECT_PLAN.md), [DATA_SCALING_PLAN.md](docs/DATA_SCALING_PLAN.md), and [PAPER_PLAN.md](docs/PAPER_PLAN.md).
