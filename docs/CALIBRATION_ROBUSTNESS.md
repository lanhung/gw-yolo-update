# Calibration robustness protocol

Calibration robustness is evaluated as a physical preprocessing stratum, not as rendered-image
augmentation. The frozen validation plan assigns one frequency-dependent complex strain response
to each `(scenario, observing run, IFO)` tuple. That response is shared by every background window
and injection in the tuple, so calibration uncertainty is not incorrectly treated as independent
pixel or event noise.

`multiplicative_rfft_strain_response_v1` linearly interpolates amplitude and phase at predeclared
frequency anchors, multiplies the real-strain Fourier coefficients, and returns a real time series.
The scorer then performs whitening and a fresh multi-Q transform. For noise-reference whitening,
the same response is applied to the mixture and PSD-reference strain.

Freeze the score-blind validation plan only after the candidate-calibration and injection-validation
GPS purposes have been separated:

```bash
python -m gwyolo.cli calibration-perturbation-plan-freeze \
  --background-manifest /artifacts/candidate_calibration/background_windows.jsonl \
  --injection-manifest /artifacts/injections/materialized_arrivals.jsonl \
  --config configs/calibration_perturbation_o4a_validation.yaml \
  --output /artifacts/calibration_perturbation_plan.json
```

Score one frozen scenario without changing model parameters or detector availability:

```bash
python -m gwyolo.cli trigger-score \
  --manifest /artifacts/candidate_calibration/background_windows.jsonl \
  --checkpoint /artifacts/model.pt \
  --config configs/physical_overlap_finetune.yaml \
  --output-dir /artifacts/calibration/envelope_plus/background \
  --required-split val \
  --coherence-config configs/physics_coherent_yolo_pilot.yaml \
  --calibration-plan /artifacts/calibration_perturbation_plan.json \
  --calibration-scenario envelope_plus

python -m gwyolo.cli injection-score \
  --manifest /artifacts/injections/materialized_arrivals.jsonl \
  --checkpoint /artifacts/model.pt \
  --config configs/physical_overlap_finetune.yaml \
  --output-dir /artifacts/calibration/envelope_plus/injections \
  --required-split val \
  --coherence-config configs/physics_coherent_yolo_pilot.yaml \
  --calibration-plan /artifacts/calibration_perturbation_plan.json \
  --calibration-scenario envelope_plus
```

The committed O4a envelope is deliberately labelled a conservative bounded stress test. It is not
an official calibration-posterior sample and cannot be described as one. A paper may either retain
the stress-test interpretation or replace it before freezing with official, hash-bound per-run/IFO
envelopes. Scenario thresholds must never be refit: candidate extraction and time slides are rerun,
but FAR and injection efficiency are evaluated with the original frozen validation threshold.
