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

If the frozen empirical timing calibration was produced by an earlier code commit, do not use the
generic whole-scorer compatibility escape hatch: the calibration-aware preprocessing is
intentionally different. First freeze a narrower implementation proof that compares the AST hashes
of the predeclared candidate timing, network-ranking and block-clustering functions:

```bash
python -m gwyolo.cli calibration-timing-transfer-compatibility-audit \
  --reference-code-dir /immutable/checkout-at-timing-commit \
  --candidate-code-dir /immutable/checkout-at-calibration-commit \
  --reference-commit <timing-code-commit> \
  --candidate-commit <calibration-code-commit> \
  --output /artifacts/calibration_timing_transfer_compatibility.json
```

Apply the original timing calibration to both scenario candidate streams only with that proof and
the exact frozen perturbation plan:

```bash
python -m gwyolo.cli candidate-timing-apply \
  --candidates /artifacts/calibration/envelope_plus/background/candidates.jsonl \
  --calibration-report /artifacts/baseline/candidate_timing_calibration.json \
  --calibration-perturbation-plan /artifacts/calibration_perturbation_plan.json \
  --calibration-timing-compatibility-report /artifacts/calibration_timing_transfer_compatibility.json \
  --output /artifacts/calibration/envelope_plus/background/calibrated.jsonl
```

The transfer audit permits only the controlled calibration preprocessing change; any change to
candidate peak extraction, network ranking or block clustering fails closed. The candidate report
also requires the same checkpoint and model-config hashes as the original timing calibration.

The committed O4a envelope is deliberately labelled a conservative bounded stress test. It is not
an official calibration-posterior sample and cannot be described as one. A paper may either retain
the stress-test interpretation or replace it before freezing with official, hash-bound per-run/IFO
envelopes. Scenario thresholds must never be refit: candidate extraction and time slides are rerun,
but FAR and injection efficiency are evaluated with the original frozen validation threshold.

After candidate extraction, application of the already-frozen timing calibration, execution of the
same score-blind block-permutation schedule, and physical injection ranking, freeze one receipt for
each scenario:

```bash
python -m gwyolo.cli calibration-perturbation-scenario-freeze \
  --plan /artifacts/calibration_perturbation_plan.json \
  --background-score-report /artifacts/calibration/envelope_plus/background/trigger_score_report.json \
  --injection-score-report /artifacts/calibration/envelope_plus/injections/injection_score_report.json \
  --background-timing-application-report /artifacts/calibration/envelope_plus/background/calibrated.jsonl.report.json \
  --injection-timing-application-report /artifacts/calibration/envelope_plus/injections/calibrated.jsonl.report.json \
  --background-search-report /artifacts/calibration/envelope_plus/search/val_candidate_time_slide_report.json \
  --injection-ranking-report /artifacts/calibration/envelope_plus/rankings/val_injection_candidate_ranking_report.json \
  --output /artifacts/calibration/envelope_plus/scenario_receipt.json
```

The receipt rejects missing/changed hashes, test rows, an altered model/config/commit, uncalibrated
candidate timing, a different timing calibration, and a background or injection manifest outside
the frozen plan. It records `threshold_fitted_or_selected: false`.

Once every frozen scenario has a receipt, evaluate all of them together. Repeat
`--scenario-receipt` exactly once per scenario:

```bash
python -m gwyolo.cli calibration-perturbation-evaluate \
  --plan /artifacts/calibration_perturbation_plan.json \
  --baseline-calibration-report /artifacts/candidate_search_calibration.json \
  --scenario-receipt /artifacts/calibration/envelope_plus/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/envelope_minus/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/amplitude_plus_phase_minus/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/random_draw_000/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/random_draw_001/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/random_draw_002/scenario_receipt.json \
  --scenario-receipt /artifacts/calibration/random_draw_003/scenario_receipt.json \
  --config configs/calibration_perturbation_o4a_validation.yaml \
  --output /artifacts/calibration/calibration_robustness.json
```

The evaluator requires the exact seven-scenario set, identical checkpoint/config/code/timing,
identical GPS-block background schedule and live time, and identical physical injection identities
and weights. It applies the original baseline validation threshold to every background and injection
row. The predeclared gate requires the paired-bootstrap lower bound on absolute weighted-efficiency
change to remain above `-0.05` and scenario FAR to remain at most twice the target FAR. Results are
also stratified by explicit detector subset. Passing this validation robustness gate does not open
O4b or GWTC-5 and does not by itself authorize a scientific claim.
