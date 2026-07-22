from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.calibration import (
    apply_frequency_dependent_calibration_response,
    freeze_calibration_perturbation_plan,
    load_calibration_perturbation_scenario,
    response_for_row,
)
from gwyolo.cli import main


def _write_config(path: Path) -> None:
    path.write_text(
        """
calibration_perturbation:
  protocol: run_correlated_frequency_response_v1
  required_split: val
  default_observing_run: O4a
  seed: 1234
  target_sample_rate: 1024
  model_ifos: [H1, L1, V1]
  anchor_frequencies_hz: [20, 100, 300, 480]
  random_draws: 4
  envelope_templates:
    test_envelope:
      maximum_amplitude_fraction: [0.08, 0.05, 0.04, 0.08]
      maximum_phase_degrees: [8, 5, 4, 8]
      source:
        identity: hand_calculated_test_envelope
        semantics: bounded_stress_test_not_posterior
  run_template_assignments:
    O4a: test_envelope
""".lstrip(),
        encoding="utf-8",
    )


def _write_manifests(tmp_path: Path, shared_block: bool = False) -> tuple[Path, Path]:
    background = tmp_path / "background.jsonl"
    injection = tmp_path / "injection.jsonl"
    background.write_text(
        json.dumps(
            {
                "window_id": "window-1",
                "split": "val",
                "observing_run": "O4a",
                "gps_block": "O4a:block-background",
                "ifos": ["H1", "L1"],
                "ranking_score": 1e9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    injection.write_text(
        json.dumps(
            {
                "injection_id": "injection-1",
                "split": "val",
                "gps_block": (
                    "O4a:block-background" if shared_block else "O4a:block-injection"
                ),
                "ifos": ["H1", "L1", "V1"],
                "ranking_score": -1e9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return background, injection


def test_frequency_dependent_calibration_has_hand_calculated_constant_gain() -> None:
    sample_rate = 1024
    time = np.arange(1024) / sample_rate
    strain = np.sin(2 * np.pi * 64 * time)
    response = {
        "application": "multiplicative_rfft_strain_response_v1",
        "anchor_frequencies_hz": [20, 100, 300, 480],
        "amplitude_fraction": [0.1, 0.1, 0.1, 0.1],
        "phase_radians": [0.0, 0.0, 0.0, 0.0],
    }

    perturbed = apply_frequency_dependent_calibration_response(
        strain, sample_rate, response
    )

    np.testing.assert_allclose(perturbed, 1.1 * strain, rtol=0, atol=1e-12)


def test_calibration_plan_is_validation_only_score_blind_and_run_correlated(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    plan_path = tmp_path / "plan.json"
    _write_config(config)
    background, injection = _write_manifests(tmp_path)

    assert (
        main(
            [
                "calibration-perturbation-plan-freeze",
                "--background-manifest",
                str(background),
                "--injection-manifest",
                str(injection),
                "--config",
                str(config),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["candidate_scores_inspected"] is False
    assert plan["test_rows_read"] == 0
    assert plan["scenario_count"] == 7
    assert plan["purpose_gps_block_overlap"] == 0
    assert plan["envelope_interpretation"].endswith("not posterior calibration samples")

    background_scenario = load_calibration_perturbation_scenario(
        plan_path,
        background,
        "background",
        "random_draw_000",
        1024,
        ("H1", "L1", "V1"),
    )
    injection_scenario = load_calibration_perturbation_scenario(
        plan_path,
        injection,
        "injection",
        "random_draw_000",
        1024,
        ("H1", "L1", "V1"),
    )
    background_row = json.loads(background.read_text(encoding="utf-8"))
    injection_row = json.loads(injection.read_text(encoding="utf-8"))
    assert response_for_row(background_scenario, background_row, "H1") == response_for_row(
        injection_scenario, injection_row, "H1"
    )


def test_calibration_plan_rejects_purpose_overlap_and_changed_manifest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    background, injection = _write_manifests(tmp_path, shared_block=True)
    with pytest.raises(ValueError, match="share GPS blocks"):
        freeze_calibration_perturbation_plan(
            background, injection, config, tmp_path / "overlap-plan.json"
        )

    background, injection = _write_manifests(tmp_path, shared_block=False)
    plan_path = tmp_path / "plan.json"
    freeze_calibration_perturbation_plan(background, injection, config, plan_path)
    with injection.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="manifest differs"):
        load_calibration_perturbation_scenario(
            plan_path,
            injection,
            "injection",
            "envelope_plus",
            1024,
            ("H1", "L1", "V1"),
        )


def test_calibration_response_rejects_nonphysical_envelope() -> None:
    with pytest.raises(ValueError, match="envelope is invalid"):
        apply_frequency_dependent_calibration_response(
            np.ones(64),
            1024,
            {
                "application": "multiplicative_rfft_strain_response_v1",
                "anchor_frequencies_hz": [20, 100, 480],
                "amplitude_fraction": [-1.0, 0.0, 0.0],
                "phase_radians": [0.0, 0.0, 0.0],
            },
        )
