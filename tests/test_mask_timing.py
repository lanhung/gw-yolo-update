from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.mask_timing import run_mask_timing_validation


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, development_passed: bool = True) -> tuple[Path, Path, Path]:
    pipeline = tmp_path / "pipeline" / "mask_search_pipeline_report.json"
    pipeline_value = {
        "status": "validation_only_end_to_end_mask_search_pipeline",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "development_gates_passed": development_passed,
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "model-config",
        "code_commit": "scoring-commit",
        "stage_reports": {},
    }
    _write_json(pipeline, pipeline_value)
    receipt = tmp_path / "mask_receipt.json"
    _write_json(
        receipt,
        {
            "status": "completed_validation_only_mask_deglitch_gate",
            "execution_passed": True,
            "development_gates_passed": development_passed,
            "scientific_claim_allowed": False,
            "locked_test_allowed": False,
            "test_rows_read": 0,
            "artifacts": {
                "pipeline_report": {
                    "path": str(pipeline),
                    "sha256": file_sha256(pipeline),
                }
            },
        },
    )
    config = tmp_path / "timing.yaml"
    config.write_text(
        """mask_timing_validation:
  required_method: local_whitened_strain_envelope_per_mask_cluster_v1
  chirp_threshold: 0.3
  minimum_bins: 1
  association_window_seconds: 0.25
  uncertainty_quantile: 0.99
  minimum_matches_per_method: 30
  maximum_empirical_timing_uncertainty_seconds: 0.01
""",
        encoding="utf-8",
    )
    if development_passed:
        base_rows = [
            {
                "injection_id": f"injection-{index}",
                "waveform_id": f"waveform-{index}",
                "source_family": "BBH",
                "gps_block": f"block-{index}",
                "vt_weight": 1.0 + index,
                "vt_weight_unit": "Mpc^3 yr",
                "valid_ifos": ["H1", "L1"],
                "detector_arrival_gps": {"H1": 1000.0 + index, "L1": 1000.01 + index},
            }
            for index in range(100)
        ]
        for arm in ("contaminated_raw", "contaminated_mask"):
            triggers = tmp_path / "pipeline" / arm / "injection_triggers.jsonl"
            triggers.parent.mkdir(parents=True, exist_ok=True)
            triggers.write_text(
                "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
            )
            _write_json(
                triggers.parent / "injection_score_report.json",
                {
                    "status": "physical_waveform_real_noise_domain_transfer_diagnostic",
                    "scientific_claim_allowed": False,
                    "required_split": "val",
                    "observed_splits": ["val"],
                    "failed_injections": 0,
                    "input_injections": 100,
                    "scored_injections": 100,
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "model-config",
                    "code_commit": "scoring-commit",
                    "manifest_path": f"{arm}.jsonl",
                    "manifest_sha256": f"{arm}-sha",
                    "triggers_path": str(triggers),
                    "triggers_sha256": file_sha256(triggers),
                },
            )
            pipeline_value["stage_reports"][arm] = {
                "manifest_path": f"{arm}.jsonl",
                "manifest_sha256": f"{arm}-sha",
            }
        _write_json(pipeline, pipeline_value)
        receipt_value = json.loads(receipt.read_text())
        receipt_value["artifacts"]["pipeline_report"]["sha256"] = file_sha256(
            pipeline
        )
        _write_json(receipt, receipt_value)
    return receipt, pipeline, config


def test_mask_timing_requires_paired_raw_and_mask_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, pipeline, config = _fixture(tmp_path)

    def calibrate(trigger: Path, output: Path, *args: object) -> dict[str, object]:
        assert Path(trigger).is_file()
        assert args == (0.3, 1, 0.25, 0.99, 30, 0.01)
        report = {
            "status": "validation_only_candidate_timing_calibration",
            "methods": {
                "local_whitened_strain_envelope_per_mask_cluster_v1": {
                    "calibration_gate_passed": True
                }
            },
        }
        _write_json(Path(output), report)
        return report

    monkeypatch.setattr("gwyolo.mask_timing.run_candidate_timing_calibration", calibrate)
    result = run_mask_timing_validation(
        receipt, pipeline, config, tmp_path / "result" / "receipt.json"
    )
    assert result["paired_injections"] == 100
    assert result["raw_timing_gate_passed"] is True
    assert result["mask_timing_gate_passed"] is True
    assert result["coherent_background_scale_allowed"] is True
    assert result["test_rows_read"] == 0


def test_mask_timing_retains_negative_ranking_gate(tmp_path: Path) -> None:
    receipt, pipeline, config = _fixture(tmp_path, development_passed=False)
    result = run_mask_timing_validation(
        receipt, pipeline, config, tmp_path / "result" / "receipt.json"
    )
    assert result["timing_evaluated"] is False
    assert result["coherent_background_scale_allowed"] is False
    assert result["reason"] == "six_arm_ranking_gate_failed"


def test_mask_timing_rejects_unpaired_physical_injections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, pipeline, config = _fixture(tmp_path)
    mask_triggers = tmp_path / "pipeline" / "contaminated_mask" / "injection_triggers.jsonl"
    rows = [json.loads(line) for line in mask_triggers.read_text().splitlines()]
    rows[0]["waveform_id"] = "different-waveform"
    mask_triggers.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    score_path = mask_triggers.parent / "injection_score_report.json"
    score = json.loads(score_path.read_text())
    score["triggers_sha256"] = file_sha256(mask_triggers)
    _write_json(score_path, score)
    monkeypatch.setattr(
        "gwyolo.mask_timing.run_candidate_timing_calibration",
        lambda *args: pytest.fail("timing calibration must not run for unpaired inputs"),
    )
    with pytest.raises(ValueError, match="identical physical injections"):
        run_mask_timing_validation(
            receipt, pipeline, config, tmp_path / "result" / "receipt.json"
        )
