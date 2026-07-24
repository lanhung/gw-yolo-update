import json
from pathlib import Path

import gwyolo.catalog as catalog
import numpy as np
import pytest
from gwyolo.io import file_sha256


def test_catalog_join_and_warning(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"event": "GW000001_000001", "has_chirp": True},
        {"event": "GW000002_000002", "has_chirp": False},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(
        catalog,
        "load_gwosc_events",
        lambda _: {
            "GW000001_000001": {"network_matched_filter_snr": 9.0, "far": 0.1, "p_astro": 0.9},
            "GW000002_000002": {"network_matched_filter_snr": 21.0, "far": 0.01, "p_astro": 0.99},
        },
    )
    output = tmp_path / "summary.json"
    result = catalog.evaluate_catalog_predictions(predictions, "unused", output)
    assert result["catalog_image_hit_rate"] == 0.5
    assert result["high_snr_misses"][0]["event"] == "GW000002_000002"
    assert "not a search recall" in result["warning"]
    assert output.is_file()


def test_locked_catalog_diagnostic_uses_frozen_own_search_threshold(
    tmp_path, monkeypatch
) -> None:
    metadata = tmp_path / "gwtc5-metadata.jsonl"
    metadata.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "event": "GW000001_000001",
                    "network_matched_filter_snr": 9.0,
                    "far": 0.2,
                    "p_astro": 0.8,
                },
                {
                    "event": "GW000002_000002",
                    "network_matched_filter_snr": 21.0,
                    "far": 0.01,
                    "p_astro": 0.99,
                },
            )
        ),
        encoding="utf-8",
    )

    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    config = tmp_path / "model.yaml"
    config.write_text("model: frozen\n", encoding="utf-8")

    def binding(_plan, _access, output_key, output_path):
        return {
            "output_key": output_key,
            "output_path": str(Path(output_path).resolve()),
            "endpoints": {"catalog_search_arm": "mask_candidate_search"},
            "frozen_artifacts": {
                "model": {
                    "path": str(model.resolve()),
                    "sha256": file_sha256(model),
                },
                "config": {
                    "path": str(config.resolve()),
                    "sha256": file_sha256(config),
                },
                "catalog_metadata": {
                    "path": str(metadata.resolve()),
                    "sha256": file_sha256(metadata),
                }
            },
        }

    monkeypatch.setattr(
        "gwyolo.evaluation_lock.validate_locked_evaluation_suite_access", binding
    )
    monkeypatch.setattr(
        "gwyolo.evaluation_lock.validate_locked_evaluation_suite_input",
        lambda _plan, input_key, input_path: {
            "input_key": input_key,
            "input_path": str(Path(input_path).resolve()),
        },
    )
    source_sample_a = tmp_path / "GW000001_000001.npz"
    source_sample_b = tmp_path / "GW000002_000002.npz"
    for path, scale in ((source_sample_a, 1.0), (source_sample_b, 2.0)):
        features = np.ones((3, 3, 4, 4), dtype=np.float32) * scale
        features[2] = 0
        np.savez_compressed(
            path,
            features=features,
            ifos=np.asarray(["H1", "L1", "V1"]),
            q_values=np.asarray([4.0, 8.0, 16.0], dtype=np.float32),
            detector_availability=np.asarray([1, 1, 0], dtype=np.int8),
        )
    source_manifest = tmp_path / "catalog-source.jsonl"
    source_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "split": "test",
                    "observing_run": "O4b",
                    "event": event,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "available_ifos": ["H1", "L1"],
                    "detector_availability": [1, 1, 0],
                    "gps_start": start,
                    "gps_end": start + 8.0,
                }
            )
            + "\n"
            for event, path, start in (
                ("GW000001_000001", source_sample_a, 100.0),
                ("GW000002_000002", source_sample_b, 200.0),
            )
        ),
        encoding="utf-8",
    )
    candidate_manifest = tmp_path / "catalog-candidates.jsonl"
    candidate_report = tmp_path / "catalog-candidate-report.json"
    predictions = tmp_path / "catalog-predictions.jsonl"
    prediction_report = tmp_path / "catalog-prediction-report.json"
    catalog_output = tmp_path / "catalog-result.json"
    plan_path = tmp_path / "suite-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "outputs": {"catalog_diagnostic": str(catalog_output.resolve())},
                "inputs": {
                    "catalog_source_manifest": str(source_manifest.resolve()),
                    "catalog_candidate_manifest": str(candidate_manifest.resolve()),
                    "catalog_candidate_report": str(candidate_report.resolve()),
                    "catalog_prediction_manifest": str(predictions.resolve()),
                    "catalog_prediction_report": str(prediction_report.resolve()),
                },
            }
        ),
        encoding="utf-8",
    )
    access_path = tmp_path / "access.json"
    access_path.write_text("{}", encoding="utf-8")
    search = tmp_path / "mask-locked-search.json"
    search.write_text(
        json.dumps(
            {
                "status": "locked_candidate_search_evaluation",
                "candidate_endpoint_gates_passed": True,
                "threshold_source": "frozen_validation_candidate_search_calibration",
                "locked_suite_access": {"output_key": "mask_candidate_search"},
                "identity": {
                    "candidate_checkpoint_sha256": file_sha256(model),
                    "candidate_config_sha256": file_sha256(config),
                },
                "test_evaluation": {
                    "threshold": 0.7,
                    "background": {"far_per_year": 0.1, "ifar_years": 10.0},
                },
            }
        ),
        encoding="utf-8",
    )
    mask = tmp_path / "mask.npz"
    mask.write_bytes(b"mask")
    candidate_manifest.write_text(
        json.dumps(
            {
                "split": "test",
                "event": "GW000001_000001",
                "candidate_id": "c0",
                "ranking_score": 0.8,
                "source_sha256": file_sha256(source_sample_a),
                "gps_peak": 123.5,
                "instances": [
                    {
                        "instance_id": "i0",
                        "class_name": "chirp",
                        "confidence": 0.9,
                        "mask_path": str(mask),
                        "mask_sha256": file_sha256(mask),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog_access = binding(
        plan_path, access_path, "catalog_diagnostic", catalog_output
    )
    candidate_inputs = {
        key: {"input_key": key, "input_path": str(path.resolve())}
        for key, path in {
            "catalog_source_manifest": source_manifest,
            "catalog_candidate_manifest": candidate_manifest,
            "catalog_candidate_report": candidate_report,
        }.items()
    }
    scoring = {
        "status": "locked_catalog_candidate_scoring_complete",
        "split": "test",
        "threshold_applied": False,
        "candidate_pruning_applied": False,
        "all_instances_retained": True,
        "test_rows_scored": 2,
        "candidate_rows": 1,
        "candidate_manifest": {
            "path": str(candidate_manifest.resolve()),
            "sha256": file_sha256(candidate_manifest),
        },
        "model": {
            "path": str(model.resolve()),
            "sha256": file_sha256(model),
        },
        "config": {
            "path": str(config.resolve()),
            "sha256": file_sha256(config),
        },
        "catalog_metadata": {
            "path": str(metadata.resolve()),
            "sha256": file_sha256(metadata),
        },
        "source_manifest": {
            "path": str(source_manifest.resolve()),
            "sha256": file_sha256(source_manifest),
        },
        "locked_suite_access": catalog_access,
        "locked_suite_inputs": candidate_inputs,
    }
    candidate_report.write_text(json.dumps({**scoring, "threshold_applied": True}))
    with pytest.raises(ValueError, match="not publication eligible"):
        catalog.run_locked_catalog_prediction_manifest(
            candidate_manifest,
            candidate_report,
            plan_path,
            access_path,
            predictions,
            prediction_report,
        )
    candidate_report.write_text(json.dumps(scoring), encoding="utf-8")
    producer = catalog.run_locked_catalog_prediction_manifest(
        candidate_manifest,
        candidate_report,
        plan_path,
        access_path,
        predictions,
        prediction_report,
    )
    assert producer["candidate_count"] == 1
    assert producer["all_candidates_retained"] is True
    prediction_rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in prediction_rows] == [
        "GW000001_000001",
        "GW000002_000002",
    ]
    assert prediction_rows[1]["candidates"] == []
    result = catalog.run_locked_gwtc5_catalog_diagnostic(
        predictions,
        prediction_report,
        search,
        plan_path,
        access_path,
        catalog_output,
    )
    assert result["threshold"] == 0.7
    assert result["threshold_refits_on_catalog"] == 0
    assert result["catalog_hit_rate"] == 0.5
    assert result["own_search_far_per_year"] == 0.1
    assert result["predictions_retain_all_instances_and_masks"] is True
    assert result["locked_suite_inputs"]["catalog_candidate_manifest"][
        "input_path"
    ] == str(candidate_manifest.resolve())
    assert "not search recall" in result["warning"]
