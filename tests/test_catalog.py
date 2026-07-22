import json
from pathlib import Path

import gwyolo.catalog as catalog
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

    def binding(_plan, _access, output_key, output_path):
        return {
            "output_key": output_key,
            "output_path": str(Path(output_path).resolve()),
            "endpoints": {"catalog_search_arm": "mask_candidate_search"},
            "frozen_artifacts": {
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
    search = tmp_path / "mask-locked-search.json"
    search.write_text(
        json.dumps(
            {
                "status": "locked_candidate_search_evaluation",
                "candidate_endpoint_gates_passed": True,
                "threshold_source": "frozen_validation_candidate_search_calibration",
                "locked_suite_access": {"output_key": "mask_candidate_search"},
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
    predictions = tmp_path / "catalog-predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "split": "test",
                    "event": "GW000001_000001",
                    "candidates": [
                        {
                            "candidate_id": "c0",
                            "ranking_score": 0.8,
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
                    ],
                },
                {
                    "split": "test",
                    "event": "GW000002_000002",
                    "candidates": [],
                },
            )
        ),
        encoding="utf-8",
    )
    result = catalog.run_locked_gwtc5_catalog_diagnostic(
        predictions,
        search,
        tmp_path / "suite-plan.json",
        tmp_path / "access.json",
        tmp_path / "catalog-result.json",
    )
    assert result["threshold"] == 0.7
    assert result["threshold_refits_on_catalog"] == 0
    assert result["catalog_hit_rate"] == 0.5
    assert result["own_search_far_per_year"] == 0.1
    assert result["predictions_retain_all_instances_and_masks"] is True
    assert "not search recall" in result["warning"]
