from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.pe import (
    PAIRED_PE_LATENCY_SCOPE_V1,
    PUBLICATION_PROVENANCE_FIELDS,
    evaluate_pe_rows,
    evaluate_pe_robustness_rows,
    posterior_sky_area_equal_solid_angle,
    posterior_truth_metrics,
    promote_pe_robustness_validation,
    run_joint_pe_robustness_evaluation,
    sky_area_estimator_identity,
    validate_paired_pe_latency,
)


def test_equal_solid_angle_sky_area_matches_hand_counted_pixels() -> None:
    ra = np.asarray([0.1] * 4 + [1.7] * 3 + [3.2] * 2 + [4.8])
    dec = np.zeros(10)
    report = posterior_sky_area_equal_solid_angle(
        ra, dec, credible_level=0.7, ra_bins=4, sin_dec_bins=2
    )
    # Counts are 4, 3, 2, 1. Seven of ten samples require the two densest
    # equal-area pixels; each of the eight pixels covers 4*pi/8 steradians.
    expected = 2 * (4 * np.pi / 8) * (180 / np.pi) ** 2
    assert report["credible_pixels"] == 2
    assert report["area_deg2"] == pytest.approx(expected)
    identity = sky_area_estimator_identity(report)
    assert "credible_pixels" not in identity
    assert identity["method"] == "fixed_equal_solid_angle_histogram_v1"


def test_paired_pe_latency_contract_validates_scope_and_component_accounting() -> None:
    report = {
        "latency_scope": PAIRED_PE_LATENCY_SCOPE_V1,
        "latency_seconds": 10.0,
        "latency_components_seconds": {
            "model_load": 2.0,
            "event_preprocessing": 1.0,
            "posterior_sampling": 5.0,
            "posterior_postprocessing_and_write": 1.0,
        },
    }
    assert validate_paired_pe_latency(report)["posterior_sampling"] == 5.0
    with pytest.raises(ValueError, match="scope differs"):
        validate_paired_pe_latency({**report, "latency_scope": "sampling-only"})
    with pytest.raises(ValueError, match="components exceed"):
        validate_paired_pe_latency(
            {
                **report,
                "latency_components_seconds": {
                    **report["latency_components_seconds"],
                    "posterior_sampling": 20.0,
                },
            }
        )


def test_standalone_pe_runners_export_the_frozen_latency_scope() -> None:
    scripts = Path(__file__).parents[1] / "scripts"
    for name in ("run_dingo_common_event.py", "run_amplfi_common_event.py"):
        namespace = runpy.run_path(str(scripts / name))
        assert namespace["LATENCY_SCOPE"] == PAIRED_PE_LATENCY_SCOPE_V1


def test_dingo_event_runner_supports_only_frozen_native_and_current_apis() -> None:
    script = Path(__file__).parents[1] / "scripts/run_dingo_common_event.py"
    source = script.read_text(encoding="utf-8")
    assert 'dingo_version == "0.5.8"' in source
    assert "from dingo.core.models import PosteriorModel" in source
    assert 'dingo_version == "0.9.8"' in source
    assert "from dingo.core.posterior_models.build_model import" in source
    assert "unsupported DINGO inference API version" in source
    assert "scipy.signal.windows import tukey" in source
    assert "the_identical_scipy.signal.windows.tukey" in source


def test_posterior_truth_metrics_match_quantiles_and_bias() -> None:
    result = posterior_truth_metrics({"mass": np.asarray([0, 1, 2, 3, 4])}, {"mass": 2}, 0.8)
    assert result["mass"]["mean"] == 2
    assert result["mass"]["bias"] == 0
    assert result["mass"]["credible_interval"] == pytest.approx([0.4, 3.6])
    assert result["mass"]["covered"]
    assert result["mass"]["mean_absolute_distance_to_truth"] == 1.2


def test_posterior_truth_metrics_wrap_ra_without_a_false_large_bias() -> None:
    values = np.asarray([2 * np.pi - 0.1, 0.0, 0.1])
    result = posterior_truth_metrics({"ra": values}, {"ra": 0.0}, 0.8)["ra"]
    assert result["periodic"] is True
    assert result["period"] == pytest.approx(2 * np.pi)
    assert result["mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["absolute_bias"] == pytest.approx(0.0, abs=1e-12)
    assert result["mean_absolute_distance_to_truth"] == pytest.approx(0.2 / 3)
    assert result["credible_width"] == pytest.approx(0.16)
    assert result["covered"] is True


def test_pe_evaluation_requires_and_compares_raw_cleaned_pairs(tmp_path) -> None:
    raw = tmp_path / "raw.npz"
    cleaned = tmp_path / "cleaned.npz"
    np.savez(raw, mass=np.asarray([2, 3, 4]))
    np.savez(cleaned, mass=np.asarray([1, 2, 3]))
    rows = [
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": "raw",
            "posterior_path": str(raw),
            "latency_seconds": 2.0,
            "truth": {"mass": 2.0},
        },
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": "cleaned",
            "posterior_path": str(cleaned),
            "latency_seconds": 2.5,
            "truth": {"mass": 2.0},
        },
    ]
    report = evaluate_pe_rows(rows, credible_level=0.8, bootstrap_replicates=20)
    comparison = report["comparisons"][0]
    assert comparison["parameters"]["mass"]["absolute_bias_change_cleaned_minus_raw"] == -1
    assert comparison["cleaning_latency_overhead_seconds"] == 0.5
    assert report["coverage"]["DINGO"]["cleaned"]["mass"]["rate"] == 1.0
    summary = report["paired_summaries"]["DINGO"]
    assert summary["parameters"]["mass"][
        "absolute_bias_change_cleaned_minus_raw"
    ]["paired_bootstrap_95"] == [-1.0, -1.0]
    assert summary["cleaning_latency_overhead_seconds"]["mean"] == 0.5
    assert summary["parameters"]["mass"]["coverage_transitions"] == {"0->1": 1}


def test_pe_evaluation_rejects_missing_pair(tmp_path) -> None:
    posterior = tmp_path / "raw.npz"
    np.savez(posterior, mass=np.asarray([1, 2, 3]))
    with pytest.raises(ValueError, match="Missing raw/cleaned"):
        evaluate_pe_rows(
            [
                {
                    "backend": "AMPLFI",
                    "injection_id": "i-1",
                    "condition": "raw",
                    "posterior_path": str(posterior),
                    "latency_seconds": 1.0,
                    "truth": {"mass": 2.0},
                }
            ]
        )


def test_pe_publication_gate_requires_and_matches_provenance(tmp_path) -> None:
    raw = tmp_path / "raw.npz"
    cleaned = tmp_path / "cleaned.npz"
    np.savez(raw, mass=np.asarray([1, 2, 3]))
    np.savez(cleaned, mass=np.asarray([1, 2, 3]))
    provenance = {field: f"fixed-{field}" for field in PUBLICATION_PROVENANCE_FIELDS}
    provenance["detector_set"] = ["H1", "L1"]
    base = {
        "backend": "AMPLFI",
        "injection_id": "i-1",
        "latency_seconds": 1.0,
        "truth": {"mass": 2.0},
        **provenance,
    }
    rows = [
        {**base, "condition": "raw", "posterior_path": str(raw)},
        {**base, "condition": "cleaned", "posterior_path": str(cleaned)},
    ]
    report = evaluate_pe_rows(
        rows,
        bootstrap_replicates=20,
        require_publication_provenance=True,
    )
    assert report["publication_provenance_required"]

    rows[1]["prior_hash"] = "different"
    with pytest.raises(ValueError, match="publication provenance mismatch"):
        evaluate_pe_rows(
            rows,
            bootstrap_replicates=20,
            require_publication_provenance=True,
        )


def test_pe_rejects_invalid_latency(tmp_path) -> None:
    posterior = tmp_path / "posterior.npz"
    np.savez(posterior, mass=np.asarray([1, 2, 3]))
    rows = [
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": condition,
            "posterior_path": str(posterior),
            "latency_seconds": -1.0,
            "truth": {"mass": 2.0},
        }
        for condition in ("raw", "cleaned")
    ]
    with pytest.raises(ValueError, match="Invalid PE latency"):
        evaluate_pe_rows(rows, bootstrap_replicates=20)


def test_pe_robustness_triplet_recovers_hand_calculated_contamination(tmp_path) -> None:
    samples = {
        "clean": np.asarray([1.0, 2.0, 3.0]),
        "contaminated": np.asarray([2.0, 3.0, 4.0]),
        "mask_conditioned": np.asarray([1.0, 2.0, 3.0]),
    }
    latency = {"clean": 1.0, "contaminated": 2.0, "mask_conditioned": 2.5}
    ess = {"clean": 3.0, "contaminated": 2.0, "mask_conditioned": 3.0}
    sky = {"clean": 10.0, "contaminated": 30.0, "mask_conditioned": 12.0}
    rows = []
    for condition, values in samples.items():
        path = tmp_path / f"{condition}.npz"
        np.savez(path, mass=values)
        rows.append(
            {
                "backend": "DINGO",
                "injection_id": "i-1",
                "condition": condition,
                "posterior_path": str(path),
                "latency_seconds": latency[condition],
                "effective_sample_size": ess[condition],
                "sky_area_90_deg2": sky[condition],
                "truth": {"mass": 2.0},
            }
        )
    report = evaluate_pe_robustness_rows(
        rows,
        credible_level=0.8,
        bootstrap_replicates=20,
        require_publication_provenance=False,
    )
    comparison = report["comparisons"][0]
    parameter = comparison["parameters"]["mass"]
    assert parameter["contamination_absolute_bias_change"] == 1.0
    assert parameter["mask_absolute_bias_change_vs_contaminated"] == -1.0
    assert parameter["coverage_transition"] == "1->0->1"
    assert comparison["sky_area_contaminated_over_clean"] == 3.0
    assert comparison["sky_area_mask_over_contaminated"] == 0.4
    assert comparison["ess_rate_mask_over_contaminated"] == pytest.approx(1.2)
    assert comparison["latency_mask_minus_contaminated_seconds"] == 0.5


def test_publication_pe_requires_cross_backend_matched_inputs_and_lineage(tmp_path) -> None:
    files = {}
    for name, payload in (
        ("base", b"base manifest"),
        ("contamination", b"contamination manifest"),
        ("clean", b"clean strain"),
        ("contaminated", b"contaminated strain"),
        ("masked", b"mask cleaned strain"),
        ("mask", b"mask artifact"),
        ("model", b"model weights"),
        ("policy", b"mask policy"),
        ("other", b"different contaminated strain"),
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(payload)
        files[name] = path

    rows = []
    for backend in ("DINGO", "AMPLFI"):
        native_config = tmp_path / f"{backend}-native-config.yaml"
        native_config.write_text(f"backend: {backend}\n", encoding="utf-8")
        provenance = {
            "backend_version": f"{backend}-version",
            "backend_model_hash": f"{backend}-model",
            "prior_hash": "common-prior",
            "waveform_approximant": "IMRPhenomXPHM",
            "detector_set": ["H1", "L1"],
            "calibration_version": "C01",
            "source_event_hash": "event-hash",
                "hardware": "same-gpu",
                "latency_scope": "load-through-posterior",
                "sky_area_estimator": {
                    "method": "fixed_equal_solid_angle_histogram_v1",
                    "ra_bins": 360,
                    "sin_dec_bins": 180,
                },
            }
        for condition in ("clean", "contaminated", "mask_conditioned"):
            posterior = tmp_path / f"{backend}-{condition}.npz"
            np.savez(posterior, mass=np.asarray([1.0, 2.0, 3.0]))
            native = tmp_path / f"{backend}-{condition}-native.hdf5"
            native.write_bytes(f"{backend}-{condition}-native".encode())
            input_name = "masked" if condition == "mask_conditioned" else condition
            row = {
                "backend": backend,
                "injection_id": "i-1",
                "split": "val",
                "condition": condition,
                "posterior_path": str(posterior),
                "latency_seconds": 2.0,
                "effective_sample_size": 3.0,
                "sky_area_90_deg2": 10.0,
                "truth": {"mass": 2.0},
                "analysis_input_path": str(files[input_name]),
                "analysis_input_sha256": file_sha256(files[input_name]),
                "input_sample_rate_hz": 2048,
                "input_duration_seconds": 8.0,
                "input_post_trigger_seconds": 2.0,
                "input_ifos": ["H1", "L1"],
                "base_injection_manifest_path": str(files["base"]),
                "base_injection_manifest_sha256": file_sha256(files["base"]),
                "native_conditioning_path": str(native),
                "native_conditioning_sha256": file_sha256(native),
                "native_conditioning_config_path": str(native_config),
                "native_conditioning_config_sha256": file_sha256(native_config),
                **provenance,
            }
            if condition in {"contaminated", "mask_conditioned"}:
                row.update(
                    {
                        "glitch_id": "glitch-1",
                        "contamination_manifest_path": str(files["contamination"]),
                        "contamination_manifest_sha256": file_sha256(
                            files["contamination"]
                        ),
                    }
                )
            if condition == "mask_conditioned":
                row.update(
                    {
                        "mask_conditioning_mode": "cleaned_strain",
                        "mask_artifact_path": str(files["mask"]),
                        "mask_artifact_sha256": file_sha256(files["mask"]),
                        "mask_model_path": str(files["model"]),
                        "mask_model_sha256": file_sha256(files["model"]),
                        "mask_policy_path": str(files["policy"]),
                        "mask_policy_sha256": file_sha256(files["policy"]),
                    }
                )
            rows.append(row)

    report = evaluate_pe_robustness_rows(
        rows,
        credible_level=0.8,
        bootstrap_replicates=20,
        require_publication_provenance=True,
    )
    assert report["dingo_amplfi_joint_gate"] is True
    assert report["cross_backend_matched_input_gate"] is True
    assert report["common_injection_ids"] == ["i-1"]

    batch_reports = {}
    for backend, status in (
        ("DINGO", "real_dingo_common_batch_complete"),
        ("AMPLFI", "real_amplfi_common_batch_complete"),
    ):
        manifest = tmp_path / f"{backend.lower()}-batch.jsonl"
        selected = [row for row in rows if row["backend"] == backend]
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        batch_report = tmp_path / f"{backend.lower()}-batch-report.json"
        batch_report.write_text(
            json.dumps(
                {
                    "status": status,
                    "rows": len(selected),
                    "paired_injections": 1,
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                }
            ),
            encoding="utf-8",
        )
        batch_reports[backend] = batch_report
    joint = run_joint_pe_robustness_evaluation(
        batch_reports["DINGO"],
        batch_reports["AMPLFI"],
        tmp_path / "joint.jsonl",
        tmp_path / "joint-report.json",
        credible_level=0.8,
        bootstrap_replicates=20,
    )
    assert joint["status"] == "paired_dingo_amplfi_pe_robustness_evaluation_complete"
    assert joint["rows"] == 6
    assert joint["common_injection_count"] == 1
    assert file_sha256(joint["manifest_path"]) == joint["manifest_sha256"]
    assert set(joint["source_batch_reports"]) == {"DINGO", "AMPLFI"}
    promotion_config = tmp_path / "pe-promotion.yaml"
    promotion_config.write_text(
        """pe_robustness_promotion:
  required_backends: [DINGO, AMPLFI]
  required_parameters: [mass]
  minimum_paired_injections: 1
  minimum_bootstrap_replicates: 20
  coverage_noninferiority_margin_vs_clean: 0.0
  coverage_noninferiority_margin_vs_contaminated: 0.0
  maximum_normalized_bias_regression_upper: 0.0
  significant_normalized_bias_improvement_upper: 0.0
  minimum_significant_bias_improvements_per_backend: 0
  minimum_width_ratio_vs_clean_lower: 1.0
  maximum_width_ratio_vs_clean_upper: 1.0
  maximum_sky_area_ratio_upper: 1.0
  minimum_ess_rate_ratio_lower: 1.0
  maximum_latency_overhead_upper_seconds: 0.0
""",
        encoding="utf-8",
    )
    promotion = promote_pe_robustness_validation(
        tmp_path / "joint-report.json",
        promotion_config,
        tmp_path / "pe-promotion.json",
    )
    assert promotion["passed"] is True
    assert promotion["promote_to_locked_test"] is True
    assert promotion["scientific_claim_allowed"] is False

    promotion_config.write_text(
        promotion_config.read_text(encoding="utf-8").replace(
            "minimum_paired_injections: 1", "minimum_paired_injections: 2"
        ),
        encoding="utf-8",
    )
    failed_promotion = promote_pe_robustness_validation(
        tmp_path / "joint-report.json",
        promotion_config,
        tmp_path / "pe-promotion-failed.json",
    )
    assert failed_promotion["passed"] is False
    assert failed_promotion["backend_checks"]["DINGO"]["sample_size_passed"] is False

    mismatched = [dict(row) for row in rows]
    target = next(
        row
        for row in mismatched
        if row["backend"] == "AMPLFI" and row["condition"] == "contaminated"
    )
    target["analysis_input_path"] = str(files["other"])
    target["analysis_input_sha256"] = file_sha256(files["other"])
    with pytest.raises(ValueError, match="inputs differ across backends"):
        evaluate_pe_robustness_rows(
            mismatched,
            credible_level=0.8,
            bootstrap_replicates=20,
            require_publication_provenance=True,
        )
