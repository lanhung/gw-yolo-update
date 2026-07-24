from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidates import (
    run_apply_candidate_timing_calibration,
    run_candidate_timing_calibration,
    run_injection_candidate_extraction,
    run_injection_candidate_rankings,
)
from .io import atomic_write_json, file_sha256, load_yaml
from .runtime import execution_provenance
from .streaming import evict_candidate_probability_artifacts


_REQUIRED_METHOD = "local_whitened_strain_envelope_per_mask_cluster_v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_identity(
    pipeline_root: Path,
    arm: str,
    pipeline: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    score_path = pipeline_root / arm / "injection_score_report.json"
    score = _load_json(score_path)
    stage = pipeline.get("stage_reports", {}).get(arm, {})
    trigger_path = Path(str(score.get("triggers_path", "")))
    if (
        score.get("status") != "physical_waveform_real_noise_domain_transfer_diagnostic"
        or score.get("scientific_claim_allowed") is not False
        or score.get("required_split") != "val"
        or score.get("observed_splits") != ["val"]
        or int(score.get("failed_injections", -1)) != 0
        or int(score.get("input_injections", -1)) < 100
        or int(score.get("scored_injections", -1))
        != int(score.get("input_injections", -2))
        or score.get("checkpoint_sha256") != pipeline.get("checkpoint_sha256")
        or score.get("config_sha256") != pipeline.get("config_sha256")
        or score.get("code_commit") != pipeline.get("code_commit")
        or score.get("manifest_path") != stage.get("manifest_path")
        or score.get("manifest_sha256") != stage.get("manifest_sha256")
        or not trigger_path.is_file()
        or score.get("triggers_sha256") != file_sha256(trigger_path)
    ):
        raise ValueError(f"{arm} score report failed mask timing replay")
    rows = _load_jsonl(trigger_path)
    if len(rows) != int(score["scored_injections"]):
        raise ValueError(f"{arm} trigger count differs from its score report")
    return score_path, trigger_path, score, rows


def _paired_injection_identity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for row in rows:
        injection_id = str(row["injection_id"])
        identity = {
            "waveform_id": str(row["waveform_id"]),
            "source_family": str(row["source_family"]),
            "gps_block": str(row["gps_block"]),
            "vt_weight": float(row["vt_weight"]),
            "vt_weight_unit": str(row.get("vt_weight_unit", "")),
            "valid_ifos": sorted(str(value) for value in row["valid_ifos"]),
            "detector_arrival_gps": {
                str(key): float(value)
                for key, value in sorted(row["detector_arrival_gps"].items())
            },
        }
        if injection_id in identities:
            raise ValueError(f"duplicate mask timing injection ID: {injection_id}")
        identities[injection_id] = identity
    return identities


def _method_gate(report: dict[str, Any], required_method: str) -> bool:
    methods = report.get("methods", {})
    passed = [
        str(name)
        for name, values in methods.items()
        if values.get("calibration_gate_passed") is True
    ]
    return passed == [required_method]


def run_mask_timing_validation(
    mask_validation_receipt_path: str | Path,
    pipeline_report_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Require separate raw/mask candidate timing gates before coherent scaling."""

    if Path(output_path).exists():
        raise FileExistsError("mask timing validation receipts are immutable")
    receipt_path = Path(mask_validation_receipt_path)
    pipeline_path = Path(pipeline_report_path)
    receipt = _load_json(receipt_path)
    pipeline = _load_json(pipeline_path)
    pipeline_identity = receipt.get("artifacts", {}).get("pipeline_report", {})
    if (
        receipt.get("status") != "completed_validation_only_mask_deglitch_gate"
        or receipt.get("execution_passed") is not True
        or receipt.get("scientific_claim_allowed") is not False
        or receipt.get("locked_test_allowed") is not False
        or receipt.get("test_rows_read") != 0
        or pipeline_identity.get("path") != str(pipeline_path)
        or pipeline_identity.get("sha256") != file_sha256(pipeline_path)
        or pipeline.get("status") != "validation_only_end_to_end_mask_search_pipeline"
        or pipeline.get("scientific_claim_allowed") is not False
        or pipeline.get("promotion_allowed") is not False
        or pipeline.get("test_rows_read") != 0
        or pipeline.get("test_evaluation") is not None
        or bool(receipt.get("development_gates_passed"))
        != bool(pipeline.get("development_gates_passed"))
    ):
        raise ValueError("mask timing inputs do not replay one validation-only six-arm gate")

    config = load_yaml(config_path)
    settings = config.get("mask_timing_validation")
    if not isinstance(settings, dict):
        raise ValueError("mask timing validation configuration is missing")
    required_method = str(settings.get("required_method", ""))
    chirp_threshold = float(settings["chirp_threshold"])
    minimum_bins = int(settings["minimum_bins"])
    association = float(settings["association_window_seconds"])
    quantile = float(settings["uncertainty_quantile"])
    minimum_matches = int(settings["minimum_matches_per_method"])
    maximum_uncertainty = float(
        settings["maximum_empirical_timing_uncertainty_seconds"]
    )
    reference_ifo = str(settings["reference_ifo"])
    second_ifo = str(settings["second_ifo"])
    physical_delay = float(settings["physical_delay_limit_seconds"])
    truth_association = float(settings["truth_association_window_seconds"])
    if (
        required_method != _REQUIRED_METHOD
        or not 0 < chirp_threshold < 1
        or minimum_bins <= 0
        or association <= 0
        or not 0.5 <= quantile < 1
        or minimum_matches <= 0
        or not 0 < maximum_uncertainty <= 0.01
        or reference_ifo == second_ifo
        or {reference_ifo, second_ifo} != {"H1", "L1"}
        or not 0 < physical_delay <= 0.01
        or truth_association <= 0
    ):
        raise ValueError("mask timing validation settings are invalid")

    base_result: dict[str, Any] = {
        "status": "completed_validation_only_mask_timing_gate",
        "scientific_claim_allowed": False,
        "locked_test_allowed": False,
        "test_rows_read": 0,
        "mask_validation_receipt_path": str(receipt_path),
        "mask_validation_receipt_sha256": file_sha256(receipt_path),
        "pipeline_report_path": str(pipeline_path),
        "pipeline_report_sha256": file_sha256(pipeline_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "pipeline_code_commit": pipeline.get("code_commit"),
        "ranking_development_gates_passed": bool(
            pipeline.get("development_gates_passed")
        ),
    }
    if not pipeline.get("development_gates_passed"):
        result = {
            **base_result,
            "timing_evaluated": False,
            "raw_timing_gate_passed": False,
            "mask_timing_gate_passed": False,
            "coherent_background_scale_allowed": False,
            "reason": "six_arm_ranking_gate_failed",
            **execution_provenance(),
        }
        atomic_write_json(output_path, result)
        return result

    pipeline_root = pipeline_path.parent
    arms = {}
    identities = {}
    for arm in ("contaminated_raw", "contaminated_mask"):
        score_path, trigger_path, score, rows = _score_identity(
            pipeline_root, arm, pipeline
        )
        arms[arm] = {
            "score_path": score_path,
            "trigger_path": trigger_path,
            "score": score,
            "rows": rows,
        }
        identities[arm] = _paired_injection_identity(rows)
    if identities["contaminated_raw"] != identities["contaminated_mask"]:
        raise ValueError("raw/mask timing arms do not contain identical physical injections")

    output = Path(output_path)
    timing_dir = output.parent / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_reports = {}
    injection_ranking_reports = {}
    probability_eviction_reports = {}
    for condition, arm in (
        ("raw", "contaminated_raw"),
        ("mask", "contaminated_mask"),
    ):
        timing_path = timing_dir / f"{condition}_candidate_timing_calibration.json"
        report = run_candidate_timing_calibration(
            arms[arm]["trigger_path"],
            timing_path,
            chirp_threshold,
            minimum_bins,
            association,
            quantile,
            minimum_matches,
            maximum_uncertainty,
        )
        timing_reports[condition] = {
            "path": str(timing_path),
            "sha256": file_sha256(timing_path),
            "gate_passed": _method_gate(report, required_method),
            "report": report,
        }
        candidate_dir = timing_dir / f"{condition}_injection_candidates"
        candidate_report = run_injection_candidate_extraction(
            arms[arm]["trigger_path"],
            candidate_dir,
            chirp_threshold,
            minimum_bins,
        )
        calibrated_path = timing_dir / f"{condition}_injection_candidates_calibrated.jsonl"
        calibrated = run_apply_candidate_timing_calibration(
            candidate_report["manifest_path"], timing_path, calibrated_path
        )
        if int(calibrated.get("uncalibrated_candidates", -1)) != 0:
            raise ValueError(f"{condition} injection candidates exceed timing support")
        method = report["methods"][required_method]
        ranking_dir = timing_dir / f"{condition}_injection_rankings"
        ranking = run_injection_candidate_rankings(
            arms[arm]["trigger_path"],
            calibrated_path,
            ranking_dir,
            "val",
            reference_ifo,
            second_ifo,
            physical_delay,
            float(method["empirical_timing_uncertainty_seconds"]),
            truth_association,
        )
        ranking_path = ranking_dir / "val_injection_candidate_ranking_report.json"
        if (
            ranking.get("status") != "physical_network_injection_candidate_rankings"
            or ranking.get("split") != "val"
            or int(ranking.get("input_injections", -1))
            != len(identities[arm])
            or ranking.get("timing_calibration_report_sha256")
            != file_sha256(timing_path)
            or ranking.get("candidate_checkpoint_sha256")
            != pipeline.get("checkpoint_sha256")
            or ranking.get("candidate_config_sha256")
            != pipeline.get("config_sha256")
            or ranking.get("candidate_code_commit") != pipeline.get("code_commit")
            or ranking.get("candidate_scoring_provenance_consistent") is not True
            or file_sha256(ranking["manifest_path"])
            != ranking.get("manifest_sha256")
        ):
            raise ValueError(f"{condition} injection candidate ranking failed replay")
        injection_ranking_reports[condition] = {
            "path": str(ranking_path),
            "sha256": file_sha256(ranking_path),
            "manifest_path": ranking["manifest_path"],
            "manifest_sha256": ranking["manifest_sha256"],
            "candidate_extraction_report_path": str(
                candidate_dir / "injection_candidate_extraction_report.json"
            ),
            "candidate_extraction_report_sha256": file_sha256(
                candidate_dir / "injection_candidate_extraction_report.json"
            ),
            "report": ranking,
        }
        candidate_report_path = (
            candidate_dir / "injection_candidate_extraction_report.json"
        )
        eviction_path = timing_dir / f"{condition}_probability_eviction_report.json"
        eviction = evict_candidate_probability_artifacts(
            candidate_report_path,
            arms[arm]["score_path"],
            arms[arm]["score_path"].parent / "probabilities",
            eviction_path,
        )
        if (
            eviction.get("status") != "verified_candidate_probability_eviction"
            or eviction.get("candidate_extraction_report_sha256")
            != file_sha256(candidate_report_path)
            or eviction.get("score_report_sha256")
            != file_sha256(arms[arm]["score_path"])
        ):
            raise ValueError(f"{condition} injection probability eviction failed replay")
        probability_eviction_reports[condition] = {
            "path": str(eviction_path),
            "sha256": file_sha256(eviction_path),
            "removed_files": int(eviction["removed_files"]),
            "removed_bytes": int(eviction["removed_bytes"]),
        }
    result = {
        **base_result,
        "timing_evaluated": True,
        "paired_injections": len(identities["contaminated_raw"]),
        "required_method": required_method,
        "raw_score_report": {
            "path": str(arms["contaminated_raw"]["score_path"]),
            "sha256": file_sha256(arms["contaminated_raw"]["score_path"]),
        },
        "mask_score_report": {
            "path": str(arms["contaminated_mask"]["score_path"]),
            "sha256": file_sha256(arms["contaminated_mask"]["score_path"]),
        },
        "timing_reports": timing_reports,
        "injection_ranking_reports": injection_ranking_reports,
        "probability_eviction_reports": probability_eviction_reports,
        "reference_ifo": reference_ifo,
        "second_ifo": second_ifo,
        "physical_delay_limit_seconds": physical_delay,
        "truth_association_window_seconds": truth_association,
        "raw_timing_gate_passed": timing_reports["raw"]["gate_passed"],
        "mask_timing_gate_passed": timing_reports["mask"]["gate_passed"],
        "coherent_background_scale_allowed": all(
            timing_reports[condition]["gate_passed"] for condition in ("raw", "mask")
        ),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result
