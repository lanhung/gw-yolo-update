from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .runtime import execution_provenance


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bounded_file(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).resolve()
    if root in {Path("/").resolve(), Path("/root").resolve()} or len(root.parts) < 3:
        raise ValueError("streaming eviction root is too broad")
    if root != path.parent and root not in path.parents:
        raise ValueError(f"eviction target lies outside the explicit cache root: {path}")
    return path


def _prepare_output(output: str | Path) -> tuple[Path, Path]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intent_path = output_path.with_suffix(output_path.suffix + ".intent.json")
    if output_path.exists() or intent_path.exists():
        raise FileExistsError(
            "streaming eviction reports are immutable; choose a new output path"
        )
    return output_path, intent_path


def evict_candidate_probability_artifacts(
    candidate_extraction_report: str | Path,
    score_report: str | Path,
    probability_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Release saved probability/strain artifacts after candidate extraction is certified."""

    output_path, intent_path = _prepare_output(output)
    candidate = _load_json(candidate_extraction_report)
    score = _load_json(score_report)
    if candidate.get("status") not in {
        "subwindow_cluster_integration_only",
        "single_ifo_physical_injection_candidates",
    }:
        raise ValueError("candidate extraction report has the wrong status")
    source = candidate.get("source_scoring_provenance", {})
    if (
        not source.get("available")
        or str(source.get("score_report_sha256")) != file_sha256(score_report)
    ):
        raise ValueError("candidate extraction does not bind the supplied score report")
    if not score.get("probabilities_saved") or int(score.get("failed_windows", 0)):
        raise ValueError("probability eviction requires complete probability-saving background score")
    if int(score.get("failed_injections", 0)):
        raise ValueError("probability eviction requires complete injection score")
    trigger_path = Path(score["triggers_path"])
    if file_sha256(trigger_path) != str(score["triggers_sha256"]):
        raise ValueError("scored trigger manifest hash mismatch before probability eviction")
    rows = _load_jsonl(trigger_path)
    expected_inputs = candidate.get("input_windows", candidate.get("input_injections"))
    if expected_inputs is None or int(expected_inputs) != len(rows):
        raise ValueError("candidate extraction did not consume every scored row")
    root = Path(probability_root).resolve()
    validated = []
    seen_paths = set()
    for row in rows:
        path = _bounded_file(row["probability_path"], root)
        if path in seen_paths:
            raise ValueError(f"duplicate probability artifact in trigger manifest: {path}")
        seen_paths.add(path)
        expected_sha = str(row["probability_sha256"])
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"probability artifact hash mismatch before eviction: {path}")
        validated.append((path, path.stat().st_size, expected_sha))
    atomic_write_json(
        intent_path,
        {
            "status": "validated_probability_eviction_intent",
            "candidate_extraction_report_sha256": file_sha256(
                candidate_extraction_report
            ),
            "score_report_sha256": file_sha256(score_report),
            "targets": [str(path) for path, _, _ in validated],
        },
    )
    removed = []
    for path, size, sha256 in validated:
        path.unlink()
        removed.append({"path": str(path), "bytes": size, "sha256": sha256})
    result = {
        "status": "verified_candidate_probability_eviction",
        "recoverable": True,
        "recovery": (
            "re-run the hash-bound scorer with --save-probabilities, then re-run candidate "
            "extraction before any publication audit"
        ),
        "candidate_extraction_report_path": str(candidate_extraction_report),
        "candidate_extraction_report_sha256": file_sha256(candidate_extraction_report),
        "score_report_path": str(score_report),
        "score_report_sha256": file_sha256(score_report),
        "probability_root": str(root),
        "removed_files": len(removed),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed": removed,
        "intent_path": str(intent_path),
        "intent_sha256": file_sha256(intent_path),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def validate_mask_conditioned_stream_gate(
    mask_validation_receipt: str | Path,
    mask_timing_receipt: str | Path,
    checkpoint: str | Path,
    config: str | Path,
) -> dict[str, Any]:
    """Replay the ranking and separate raw/mask timing gates for continuous streaming."""

    ranking_receipt_path = Path(mask_validation_receipt).resolve()
    timing_path = Path(mask_timing_receipt).resolve()
    ranking = _load_json(ranking_receipt_path)
    timing = _load_json(timing_path)
    pipeline_identity = ranking.get("artifacts", {}).get("pipeline_report", {})
    pipeline_path = Path(str(pipeline_identity.get("path", ""))).resolve()
    if (
        ranking.get("status") != "completed_validation_only_mask_deglitch_gate"
        or ranking.get("execution_passed") is not True
        or ranking.get("development_gates_passed") is not True
        or ranking.get("scientific_claim_allowed") is not False
        or ranking.get("locked_test_allowed") is not False
        or ranking.get("test_rows_read") != 0
        or ranking.get("coherent_background_scale_allowed") is not False
        or not pipeline_path.is_file()
        or pipeline_identity.get("sha256") != file_sha256(pipeline_path)
    ):
        raise ValueError("mask ranking receipt does not authorize validation scaling")
    pipeline = _load_json(pipeline_path)
    if (
        pipeline.get("status") != "validation_only_end_to_end_mask_search_pipeline"
        or pipeline.get("development_gates_passed") is not True
        or pipeline.get("scientific_claim_allowed") is not False
        or pipeline.get("promotion_allowed") is not False
        or pipeline.get("test_rows_read") != 0
        or pipeline.get("test_evaluation") is not None
        or pipeline.get("checkpoint_sha256") != file_sha256(checkpoint)
        or pipeline.get("config_sha256") != file_sha256(config)
    ):
        raise ValueError("mask ranking pipeline differs from the selected stream model")
    strength = float(pipeline.get("strength", -1.0))
    if not 0 <= strength <= 1:
        raise ValueError("mask ranking pipeline has an invalid deglitch strength")
    if (
        timing.get("status") != "completed_validation_only_mask_timing_gate"
        or timing.get("scientific_claim_allowed") is not False
        or timing.get("locked_test_allowed") is not False
        or timing.get("test_rows_read") != 0
        or timing.get("ranking_development_gates_passed") is not True
        or timing.get("timing_evaluated") is not True
        or timing.get("raw_timing_gate_passed") is not True
        or timing.get("mask_timing_gate_passed") is not True
        or timing.get("coherent_background_scale_allowed") is not True
        or Path(str(timing.get("mask_validation_receipt_path", ""))).resolve()
        != ranking_receipt_path
        or timing.get("mask_validation_receipt_sha256")
        != file_sha256(ranking_receipt_path)
        or Path(str(timing.get("pipeline_report_path", ""))).resolve()
        != pipeline_path
        or timing.get("pipeline_report_sha256") != file_sha256(pipeline_path)
    ):
        raise ValueError("mask timing receipt does not authorize coherent dual-arm scaling")
    required_method = str(timing.get("required_method", ""))
    if required_method != "local_whitened_strain_envelope_per_mask_cluster_v1":
        raise ValueError("mask timing receipt uses an unsupported candidate timing method")
    timing_reports = {}
    injection_ranking_reports = {}
    probability_eviction_reports = {}
    for condition in ("raw", "mask"):
        identity = timing.get("timing_reports", {}).get(condition, {})
        report_path = Path(str(identity.get("path", ""))).resolve()
        if (
            identity.get("gate_passed") is not True
            or not report_path.is_file()
            or identity.get("sha256") != file_sha256(report_path)
        ):
            raise ValueError(f"{condition} timing report failed its hash gate")
        report = _load_json(report_path)
        method = report.get("methods", {}).get(required_method, {})
        source = report.get("source_scoring_provenance", {})
        if (
            report.get("status") != "validation_only_candidate_timing_calibration"
            or report.get("scientific_claim_allowed") is not False
            or report.get("selection_data") != "validation_injections_only"
            or report.get("test_evaluation") is not None
            or method.get("calibration_gate_passed") is not True
            or int(method.get("matches", 0)) < 30
            or float(method.get("empirical_timing_uncertainty_seconds", 1.0)) > 0.01
            or source.get("available") is not True
            or source.get("checkpoint_sha256") != pipeline.get("checkpoint_sha256")
            or source.get("config_sha256") != pipeline.get("config_sha256")
            or source.get("code_commit") != pipeline.get("code_commit")
        ):
            raise ValueError(f"{condition} timing calibration failed replay")
        timing_reports[condition] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "source_code_commit": str(source["code_commit"]),
        }
        ranking_identity = timing.get("injection_ranking_reports", {}).get(
            condition, {}
        )
        ranking_path = Path(str(ranking_identity.get("path", ""))).resolve()
        candidate_report_path = Path(
            str(ranking_identity.get("candidate_extraction_report_path", ""))
        ).resolve()
        if (
            not ranking_path.is_file()
            or ranking_identity.get("sha256") != file_sha256(ranking_path)
            or not candidate_report_path.is_file()
            or ranking_identity.get("candidate_extraction_report_sha256")
            != file_sha256(candidate_report_path)
        ):
            raise ValueError(f"{condition} injection ranking failed its hash gate")
        ranking = _load_json(ranking_path)
        manifest_path = Path(str(ranking.get("manifest_path", ""))).resolve()
        if (
            ranking.get("status") != "physical_network_injection_candidate_rankings"
            or ranking.get("split") != "val"
            or ranking.get("timing_calibration_report_sha256")
            != identity.get("sha256")
            or ranking.get("candidate_checkpoint_sha256")
            != pipeline.get("checkpoint_sha256")
            or ranking.get("candidate_config_sha256")
            != pipeline.get("config_sha256")
            or ranking.get("candidate_code_commit") != pipeline.get("code_commit")
            or ranking.get("candidate_scoring_provenance_consistent") is not True
            or not manifest_path.is_file()
            or ranking.get("manifest_sha256") != file_sha256(manifest_path)
        ):
            raise ValueError(f"{condition} injection ranking failed replay")
        injection_ranking_reports[condition] = {
            "path": str(ranking_path),
            "sha256": file_sha256(ranking_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        }
        eviction_identity = timing.get("probability_eviction_reports", {}).get(
            condition, {}
        )
        eviction_path = Path(str(eviction_identity.get("path", ""))).resolve()
        score_identity = timing.get(f"{condition}_score_report", {})
        score_path = Path(str(score_identity.get("path", ""))).resolve()
        if (
            not score_path.is_file()
            or score_identity.get("sha256") != file_sha256(score_path)
            or not eviction_path.is_file()
            or eviction_identity.get("sha256") != file_sha256(eviction_path)
        ):
            raise ValueError(f"{condition} probability eviction failed its hash gate")
        eviction = _load_json(eviction_path)
        if (
            eviction.get("status") != "verified_candidate_probability_eviction"
            or eviction.get("recoverable") is not True
            or eviction.get("score_report_sha256") != file_sha256(score_path)
            or eviction.get("candidate_extraction_report_sha256")
            != file_sha256(candidate_report_path)
            or int(eviction.get("removed_files", -1))
            != int(timing.get("paired_injections", -2))
            or int(eviction.get("removed_bytes", 0)) <= 0
        ):
            raise ValueError(f"{condition} probability eviction failed replay")
        probability_eviction_reports[condition] = {
            "path": str(eviction_path),
            "sha256": file_sha256(eviction_path),
            "removed_files": int(eviction["removed_files"]),
            "removed_bytes": int(eviction["removed_bytes"]),
        }
    reference_ifo = str(timing.get("reference_ifo", ""))
    second_ifo = str(timing.get("second_ifo", ""))
    physical_delay = float(timing.get("physical_delay_limit_seconds", -1.0))
    if (
        reference_ifo == second_ifo
        or {reference_ifo, second_ifo} != {"H1", "L1"}
        or not 0 < physical_delay <= 0.01
    ):
        raise ValueError("mask timing receipt has an invalid H1/L1 physical delay contract")
    return {
        "mask_validation_receipt_path": str(ranking_receipt_path),
        "mask_validation_receipt_sha256": file_sha256(ranking_receipt_path),
        "mask_timing_receipt_path": str(timing_path),
        "mask_timing_receipt_sha256": file_sha256(timing_path),
        "pipeline_report_path": str(pipeline_path),
        "pipeline_report_sha256": file_sha256(pipeline_path),
        "pipeline_code_commit": str(pipeline["code_commit"]),
        "deglitch_strength": strength,
        "timing_reports": timing_reports,
        "injection_ranking_reports": injection_ranking_reports,
        "probability_eviction_reports": probability_eviction_reports,
        "reference_ifo": reference_ifo,
        "second_ifo": second_ifo,
        "physical_delay_limit_seconds": physical_delay,
    }


def evict_mask_conditioned_background_overrides(
    cleaning_report: str | Path,
    mask_score_report: str | Path,
    mask_candidate_report: str | Path,
    override_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Release cleaned strain only after the mask arm is fully scored and reduced."""

    output_path, intent_path = _prepare_output(output)
    cleaning = _load_json(cleaning_report)
    score = _load_json(mask_score_report)
    candidates = _load_json(mask_candidate_report)
    manifest_path = Path(str(cleaning.get("manifest_path", "")))
    trigger_path = Path(str(score.get("triggers_path", "")))
    candidate_manifest = Path(str(candidates.get("manifest_path", "")))
    source = candidates.get("source_scoring_provenance", {})
    if (
        cleaning.get("status") != "learned_mask_background_analysis_overrides"
        or not manifest_path.is_file()
        or cleaning.get("manifest_sha256") != file_sha256(manifest_path)
        or score.get("status") != "real_o4a_domain_transfer_diagnostic"
        or score.get("probabilities_saved") is not True
        or int(score.get("failed_windows", -1)) != 0
        or int(score.get("analysis_override_windows", -1))
        != int(cleaning.get("windows", -2))
        or score.get("manifest_sha256") != file_sha256(manifest_path)
        or not trigger_path.is_file()
        or score.get("triggers_sha256") != file_sha256(trigger_path)
        or candidates.get("status") != "subwindow_cluster_integration_only"
        or source.get("available") is not True
        or source.get("score_report_sha256") != file_sha256(mask_score_report)
        or not candidate_manifest.is_file()
        or candidates.get("manifest_sha256") != file_sha256(candidate_manifest)
    ):
        raise ValueError("mask-conditioned override eviction lacks a complete reduced arm")
    cleaned_rows = _load_jsonl(manifest_path)
    trigger_rows = _load_jsonl(trigger_path)
    cleaned_by_id = {str(row["window_id"]): row for row in cleaned_rows}
    trigger_by_id = {str(row["window_id"]): row for row in trigger_rows}
    if (
        len(cleaned_by_id) != len(cleaned_rows)
        or len(trigger_by_id) != len(trigger_rows)
        or set(cleaned_by_id) != set(trigger_by_id)
        or len(cleaned_rows) != int(cleaning["windows"])
        or len(trigger_rows) != int(score.get("scored_windows", -1))
        or int(candidates.get("input_windows", -1)) != len(trigger_rows)
    ):
        raise ValueError("mask-conditioned override IDs are incomplete or duplicated")
    root = Path(override_root).resolve()
    validated = []
    for window_id, row in sorted(cleaned_by_id.items()):
        path = _bounded_file(row["analysis_override_path"], root)
        expected_sha = str(row["analysis_override_sha256"])
        trigger = trigger_by_id[window_id]
        if (
            not path.is_file()
            or file_sha256(path) != expected_sha
            or trigger.get("analysis_override_sha256") != expected_sha
        ):
            raise ValueError(f"mask-conditioned override changed before eviction: {path}")
        validated.append((path, path.stat().st_size, expected_sha))
    atomic_write_json(
        intent_path,
        {
            "status": "validated_mask_override_eviction_intent",
            "cleaning_report_sha256": file_sha256(cleaning_report),
            "mask_score_report_sha256": file_sha256(mask_score_report),
            "mask_candidate_report_sha256": file_sha256(mask_candidate_report),
            "targets": [str(path) for path, _, _ in validated],
        },
    )
    removed = []
    for path, size, sha256 in validated:
        path.unlink()
        removed.append({"path": str(path), "bytes": size, "sha256": sha256})
    result = {
        "status": "verified_mask_conditioned_override_eviction",
        "recoverable": True,
        "recovery": (
            "re-run the hash-bound raw scorer and learned background deglitch stage before "
            "re-scoring the mask-conditioned arm"
        ),
        "cleaning_report_path": str(Path(cleaning_report).resolve()),
        "cleaning_report_sha256": file_sha256(cleaning_report),
        "mask_score_report_path": str(Path(mask_score_report).resolve()),
        "mask_score_report_sha256": file_sha256(mask_score_report),
        "mask_candidate_report_path": str(Path(mask_candidate_report).resolve()),
        "mask_candidate_report_sha256": file_sha256(mask_candidate_report),
        "override_root": str(root),
        "removed_files": len(removed),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed": removed,
        "intent_path": str(intent_path),
        "intent_sha256": file_sha256(intent_path),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def evict_scored_background_batch_sources(
    batch_download_report: str | Path,
    background_plan_report: str | Path,
    score_reports: Iterable[str | Path],
    candidate_extraction_reports: Iterable[str | Path],
    cache_root: str | Path,
    output: str | Path,
    required_splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, Any]:
    """Release verified public HDF sources after every required split has candidates."""

    output_path, intent_path = _prepare_output(output)
    batch = _load_json(batch_download_report)
    plan = _load_json(background_plan_report)
    if batch.get("status") != "verified_development_strain_batch" or not batch.get("passed"):
        raise ValueError("source eviction requires a verified GWOSC batch")
    if plan.get("status") != "verified_multi_segment_development_background" or not plan.get(
        "passed"
    ):
        raise ValueError("source eviction requires a passing background plan")
    if plan.get("split_strategy") != "hash_threshold_v1":
        raise ValueError("streamed source eviction requires stable hash-threshold splits")
    batch_sha = file_sha256(batch_download_report)
    if batch_sha not in set(str(value) for value in plan["source_batch_report_sha256s"]):
        raise ValueError("background plan does not bind the supplied batch report")
    manifest_path = Path(plan["manifest_path"])
    if file_sha256(manifest_path) != str(plan["manifest_sha256"]):
        raise ValueError("background plan manifest hash mismatch before source eviction")
    background_rows = _load_jsonl(manifest_path)
    required_ids = {
        str(row["window_id"])
        for row in background_rows
        if str(row["split"]) in set(required_splits)
    }
    score_paths = [Path(path) for path in score_reports]
    candidate_paths = [Path(path) for path in candidate_extraction_reports]
    scores = [_load_json(path) for path in score_paths]
    candidates = [_load_json(path) for path in candidate_paths]
    expected_splits = {
        str(row["split"])
        for row in background_rows
        if str(row["split"]) in set(required_splits)
    }
    if len(scores) != len(candidates):
        raise ValueError("source eviction requires one candidate report per score report")
    if bool(scores) != bool(expected_splits):
        raise ValueError("score reports must cover exactly the non-empty required splits")
    scored_ids = set()
    observed_splits = set()
    score_hashes = set()
    for score_path, score in zip(score_paths, scores):
        if int(score.get("failed_windows", 0)) or not score.get("probabilities_saved"):
            raise ValueError("source eviction requires complete probability-saving scores")
        split = str(score.get("required_split"))
        observed_splits.add(split)
        trigger_path = Path(score["triggers_path"])
        if file_sha256(trigger_path) != str(score["triggers_sha256"]):
            raise ValueError("trigger manifest hash mismatch before source eviction")
        trigger_rows = _load_jsonl(trigger_path)
        if any(str(row["split"]) != split for row in trigger_rows):
            raise ValueError("score report trigger split differs from required split")
        scored_ids.update(str(row["window_id"]) for row in trigger_rows)
        score_hashes.add(file_sha256(score_path))
    if observed_splits != expected_splits or scored_ids != required_ids:
        raise ValueError("not every required background window has a complete score")
    bound_score_hashes = set()
    for candidate_path, candidate in zip(candidate_paths, candidates):
        if candidate.get("status") != "subwindow_cluster_integration_only":
            raise ValueError("background source eviction requires background candidates")
        source = candidate.get("source_scoring_provenance", {})
        if not source.get("available"):
            raise ValueError("candidate report lacks scoring provenance")
        bound_score_hashes.add(str(source["score_report_sha256"]))
        if file_sha256(candidate["manifest_path"]) != str(candidate["manifest_sha256"]):
            raise ValueError("candidate manifest hash mismatch before source eviction")
    if bound_score_hashes != score_hashes:
        raise ValueError("candidate reports do not bind every supplied score report")
    batch_sources = {
        str(Path(row["path"]).resolve()): str(row["sha256"]) for row in batch["files"]
    }
    manifest_sources = {
        str(Path(source["path"]).resolve()): str(source["sha256"])
        for row in background_rows
        for source in row["source_files"].values()
    }
    if manifest_sources != batch_sources:
        raise ValueError("background manifest source set differs from batch source set")
    root = Path(cache_root).resolve()
    validated = []
    for path_value, expected_sha in sorted(batch_sources.items()):
        path = _bounded_file(path_value, root)
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"GWOSC source hash mismatch before eviction: {path}")
        validated.append((path, path.stat().st_size, expected_sha))
    atomic_write_json(
        intent_path,
        {
            "status": "validated_scored_source_eviction_intent",
            "batch_download_report_sha256": batch_sha,
            "background_plan_report_sha256": file_sha256(background_plan_report),
            "required_splits": list(required_splits),
            "targets": [str(path) for path, _, _ in validated],
        },
    )
    removed = []
    for path, size, sha256 in validated:
        path.unlink()
        removed.append({"path": str(path), "bytes": size, "sha256": sha256})
    result = {
        "status": "verified_scored_gwosc_source_eviction",
        "recoverable": True,
        "recovery": (
            "re-run gwosc-batch-download with the batch report's hash-bound public plan"
        ),
        "batch_download_report_path": str(batch_download_report),
        "batch_download_report_sha256": batch_sha,
        "background_plan_report_path": str(background_plan_report),
        "background_plan_report_sha256": file_sha256(background_plan_report),
        "required_splits": list(required_splits),
        "observed_required_splits": sorted(expected_splits),
        "scored_windows": len(scored_ids),
        "unscored_training_windows": sum(
            str(row["split"]) == "train" for row in background_rows
        ),
        "cache_root": str(root),
        "removed_files": len(removed),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed": removed,
        "intent_path": str(intent_path),
        "intent_sha256": file_sha256(intent_path),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def evict_amplfi_background_batch_sources(
    batch_download_report: str | Path,
    background_plan_report: str | Path,
    amplfi_export_report: str | Path,
    cache_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Release public source HDFs after a hash-complete AMPLFI background export."""

    output_path, intent_path = _prepare_output(output)
    batch = _load_json(batch_download_report)
    plan = _load_json(background_plan_report)
    export = _load_json(amplfi_export_report)
    if batch.get("status") != "verified_development_strain_batch" or not batch.get("passed"):
        raise ValueError("AMPLFI source eviction requires a verified GWOSC batch")
    if (
        plan.get("status") != "verified_multi_segment_development_background"
        or not plan.get("passed")
        or plan.get("split_strategy") != "hash_threshold_v1"
        or int(plan.get("splits", {}).get("test", {}).get("windows", -1)) != 0
    ):
        raise ValueError("AMPLFI source eviction requires a stable train/validation plan")
    if export.get("status") != "group_safe_amplfi_background":
        raise ValueError("AMPLFI source eviction requires a complete group-safe export")
    batch_sha = file_sha256(batch_download_report)
    bound_batch_hashes = set(
        str(value) for value in plan.get("source_batch_report_sha256s", [])
    )
    if plan.get("source_batch_report_sha256"):
        bound_batch_hashes.add(str(plan["source_batch_report_sha256"]))
    if batch_sha not in bound_batch_hashes:
        raise ValueError("AMPLFI background plan does not bind the supplied batch")
    manifest_path = Path(str(plan.get("manifest_path", ""))).resolve()
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != plan.get("manifest_sha256")
        or Path(str(export.get("manifest_path", ""))).resolve() != manifest_path
        or export.get("manifest_sha256") != plan.get("manifest_sha256")
        or int(export.get("split_file_counts", {}).get("test", -1)) != 0
    ):
        raise ValueError("AMPLFI export does not bind the train/validation background manifest")
    for row in export.get("files", []):
        path = Path(str(row.get("path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != row.get("sha256"):
            raise ValueError(f"AMPLFI exported background hash mismatch: {path}")
    batch_sources = {
        str(Path(row["path"]).resolve()): str(row["sha256"]) for row in batch["files"]
    }
    manifest_sources = {
        str(Path(source["path"]).resolve()): str(source["sha256"])
        for row in _load_jsonl(manifest_path)
        for source in row["source_files"].values()
    }
    export_sources = {
        str(Path(source["path"]).resolve()): str(source["sha256"])
        for row in export.get("files", [])
        for source in row["source_files"].values()
    }
    if manifest_sources != batch_sources or export_sources != batch_sources:
        raise ValueError("AMPLFI export does not cover every verified batch source")
    root = Path(cache_root).resolve()
    validated = []
    for path_value, expected_sha in sorted(batch_sources.items()):
        path = _bounded_file(path_value, root)
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"GWOSC source hash mismatch before AMPLFI eviction: {path}")
        validated.append((path, path.stat().st_size, expected_sha))
    atomic_write_json(
        intent_path,
        {
            "status": "validated_amplfi_source_eviction_intent",
            "batch_download_report_sha256": batch_sha,
            "background_plan_report_sha256": file_sha256(background_plan_report),
            "amplfi_export_report_sha256": file_sha256(amplfi_export_report),
            "targets": [str(path) for path, _, _ in validated],
        },
    )
    removed = []
    for path, size, sha256 in validated:
        path.unlink()
        removed.append({"path": str(path), "bytes": size, "sha256": sha256})
    result = {
        "status": "verified_exported_amplfi_source_eviction",
        "recoverable": True,
        "recovery": "re-run gwosc-batch-download from the hash-bound public plan",
        "batch_download_report_path": str(Path(batch_download_report).resolve()),
        "batch_download_report_sha256": batch_sha,
        "background_plan_report_path": str(Path(background_plan_report).resolve()),
        "background_plan_report_sha256": file_sha256(background_plan_report),
        "amplfi_export_report_path": str(Path(amplfi_export_report).resolve()),
        "amplfi_export_report_sha256": file_sha256(amplfi_export_report),
        "cache_root": str(root),
        "removed_files": len(removed),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed": removed,
        "intent_path": str(intent_path),
        "intent_sha256": file_sha256(intent_path),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def run_streamed_background_shard(
    parent_plan: str | Path,
    event_exclusions: str | Path,
    timing_calibration_report: str | Path | None,
    checkpoint: str | Path,
    config: str | Path,
    coherence_config: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    shard_index: int,
    pairs_per_shard: int = 1,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 20260720,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
    download_workers: int = 8,
    allow_uncalibrated_morphology_baseline: bool = False,
    verified_source_inventories: Iterable[str | Path] = (),
    mask_validation_receipt: str | Path | None = None,
    mask_timing_receipt: str | Path | None = None,
    scoring_compatibility_report: str | Path | None = None,
) -> dict[str, Any]:
    """Download, score, reduce, and safely release one stable-split background shard."""

    from .background import run_batch_background_plan
    from .candidates import (
        run_apply_candidate_timing_calibration,
        run_candidate_extraction,
    )
    from .gwosc import run_gwosc_batch_download, run_gwosc_plan_shard
    from .manifests import select_jsonl_split
    from .trigger import score_background_manifest

    mask_values = (mask_validation_receipt, mask_timing_receipt)
    mask_mode = any(value is not None for value in mask_values)
    if mask_mode and not all(value is not None for value in mask_values):
        raise ValueError("mask streaming requires both ranking and timing receipts")
    if mask_mode and allow_uncalibrated_morphology_baseline:
        raise ValueError("mask streaming cannot use the uncalibrated morphology mode")
    mask_gate = None
    if mask_mode:
        if test_fraction != 0:
            raise ValueError("mask-conditioned development streaming requires test_fraction=0")
        mask_gate = validate_mask_conditioned_stream_gate(
            mask_validation_receipt,  # type: ignore[arg-type]
            mask_timing_receipt,  # type: ignore[arg-type]
            checkpoint,
            config,
        )
        timing_calibration_report = mask_gate["timing_reports"]["raw"]["path"]
        source_commits = {
            str(value["source_code_commit"])
            for value in mask_gate["timing_reports"].values()
        }
        current_commit = str(execution_provenance()["code_commit"])
        if source_commits != {current_commit}:
            if len(source_commits) != 1 or scoring_compatibility_report is None:
                raise ValueError(
                    "cross-commit mask streaming requires one scoring compatibility report"
                )
            from .code_compatibility import validate_candidate_scoring_compatibility

            validate_candidate_scoring_compatibility(
                scoring_compatibility_report,
                next(iter(source_commits)),
                current_commit,
            )
    if allow_uncalibrated_morphology_baseline:
        if timing_calibration_report is not None:
            raise ValueError("morphology-only streaming cannot accept timing calibration")
        if test_fraction != 0:
            raise ValueError("morphology-only development streaming requires test_fraction=0")
    elif timing_calibration_report is None:
        raise ValueError("coherent streaming requires a timing calibration report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inventory_paths = [Path(path) for path in verified_source_inventories]
    identity = {
        "parent_plan_sha256": file_sha256(parent_plan),
        "event_exclusions_sha256": file_sha256(event_exclusions),
        "timing_calibration_report_sha256": (
            file_sha256(timing_calibration_report)
            if timing_calibration_report is not None
            else None
        ),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "coherence_config_sha256": file_sha256(coherence_config),
        "shard_index": shard_index,
        "pairs_per_shard": pairs_per_shard,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "download_workers": download_workers,
        "verified_source_inventory_sha256s": [
            file_sha256(path) for path in inventory_paths
        ],
        "streaming_mode": (
            "morphology_only_validation"
            if allow_uncalibrated_morphology_baseline
            else "empirically_calibrated_timing"
        ),
        "code_commit": execution_provenance()["code_commit"],
    }
    if mask_gate is not None:
        identity.update(
            {
                "streaming_mode": "paired_raw_mask_empirically_calibrated_timing",
                "mask_validation_receipt_sha256": mask_gate[
                    "mask_validation_receipt_sha256"
                ],
                "mask_timing_receipt_sha256": mask_gate[
                    "mask_timing_receipt_sha256"
                ],
                "mask_pipeline_report_sha256": mask_gate[
                    "pipeline_report_sha256"
                ],
                "raw_timing_calibration_report_sha256": mask_gate[
                    "timing_reports"
                ]["raw"]["sha256"],
                "mask_timing_calibration_report_sha256": mask_gate[
                    "timing_reports"
                ]["mask"]["sha256"],
                "raw_injection_ranking_report_sha256": mask_gate[
                    "injection_ranking_reports"
                ]["raw"]["sha256"],
                "mask_injection_ranking_report_sha256": mask_gate[
                    "injection_ranking_reports"
                ]["mask"]["sha256"],
                "reference_ifo": mask_gate["reference_ifo"],
                "second_ifo": mask_gate["second_ifo"],
                "physical_delay_limit_seconds": mask_gate[
                    "physical_delay_limit_seconds"
                ],
                "deglitch_strength": mask_gate["deglitch_strength"],
                "scoring_compatibility_report_sha256": (
                    file_sha256(scoring_compatibility_report)
                    if scoring_compatibility_report is not None
                    else None
                ),
            }
        )
    final_path = output / "streamed_background_shard_report.json"
    if final_path.is_file():
        prior = _load_json(final_path)
        if prior.get("run_identity") != identity:
            raise ValueError("completed streamed background shard has another identity")
        return prior

    shard_plan_path = output / "acquisition_plan_shard.json"
    if shard_plan_path.is_file():
        shard_plan = _load_json(shard_plan_path)
        if (
            shard_plan.get("parent_plan_sha256") != identity["parent_plan_sha256"]
            or int(shard_plan.get("shard_index", -1)) != shard_index
            or int(shard_plan.get("pairs_per_shard", -1)) != pairs_per_shard
        ):
            raise ValueError("existing acquisition shard plan has another identity")
    else:
        shard_plan = run_gwosc_plan_shard(
            parent_plan, shard_plan_path, shard_index, pairs_per_shard
        )

    batch_dir = output / "download"
    batch_report_path = batch_dir / "batch_download_report.json"
    if batch_report_path.is_file():
        batch = _load_json(batch_report_path)
        if batch.get("plan_sha256") != file_sha256(shard_plan_path):
            raise ValueError("existing download report belongs to another shard plan")
    else:
        batch = run_gwosc_batch_download(
            shard_plan_path,
            cache_root,
            batch_dir,
            None,
            download_workers,
            1_048_576,
            inventory_paths,
        )

    background_dir = output / "background"
    background_report_path = background_dir / "background_plan_report.json"
    if background_report_path.is_file():
        background = _load_json(background_report_path)
        if (
            background.get("split_strategy") != "hash_threshold_v1"
            or file_sha256(batch_report_path)
            not in set(background.get("source_batch_report_sha256s", []))
        ):
            raise ValueError("existing background plan belongs to another streamed shard")
    else:
        background = run_batch_background_plan(
            batch_report_path,
            event_exclusions,
            background_dir,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            split_strategy="hash_threshold_v1",
        )

    background_rows = _load_jsonl(background["manifest_path"])
    split_counts = {
        split: sum(str(row["split"]) == split for row in background_rows)
        for split in ("train", "val", "test")
    }
    score_report_paths = []
    candidate_report_paths = []
    split_artifacts = {}
    for split in ("val", "test"):
        if not split_counts[split]:
            continue
        split_dir = output / split
        if mask_gate is not None:
            split_manifest_report = select_jsonl_split(
                background["manifest_path"], split, split_dir / "manifest"
            )
            reduction_path = split_dir / "paired_raw_mask_reduction_report.json"
            raw_root = split_dir / "raw"
            mask_root = split_dir / "mask"
            cleaning_root = split_dir / "cleaning"
            raw_score_path = raw_root / "score" / "trigger_score_report.json"
            raw_candidate_path = (
                raw_root / "candidates" / "candidate_extraction_report.json"
            )
            raw_calibrated_path = raw_root / "candidates_calibrated.jsonl"
            raw_calibrated_report_path = raw_calibrated_path.with_suffix(
                raw_calibrated_path.suffix + ".report.json"
            )
            cleaning_report_path = (
                cleaning_root / "learned_background_deglitch_report.json"
            )
            mask_score_path = mask_root / "score" / "trigger_score_report.json"
            mask_candidate_path = (
                mask_root / "candidates" / "candidate_extraction_report.json"
            )
            mask_calibrated_path = mask_root / "candidates_calibrated.jsonl"
            mask_calibrated_report_path = mask_calibrated_path.with_suffix(
                mask_calibrated_path.suffix + ".report.json"
            )
            if reduction_path.is_file():
                reduction = _load_json(reduction_path)
                expected = {
                    "raw_score_report_sha256": file_sha256(raw_score_path),
                    "raw_candidate_report_sha256": file_sha256(raw_candidate_path),
                    "raw_calibrated_report_sha256": file_sha256(
                        raw_calibrated_report_path
                    ),
                    "cleaning_report_sha256": file_sha256(cleaning_report_path),
                    "mask_score_report_sha256": file_sha256(mask_score_path),
                    "mask_candidate_report_sha256": file_sha256(mask_candidate_path),
                    "mask_calibrated_report_sha256": file_sha256(
                        mask_calibrated_report_path
                    ),
                }
                if (
                    reduction.get("status")
                    != "complete_paired_raw_mask_candidate_reduction"
                    or reduction.get("split") != split
                    or any(reduction.get(key) != value for key, value in expected.items())
                ):
                    raise ValueError(f"existing {split} raw/mask reduction is inconsistent")
                raw_score = _load_json(raw_score_path)
                raw_candidates = _load_json(raw_candidate_path)
                raw_calibrated = _load_json(raw_calibrated_report_path)
                cleaning = _load_json(cleaning_report_path)
                mask_score = _load_json(mask_score_path)
                mask_candidates = _load_json(mask_candidate_path)
                mask_calibrated = _load_json(mask_calibrated_report_path)
            else:
                raw_score = score_background_manifest(
                    split_manifest_report["manifest_path"],
                    checkpoint,
                    config,
                    raw_root / "score",
                    model_ifos,
                    q_values,
                    target_sample_rate,
                    context_duration,
                    True,
                    split,
                    None,
                    coherence_config,
                )
                raw_candidates = run_candidate_extraction(
                    raw_score["triggers_path"],
                    raw_root / "candidates",
                    chirp_threshold,
                    minimum_bins,
                )
                raw_calibrated = run_apply_candidate_timing_calibration(
                    raw_candidates["manifest_path"],
                    mask_gate["timing_reports"]["raw"]["path"],
                    raw_calibrated_path,
                    scoring_compatibility_report,
                )
                if raw_calibrated["uncalibrated_candidates"]:
                    raise RuntimeError(
                        f"{split} raw candidates exceed their frozen timing support"
                    )
                from .learned_deglitch import run_learned_background_deglitch

                cleaning = run_learned_background_deglitch(
                    split_manifest_report["manifest_path"],
                    raw_score["triggers_path"],
                    cleaning_root,
                    float(mask_gate["deglitch_strength"]),
                    model_ifos,
                    target_sample_rate,
                    context_duration,
                    split,
                )
                mask_score = score_background_manifest(
                    cleaning["manifest_path"],
                    checkpoint,
                    config,
                    mask_root / "score",
                    model_ifos,
                    q_values,
                    target_sample_rate,
                    context_duration,
                    True,
                    split,
                    None,
                    coherence_config,
                )
                mask_candidates = run_candidate_extraction(
                    mask_score["triggers_path"],
                    mask_root / "candidates",
                    chirp_threshold,
                    minimum_bins,
                )
                mask_calibrated = run_apply_candidate_timing_calibration(
                    mask_candidates["manifest_path"],
                    mask_gate["timing_reports"]["mask"]["path"],
                    mask_calibrated_path,
                    scoring_compatibility_report,
                )
                if mask_calibrated["uncalibrated_candidates"]:
                    raise RuntimeError(
                        f"{split} mask candidates exceed their frozen timing support"
                    )
                reduction = {
                    "status": "complete_paired_raw_mask_candidate_reduction",
                    "split": split,
                    "windows": split_counts[split],
                    "raw_score_report_sha256": file_sha256(raw_score_path),
                    "raw_candidate_report_sha256": file_sha256(raw_candidate_path),
                    "raw_calibrated_report_sha256": file_sha256(
                        raw_calibrated_report_path
                    ),
                    "cleaning_report_sha256": file_sha256(cleaning_report_path),
                    "mask_score_report_sha256": file_sha256(mask_score_path),
                    "mask_candidate_report_sha256": file_sha256(mask_candidate_path),
                    "mask_calibrated_report_sha256": file_sha256(
                        mask_calibrated_report_path
                    ),
                    **execution_provenance(),
                }
                atomic_write_json(reduction_path, reduction)
            raw_eviction_path = raw_root / "probability_eviction_report.json"
            if raw_eviction_path.is_file():
                raw_eviction = _load_json(raw_eviction_path)
            else:
                raw_eviction = evict_candidate_probability_artifacts(
                    raw_candidate_path,
                    raw_score_path,
                    raw_root / "score" / "probabilities",
                    raw_eviction_path,
                )
            if (
                raw_eviction.get("status")
                != "verified_candidate_probability_eviction"
                or raw_eviction.get("score_report_sha256")
                != file_sha256(raw_score_path)
                or raw_eviction.get("candidate_extraction_report_sha256")
                != file_sha256(raw_candidate_path)
            ):
                raise ValueError(f"existing {split} raw probability eviction is inconsistent")
            mask_eviction_path = mask_root / "probability_eviction_report.json"
            if mask_eviction_path.is_file():
                mask_eviction = _load_json(mask_eviction_path)
            else:
                mask_eviction = evict_candidate_probability_artifacts(
                    mask_candidate_path,
                    mask_score_path,
                    mask_root / "score" / "probabilities",
                    mask_eviction_path,
                )
            if (
                mask_eviction.get("status")
                != "verified_candidate_probability_eviction"
                or mask_eviction.get("score_report_sha256")
                != file_sha256(mask_score_path)
                or mask_eviction.get("candidate_extraction_report_sha256")
                != file_sha256(mask_candidate_path)
            ):
                raise ValueError(f"existing {split} mask probability eviction is inconsistent")
            override_eviction_path = cleaning_root / "override_eviction_report.json"
            if override_eviction_path.is_file():
                override_eviction = _load_json(override_eviction_path)
            else:
                override_eviction = evict_mask_conditioned_background_overrides(
                    cleaning_report_path,
                    mask_score_path,
                    mask_candidate_path,
                    cleaning_root / "arrays",
                    override_eviction_path,
                )
            if (
                override_eviction.get("status")
                != "verified_mask_conditioned_override_eviction"
                or override_eviction.get("cleaning_report_sha256")
                != file_sha256(cleaning_report_path)
                or override_eviction.get("mask_score_report_sha256")
                != file_sha256(mask_score_path)
                or override_eviction.get("mask_candidate_report_sha256")
                != file_sha256(mask_candidate_path)
            ):
                raise ValueError(f"existing {split} mask override eviction is inconsistent")
            score_report_paths.extend((raw_score_path, mask_score_path))
            candidate_report_paths.extend((raw_candidate_path, mask_candidate_path))
            split_artifacts[split] = {
                "windows": split_counts[split],
                "paired_reduction_report_sha256": file_sha256(reduction_path),
                "cleaning_report_sha256": file_sha256(cleaning_report_path),
                "override_eviction_report_sha256": file_sha256(
                    override_eviction_path
                ),
                "override_files_removed": int(override_eviction["removed_files"]),
                "arms": {
                    "raw": {
                        "score_report_sha256": file_sha256(raw_score_path),
                        "candidate_report_sha256": file_sha256(raw_candidate_path),
                        "candidate_manifest_path": str(
                            raw_candidates["manifest_path"]
                        ),
                        "candidate_manifest_sha256": file_sha256(
                            raw_candidates["manifest_path"]
                        ),
                        "calibrated_candidate_manifest_path": str(
                            raw_calibrated_path
                        ),
                        "calibrated_candidate_manifest_sha256": file_sha256(
                            raw_calibrated_path
                        ),
                        "calibrated_candidate_report_sha256": file_sha256(
                            raw_calibrated_report_path
                        ),
                        "probability_eviction_report_sha256": file_sha256(
                            raw_eviction_path
                        ),
                        "probability_files_removed": int(
                            raw_eviction["removed_files"]
                        ),
                    },
                    "mask": {
                        "score_report_sha256": file_sha256(mask_score_path),
                        "candidate_report_sha256": file_sha256(mask_candidate_path),
                        "candidate_manifest_path": str(
                            mask_candidates["manifest_path"]
                        ),
                        "candidate_manifest_sha256": file_sha256(
                            mask_candidates["manifest_path"]
                        ),
                        "calibrated_candidate_manifest_path": str(
                            mask_calibrated_path
                        ),
                        "calibrated_candidate_manifest_sha256": file_sha256(
                            mask_calibrated_path
                        ),
                        "calibrated_candidate_report_sha256": file_sha256(
                            mask_calibrated_report_path
                        ),
                        "probability_eviction_report_sha256": file_sha256(
                            mask_eviction_path
                        ),
                        "probability_files_removed": int(
                            mask_eviction["removed_files"]
                        ),
                    },
                },
            }
            continue
        score_report_path = split_dir / "score" / "trigger_score_report.json"
        candidate_report_path = (
            split_dir / "candidates" / "candidate_extraction_report.json"
        )
        calibrated_path = split_dir / "candidates_calibrated.jsonl"
        calibrated_report_path = calibrated_path.with_suffix(
            calibrated_path.suffix + ".report.json"
        )
        eviction_path = split_dir / "probability_eviction_report.json"
        if eviction_path.is_file():
            eviction = _load_json(eviction_path)
            if (
                eviction.get("status") != "verified_candidate_probability_eviction"
                or eviction.get("score_report_sha256") != file_sha256(score_report_path)
                or eviction.get("candidate_extraction_report_sha256")
                != file_sha256(candidate_report_path)
            ):
                raise ValueError(f"existing {split} probability eviction is inconsistent")
            calibrated = (
                None
                if allow_uncalibrated_morphology_baseline
                else _load_json(calibrated_report_path)
            )
            candidates = _load_json(candidate_report_path)
        else:
            split_manifest_report = select_jsonl_split(
                background["manifest_path"], split, split_dir / "manifest"
            )
            score = score_background_manifest(
                split_manifest_report["manifest_path"],
                checkpoint,
                config,
                split_dir / "score",
                model_ifos,
                q_values,
                target_sample_rate,
                context_duration,
                True,
                split,
                None,
                coherence_config,
            )
            candidates = run_candidate_extraction(
                score["triggers_path"],
                split_dir / "candidates",
                chirp_threshold,
                minimum_bins,
            )
            calibrated = None
            if not allow_uncalibrated_morphology_baseline:
                calibrated = run_apply_candidate_timing_calibration(
                    candidates["manifest_path"], timing_calibration_report, calibrated_path
                )
                if calibrated["uncalibrated_candidates"]:
                    raise RuntimeError(
                        f"{split} shard candidates do not match the frozen timing calibration"
                    )
            eviction = evict_candidate_probability_artifacts(
                candidate_report_path,
                score_report_path,
                split_dir / "score" / "probabilities",
                eviction_path,
            )
        score_report_paths.append(score_report_path)
        candidate_report_paths.append(candidate_report_path)
        split_artifacts[split] = {
            "windows": split_counts[split],
            "score_report_sha256": file_sha256(score_report_path),
            "candidate_report_sha256": file_sha256(candidate_report_path),
            "candidate_manifest_path": str(candidates["manifest_path"]),
            "candidate_manifest_sha256": file_sha256(candidates["manifest_path"]),
            "calibrated_candidate_manifest_path": (
                str(calibrated_path) if calibrated is not None else None
            ),
            "calibrated_candidate_manifest_sha256": (
                file_sha256(calibrated_path) if calibrated is not None else None
            ),
            "calibrated_candidate_report_sha256": (
                file_sha256(calibrated_report_path) if calibrated is not None else None
            ),
            "probability_eviction_report_sha256": file_sha256(eviction_path),
            "probability_files_removed": int(eviction["removed_files"]),
        }

    source_eviction_path = output / "source_eviction_report.json"
    if source_eviction_path.is_file():
        source_eviction = _load_json(source_eviction_path)
        if (
            source_eviction.get("status") != "verified_scored_gwosc_source_eviction"
            or source_eviction.get("batch_download_report_sha256")
            != file_sha256(batch_report_path)
            or source_eviction.get("background_plan_report_sha256")
            != file_sha256(background_report_path)
        ):
            raise ValueError("existing source eviction report is inconsistent")
    else:
        source_eviction = evict_scored_background_batch_sources(
            batch_report_path,
            background_report_path,
            score_report_paths,
            candidate_report_paths,
            cache_root,
            source_eviction_path,
        )
    result = {
        "status": (
            "verified_streamed_raw_mask_candidate_background_shard"
            if mask_gate is not None
            else (
                "verified_streamed_morphology_background_shard"
                if allow_uncalibrated_morphology_baseline
                else "verified_streamed_candidate_background_shard"
            )
        ),
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "timing_empirically_calibrated": not allow_uncalibrated_morphology_baseline,
        "paired_raw_mask_arms": mask_gate is not None,
        "scientific_blocker": (
            "morphology-only validation baseline; no test scoring or network-coherence claim is "
            "allowed until a separate empirical timing calibration passes"
            if allow_uncalibrated_morphology_baseline
            else "merge disjoint stable-hash shards, construct adequate time-slide exposure, "
            "freeze the validation threshold, and evaluate the independently locked test partition"
        ),
        "run_identity": identity,
        "run_identity_hash": canonical_hash(identity, 64),
        "split_strategy": "hash_threshold_v1",
        "acquisition_shard_plan_sha256": file_sha256(shard_plan_path),
        "parent_selected_pairs": int(shard_plan["parent_selected_pairs"]),
        "shard_count": int(shard_plan["shard_count"]),
        "pair_index_start_inclusive": int(shard_plan["pair_index_start_inclusive"]),
        "pair_index_stop_exclusive": int(shard_plan["pair_index_stop_exclusive"]),
        "selected_pair_ids_hash": str(shard_plan["selected_pair_ids_hash"]),
        "batch_download_report_sha256": file_sha256(batch_report_path),
        "background_plan_report_sha256": file_sha256(background_report_path),
        "background_manifest_path": str(background["manifest_path"]),
        "background_manifest_sha256": file_sha256(background["manifest_path"]),
        "split_counts": split_counts,
        "split_artifacts": split_artifacts,
        "source_eviction_report_sha256": file_sha256(source_eviction_path),
        "source_files_removed": int(source_eviction["removed_files"]),
        "source_bytes_removed": int(source_eviction["removed_bytes"]),
        "recoverable": True,
        **execution_provenance(),
    }
    atomic_write_json(final_path, result)
    return result


def run_streamed_morphology_background_shard(
    parent_plan: str | Path,
    event_exclusions: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    coherence_config: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    shard_index: int,
    pairs_per_shard: int = 1,
    validation_fraction: float = 0.2,
    seed: int = 20260720,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
    download_workers: int = 8,
    verified_source_inventories: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Stream a validation-only morphology baseline without implying timing coherence."""
    return run_streamed_background_shard(
        parent_plan,
        event_exclusions,
        None,
        checkpoint,
        config,
        coherence_config,
        cache_root,
        output_dir,
        shard_index,
        pairs_per_shard,
        validation_fraction,
        0.0,
        seed,
        model_ifos,
        q_values,
        target_sample_rate,
        context_duration,
        chirp_threshold,
        minimum_bins,
        download_workers,
        True,
        verified_source_inventories,
    )


def merge_streamed_background_shards(
    shard_reports: Iterable[str | Path],
    output_dir: str | Path,
    parent_plan: str | Path | None = None,
) -> dict[str, Any]:
    """Merge stable-split background/candidate shards with global overlap checks."""

    from .background import _union_duration

    report_paths = [Path(path) for path in shard_reports]
    if not report_paths:
        raise ValueError("at least one streamed background shard is required")
    reports = [_load_json(path) for path in report_paths]
    statuses = {str(report.get("status")) for report in reports}
    allowed_statuses = {
        "verified_streamed_candidate_background_shard",
        "verified_streamed_morphology_background_shard",
    }
    if len(statuses) != 1 or not statuses <= allowed_statuses:
        raise ValueError("streamed background shards mix calibrated and morphology modes")
    morphology_only = statuses == {"verified_streamed_morphology_background_shard"}
    for report in reports:
        if report.get("status") not in allowed_statuses:
            raise ValueError("streamed background shard report has the wrong status")
        if report.get("split_strategy") != "hash_threshold_v1":
            raise ValueError("only stable hash-threshold shards can be merged")
    common_fields = (
        "event_exclusions_sha256",
        "timing_calibration_report_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "coherence_config_sha256",
        "validation_fraction",
        "test_fraction",
        "seed",
        "model_ifos",
        "q_values",
        "target_sample_rate",
        "context_duration",
        "chirp_threshold",
        "minimum_bins",
        "pairs_per_shard",
        "streaming_mode",
        "code_commit",
    )
    reference = reports[0]["run_identity"]
    if any(
        any(report["run_identity"].get(field) != reference.get(field) for field in common_fields)
        for report in reports[1:]
    ):
        raise ValueError("streamed shards do not share one scoring/split identity")
    reported_parent_hashes = {
        str(report["run_identity"]["parent_plan_sha256"]) for report in reports
    }
    authoritative_plan = None
    authoritative_plan_path = None
    authoritative_plan_sha256 = None
    parent_pair_ids = None
    base_parent_sha256 = None
    base_parent_count = None
    if parent_plan is not None:
        authoritative_plan_path = Path(parent_plan).resolve()
        authoritative_plan = _load_json(authoritative_plan_path)
        parent_pairs = list(authoritative_plan.get("pairs", []))
        parent_pair_ids = [str(row.get("pair_id", "")) for row in parent_pairs]
        if (
            authoritative_plan.get("status") != "development_acquisition_plan"
            or authoritative_plan.get("locked_evaluation_data") is not False
            or not parent_pairs
            or any(not pair_id for pair_id in parent_pair_ids)
            or len(set(parent_pair_ids)) != len(parent_pair_ids)
            or int(authoritative_plan.get("selected_pairs", -1)) != len(parent_pairs)
        ):
            raise ValueError("stream merge parent is not a complete development plan")
        authoritative_plan_sha256 = file_sha256(authoritative_plan_path)
        base_value = authoritative_plan.get("base_parent_plan_sha256")
        allowed_parent_hashes = {authoritative_plan_sha256}
        if base_value is not None:
            base_parent_sha256 = str(base_value)
            base_parent_count = int(authoritative_plan.get("base_selected_pairs", 0))
            if (
                authoritative_plan.get("selection_rule")
                != "frozen_prefix_stratified_complement_v1"
                or authoritative_plan.get("candidate_scores_inspected") is not False
                or not 0 < base_parent_count < len(parent_pair_ids)
            ):
                raise ValueError("extended stream parent has an invalid frozen-prefix count")
            if canonical_hash(parent_pair_ids[:base_parent_count], 64) != str(
                authoritative_plan.get("base_pair_ids_hash", "")
            ):
                raise ValueError("extended stream parent frozen-prefix hash mismatch")
            base_path = Path(
                str(authoritative_plan.get("base_parent_plan_path", ""))
            ).resolve()
            if not base_path.is_file() or file_sha256(base_path) != base_parent_sha256:
                raise ValueError("extended stream parent base artifact hash mismatch")
            base_plan = _load_json(base_path)
            base_ids = [str(row.get("pair_id", "")) for row in base_plan.get("pairs", [])]
            if (
                base_plan.get("status") != "development_acquisition_plan"
                or base_plan.get("locked_evaluation_data") is not False
                or int(base_plan.get("selected_pairs", -1)) != base_parent_count
                or base_ids != parent_pair_ids[:base_parent_count]
            ):
                raise ValueError("extended stream parent is not an exact base-plan prefix")
            allowed_parent_hashes.add(base_parent_sha256)
        if not reported_parent_hashes <= allowed_parent_hashes:
            raise ValueError("streamed shard parent is outside the declared plan lineage")
    elif len(reported_parent_hashes) != 1:
        raise ValueError("streamed shards do not share one parent plan")
    indices = [int(report["run_identity"]["shard_index"]) for report in reports]
    if len(indices) != len(set(indices)):
        raise ValueError("streamed shard indices repeat")
    parent_counts = {int(report["parent_selected_pairs"]) for report in reports}
    shard_counts = {int(report["shard_count"]) for report in reports}
    if authoritative_plan is None and (len(parent_counts) != 1 or len(shard_counts) != 1):
        raise ValueError("streamed shards disagree on parent plan size")
    ranges = sorted(
        (
            int(report["pair_index_start_inclusive"]),
            int(report["pair_index_stop_exclusive"]),
        )
        for report in reports
    )
    if any(start >= stop for start, stop in ranges) or any(
        right_start < left_stop
        for (_, left_stop), (right_start, _) in zip(ranges, ranges[1:])
    ):
        raise ValueError("streamed acquisition pair ranges overlap or are empty")
    if authoritative_plan is not None:
        if parent_pair_ids is None:
            raise RuntimeError("authoritative stream parent pair inventory is missing")
        for report in reports:
            parent_hash = str(report["run_identity"]["parent_plan_sha256"])
            start = int(report["pair_index_start_inclusive"])
            stop = int(report["pair_index_stop_exclusive"])
            declared_count = int(report["parent_selected_pairs"])
            expected_count = (
                base_parent_count
                if parent_hash == base_parent_sha256
                else len(parent_pair_ids)
            )
            pairs_per_shard = int(report["run_identity"].get("pairs_per_shard", 0))
            shard_index = int(report["run_identity"].get("shard_index", -1))
            expected_shards = (
                math.ceil(expected_count / pairs_per_shard)
                if expected_count is not None and pairs_per_shard > 0
                else -1
            )
            if (
                declared_count != expected_count
                or stop > expected_count
                or int(report["shard_count"]) != expected_shards
                or start != shard_index * pairs_per_shard
            ):
                raise ValueError("streamed shard range exceeds its declared lineage parent")
            if canonical_hash(parent_pair_ids[start:stop], 64) != str(
                report.get("selected_pair_ids_hash", "")
            ):
                raise ValueError("streamed shard pair IDs differ from the authoritative plan")

    background_rows = []
    candidates_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    for report in reports:
        manifest = Path(report["background_manifest_path"])
        if file_sha256(manifest) != str(report["background_manifest_sha256"]):
            raise ValueError("streamed background manifest hash mismatch")
        background_rows.extend(_load_jsonl(manifest))
        for split, artifact in report.get("split_artifacts", {}).items():
            if split not in candidates_by_split:
                raise ValueError(f"unexpected streamed candidate split: {split}")
            path_field = (
                "candidate_manifest_path"
                if morphology_only
                else "calibrated_candidate_manifest_path"
            )
            hash_field = (
                "candidate_manifest_sha256"
                if morphology_only
                else "calibrated_candidate_manifest_sha256"
            )
            candidate_manifest = Path(artifact[path_field])
            if file_sha256(candidate_manifest) != str(artifact[hash_field]):
                raise ValueError("streamed candidate manifest hash mismatch")
            split_rows = _load_jsonl(candidate_manifest)
            if any(str(row["split"]) != split for row in split_rows):
                raise ValueError("calibrated candidate appears in the wrong split")
            candidates_by_split[split].extend(split_rows)
    window_ids = [str(row["window_id"]) for row in background_rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("streamed background shards repeat window IDs")
    window_intervals = [
        (float(row["gps_start"]), float(row["gps_end"])) for row in background_rows
    ]
    if len(window_intervals) != len(set(window_intervals)):
        raise ValueError("streamed background shards repeat GPS windows")
    block_splits: dict[str, str] = {}
    for row in background_rows:
        if not str(row.get("observing_run", "")):
            raise ValueError("streamed background window lacks observing-run identity")
        ifos = row.get("ifos")
        if not isinstance(ifos, list) or not ifos or len(set(ifos)) != len(ifos):
            raise ValueError("streamed background window lacks explicit detector availability")
        block = str(row["gps_block"])
        split = str(row["split"])
        prior = block_splits.setdefault(block, split)
        if prior != split:
            raise ValueError(f"GPS block {block} crosses streamed splits")
    known_windows = set(window_ids)
    candidate_ids = []
    for split, rows in candidates_by_split.items():
        for row in rows:
            if str(row["window_id"]) not in known_windows:
                raise ValueError("streamed candidate references an unknown window")
            if not morphology_only and not row.get("timing_empirically_calibrated"):
                raise ValueError("streamed candidate lacks empirical timing calibration")
            candidate_ids.append(str(row["candidate_id"]))
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("streamed candidate IDs repeat")

    output = Path(output_dir)
    report_path = output / "streamed_background_merge_report.json"
    if report_path.exists():
        raise FileExistsError("streamed background merge reports are immutable")
    output.mkdir(parents=True, exist_ok=True)
    background_path = output / "background_windows.jsonl"
    ordered_background = sorted(
        background_rows, key=lambda row: (float(row["gps_start"]), str(row["window_id"]))
    )
    atomic_write_text(
        background_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_background),
    )
    candidate_outputs = {}
    for split, rows in candidates_by_split.items():
        path = output / (
            f"{split}_morphology_candidates.jsonl"
            if morphology_only
            else f"{split}_calibrated_candidates.jsonl"
        )
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["gps_peak"]),
                str(row["ifo"]),
                str(row["candidate_id"]),
            ),
        )
        atomic_write_text(
            path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
        )
        candidate_outputs[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "candidates": len(ordered),
        }
    parent_count = (
        len(parent_pair_ids)
        if parent_pair_ids is not None
        else next(iter(parent_counts))
    )
    covered_indices = {index for start, stop in ranges for index in range(start, stop)}
    complete_parent = covered_indices == set(range(parent_count))
    split_counts = {
        split: sum(str(row["split"]) == split for row in background_rows)
        for split in ("train", "val", "test")
    }
    split_live_time_seconds = {}
    split_detector_time_seconds = {}
    for split in ("train", "val", "test"):
        selected = [row for row in background_rows if str(row["split"]) == split]
        split_live_time_seconds[split] = _union_duration(
            (float(row["gps_start"]), float(row["gps_end"])) for row in selected
        )
        intervals_by_ifo: dict[str, list[tuple[float, float]]] = {}
        for row in selected:
            for ifo in row["ifos"]:
                intervals_by_ifo.setdefault(str(ifo), []).append(
                    (float(row["gps_start"]), float(row["gps_end"]))
                )
        split_detector_time_seconds[split] = sum(
            _union_duration(intervals) for intervals in intervals_by_ifo.values()
        )
    common_run_identity = {field: reference.get(field) for field in common_fields}
    common_run_identity["parent_plan_sha256"] = (
        authoritative_plan_sha256
        if authoritative_plan_sha256 is not None
        else next(iter(reported_parent_hashes))
    )
    pairs_per_shard = int(reference.get("pairs_per_shard") or 1)
    parent_shard_count = (
        math.ceil(parent_count / pairs_per_shard)
        if authoritative_plan is not None
        else next(iter(shard_counts))
    )
    result = {
        "status": (
            "verified_merged_streamed_morphology_background"
            if morphology_only
            else "verified_merged_streamed_candidate_background"
        ),
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "morphology_only": morphology_only,
        "scientific_blocker": (
            "empirical timing calibration and a separate locked test evaluation remain required; "
            "this merged bank supports validation morphology/FAR development only"
            if morphology_only
            else "adequate time-slide exposure, validation-only threshold freeze and an independent "
            "locked test evaluation remain required"
        ),
        "common_run_identity": common_run_identity,
        "parent_plan_lineage": (
            {
                "authoritative_parent_plan_path": str(authoritative_plan_path),
                "authoritative_parent_plan_sha256": authoritative_plan_sha256,
                "base_parent_plan_sha256": base_parent_sha256,
                "base_selected_pairs": base_parent_count,
                "reported_parent_plan_sha256s": sorted(reported_parent_hashes),
            }
            if authoritative_plan is not None
            else None
        ),
        "shard_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in report_paths
        ],
        "shard_count_merged": len(reports),
        "parent_shard_count": parent_shard_count,
        "parent_selected_pairs": parent_count,
        "covered_pair_ranges": [list(value) for value in ranges],
        "covered_parent_pair_indices": len(covered_indices),
        "complete_parent_plan": complete_parent,
        "background_windows": len(background_rows),
        "gps_blocks": len(block_splits),
        "observing_runs": dict(
            sorted(
                Counter(str(row["observing_run"]) for row in background_rows).items()
            )
        ),
        "available_ifos": dict(
            sorted(
                Counter(
                    str(ifo) for row in background_rows for ifo in row["ifos"]
                ).items()
            )
        ),
        "detector_subset_counts": dict(
            sorted(
                Counter(
                    "".join(str(ifo) for ifo in row["ifos"])
                    for row in background_rows
                ).items()
            )
        ),
        "zero_lag_live_time_seconds": _union_duration(window_intervals),
        "detector_time_seconds": sum(split_detector_time_seconds.values()),
        "split_live_time_seconds": split_live_time_seconds,
        "split_detector_time_seconds": split_detector_time_seconds,
        "cross_split_gps_block_overlap": False,
        "split_counts": split_counts,
        "background_manifest_path": str(background_path),
        "background_manifest_sha256": file_sha256(background_path),
        "candidate_manifests": candidate_outputs,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def merge_raw_mask_streamed_background_shards(
    shard_reports: Iterable[str | Path],
    output_dir: str | Path,
    parent_plan: str | Path | None = None,
) -> dict[str, Any]:
    """Merge paired raw/mask shards through two independently calibrated arm merges."""

    report_paths = [Path(path).resolve() for path in shard_reports]
    if not report_paths:
        raise ValueError("at least one paired raw/mask shard is required")
    reports = [_load_json(path) for path in report_paths]
    if any(
        report.get("status")
        != "verified_streamed_raw_mask_candidate_background_shard"
        or report.get("paired_raw_mask_arms") is not True
        or report.get("split_strategy") != "hash_threshold_v1"
        or int(report.get("split_counts", {}).get("test", -1)) != 0
        for report in reports
    ):
        raise ValueError("paired raw/mask merge requires validation-only dual-arm shards")
    common_gate_fields = (
        "mask_validation_receipt_sha256",
        "mask_timing_receipt_sha256",
        "mask_pipeline_report_sha256",
        "raw_timing_calibration_report_sha256",
        "mask_timing_calibration_report_sha256",
        "raw_injection_ranking_report_sha256",
        "mask_injection_ranking_report_sha256",
        "reference_ifo",
        "second_ifo",
        "physical_delay_limit_seconds",
        "deglitch_strength",
        "scoring_compatibility_report_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "coherence_config_sha256",
        "code_commit",
    )
    reference = reports[0]["run_identity"]
    if any(
        any(report["run_identity"].get(field) != reference.get(field) for field in common_gate_fields)
        for report in reports[1:]
    ):
        raise ValueError("paired raw/mask shards do not share one frozen gate identity")
    output = Path(output_dir)
    final_path = output / "raw_mask_streamed_background_merge_report.json"
    input_identities = [
        {"path": str(path), "sha256": file_sha256(path)} for path in report_paths
    ]
    if final_path.is_file():
        prior = _load_json(final_path)
        if (
            prior.get("status")
            != "verified_merged_streamed_raw_mask_candidate_background"
            or prior.get("shard_reports") != input_identities
        ):
            raise ValueError("existing paired raw/mask merge has another identity")
        return prior
    output.mkdir(parents=True, exist_ok=True)
    proxy_root = output / "arm-proxies"
    proxy_root.mkdir(parents=True, exist_ok=True)
    arm_results = {}
    for arm in ("raw", "mask"):
        proxy_paths = []
        for index, (source_path, report) in enumerate(zip(report_paths, reports)):
            proxy_path = proxy_root / arm / f"shard-{index:05d}.json"
            proxy_path.parent.mkdir(parents=True, exist_ok=True)
            run_identity = dict(report["run_identity"])
            run_identity["timing_calibration_report_sha256"] = run_identity[
                f"{arm}_timing_calibration_report_sha256"
            ]
            run_identity["streaming_mode"] = f"paired_raw_mask_arm_{arm}"
            proxy = {
                **report,
                "status": "verified_streamed_candidate_background_shard",
                "paired_raw_mask_arms": False,
                "run_identity": run_identity,
                "run_identity_hash": canonical_hash(run_identity, 64),
                "split_artifacts": {
                    split: {
                        "windows": artifact["windows"],
                        **artifact["arms"][arm],
                    }
                    for split, artifact in report.get("split_artifacts", {}).items()
                },
                "paired_source_shard_report_path": str(source_path),
                "paired_source_shard_report_sha256": file_sha256(source_path),
                "selected_arm": arm,
            }
            if proxy_path.is_file():
                if _load_json(proxy_path) != proxy:
                    raise ValueError("existing paired arm proxy has another identity")
            else:
                atomic_write_json(proxy_path, proxy)
            proxy_paths.append(proxy_path)
        arm_output = output / arm
        arm_report_path = arm_output / "streamed_background_merge_report.json"
        if arm_report_path.is_file():
            merged = _load_json(arm_report_path)
            if merged.get("status") != "verified_merged_streamed_candidate_background":
                raise ValueError(f"existing {arm} merge report has the wrong status")
        else:
            merged = merge_streamed_background_shards(
                proxy_paths, arm_output, parent_plan
            )
        arm_results[arm] = {
            "report": merged,
            "report_path": str(arm_report_path),
            "report_sha256": file_sha256(arm_report_path),
        }
    raw = arm_results["raw"]["report"]
    mask = arm_results["mask"]["report"]
    paired_fields = (
        "parent_selected_pairs",
        "covered_pair_ranges",
        "covered_parent_pair_indices",
        "complete_parent_plan",
        "background_windows",
        "gps_blocks",
        "observing_runs",
        "available_ifos",
        "detector_subset_counts",
        "zero_lag_live_time_seconds",
        "detector_time_seconds",
        "split_live_time_seconds",
        "split_detector_time_seconds",
        "split_counts",
        "background_manifest_sha256",
    )
    if any(raw.get(field) != mask.get(field) for field in paired_fields):
        raise ValueError("raw and mask merges do not cover identical physical background")
    if int(raw.get("split_counts", {}).get("test", -1)) != 0:
        raise ValueError("paired raw/mask development merge unexpectedly contains test rows")
    result = {
        "status": "verified_merged_streamed_raw_mask_candidate_background",
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "scientific_blocker": (
            "fit paired validation-only time-slide thresholds, verify adequate exposure and "
            "clean non-inferiority, then evaluate the independently locked test corpus once"
        ),
        "shard_reports": input_identities,
        "common_gate_identity": {
            field: reference.get(field) for field in common_gate_fields
        },
        "background_manifest_path": raw["background_manifest_path"],
        "background_manifest_sha256": raw["background_manifest_sha256"],
        "split_counts": raw["split_counts"],
        "split_live_time_seconds": raw["split_live_time_seconds"],
        "split_detector_time_seconds": raw["split_detector_time_seconds"],
        "zero_lag_live_time_seconds": raw["zero_lag_live_time_seconds"],
        "detector_time_seconds": raw["detector_time_seconds"],
        "complete_parent_plan": raw["complete_parent_plan"],
        "arm_merges": {
            arm: {
                "report_path": values["report_path"],
                "report_sha256": values["report_sha256"],
                "candidate_manifests": values["report"]["candidate_manifests"],
            }
            for arm, values in arm_results.items()
        },
        **execution_provenance(),
    }
    atomic_write_json(final_path, result)
    return result


def calibrate_streamed_morphology_candidate_rate(
    merge_report_path: str | Path,
    target_rate_per_detector_year: float,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze a validation-only single-IFO morphology trigger-rate threshold.

    Detector-time exposure is the union of valid window intervals for each IFO. This is
    deliberately not a network-coincident FAR or a search-sensitivity measurement.
    """
    from .background import SECONDS_PER_YEAR, _union_duration
    from .search import calibrate_threshold

    if target_rate_per_detector_year < 0:
        raise ValueError("target morphology trigger rate must be non-negative")
    merge_path = Path(merge_report_path)
    merged = _load_json(merge_path)
    if merged.get("status") != "verified_merged_streamed_morphology_background":
        raise ValueError("morphology calibration requires a merged morphology background")
    if not merged.get("morphology_only") or merged.get("test_evaluation") is not None:
        raise ValueError("morphology calibration accepts validation-only morphology data")
    if int(merged.get("split_counts", {}).get("test", 0)) != 0:
        raise ValueError("morphology calibration refuses a merge containing test windows")

    background_path = Path(merged["background_manifest_path"])
    if file_sha256(background_path) != str(merged["background_manifest_sha256"]):
        raise ValueError("merged morphology background manifest hash mismatch")
    candidate_item = merged["candidate_manifests"]["val"]
    candidate_path = Path(candidate_item["path"])
    if file_sha256(candidate_path) != str(candidate_item["sha256"]):
        raise ValueError("merged morphology candidate manifest hash mismatch")
    windows = [
        row for row in _load_jsonl(background_path) if str(row.get("split")) == "val"
    ]
    candidates = _load_jsonl(candidate_path)
    if not windows:
        raise ValueError("morphology calibration has no validation windows")
    if any(str(row.get("split")) != "val" for row in candidates):
        raise ValueError("morphology calibration candidate appears outside validation")

    windows_by_id = {str(row["window_id"]): row for row in windows}
    if len(windows_by_id) != len(windows):
        raise ValueError("morphology calibration repeats validation window IDs")
    intervals_by_ifo: dict[str, list[tuple[float, float]]] = {}
    for row in windows:
        start = float(row["gps_start"])
        stop = float(row["gps_end"])
        if stop <= start or not row.get("ifos"):
            raise ValueError("morphology validation window has invalid exposure metadata")
        for ifo in row["ifos"]:
            intervals_by_ifo.setdefault(str(ifo), []).append((start, stop))
    exposure_by_ifo = {
        ifo: _union_duration(intervals) for ifo, intervals in sorted(intervals_by_ifo.items())
    }
    detector_time_seconds = sum(exposure_by_ifo.values())
    detector_time_years = detector_time_seconds / SECONDS_PER_YEAR
    if detector_time_years <= 0:
        raise ValueError("morphology calibration has no detector-time exposure")

    for row in candidates:
        window = windows_by_id.get(str(row["window_id"]))
        if window is None or str(row["ifo"]) not in {str(value) for value in window["ifos"]}:
            raise ValueError("morphology candidate is not covered by its detector window")
        if row.get("timing_empirically_calibrated"):
            raise ValueError("morphology-only calibration received timing-calibrated candidates")
    scores = [float(row["chirp_score"]) for row in candidates]
    calibration = calibrate_threshold(
        scores, detector_time_years, target_rate_per_detector_year
    )
    selected = [
        row for row in candidates if float(row["chirp_score"]) >= calibration["threshold"]
    ]
    selected_by_ifo = Counter(str(row["ifo"]) for row in selected)
    per_ifo = {}
    for ifo, seconds in exposure_by_ifo.items():
        years = seconds / SECONDS_PER_YEAR
        count = int(selected_by_ifo.get(ifo, 0))
        per_ifo[ifo] = {
            "exposure_seconds": seconds,
            "exposure_years": years,
            "surviving_candidates": count,
            "rate_per_detector_year": count / years,
            "rate_90_upper_limit_if_zero": (
                -math.log(0.1) / years if count == 0 else None
            ),
        }
    extraction_floor = float(merged["common_run_identity"]["chirp_threshold"])
    result = {
        "status": "validation_only_morphology_candidate_rate_frozen",
        "scientific_claim_allowed": False,
        "network_far_claim_allowed": False,
        "test_evaluation": None,
        "scientific_blocker": (
            "single-IFO detector-time morphology trigger rate is not a network FAR; empirical "
            "timing calibration, physical coincidence and a locked test are still required"
        ),
        "merge_report_path": str(merge_path.resolve()),
        "merge_report_sha256": file_sha256(merge_path),
        "background_manifest_path": str(background_path.resolve()),
        "background_manifest_sha256": file_sha256(background_path),
        "candidate_manifest_path": str(candidate_path.resolve()),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "validation_windows": len(windows),
        "validation_candidates": len(candidates),
        "extraction_floor": extraction_floor,
        "selected_threshold_respects_extraction_floor": (
            float(calibration["threshold"]) >= extraction_floor
        ),
        "detector_time_seconds": detector_time_seconds,
        "detector_time_years": detector_time_years,
        "exposure_by_ifo_seconds": exposure_by_ifo,
        "target_expected_count": target_rate_per_detector_year * detector_time_years,
        "target_rate_exposure_gate_one_expected_count": (
            target_rate_per_detector_year * detector_time_years >= 1.0
        ),
        "calibration": calibration,
        "per_ifo_at_frozen_threshold": per_ifo,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result
