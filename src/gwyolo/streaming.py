from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_write_json, file_sha256
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
