from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_write_json, canonical_hash, file_sha256
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


def run_streamed_background_shard(
    parent_plan: str | Path,
    event_exclusions: str | Path,
    timing_calibration_report: str | Path,
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

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "parent_plan_sha256": file_sha256(parent_plan),
        "event_exclusions_sha256": file_sha256(event_exclusions),
        "timing_calibration_report_sha256": file_sha256(timing_calibration_report),
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
        "code_commit": execution_provenance()["code_commit"],
    }
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
            calibrated = _load_json(calibrated_report_path)
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
            "calibrated_candidate_manifest_path": str(calibrated_path),
            "calibrated_candidate_manifest_sha256": file_sha256(calibrated_path),
            "calibrated_candidate_report_sha256": file_sha256(calibrated_report_path),
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
        "status": "verified_streamed_candidate_background_shard",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "merge disjoint stable-hash shards, construct adequate time-slide exposure, freeze "
            "the validation threshold, and evaluate the independently locked test partition"
        ),
        "run_identity": identity,
        "run_identity_hash": canonical_hash(identity, 64),
        "split_strategy": "hash_threshold_v1",
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
