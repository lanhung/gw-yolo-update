from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .runtime import execution_provenance


SECONDS_PER_YEAR = 365.25 * 24 * 3600


def validate_source_verification(
    files: dict[str, str | Path], verification_report: str | Path
) -> dict[str, Any]:
    report_path = Path(verification_report)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("passed") or report.get("status") != "verified":
        raise ValueError("GWOSC source verification report did not pass")
    detector_reports = report.get("detectors", {})
    verified = {}
    for ifo, path in sorted(files.items()):
        detector = detector_reports.get(ifo)
        if not detector or not detector.get("passed"):
            raise ValueError(f"GWOSC source verification is missing a passing {ifo} record")
        observed_sha = file_sha256(path)
        expected_sha = detector.get("sha256")
        if observed_sha != expected_sha:
            raise ValueError(
                f"GWOSC source hash differs from verification report for {ifo}: "
                f"{observed_sha} != {expected_sha}"
            )
        verified[ifo] = observed_sha
    return {
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "event": report.get("event"),
        "detector_sha256": verified,
    }


def _read_quality(path: str | Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Background planning requires the optional h5py dependency") from exc
    with h5py.File(path, "r") as handle:
        gps_start = int(handle["meta/GPSstart"][()])
        duration = int(handle["meta/Duration"][()])
        dqmask = np.asarray(handle["quality/simple/DQmask"], dtype=np.int64)
        injection_path = "quality/injections/Injmask"
        injmask = (
            np.asarray(handle[injection_path], dtype=np.int64)
            if injection_path in handle
            else np.full(duration, -1, dtype=np.int64)
        )
    if dqmask.size < duration or injmask.size < duration:
        raise ValueError(f"Quality vectors in {path} are shorter than metadata duration")
    return {
        "gps_start": gps_start,
        "gps_end": gps_start + duration,
        "duration": duration,
        "dqmask": dqmask[:duration],
        "injmask": injmask[:duration],
    }


def _overlaps(start: float, end: float, intervals: Iterable[tuple[float, float]]) -> bool:
    return any(start < excluded_end and end > excluded_start for excluded_start, excluded_end in intervals)


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _assign_blocks(
    block_ids: Iterable[str], validation_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be non-negative and sum below one")
    ordered = sorted(
        set(block_ids), key=lambda value: canonical_hash({"block_id": value, "seed": seed}, 64)
    )
    validation_count = round(len(ordered) * validation_fraction)
    test_count = round(len(ordered) * test_fraction)
    mapping = {}
    for index, block_id in enumerate(ordered):
        if index < validation_count:
            split = "val"
        elif index < validation_count + test_count:
            split = "test"
        else:
            split = "train"
        mapping[block_id] = split
    return mapping


def plan_background_windows(
    files: dict[str, str | Path],
    window_duration: int = 8,
    stride: int = 8,
    block_duration: int = 256,
    required_context_duration: int | None = None,
    required_dq_bits: int = 1,
    required_injection_bits: int = 0,
    excluded_intervals: Iterable[tuple[float, float]] = (),
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 20260719,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not files:
        raise ValueError("At least one detector file is required")
    if min(window_duration, stride, block_duration) <= 0:
        raise ValueError("window, stride, and block durations must be positive")
    if window_duration > block_duration:
        raise ValueError("window duration cannot exceed block duration")
    context_duration = required_context_duration or window_duration
    if context_duration < window_duration:
        raise ValueError("required context duration cannot be shorter than the window")
    quality = {ifo: _read_quality(path) for ifo, path in sorted(files.items())}
    common_start = max(item["gps_start"] for item in quality.values())
    common_end = min(item["gps_end"] for item in quality.values())
    if common_end - common_start < window_duration:
        raise ValueError("Detector files have no usable common interval")
    exclusions = list(excluded_intervals)
    source_files = {
        ifo: {"path": str(files[ifo]), "sha256": file_sha256(files[ifo])}
        for ifo in sorted(files)
    }
    candidates = []
    rejections: Counter[str] = Counter()
    grid_windows = 0
    first_start = int(math.ceil(common_start / stride) * stride)
    for gps_start in range(first_start, common_end - window_duration + 1, stride):
        grid_windows += 1
        gps_end = gps_start + window_duration
        center = (gps_start + gps_end) / 2
        if (
            center - context_duration / 2 < common_start
            or center + context_duration / 2 > common_end
        ):
            rejections["insufficient_preprocessing_context"] += 1
            continue
        block_index = (gps_start - common_start) // block_duration
        block_start = common_start + block_index * block_duration
        if gps_end > block_start + block_duration:
            rejections["crosses_gps_block_boundary"] += 1
            continue
        if _overlaps(gps_start, gps_end, exclusions):
            rejections["catalog_or_declared_exclusion"] += 1
            continue
        dq_values = []
        injection_values = []
        valid = True
        for item in quality.values():
            context_start = int(math.floor(center - context_duration / 2))
            context_stop = int(math.ceil(center + context_duration / 2))
            start_index = context_start - item["gps_start"]
            stop_index = context_stop - item["gps_start"]
            dq = item["dqmask"][start_index:stop_index]
            injection = item["injmask"][start_index:stop_index]
            expected_context_seconds = context_stop - context_start
            if dq.size != expected_context_seconds or injection.size != expected_context_seconds:
                rejections["incomplete_quality_context"] += 1
                valid = False
                break
            if required_dq_bits and np.any((dq & required_dq_bits) != required_dq_bits):
                rejections["required_dq_bits_missing_in_context"] += 1
                valid = False
                break
            if required_injection_bits and np.any(
                (injection & required_injection_bits) != required_injection_bits
            ):
                rejections["required_no_injection_bits_missing_in_context"] += 1
                valid = False
                break
            dq_values.extend(int(value) for value in dq)
            injection_values.extend(int(value) for value in injection)
        if not valid:
            continue
        block_id = f"gps:{block_start}:{block_duration}"
        candidates.append(
            {
                "window_id": f"background-{canonical_hash({'gps': gps_start, 'ifos': sorted(files)}, 20)}",
                "gps_start": gps_start,
                "gps_end": gps_end,
                "duration": window_duration,
                "ifos": sorted(files),
                "gps_block": block_id,
                "dq_bitwise_and": int(np.bitwise_and.reduce(dq_values)),
                "inj_bitwise_and": int(np.bitwise_and.reduce(injection_values)),
                "source_files": source_files,
            }
        )
    block_mapping = _assign_blocks(
        (row["gps_block"] for row in candidates), validation_fraction, test_fraction, seed
    )
    for row in candidates:
        row["split"] = block_mapping[row["gps_block"]]
    split_blocks = {
        split: {row["gps_block"] for row in candidates if row["split"] == split}
        for split in ("train", "val", "test")
    }
    overlaps = {
        f"{left}:{right}": sorted(split_blocks[left] & split_blocks[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    split_summary = {}
    for split in ("train", "val", "test"):
        rows = [row for row in candidates if row["split"] == split]
        live_seconds = _union_duration((row["gps_start"], row["gps_end"]) for row in rows)
        split_summary[split] = {
            "windows": len(rows),
            "gps_blocks": len(split_blocks[split]),
            "live_time_seconds": live_seconds,
            "live_time_years": live_seconds / SECONDS_PER_YEAR,
        }
    report = {
        "passed": all(not value for value in overlaps.values()),
        "ifos": sorted(files),
        "common_gps_interval": [common_start, common_end],
        "window_duration": window_duration,
        "stride": stride,
        "block_duration": block_duration,
        "required_context_duration": context_duration,
        "required_dq_bits": required_dq_bits,
        "required_injection_bits": required_injection_bits,
        "excluded_intervals": exclusions,
        "windows": len(candidates),
        "candidate_grid_windows": grid_windows,
        "rejected_windows": sum(rejections.values()),
        "rejection_counts": dict(sorted(rejections.items())),
        "unique_gps_blocks": len(block_mapping),
        "cross_split_block_overlaps": overlaps,
        "splits": split_summary,
    }
    return candidates, report


def run_background_plan(
    files: dict[str, str | Path],
    output_dir: str | Path,
    source_verification_report: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    verification = validate_source_verification(files, source_verification_report)
    rows, report = plan_background_windows(files, **kwargs)
    if not report["passed"]:
        raise ValueError(f"Background split audit failed: {report}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "background_windows.jsonl"
    for row in rows:
        for source in row["source_files"].values():
            source["verification_report_sha256"] = verification["report_sha256"]
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    result = {
        **report,
        "source_verification": verification,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / "background_plan_report.json", result)
    return result


def run_batch_background_plan(
    batch_report_path: str | Path | Iterable[str | Path],
    event_exclusions_path: str | Path,
    output_dir: str | Path,
    window_duration: int = 8,
    stride: int = 8,
    block_duration: int = 256,
    required_context_duration: int = 64,
    required_dq_bits: int = 1,
    required_injection_bits: int = 23,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 20260719,
) -> dict[str, Any]:
    report_paths = (
        [Path(batch_report_path)]
        if isinstance(batch_report_path, (str, Path))
        else [Path(path) for path in batch_report_path]
    )
    if not report_paths:
        raise ValueError("At least one batch strain report is required")
    batches = []
    for report_path in report_paths:
        with report_path.open("r", encoding="utf-8") as handle:
            batch = json.load(handle)
        if batch.get("status") != "verified_development_strain_batch" or not batch.get(
            "passed"
        ):
            raise ValueError("Batch strain report is not fully verified")
        batches.append((batch, file_sha256(report_path)))
    runs = {str(batch.get("run")) for batch, _ in batches}
    if len(runs) != 1:
        raise ValueError("Batch strain reports must belong to one observing run")
    run = next(iter(runs))
    with Path(event_exclusions_path).open("r", encoding="utf-8") as handle:
        exclusion_report = json.load(handle)
    if exclusion_report.get("status") != "development_catalog_event_exclusions":
        raise ValueError("Event exclusions are not a development catalog report")
    if exclusion_report.get("run") != run:
        raise ValueError("Batch strain and event-exclusion runs differ")
    excluded_intervals = [
        (float(row["exclusion_start"]), float(row["exclusion_end"]))
        for row in exclusion_report.get("intervals", [])
    ]
    by_pair: dict[str, dict[str, str]] = {}
    pair_gps = {}
    pair_batch_sha = {}
    source_files = 0
    for batch, batch_sha in batches:
        for source in batch.get("files", []):
            source_files += 1
            if not source.get("verification", {}).get("passed"):
                raise ValueError(f"Source verification did not pass: {source.get('path')}")
            if file_sha256(source["path"]) != source["sha256"]:
                raise ValueError(f"Batch source hash mismatch: {source['path']}")
            pair_id = str(source["pair_id"])
            if pair_id in pair_batch_sha and pair_batch_sha[pair_id] != batch_sha:
                raise ValueError(f"Pair {pair_id} appears in multiple batch reports")
            pair_batch_sha[pair_id] = batch_sha
            ifo = str(source["detector"])
            if ifo in by_pair.setdefault(pair_id, {}):
                raise ValueError(f"Duplicate detector {ifo} in pair {pair_id}")
            by_pair[pair_id][ifo] = str(source["path"])
            pair_gps[pair_id] = int(source["gps_start"])
    detector_sets = {tuple(sorted(files)) for files in by_pair.values()}
    if len(detector_sets) != 1 or not detector_sets or len(next(iter(detector_sets))) < 2:
        raise ValueError("Batch report does not contain complete, consistent multi-IFO pairs")
    rows = []
    aggregate_grid_windows = 0
    aggregate_rejections: Counter[str] = Counter()
    for pair_id in sorted(by_pair, key=lambda value: pair_gps[value]):
        pair_rows, pair_report = plan_background_windows(
            by_pair[pair_id],
            window_duration=window_duration,
            stride=stride,
            block_duration=block_duration,
            required_context_duration=required_context_duration,
            required_dq_bits=required_dq_bits,
            required_injection_bits=required_injection_bits,
            excluded_intervals=excluded_intervals,
            validation_fraction=0,
            test_fraction=0,
            seed=seed,
        )
        aggregate_grid_windows += int(pair_report["candidate_grid_windows"])
        aggregate_rejections.update(pair_report["rejection_counts"])
        for row in pair_rows:
            row["pair_id"] = pair_id
            row["observing_run"] = run
            for source in row["source_files"].values():
                source["verification_report_sha256"] = pair_batch_sha[pair_id]
        rows.extend(pair_rows)
    if not rows:
        raise ValueError("Verified batch produced no DQ-safe background windows")
    block_mapping = _assign_blocks(
        (row["gps_block"] for row in rows), validation_fraction, test_fraction, seed
    )
    for row in rows:
        row["split"] = block_mapping[row["gps_block"]]
    split_blocks = {
        split: {row["gps_block"] for row in rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    overlaps = {
        f"{left}:{right}": sorted(split_blocks[left] & split_blocks[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    split_summary = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        live_seconds = _union_duration(
            (float(row["gps_start"]), float(row["gps_end"])) for row in selected
        )
        split_summary[split] = {
            "windows": len(selected),
            "gps_blocks": len(split_blocks[split]),
            "live_time_seconds": live_seconds,
            "live_time_years": live_seconds / SECONDS_PER_YEAR,
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "background_windows.jsonl"
    atomic_write_text(
        manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        "status": "verified_multi_segment_development_background",
        "scientific_claim_allowed": False,
        "passed": all(not values for values in overlaps.values()),
        "run": run,
        "ifos": list(next(iter(detector_sets))),
        "source_pairs": len(by_pair),
        "source_files": source_files,
        "source_batch_report_sha256s": [sha for _, sha in batches],
        "event_exclusions_sha256": file_sha256(event_exclusions_path),
        "catalog_events_excluded": int(exclusion_report.get("events", 0)),
        "event_padding_seconds": float(exclusion_report["padding_seconds"]),
        "window_duration": window_duration,
        "stride": stride,
        "block_duration": block_duration,
        "required_context_duration": required_context_duration,
        "required_dq_bits": required_dq_bits,
        "required_injection_bits": required_injection_bits,
        "windows": len(rows),
        "candidate_grid_windows": aggregate_grid_windows,
        "rejected_windows": sum(aggregate_rejections.values()),
        "rejection_counts": dict(sorted(aggregate_rejections.items())),
        "unique_gps_blocks": len(block_mapping),
        "cross_split_block_overlaps": overlaps,
        "splits": split_summary,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / "background_plan_report.json", result)
    if not result["passed"]:
        raise RuntimeError("Batch background split audit failed")
    return result
