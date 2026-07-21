from __future__ import annotations

import json
import math
import os
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .io import atomic_write_json, canonical_hash, file_sha256
from .runtime import execution_provenance


API_ROOT = "https://gwosc.org/api/v2"
USER_AGENT = "GW-YOLO-research/0.1"


def _api_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object from {url}")
    return value


def resolve_event(event: str) -> dict[str, Any]:
    event_record = _api_json(f"{API_ROOT}/events/{event}")
    versions = event_record.get("versions", [])
    if not versions:
        raise ValueError(f"GWOSC returned no versions for {event}")
    preferred = next(
        (item for item in versions if item.get("catalog") == "O4_Discovery_Papers"),
        versions[0],
    )
    detail = _api_json(str(preferred["detail_url"]))
    return {
        "event": event,
        "gps": float(detail["gps"]),
        "run": str(detail["run"]),
        "version": int(detail["version"]),
        "catalog": str(detail["catalog"]),
        "detectors": [str(item) for item in detail.get("detectors", [])],
    }


def event_strain_files(
    event: str,
    detectors: Iterable[str] | None = None,
    sample_rate_khz: int = 4,
) -> list[dict[str, Any]]:
    wanted = set(detectors or [])
    payload = _api_json(f"{API_ROOT}/events/{event}/strain-files")
    records = []
    for item in payload.get("results", []):
        if int(item["sample_rate_kHz"]) != sample_rate_khz:
            continue
        if wanted and str(item["detector"]) not in wanted:
            continue
        records.append(
            {
                "detector": str(item["detector"]),
                "sample_rate": sample_rate_khz * 1024,
                "gps_start": int(item["gps_start"]),
                "hdf5_url": str(item["hdf5_url"]),
                "detail_url": str(item["detail_url"]),
            }
        )
    records.sort(key=lambda item: item["detector"])
    if wanted - {record["detector"] for record in records}:
        raise ValueError(f"Missing GWOSC strain for detectors: {sorted(wanted - {record['detector'] for record in records})}")
    if not records:
        raise ValueError(f"No {sample_rate_khz} kHz strain files found for {event}")
    return records


def _api_results(url: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    page_count = 0
    results_count = None
    while url:
        payload = _api_json(url)
        page_count += 1
        if results_count is None:
            results_count = int(payload.get("results_count", 0))
        rows.extend(payload.get("results", []))
        next_url = payload.get("next")
        url = str(next_url) if next_url else ""
    return rows, {"api_results_count": results_count, "api_pages": page_count}


def _stratified_records(
    records: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(records):
        return records
    if maximum <= 0:
        raise ValueError("maximum pair count must be positive")
    boundaries = np.linspace(0, len(records), maximum + 1, dtype=int)
    selected = []
    for index in range(maximum):
        candidates = records[boundaries[index] : boundaries[index + 1]]
        selected.append(
            min(
                candidates,
                key=lambda row: canonical_hash(
                    {"gps_start": row["gps_start"], "seed": seed}, 64
                ),
            )
        )
    return sorted(selected, key=lambda row: int(row["gps_start"]))


def plan_run_strain_pairs(
    run: str,
    detectors: Iterable[str] = ("H1", "L1"),
    sample_rate_khz: int = 4,
    maximum_pairs: int | None = None,
    seed: int = 20260719,
) -> dict[str, Any]:
    if run.lower().startswith("o4b"):
        raise ValueError("O4b is locked evaluation data and cannot enter a development plan")
    wanted = tuple(sorted(set(str(ifo).upper() for ifo in detectors)))
    if len(wanted) < 2:
        raise ValueError("A run plan requires at least two detectors")
    endpoint = (
        f"{API_ROOT}/runs/{run}/strain-files?sample-rate={sample_rate_khz}&pagesize=500"
    )
    records, api_summary = _api_results(endpoint)
    by_gps: dict[int, dict[str, dict[str, Any]]] = {}
    for item in records:
        if int(item["sample_rate_kHz"]) != sample_rate_khz:
            continue
        ifo = str(item["detector"])
        if ifo not in wanted:
            continue
        gps_start = int(item["gps_start"])
        if ifo in by_gps.setdefault(gps_start, {}):
            raise ValueError(f"Duplicate {ifo} strain record at GPS {gps_start}")
        by_gps[gps_start][ifo] = {
            "detector": ifo,
            "gps_start": gps_start,
            "sample_rate": sample_rate_khz * 1024,
            "hdf5_url": str(item["hdf5_url"]),
            "detail_url": str(item["detail_url"]),
        }
    aligned = [
        {
            "pair_id": f"{run}-{gps_start}-{'-'.join(wanted)}",
            "run": run,
            "gps_start": gps_start,
            "detectors": {ifo: by_gps[gps_start][ifo] for ifo in wanted},
        }
        for gps_start in sorted(by_gps)
        if set(by_gps[gps_start]) == set(wanted)
    ]
    if not aligned:
        raise ValueError(f"No aligned {wanted} strain-file pairs found for {run}")
    selected = _stratified_records(aligned, maximum_pairs, seed)
    return {
        "status": "development_acquisition_plan",
        "locked_evaluation_data": False,
        "run": run,
        "detectors": list(wanted),
        "sample_rate_khz": sample_rate_khz,
        "seed": seed,
        "source_endpoint": endpoint,
        **api_summary,
        "aligned_pairs_available": len(aligned),
        "selected_pairs": len(selected),
        "selected_gps_span": [selected[0]["gps_start"], selected[-1]["gps_start"]],
        "pairs": selected,
    }


def run_gwosc_run_plan(
    run: str,
    detectors: Iterable[str],
    output: str | Path,
    sample_rate_khz: int = 4,
    maximum_pairs: int | None = None,
    seed: int = 20260719,
) -> dict[str, Any]:
    result = {
        **plan_run_strain_pairs(run, detectors, sample_rate_khz, maximum_pairs, seed),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return {**result, "plan_path": str(output), "plan_sha256": file_sha256(output)}


def extend_gwosc_run_plan(
    base_plan_path: str | Path,
    output: str | Path,
    target_pairs: int,
    extension_seed: int | None = None,
) -> dict[str, Any]:
    """Append score-blind pairs while preserving a frozen parent as an exact prefix."""

    base_path = Path(base_plan_path).resolve()
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("extended GWOSC acquisition plans are immutable")
    with base_path.open("r", encoding="utf-8") as handle:
        base = json.load(handle)
    base_pairs = list(base.get("pairs", []))
    base_ids = [str(row.get("pair_id", "")) for row in base_pairs]
    if (
        base.get("status") != "development_acquisition_plan"
        or base.get("locked_evaluation_data") is not False
        or str(base.get("run", "")).lower().startswith("o4b")
        or not base_pairs
        or any(not pair_id for pair_id in base_ids)
        or len(set(base_ids)) != len(base_ids)
        or int(base.get("selected_pairs", -1)) != len(base_pairs)
    ):
        raise ValueError("GWOSC extension requires a complete unlocked development parent")
    if base.get("base_parent_plan_sha256") is not None:
        raise ValueError("GWOSC extension currently requires a root parent plan")
    if target_pairs <= len(base_pairs):
        raise ValueError("GWOSC extension target must exceed the frozen parent size")

    run = str(base["run"])
    detectors = tuple(str(value) for value in base["detectors"])
    sample_rate_khz = int(base["sample_rate_khz"])
    base_seed = int(base["seed"])
    full = plan_run_strain_pairs(
        run,
        detectors,
        sample_rate_khz,
        maximum_pairs=None,
        seed=base_seed,
    )
    identity_fields = ("run", "detectors", "sample_rate_khz", "source_endpoint")
    if any(full.get(field) != base.get(field) for field in identity_fields):
        raise ValueError("current GWOSC inventory does not match the frozen parent identity")
    full_pairs = list(full["pairs"])
    if target_pairs > len(full_pairs):
        raise ValueError("GWOSC extension target exceeds aligned source-pair availability")
    full_by_id = {str(row["pair_id"]): row for row in full_pairs}
    if len(full_by_id) != len(full_pairs):
        raise ValueError("current GWOSC inventory repeats source-pair IDs")
    for row in base_pairs:
        pair_id = str(row["pair_id"])
        if full_by_id.get(pair_id) != row:
            raise ValueError(
                f"frozen parent pair {pair_id} is absent or changed in the current inventory"
            )

    seed = base_seed if extension_seed is None else extension_seed
    base_id_set = set(base_ids)
    complement = [row for row in full_pairs if str(row["pair_id"]) not in base_id_set]
    additional = _stratified_records(complement, target_pairs - len(base_pairs), seed)
    extended_pairs = [*base_pairs, *additional]
    extended_ids = [str(row["pair_id"]) for row in extended_pairs]
    if len(extended_pairs) != target_pairs or len(set(extended_ids)) != target_pairs:
        raise RuntimeError("GWOSC extension did not produce the requested unique pair count")
    gps_values = [int(row["gps_start"]) for row in extended_pairs]
    result = {
        **{key: value for key, value in full.items() if key != "pairs"},
        "selected_pairs": len(extended_pairs),
        "selected_gps_span": [min(gps_values), max(gps_values)],
        "pairs": extended_pairs,
        "selection_rule": "frozen_prefix_stratified_complement_v1",
        "selection_data": "GWOSC strain-file metadata only",
        "candidate_scores_inspected": False,
        "base_parent_plan_path": str(base_path),
        "base_parent_plan_sha256": file_sha256(base_path),
        "base_selected_pairs": len(base_pairs),
        "base_pair_ids_hash": canonical_hash(base_ids, 64),
        "extension_seed": seed,
        "extension_target_pairs": target_pairs,
        "extension_pairs": len(additional),
        "extension_pair_ids_hash": canonical_hash(
            [str(row["pair_id"]) for row in additional], 64
        ),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return {**result, "plan_path": str(target), "plan_sha256": file_sha256(target)}


def run_gwosc_plan_shard(
    plan_path: str | Path,
    output: str | Path,
    shard_index: int,
    pairs_per_shard: int = 1,
) -> dict[str, Any]:
    """Select one immutable, non-overlapping slice of a frozen acquisition plan."""

    if shard_index < 0 or pairs_per_shard <= 0:
        raise ValueError("shard index must be non-negative and pairs per shard must be positive")
    with Path(plan_path).open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("status") != "development_acquisition_plan":
        raise ValueError("GWOSC plan shard requires a development acquisition plan")
    if plan.get("locked_evaluation_data") or str(plan.get("run", "")).lower().startswith(
        "o4b"
    ):
        raise ValueError("O4b is locked evaluation data and cannot enter a development shard")
    pairs = list(plan.get("pairs", []))
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if not pairs or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("parent acquisition plan has no pairs or repeats pair IDs")
    if int(plan.get("selected_pairs", len(pairs))) != len(pairs):
        raise ValueError("parent acquisition plan selected-pair count is inconsistent")
    start = shard_index * pairs_per_shard
    stop = min(start + pairs_per_shard, len(pairs))
    if start >= len(pairs):
        raise ValueError(
            f"shard {shard_index} starts beyond the {len(pairs)} parent plan pairs"
        )
    selected = pairs[start:stop]
    result = {
        "status": "development_acquisition_plan",
        "locked_evaluation_data": False,
        "run": plan["run"],
        "detectors": list(plan["detectors"]),
        "sample_rate_khz": int(plan["sample_rate_khz"]),
        "seed": int(plan["seed"]),
        "source_endpoint": plan["source_endpoint"],
        "aligned_pairs_available": int(plan["aligned_pairs_available"]),
        "selected_pairs": len(selected),
        "selected_gps_span": [selected[0]["gps_start"], selected[-1]["gps_start"]],
        "pairs": selected,
        "parent_plan_path": str(plan_path),
        "parent_plan_sha256": file_sha256(plan_path),
        "parent_selected_pairs": len(pairs),
        "shard_index": shard_index,
        "pairs_per_shard": pairs_per_shard,
        "pair_index_start_inclusive": start,
        "pair_index_stop_exclusive": stop,
        "shard_count": (len(pairs) + pairs_per_shard - 1) // pairs_per_shard,
        "selected_pair_ids_hash": canonical_hash(pair_ids[start:stop], 64),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return {**result, "plan_path": str(output), "plan_sha256": file_sha256(output)}


def run_gwosc_event_exclusions(
    run: str,
    output: str | Path,
    padding_seconds: float = 16.0,
    workers: int = 4,
) -> dict[str, Any]:
    if run.lower().startswith("o4b"):
        raise ValueError("O4b is locked evaluation data and cannot enter development exclusions")
    if padding_seconds <= 0 or workers <= 0:
        raise ValueError("padding and workers must be positive")
    endpoint = f"{API_ROOT}/runs/{run}/events?pagesize=500"
    events, api_summary = _api_results(endpoint)
    selected = []
    for event in events:
        versions = list(event.get("versions", []))
        if not versions:
            raise ValueError(f"GWOSC event has no versions: {event.get('name')}")
        preferred = max(versions, key=lambda row: int(row["version"]))
        selected.append((str(event["name"]), str(preferred["detail_url"])))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        details = list(executor.map(lambda item: _api_json(item[1]), selected))
    rows = []
    for (name, detail_url), detail in zip(selected, details):
        if str(detail["run"]) != run:
            raise ValueError(f"Event {name} reports run {detail['run']}, expected {run}")
        gps = float(detail["gps"])
        rows.append(
            {
                "event": name,
                "gps": gps,
                "run": run,
                "catalog": detail.get("catalog"),
                "version": detail.get("version"),
                "detail_url": detail_url,
                "exclusion_start": gps - padding_seconds,
                "exclusion_end": gps + padding_seconds,
            }
        )
    rows.sort(key=lambda row: row["gps"])
    result = {
        "status": "development_catalog_event_exclusions",
        "locked_evaluation_data": False,
        "run": run,
        "padding_seconds": padding_seconds,
        "source_endpoint": endpoint,
        **api_summary,
        "events": len(rows),
        "intervals": rows,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return {**result, "output_path": str(output), "output_sha256": file_sha256(output)}


def run_gwosc_batch_download(
    plan_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    maximum_pairs: int | None = None,
    download_workers: int = 8,
    chunk_samples: int = 1_048_576,
) -> dict[str, Any]:
    with Path(plan_path).open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if str(plan.get("run", "")).lower().startswith("o4b"):
        raise ValueError("O4b is locked evaluation data and cannot be downloaded for development")
    pairs = _stratified_records(list(plan.get("pairs", [])), maximum_pairs, int(plan["seed"]))
    if not pairs:
        raise ValueError("Acquisition plan contains no selected pairs")
    cache = Path(cache_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "plan_sha256": file_sha256(plan_path),
        "selected_pair_ids_hash": canonical_hash([row["pair_id"] for row in pairs], 64),
        "download_workers": download_workers,
        "chunk_samples": chunk_samples,
    }
    state_path = output / "batch_download_state.json"
    partial_path = output / "batch_download_partial.json"
    completed = []
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing batch-download state belongs to a different run")
        if partial_path.is_file():
            with partial_path.open("r", encoding="utf-8") as handle:
                completed = json.load(handle).get("files", [])
    completed_keys = set()
    for row in completed:
        key = (str(row["pair_id"]), str(row["detector"]))
        if key in completed_keys or file_sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Invalid resumable batch-download entry: {key}")
        completed_keys.add(key)
    for pair in pairs:
        for ifo, record in sorted(pair["detectors"].items()):
            key = (str(pair["pair_id"]), str(ifo))
            if key in completed_keys:
                continue
            filename = Path(urlparse(record["hdf5_url"]).path).name
            download = download_resumable(
                record["hdf5_url"], cache / str(plan["run"]) / filename, workers=download_workers
            )
            verification = verify_hdf5_against_detail(
                download["path"], _api_json(record["detail_url"]), chunk_samples
            )
            entry = {
                "pair_id": pair["pair_id"],
                "run": plan["run"],
                "gps_start": pair["gps_start"],
                "detector": ifo,
                "path": download["path"],
                "sha256": download["sha256"],
                "bytes": download["bytes"],
                "downloaded": download["downloaded"],
                "detail_url": record["detail_url"],
                "verification": verification,
            }
            completed.append(entry)
            completed_keys.add(key)
            atomic_write_json(partial_path, {"run_identity": run_identity, "files": completed})
            atomic_write_json(
                state_path,
                {
                    "status": "in_progress" if verification["passed"] else "failed",
                    "run_identity": run_identity,
                    "completed_files": len(completed),
                    "requested_files": len(pairs) * len(plan["detectors"]),
                },
            )
            if not verification["passed"]:
                raise RuntimeError(f"Full-file verification failed for {key}")
    result = {
        "status": "verified_development_strain_batch",
        "passed": all(row["verification"]["passed"] for row in completed),
        "run": plan["run"],
        "plan_path": str(plan_path),
        "plan_sha256": run_identity["plan_sha256"],
        "selected_pairs": len(pairs),
        "verified_files": len(completed),
        "files": completed,
        **execution_provenance(),
    }
    report_path = output / "batch_download_report.json"
    atomic_write_json(report_path, result)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed_files": len(completed),
            "requested_files": len(pairs) * len(plan["detectors"]),
            "report_sha256": file_sha256(report_path),
        },
    )
    return result


def verify_hdf5_against_detail(
    path: str | Path, detail: dict[str, Any], chunk_samples: int = 1_048_576
) -> dict[str, Any]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("GWOSC verification requires the optional h5py dependency") from exc
    source = Path(path)
    failures = []
    try:
        with h5py.File(source, "r") as handle:
            dataset = handle["strain/Strain"]
            count = 0
            total = np.longdouble(0.0)
            total_square = np.longdouble(0.0)
            minimum = math.inf
            maximum = -math.inf
            nonfinite = 0
            for start in range(0, dataset.shape[0], chunk_samples):
                values = np.asarray(dataset[start : start + chunk_samples], dtype=np.float64)
                finite = np.isfinite(values)
                nonfinite += int((~finite).sum())
                valid = values[finite]
                if valid.size:
                    count += int(valid.size)
                    total += np.sum(valid, dtype=np.longdouble)
                    total_square += np.sum(
                        valid.astype(np.longdouble) ** 2, dtype=np.longdouble
                    )
                    minimum = min(minimum, float(np.min(valid)))
                    maximum = max(maximum, float(np.max(valid)))
            if count == 0:
                raise ValueError("strain dataset has no finite samples")
            mean = float(total / count)
            variance = max(float(total_square / count - np.longdouble(mean) ** 2), 0.0)
            standard_deviation = math.sqrt(variance)
            dqmask = np.asarray(handle["quality/simple/DQmask"], dtype=np.int64)
            injmask = np.asarray(handle["quality/injections/Injmask"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        return {
            "passed": False,
            "path": str(source),
            "bytes": source.stat().st_size if source.exists() else None,
            "sha256": file_sha256(source) if source.is_file() else None,
            "read_error": str(exc),
            "failures": ["full_hdf5_chunk_scan_failed"],
        }
    observed = {
        "filesize_bytes": source.stat().st_size,
        "mean_strain": mean,
        "stdev_strain": standard_deviation,
        "min_strain": minimum,
        "max_strain": maximum,
        "nans_fraction": nonfinite / (count + nonfinite),
    }
    tolerances = {
        "mean_strain": (1e-6, 1e-30),
        "stdev_strain": (1e-10, 0.0),
        "min_strain": (1e-12, 0.0),
        "max_strain": (1e-12, 0.0),
        "nans_fraction": (0.0, 1e-15),
    }
    if observed["filesize_bytes"] != int(detail["filesize_bytes"]):
        failures.append("filesize_bytes_mismatch")
    for field, (relative, absolute) in tolerances.items():
        if not math.isclose(
            float(observed[field]), float(detail[field]), rel_tol=relative, abs_tol=absolute
        ):
            failures.append(f"{field}_mismatch")
    observed_bits = {}
    for record in detail.get("bitsums", []):
        bit = int(record["bit"])
        vector = dqmask if bit < 32 else injmask
        local_bit = bit if bit < 32 else bit - 32
        bit_sum = int(np.count_nonzero(vector & (1 << local_bit)))
        observed_bits[str(bit)] = bit_sum
        if bit_sum != int(record["sum"]):
            failures.append(f"bit_{bit}_sum_mismatch")
    return {
        "passed": not failures,
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
        "strain_samples": count + nonfinite,
        "observed": observed,
        "expected": {key: detail[key] for key in observed},
        "observed_bitsums": observed_bits,
        "failures": failures,
    }


def run_gwosc_verification(
    event: str,
    files: dict[str, str | Path],
    output_path: str | Path,
    chunk_samples: int = 1_048_576,
) -> dict[str, Any]:
    records = {record["detector"]: record for record in event_strain_files(event, files)}
    missing = sorted(set(files) - set(records))
    if missing:
        raise ValueError(f"GWOSC metadata lacks detectors: {missing}")
    detector_reports = {}
    for ifo, path in sorted(files.items()):
        detail = _api_json(records[ifo]["detail_url"])
        detector_reports[ifo] = verify_hdf5_against_detail(path, detail, chunk_samples)
    report = {
        "status": (
            "verified" if all(row["passed"] for row in detector_reports.values()) else "failed"
        ),
        "passed": all(row["passed"] for row in detector_reports.values()),
        "event": event,
        "detectors": detector_reports,
        "chunk_samples": chunk_samples,
    }
    atomic_write_json(output_path, report)
    if not report["passed"]:
        raise RuntimeError(f"GWOSC full-file verification failed; inspect {output_path}")
    return report


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value else None


def _download_range(
    url: str,
    path: Path,
    start: int,
    stop: int,
    chunk_size: int,
    max_attempts: int = 50,
) -> None:
    expected = stop - start + 1
    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        present = path.stat().st_size if path.exists() else 0
        if present == expected:
            return
        if present > expected:
            raise IOError(f"Range cache {path} is larger than expected")
        request_start = start + present
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={request_start}-{stop}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise IOError(
                        f"Server ignored range request {request_start}-{stop}: HTTP {response.status}"
                    )
                with path.open("ab") as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
        except (OSError, TimeoutError) as exc:
            last_error = exc
        if path.stat().st_size == expected:
            return
        time.sleep(min(0.25 * (attempt + 1), 2.0))
    raise IOError(
        f"Incomplete range {start}-{stop} after {max_attempts} attempts: "
        f"{path.stat().st_size} != {expected}; last_error={last_error}"
    )


def download_resumable(
    url: str,
    destination: str | Path,
    chunk_size: int = 1024 * 1024,
    workers: int = 4,
) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = _remote_size(url)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": file_sha256(target),
            "downloaded": False,
        }

    if expected_size is None:
        raise IOError(f"Parallel resumable download requires Content-Length: {url}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    prefix_size = target.stat().st_size if target.exists() else 0
    if prefix_size > expected_size:
        raise IOError(f"Existing file is larger than remote object: {prefix_size} > {expected_size}")
    remaining = expected_size - prefix_size
    ranges = []
    if remaining:
        worker_count = min(workers, remaining)
        base = remaining // worker_count
        extra = remaining % worker_count
        cursor = prefix_size
        for index in range(worker_count):
            length = base + (1 if index < extra else 0)
            start = cursor
            stop = start + length - 1
            part = target.with_name(f".{target.name}.range-{start}-{stop}.part")
            ranges.append((part, start, stop))
            cursor = stop + 1
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_download_range, url, part, start, stop, chunk_size)
                for part, start, stop in ranges
            ]
            for future in futures:
                future.result()

    descriptor, assembled_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".assembled", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as assembled:
            if prefix_size:
                with target.open("rb") as prefix:
                    remaining_prefix = prefix_size
                    while remaining_prefix:
                        chunk = prefix.read(min(chunk_size, remaining_prefix))
                        if not chunk:
                            raise IOError(
                                f"Existing prefix ended before {prefix_size} bytes"
                            )
                        assembled.write(chunk)
                        remaining_prefix -= len(chunk)
            for part, _, _ in ranges:
                with part.open("rb") as source:
                    while chunk := source.read(chunk_size):
                        assembled.write(chunk)
        actual_size = Path(assembled_name).stat().st_size
        if actual_size != expected_size:
            raise IOError(f"Incomplete download for {url}: {actual_size} != {expected_size}")
        os.replace(assembled_name, target)
        for part, _, _ in ranges:
            part.unlink()
    except BaseException:
        try:
            os.unlink(assembled_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": file_sha256(target),
        "downloaded": True,
    }


def _hdf_scalar(handle: Any, path: str) -> Any:
    value = handle[path][()]
    return value.item() if hasattr(value, "item") else value


def read_hdf5_segment(path: str | Path, gps_center: float, duration: float) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Reading GWOSC HDF5 requires the optional 'h5py' dependency") from exc

    with h5py.File(path, "r") as handle:
        gps_start = float(_hdf_scalar(handle, "meta/GPSstart"))
        dataset = handle["strain/Strain"]
        spacing = float(dataset.attrs["Xspacing"])
        sample_rate = int(round(1.0 / spacing))
        start = int(round((gps_center - duration / 2 - gps_start) * sample_rate))
        stop = start + int(round(duration * sample_rate))
        if start < 0 or stop > dataset.shape[0]:
            raise ValueError(f"Requested [{start}:{stop}] outside strain file with {dataset.shape[0]} samples")
        strain = np.asarray(dataset[start:stop], dtype=np.float64)
        quality: dict[str, np.ndarray] = {}
        quality_paths = {
            "DQmask": "quality/simple/DQmask",
            "Injmask": "quality/injections/Injmask",
        }
        for key, dataset_path in quality_paths.items():
            if dataset_path in handle:
                second_start = int(np.floor(gps_center - duration / 2 - gps_start))
                second_stop = int(np.ceil(gps_center + duration / 2 - gps_start))
                quality[key] = np.asarray(handle[dataset_path][second_start:second_stop])
    return {
        "strain": strain,
        "sample_rate": sample_rate,
        "gps_start": gps_center - duration / 2,
        "quality": quality,
    }


def _fft_downsample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.copy()
    if source_rate % target_rate:
        raise ValueError("source sample rate must be an integer multiple of target rate")
    ratio = source_rate // target_rate
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / source_rate)
    spectrum[frequencies >= target_rate * 0.45] = 0
    filtered = np.fft.irfft(spectrum, n=signal.size)
    return filtered[::ratio]


def _whiten(signal: np.ndarray, smoothing_bins: int = 129) -> np.ndarray:
    return _whiten_with_reference(signal, signal, smoothing_bins)


def _whiten_with_reference(
    reference_noise: np.ndarray,
    values: np.ndarray,
    smoothing_bins: int = 129,
    component: bool = False,
) -> np.ndarray:
    reference = np.asarray(reference_noise, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    if reference.shape != target.shape or reference.ndim != 1:
        raise ValueError("Whitening reference and values must have the same 1D shape")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("Whitening input contains non-finite samples")
    reference_spectrum = np.fft.rfft(reference - np.median(reference))
    raw_psd = np.abs(reference_spectrum) ** 2
    width = min(smoothing_bins, max(3, raw_psd.size // 16 * 2 + 1))
    kernel = np.ones(width, dtype=np.float64) / width
    psd = np.convolve(raw_psd, kernel, mode="same")
    floor = max(float(np.median(psd)) * 1e-6, np.finfo(np.float64).tiny)
    denominator = np.sqrt(np.maximum(psd, floor))
    whitened_reference = np.fft.irfft(
        reference_spectrum / denominator, n=reference.size
    )
    scale = float(np.std(whitened_reference))
    if (
        not np.isfinite(whitened_reference).all()
        or not math.isfinite(scale)
        or scale <= 1e-12
    ):
        raise ValueError("Whitening produced non-finite or zero-variance output")
    target_centered = target if component else target - np.median(reference)
    target_spectrum = np.fft.rfft(target_centered)
    whitened = np.fft.irfft(target_spectrum / denominator, n=target.size)
    if not np.isfinite(whitened).all():
        raise ValueError("Whitening target produced non-finite output")
    return (whitened / scale).astype(np.float32)


def run_gwosc_pilot(
    event: str,
    cache_dir: str | Path,
    output_dir: str | Path,
    detectors: Iterable[str] | None = None,
    context_duration: float = 64.0,
    output_duration: float = 8.0,
    target_sample_rate: int = 1024,
    download_workers: int = 4,
    allow_locked_evaluation_data: bool = False,
) -> dict[str, Any]:
    event_record = resolve_event(event)
    if str(event_record["run"]).lower().startswith("o4b") and not allow_locked_evaluation_data:
        raise ValueError("O4b is locked evaluation data; pass explicit unlock only for a frozen evaluation")
    wanted = list(detectors or event_record["detectors"])
    files = event_strain_files(event, wanted, sample_rate_khz=4)
    cache = Path(cache_dir)
    output = Path(output_dir)
    downloads = []
    raw_segments = []
    quality = {}
    for record in files:
        filename = Path(record["hdf5_url"]).name
        download = download_resumable(
            record["hdf5_url"], cache / filename, workers=download_workers
        )
        downloads.append({**record, **download})
        segment = read_hdf5_segment(download["path"], event_record["gps"], context_duration)
        resampled = _fft_downsample(segment["strain"], segment["sample_rate"], target_sample_rate)
        raw_segments.append(resampled)
        quality[record["detector"]] = {
            key: value.astype(int).tolist() for key, value in segment["quality"].items()
        }

    raw = np.stack(raw_segments).astype(np.float32)
    whitened_context = np.stack([_whiten(item) for item in raw])
    output_samples = int(round(output_duration * target_sample_rate))
    context_center = whitened_context.shape[1] // 2
    selection = slice(context_center - output_samples // 2, context_center + output_samples // 2)
    whitened = whitened_context[:, selection]
    raw_selected = raw[:, selection]
    q_values = (4.0, 8.0, 16.0)
    power = multiresolution_power(
        whitened,
        target_sample_rate,
        q_values,
        frequency_bins=96,
        time_bins=96,
        fmin=16.0,
        fmax=500.0,
    )

    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / f"{event}_real_o4a.npz"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{tensor_path.name}.", suffix=".npz", dir=output)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            features=_normalize_power(power),
            whitened_strain=whitened,
            raw_strain=raw_selected,
            ifos=np.asarray([record["detector"] for record in files]),
            q_values=np.asarray(q_values, dtype=np.float32),
            sample_rate=np.asarray(target_sample_rate, dtype=np.int32),
            event_gps=np.asarray(event_record["gps"], dtype=np.float64),
        )
        os.replace(temporary, tensor_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    report = {
        "event": event_record,
        "development_only": True,
        "detectors": [record["detector"] for record in files],
        "context_duration": context_duration,
        "output_duration": output_duration,
        "target_sample_rate": target_sample_rate,
        "tensor_path": str(tensor_path),
        "tensor_sha256": file_sha256(tensor_path),
        "tensor_shape": list(power.shape),
        "quality": quality,
        "downloads": downloads,
        "preprocessing": "FFT anti-alias downsample; context PSD whitening; Q-conditioned STFT",
    }
    atomic_write_json(output / f"{event}_report.json", report)
    return report
