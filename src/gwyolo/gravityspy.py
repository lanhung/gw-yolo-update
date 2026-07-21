from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shlex
import sys
import random
import re
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import (
    API_ROOT,
    USER_AGENT,
    _api_json,
    _api_results,
    _fft_downsample,
    _whiten,
    download_resumable,
    read_hdf5_segment,
    verify_hdf5_against_detail,
)
from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    load_yaml,
)


ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_EXCLUDED_LABELS = ("Chirp", "No_Glitch", "None_of_the_Above")


def _execution_provenance() -> dict[str, Any]:
    return {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def gravityspy_weak_mask(
    ifo: str,
    model_ifos: tuple[str, ...],
    q_values: tuple[float, ...],
    frequency_bins: int,
    time_bins: int,
    fmin: float,
    fmax: float,
    duration: float,
    peak_frequency: float,
    quality_factor: float,
    output_duration: float,
) -> np.ndarray:
    """Construct a conservative metadata-derived mask for weak supervision.

    Gravity Spy provides a trigger duration, peak frequency and Q value, but not a
    pixel-level annotation.  This mask must therefore never be treated as human
    ground truth.  Its deterministic geometry makes the approximation auditable.
    """
    if ifo not in model_ifos:
        raise ValueError(f"Gravity Spy IFO {ifo} is absent from model IFOs")
    if min(frequency_bins, time_bins) <= 0 or output_duration <= 0:
        raise ValueError("weak-mask dimensions and output duration must be positive")
    if not 0 <= fmin < fmax or duration <= 0 or peak_frequency <= 0 or quality_factor <= 0:
        raise ValueError("invalid Gravity Spy weak-mask metadata")
    times = np.linspace(
        -output_duration / 2,
        output_duration / 2,
        time_bins,
        endpoint=False,
        dtype=np.float64,
    )
    frequencies = np.linspace(fmin, fmax, frequency_bins, dtype=np.float64)
    half_time = min(max(duration / 2, output_duration / time_bins), output_duration / 2)
    half_frequency = max(
        (fmax - fmin) / max(frequency_bins - 1, 1),
        peak_frequency / quality_factor,
    )
    time_support = np.abs(times) <= half_time
    frequency_support = np.abs(frequencies - peak_frequency) <= half_frequency
    support = frequency_support[:, None] & time_support[None, :]
    mask = np.zeros(
        (len(model_ifos), len(q_values), frequency_bins, time_bins), dtype=np.uint8
    )
    mask[model_ifos.index(ifo), :, support] = 1
    return mask


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def match_glitch_to_strain_file(
    event_time: float,
    records: list[dict[str, Any]],
    context_duration: float,
) -> dict[str, Any] | None:
    if context_duration <= 0:
        raise ValueError("glitch context duration must be positive")
    if not records:
        return None
    ordered = sorted(records, key=lambda row: int(row["gps_start"]))
    starts = [int(row["gps_start"]) for row in ordered]
    index = bisect_right(starts, event_time) - 1
    if index < 0:
        return None
    record = ordered[index]
    margin = context_duration / 2.0
    if event_time - margin < float(record["gps_start"]):
        return None
    if event_time + margin > float(record["gps_start"]) + float(record["duration"]):
        return None
    return record


def plan_gravityspy_strain(
    manifest_path: str | Path,
    output_dir: str | Path,
    sample_rate_khz: int = 4,
    context_duration: float = 64.0,
) -> dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Gravity Spy strain plan requires a non-empty manifest")
    combinations = sorted({(str(row["observing_run"]), str(row["ifo"])) for row in rows})
    records_by_key = {}
    api_summaries = []
    for observing_run, ifo in combinations:
        endpoint = (
            f"{API_ROOT}/runs/{observing_run}/strain-files?sample-rate={sample_rate_khz}"
            f"&detector={ifo}&pagesize=500"
        )
        api_rows, api_summary = _api_results(endpoint)
        records = []
        for item in api_rows:
            url = str(item["hdf5_url"])
            match = re.search(r"-(\d+)\.hdf5$", url)
            if match is None:
                raise ValueError(f"Cannot infer strain-file duration from {url}")
            records.append(
                {
                    "detector": ifo,
                    "observing_run": observing_run,
                    "gps_start": int(item["gps_start"]),
                    "duration": int(match.group(1)),
                    "sample_rate": sample_rate_khz * 1024,
                    "hdf5_url": url,
                    "detail_url": str(item["detail_url"]),
                }
            )
        records_by_key[(observing_run, ifo)] = records
        api_summaries.append(
            {
                "observing_run": observing_run,
                "ifo": ifo,
                "endpoint": endpoint,
                "records": len(records),
                **api_summary,
            }
        )
    planned = []
    rejected = []
    for row in rows:
        key = (str(row["observing_run"]), str(row["ifo"]))
        record = match_glitch_to_strain_file(
            float(row["event_time"]), records_by_key[key], context_duration
        )
        if record is None:
            rejected.append(
                {
                    "glitch_id": row["glitch_id"],
                    "reason": "no_single_file_with_full_context",
                }
            )
            continue
        planned.append(
            {
                **row,
                "strain_source": record,
                "context_duration": context_duration,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "gravityspy_strain_plan.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in planned)
    )
    unique_files = {row["strain_source"]["hdf5_url"] for row in planned}
    report = {
        "status": "gravityspy_strain_acquisition_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "source files still require resumable download, full hash/DQ verification, numeric "
            "mask construction and split-frozen evaluation"
        ),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(target),
        "manifest_sha256": file_sha256(target),
        "input_rows": len(rows),
        "planned_rows": len(planned),
        "rejected_rows": len(rejected),
        "coverage": len(planned) / len(rows),
        "rejection_reason_counts": dict(
            sorted(Counter(row["reason"] for row in rejected).items())
        ),
        "rejection_examples": rejected[:20],
        "unique_source_files": len(unique_files),
        "context_duration": context_duration,
        "sample_rate_khz": sample_rate_khz,
        "api_queries": api_summaries,
    }
    atomic_write_json(output / "gravityspy_strain_plan_report.json", report)
    return report


def plan_gravityspy_network_strain(
    manifest_path: str | Path,
    output_dir: str | Path,
    detectors: Iterable[str] = ("H1", "L1", "V1"),
    sample_rate_khz: int = 4,
    context_duration: float = 64.0,
    minimum_detectors: int = 2,
) -> dict[str, Any]:
    """Match each glitch GPS to explicit companion-detector GWOSC strain files."""

    wanted = tuple(dict.fromkeys(str(value).upper() for value in detectors))
    if len(wanted) < 2 or minimum_detectors < 2 or minimum_detectors > len(wanted):
        raise ValueError("Network Gravity Spy planning requires a valid detector subset gate")
    if sample_rate_khz <= 0 or context_duration <= 0:
        raise ValueError("Network Gravity Spy sample rate and context duration must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Network Gravity Spy planning requires a non-empty manifest")
    if any(str(row["ifo"]) not in wanted for row in rows):
        raise ValueError("A Gravity Spy event IFO is absent from the requested detector slots")
    glitch_ids = [str(row["glitch_id"]) for row in rows]
    if len(glitch_ids) != len(set(glitch_ids)):
        raise ValueError("Network Gravity Spy source manifest contains duplicate glitch IDs")

    runs = sorted({str(row["observing_run"]) for row in rows})
    records_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    api_queries = []
    for observing_run in runs:
        for ifo in wanted:
            endpoint = (
                f"{API_ROOT}/runs/{observing_run}/strain-files?sample-rate="
                f"{sample_rate_khz}&detector={ifo}&pagesize=500"
            )
            api_rows, api_summary = _api_results(endpoint)
            records = []
            for item in api_rows:
                if str(item["detector"]) != ifo:
                    raise ValueError(f"GWOSC detector mismatch for {observing_run}/{ifo}")
                if int(item["sample_rate_kHz"]) != sample_rate_khz:
                    continue
                url = str(item["hdf5_url"])
                match = re.search(r"-(\d+)\.hdf5$", url)
                if match is None:
                    raise ValueError(f"Cannot infer strain-file duration from {url}")
                records.append(
                    {
                        "detector": ifo,
                        "observing_run": observing_run,
                        "gps_start": int(item["gps_start"]),
                        "duration": int(match.group(1)),
                        "sample_rate": sample_rate_khz * 1024,
                        "hdf5_url": url,
                        "detail_url": str(item["detail_url"]),
                    }
                )
            records_by_key[(observing_run, ifo)] = records
            api_queries.append(
                {
                    "observing_run": observing_run,
                    "ifo": ifo,
                    "endpoint": endpoint,
                    "records": len(records),
                    **api_summary,
                }
            )

    planned = []
    rejected = []
    availability_counts: Counter[str] = Counter()
    for row in rows:
        observing_run = str(row["observing_run"])
        sources = {}
        for ifo in wanted:
            match = match_glitch_to_strain_file(
                float(row["event_time"]),
                records_by_key[(observing_run, ifo)],
                context_duration,
            )
            if match is not None:
                sources[ifo] = match
        event_ifo = str(row["ifo"])
        if event_ifo not in sources:
            rejected.append(
                {
                    "glitch_id": row["glitch_id"],
                    "reason": "event_ifo_lacks_full_context",
                    "available_ifos": sorted(sources),
                }
            )
            continue
        if len(sources) < minimum_detectors:
            rejected.append(
                {
                    "glitch_id": row["glitch_id"],
                    "reason": "insufficient_companion_detectors",
                    "available_ifos": sorted(sources),
                }
            )
            continue
        if row.get("strain_source"):
            previous = row["strain_source"]
            if str(previous["hdf5_url"]) != str(sources[event_ifo]["hdf5_url"]):
                raise ValueError("Existing event-IFO strain source differs from network match")
        available_ifos = [ifo for ifo in wanted if ifo in sources]
        detector_availability = [int(ifo in sources) for ifo in wanted]
        subset = "".join(available_ifos)
        availability_counts[subset] += 1
        planned.append(
            {
                **row,
                "network_strain_sources": sources,
                "available_ifos": available_ifos,
                "detector_availability": detector_availability,
                "context_duration": context_duration,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "gravityspy_network_strain_plan.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in planned)
    )
    report = {
        "status": "gravityspy_aligned_companion_strain_acquisition_plan",
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "scientific_blocker": (
            "all companion files still require full-file hash/DQ verification and aligned "
            "numeric materialization; event-local weak masks require human audit"
        ),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "input_rows": len(rows),
        "planned_rows": len(planned),
        "rejected_rows": len(rejected),
        "coverage": len(planned) / len(rows),
        "minimum_detectors": minimum_detectors,
        "detector_slots": list(wanted),
        "detector_subset_counts": dict(sorted(availability_counts.items())),
        "unique_glitches": len({str(row["glitch_id"]) for row in planned}),
        "unique_network_gps_blocks": len(
            {str(row["network_gps_block"]) for row in planned}
        ),
        "unique_source_files": len(
            {
                str(source["hdf5_url"])
                for row in planned
                for source in row["network_strain_sources"].values()
            }
        ),
        "context_duration": context_duration,
        "sample_rate_khz": sample_rate_khz,
        "rejection_reason_counts": dict(
            sorted(Counter(row["reason"] for row in rejected).items())
        ),
        "rejection_examples": rejected[:20],
        "api_queries": api_queries,
        **_execution_provenance(),
    }
    atomic_write_json(output / "gravityspy_network_strain_plan_report.json", report)
    return report


def shard_gravityspy_network_strain_plan(
    manifest_path: str | Path,
    output_dir: str | Path,
    files_per_shard: int = 16,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Shard connected source-file components without downloading a file twice."""

    if files_per_shard < 2:
        raise ValueError("Network Gravity Spy shards require at least two source files")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Network Gravity Spy sharding requires a non-empty plan")
    glitch_ids = [str(row["glitch_id"]) for row in rows]
    if len(glitch_ids) != len(set(glitch_ids)):
        raise ValueError("Network Gravity Spy sharding found duplicate glitch IDs")

    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    source_owner: dict[str, int] = {}
    row_sources: list[set[str]] = []
    for index, row in enumerate(rows):
        sources = {
            str(source["hdf5_url"])
            for source in row["network_strain_sources"].values()
        }
        if len(sources) < 2:
            raise ValueError("A network Gravity Spy row has fewer than two source files")
        row_sources.append(sources)
        for url in sources:
            if url in source_owner:
                union(index, source_owner[url])
            else:
                source_owner[url] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    component_records = []
    for indices in components.values():
        sources = set().union(*(row_sources[index] for index in indices))
        if len(sources) > files_per_shard:
            raise ValueError(
                f"Connected network source component has {len(sources)} files, "
                f"exceeding files_per_shard={files_per_shard}"
            )
        component_records.append(
            {
                "indices": indices,
                "sources": sources,
                "tie_break": canonical_hash(
                    {"seed": seed, "source_files": sorted(sources)}, 64
                ),
            }
        )
    component_records.sort(key=lambda item: str(item["tie_break"]))
    shards: list[dict[str, Any]] = []
    for component in component_records:
        if not shards or len(shards[-1]["sources"] | component["sources"]) > files_per_shard:
            shards.append({"sources": set(), "indices": []})
        shards[-1]["sources"].update(component["sources"])
        shards[-1]["indices"].extend(component["indices"])
    assignment = {
        index: shard_index
        for shard_index, shard in enumerate(shards)
        for index in shard["indices"]
    }
    sharded = [
        {**row, "network_strain_shard": assignment[index]}
        for index, row in enumerate(rows)
    ]
    sharded.sort(
        key=lambda row: (int(row["network_strain_shard"]), str(row["glitch_id"]))
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "gravityspy_network_strain_shards.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sharded)
    )
    source_shards: dict[str, set[int]] = defaultdict(set)
    for row in sharded:
        for source in row["network_strain_sources"].values():
            source_shards[str(source["hdf5_url"])].add(
                int(row["network_strain_shard"])
            )
    if any(len(values) != 1 for values in source_shards.values()):
        raise RuntimeError("Network source file was assigned to more than one shard")
    report = {
        "status": "bounded_gravityspy_network_strain_shards",
        "scientific_claim_allowed": False,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "seed": seed,
        "files_per_shard": files_per_shard,
        "rows": len(sharded),
        "unique_glitches": len(glitch_ids),
        "unique_source_files": len(source_shards),
        "connected_components": len(component_records),
        "shards": len(shards),
        "all_source_files_assigned_once": True,
        "shard_summaries": [
            {
                "shard": index,
                "rows": len(shard["indices"]),
                "unique_files": len(shard["sources"]),
            }
            for index, shard in enumerate(shards)
        ],
        **_execution_provenance(),
    }
    atomic_write_json(output / "gravityspy_network_strain_shard_report.json", report)
    return report


def _import_verified_network_sources(
    inventory_paths: Iterable[str | Path],
    run_identity: dict[str, Any],
    source_inventory: dict[str, dict[str, Any]],
    cache: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Import byte-verified source files from an equivalent interrupted run."""

    imported: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    identity_fields = (
        "source_manifest_sha256",
        "config_hash",
        "output_duration",
        "download_workers",
        "chunk_samples",
        "shard",
    )
    for inventory_path in inventory_paths:
        path = Path(inventory_path)
        if not path.is_file():
            raise ValueError(f"Verified source inventory is absent: {path}")
        inventory = json.loads(path.read_text(encoding="utf-8"))
        source_identity = inventory.get("run_identity")
        if not isinstance(source_identity, dict) or any(
            source_identity.get(field) != run_identity.get(field)
            for field in identity_fields
        ):
            raise ValueError(f"Verified source inventory run identity differs: {path}")
        sources = inventory.get("verified_sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"Verified source inventory has no sources: {path}")
        imported_urls = []
        for url, verification in sorted(sources.items()):
            source = source_inventory.get(str(url))
            if source is None:
                raise ValueError(
                    f"Verified source inventory contains an out-of-shard URL: {url}"
                )
            if (
                not isinstance(verification, dict)
                or verification.get("passed") is not True
                or verification.get("failures") not in ([], None)
            ):
                raise ValueError(f"Imported source was not fully verified: {url}")
            filename = Path(str(url).split("?", 1)[0]).name
            expected_path = (
                cache
                / str(source["observing_run"])
                / str(source["detector"])
                / filename
            )
            recorded_path = Path(str(verification.get("path", "")))
            if recorded_path.resolve() != expected_path.resolve():
                raise ValueError(
                    f"Imported source path is outside the exact cache slot: {url}"
                )
            if str(verification.get("detail_url")) != str(source["detail_url"]):
                raise ValueError(f"Imported source detail URL differs: {url}")
            expected = verification.get("expected")
            observed = verification.get("observed")
            if not isinstance(expected, dict) or not isinstance(observed, dict):
                raise ValueError(f"Imported source lacks full HDF5 statistics: {url}")
            try:
                expected_bytes = int(expected["filesize_bytes"])
                recorded_bytes = int(verification["bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Imported source lacks a valid byte count: {url}") from exc
            if expected_bytes != recorded_bytes:
                raise ValueError(f"Imported source byte counts disagree: {url}")
            if not expected_path.is_file() or expected_path.stat().st_size != recorded_bytes:
                raise ValueError(f"Imported source cache file is absent or truncated: {url}")
            if file_sha256(expected_path) != str(verification.get("sha256")):
                raise ValueError(f"Imported source cache hash mismatch: {url}")
            if not isinstance(verification.get("observed_bitsums"), dict) or int(
                verification.get("strain_samples", 0)
            ) <= 0:
                raise ValueError(f"Imported source lacks full dataset verification: {url}")
            existing = imported.get(str(url))
            if existing is not None and canonical_hash(existing) != canonical_hash(
                verification
            ):
                raise ValueError(f"Conflicting imported source verification: {url}")
            imported[str(url)] = verification
            imported_urls.append(str(url))
        evidence.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "source_code_commit": source_identity.get("code_commit"),
                "imported_urls": imported_urls,
            }
        )
    return imported, evidence


def materialize_gravityspy_network_strain(
    manifest_path: str | Path,
    config_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    output_duration: float = 8.0,
    download_workers: int = 8,
    chunk_samples: int = 1_048_576,
    shard: int | None = None,
    verified_source_inventories: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Verify and transform aligned real H1/L1/V1 contexts around catalog glitches."""

    if output_duration <= 0 or download_workers <= 0 or chunk_samples <= 0:
        raise ValueError("Invalid Gravity Spy network materialization settings")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    if shard is not None:
        if shard < 0:
            raise ValueError("Network Gravity Spy shard must be non-negative")
        if any("network_strain_shard" not in row for row in all_rows):
            raise ValueError("Requested network shard from an unsharded manifest")
        rows = [row for row in all_rows if int(row["network_strain_shard"]) == shard]
    else:
        rows = all_rows
    if not rows:
        raise ValueError("Gravity Spy network materialization requires a non-empty plan")
    glitch_ids = [str(row["glitch_id"]) for row in rows]
    if len(glitch_ids) != len(set(glitch_ids)):
        raise ValueError("Gravity Spy network plan contains duplicate glitch IDs")
    config = load_yaml(config_path)
    section_name = next(
        (name for name in ("physical_training", "numeric_training") if name in config),
        None,
    )
    if section_name is None:
        raise ValueError("Gravity Spy network materialization needs a training configuration")
    settings = config[section_name]
    tensor = settings["tensor"]
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    target_rate = int(settings["target_sample_rate"])
    for row in rows:
        if output_duration >= float(row["context_duration"]):
            raise ValueError("Output duration must be shorter than every whitening context")
        sources = row.get("network_strain_sources")
        if not isinstance(sources, dict) or len(sources) < 2:
            raise ValueError("Network materialization requires at least two source detectors")
        if any(ifo not in model_ifos for ifo in sources):
            raise ValueError("Network source detector is absent from configured model slots")
        expected = [int(ifo in sources) for ifo in model_ifos]
        if list(row.get("detector_availability", [])) != expected:
            raise ValueError("Planned detector availability does not match network sources")
        if str(row["ifo"]) not in sources:
            raise ValueError("Event IFO is absent from network sources")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    run_identity = {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "source_manifest_sha256": file_sha256(manifest_path),
        "config_hash": canonical_hash(config),
        "output_duration": output_duration,
        "download_workers": download_workers,
        "chunk_samples": chunk_samples,
        "shard": shard,
    }
    state_path = output / "materialization_state.json"
    partial_path = output / "materialization_partial.json"
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    verified_sources: dict[str, dict[str, Any]] = {}
    imported_source_inventories: list[dict[str, Any]] = []
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing network materialization belongs to another run")
        if partial_path.is_file():
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = list(partial.get("records", []))
            rejected = list(partial.get("rejected", []))
            verified_sources = dict(partial.get("verified_sources", {}))
            imported_source_inventories = list(
                partial.get("imported_verified_source_inventories", [])
            )
    completed_ids = set()
    for record in completed:
        glitch_id = str(record["glitch_id"])
        if glitch_id in completed_ids or file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Invalid resumable network sample: {glitch_id}")
        completed_ids.add(glitch_id)
    rejected_ids = {str(row["glitch_id"]) for row in rejected}
    if len(rejected_ids) != len(rejected) or completed_ids & rejected_ids:
        raise ValueError("Invalid resumable network rejection inventory")

    source_inventory: dict[str, dict[str, Any]] = {}
    for row in rows:
        for source in row["network_strain_sources"].values():
            url = str(source["hdf5_url"])
            if url in source_inventory and source_inventory[url] != source:
                raise ValueError("A network source URL has inconsistent metadata")
            source_inventory[url] = source
    imported_sources, imported_evidence = _import_verified_network_sources(
        verified_source_inventories, run_identity, source_inventory, cache
    )
    for url, verification in imported_sources.items():
        existing = verified_sources.get(url)
        if existing is not None and canonical_hash(existing) != canonical_hash(verification):
            raise ValueError(f"Imported source conflicts with current partial state: {url}")
        verified_sources[url] = verification
    evidence_by_hash = {
        str(item["sha256"]): item for item in imported_source_inventories
    }
    evidence_by_hash.update(
        {str(item["sha256"]): item for item in imported_evidence}
    )
    imported_source_inventories = [
        evidence_by_hash[key] for key in sorted(evidence_by_hash)
    ]
    if imported_evidence:
        atomic_write_json(
            partial_path,
            {
                "run_identity": run_identity,
                "verified_sources": verified_sources,
                "imported_verified_source_inventories": imported_source_inventories,
                "records": completed,
                "rejected": rejected,
            },
        )
    for url, source in sorted(source_inventory.items()):
        filename = Path(url.split("?", 1)[0]).name
        verification = verified_sources.get(url)
        cache_path = (
            cache / str(source["observing_run"]) / str(source["detector"]) / filename
        )
        if verification is not None and Path(str(verification["path"])).resolve() != (
            cache_path.resolve()
        ):
            raise ValueError(f"Verified network source uses another cache slot: {url}")
        if verification is not None and cache_path.is_file():
            if cache_path.stat().st_size != int(verification["bytes"]):
                raise ValueError(f"Cached network source size changed: {url}")
            if file_sha256(cache_path) != str(verification["sha256"]):
                raise ValueError(f"Cached network source changed after verification: {url}")
            download = {"path": str(cache_path), "downloaded": False}
        else:
            download = download_resumable(url, cache_path, workers=download_workers)
        if verification is None:
            verification = verify_hdf5_against_detail(
                download["path"], _api_json(str(source["detail_url"])), chunk_samples
            )
            if not verification["passed"]:
                raise RuntimeError(f"Full-file network verification failed for {url}")
            verified_sources[url] = {**verification, "detail_url": source["detail_url"]}
            atomic_write_json(
                partial_path,
                {
                    "run_identity": run_identity,
                    "verified_sources": verified_sources,
                    "imported_verified_source_inventories": imported_source_inventories,
                    "records": completed,
                    "rejected": rejected,
                },
            )
        elif verification["sha256"] != file_sha256(download["path"]):
            raise ValueError(f"Cached network source changed after verification: {url}")

    output_samples = int(round(output_duration * target_rate))
    for row in rows:
        glitch_id = str(row["glitch_id"])
        if glitch_id in completed_ids or glitch_id in rejected_ids:
            continue
        raw = np.zeros((len(model_ifos), output_samples), dtype=np.float64)
        whitened = np.zeros_like(raw)
        data_quality: dict[str, Any] = {}
        rejection_reason = None
        for ifo, source in row["network_strain_sources"].items():
            verification = verified_sources[str(source["hdf5_url"])]
            segment = read_hdf5_segment(
                verification["path"],
                float(row["event_time"]),
                float(row["context_duration"]),
            )
            context = np.asarray(segment["strain"], dtype=np.float64)
            dqmask = np.asarray(segment["quality"].get("DQmask", []), dtype=np.int64)
            if not np.isfinite(context).all():
                rejection_reason = f"nonfinite_strain_context_{ifo}"
                break
            if dqmask.size == 0 or not np.all(dqmask & 1):
                rejection_reason = f"data_quality_bit_missing_{ifo}"
                break
            context = _fft_downsample(context, int(segment["sample_rate"]), target_rate)
            whitened_context = _whiten(context)
            center = context.size // 2
            start = center - output_samples // 2
            stop = start + output_samples
            if start < 0 or stop > context.size:
                rejection_reason = f"short_context_{ifo}"
                break
            index = model_ifos.index(str(ifo))
            raw[index] = context[start:stop]
            whitened[index] = whitened_context[start:stop]
            data_quality[str(ifo)] = {
                "seconds": int(dqmask.size),
                "dqmask_min": int(dqmask.min()),
                "dqmask_max": int(dqmask.max()),
                "injmask_values": sorted(
                    int(value)
                    for value in np.unique(segment["quality"].get("Injmask", []))
                ),
            }
        if rejection_reason is not None:
            rejected.append({"glitch_id": glitch_id, "reason": rejection_reason})
            rejected_ids.add(glitch_id)
            atomic_write_json(
                partial_path,
                {
                    "run_identity": run_identity,
                    "verified_sources": verified_sources,
                    "imported_verified_source_inventories": imported_source_inventories,
                    "records": completed,
                    "rejected": rejected,
                },
            )
            continue
        power = multiresolution_power(
            whitened,
            target_rate,
            q_values,
            int(tensor["frequency_bins"]),
            int(tensor["time_bins"]),
            float(tensor["fmin"]),
            float(tensor["fmax"]),
        )
        features = _normalize_power(power)
        availability = np.asarray(row["detector_availability"], dtype=np.uint8)
        features[availability == 0] = 0
        glitch_mask = gravityspy_weak_mask(
            str(row["ifo"]),
            model_ifos,
            q_values,
            int(tensor["frequency_bins"]),
            int(tensor["time_bins"]),
            float(tensor["fmin"]),
            float(tensor["fmax"]),
            float(row["duration"]),
            float(row["peak_frequency"]),
            float(row["q_value"]),
            output_duration,
        )
        sample_path = output / "samples" / f"network-{canonical_hash(glitch_id, 24)}.npz"
        _atomic_savez(
            sample_path,
            {
                "features": features.astype(np.float16),
                "chirp_mask": np.zeros_like(glitch_mask, dtype=np.uint8),
                "glitch_mask": glitch_mask.astype(np.uint8),
                "raw_strain": raw.astype(np.float32),
                "whitened_strain": whitened.astype(np.float32),
                "detector_availability": availability,
                "ifos": np.asarray(model_ifos),
                "q_values": np.asarray(q_values, dtype=np.float32),
                "sample_rate": np.asarray(target_rate, dtype=np.int32),
                "event_gps": np.asarray(row["event_time"], dtype=np.float64),
            },
        )
        record = {
            **row,
            "single_ifo_numeric_path": row.get("path"),
            "single_ifo_numeric_sha256": row.get("sha256"),
            "path": str(sample_path),
            "sha256": file_sha256(sample_path),
            "mask_provenance": "weak_gravityspy_duration_peak_frequency_q_geometry_v1",
            "human_pixel_mask": False,
            "data_quality": data_quality,
            "aligned_network_context": True,
        }
        completed.append(record)
        completed_ids.add(glitch_id)
        atomic_write_json(
            partial_path,
            {
                "run_identity": run_identity,
                "verified_sources": verified_sources,
                "imported_verified_source_inventories": imported_source_inventories,
                "records": completed,
                "rejected": rejected,
            },
        )
        atomic_write_json(
            state_path,
            {
                "status": "in_progress",
                "run_identity": run_identity,
                "completed_rows": len(completed),
                "rejected_rows": len(rejected),
                "requested_rows": len(rows),
                "verified_files": len(verified_sources),
            },
        )
    if len(completed_ids) + len(rejected_ids) != len(rows):
        raise RuntimeError("Network Gravity Spy rows were not fully accounted")
    completed.sort(key=lambda row: str(row["glitch_id"]))
    manifest = output / "gravityspy_network_numeric_manifest.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in completed)
    )
    report = {
        "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "scientific_blocker": (
            "aligned strain is verified but metadata-derived glitch masks require human audit; "
            "coherence gains still require frozen continuous-background evaluation"
        ),
        "run_identity": run_identity,
        **_execution_provenance(),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "rows": len(completed),
        "shard": shard,
        "requested_rows": len(rows),
        "rejected_rows": len(rejected),
        "rejection_reason_counts": dict(
            sorted(Counter(row["reason"] for row in rejected).items())
        ),
        "unique_glitches": len(completed_ids),
        "unique_network_gps_blocks": len(
            {str(row["network_gps_block"]) for row in completed}
        ),
        "verified_files": len(verified_sources),
        "imported_verified_source_inventories": imported_source_inventories,
        "detector_subset_counts": dict(
            sorted(
                Counter("".join(row["available_ifos"]) for row in completed).items()
            )
        ),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "tensor_shape": [
            len(model_ifos),
            len(q_values),
            int(tensor["frequency_bins"]),
            int(tensor["time_bins"]),
        ],
        "mask_provenance": "weak_gravityspy_duration_peak_frequency_q_geometry_v1",
        "human_pixel_masks": 0,
        "source_cache_evicted": False,
    }
    report_path = output / "gravityspy_network_numeric_report.json"
    atomic_write_json(report_path, report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed_rows": len(completed),
            "rejected_rows": len(rejected),
            "requested_rows": len(rows),
            "verified_files": len(verified_sources),
            "report_sha256": file_sha256(report_path),
        },
    )
    return report


def select_gravityspy_source_files(
    manifest_path: str | Path,
    output_dir: str | Path,
    per_label: int,
    maximum_files: int,
    seed: int = 20260720,
    existing_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select whole source files greedily against label deficits in one frozen split."""
    if per_label <= 0 or maximum_files <= 0:
        raise ValueError("Gravity Spy label target and maximum files must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Gravity Spy source selection requires a non-empty strain plan")
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1:
        raise ValueError(f"Gravity Spy source selection cannot mix splits: {sorted(splits)}")
    split = next(iter(splits))
    glitch_ids = [str(row["glitch_id"]) for row in rows]
    if len(glitch_ids) != len(set(glitch_ids)):
        raise ValueError("Gravity Spy strain plan contains duplicate glitch IDs")

    existing_rows: list[dict[str, Any]] = []
    existing_hash = None
    if existing_manifest_path is not None:
        existing_path = Path(existing_manifest_path)
        with existing_path.open("r", encoding="utf-8") as handle:
            existing_rows = [json.loads(line) for line in handle if line.strip()]
        if any(str(row.get("split")) != split for row in existing_rows):
            raise ValueError("Existing Gravity Spy numeric data belong to another split")
        existing_hash = file_sha256(existing_path)
    existing_counts = Counter(str(row["ml_label"]) for row in existing_rows)
    existing_sources = {
        str(row["strain_source"]["hdf5_url"])
        for row in existing_rows
        if row.get("strain_source", {}).get("hdf5_url")
    }
    labels = sorted({str(row["ml_label"]) for row in rows})
    deficits = {label: max(0, per_label - existing_counts[label]) for label in labels}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        url = str(row["strain_source"]["hdf5_url"])
        if url not in existing_sources:
            by_source[url].append(row)

    source_statistics = {
        url: {
            "counts": Counter(str(row["ml_label"]) for row in source_rows),
            "rows": len(source_rows),
            "tie_break": canonical_hash({"seed": seed, "url": url}, 64),
        }
        for url, source_rows in by_source.items()
    }

    selected_sources: list[str] = []
    remaining = dict(by_source)
    while any(deficits.values()) and len(selected_sources) < maximum_files and remaining:
        scored = []
        for url in remaining:
            statistics = source_statistics[url]
            counts = statistics["counts"]
            covered = sum(min(counts[label], deficits[label]) for label in labels)
            distinct = sum(counts[label] > 0 and deficits[label] > 0 for label in labels)
            scored.append(
                (
                    covered,
                    distinct,
                    -int(statistics["rows"]),
                    str(statistics["tie_break"]),
                    url,
                    counts,
                )
            )
        covered, _, _, _, url, counts = max(scored)
        if covered <= 0:
            break
        selected_sources.append(url)
        del remaining[url]
        for label in labels:
            deficits[label] = max(0, deficits[label] - counts[label])

    selected_rows = [row for url in selected_sources for row in by_source[url]]
    selected_rows.sort(
        key=lambda row: (
            str(row["strain_source"]["hdf5_url"]),
            str(row["glitch_id"]),
        )
    )
    selected_counts = Counter(str(row["ml_label"]) for row in selected_rows)
    combined_counts = {
        label: existing_counts[label] + selected_counts[label] for label in labels
    }
    underfilled = {
        label: per_label - combined_counts[label]
        for label in labels
        if combined_counts[label] < per_label
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"gravityspy_{split}_selected_sources.jsonl"
    atomic_write_text(
        target,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_rows),
    )
    report = {
        "status": "bounded_label_deficit_gravityspy_source_selection",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "selected official strain still requires verified materialization and weak masks "
            "require a frozen human pixel-mask audit"
        ),
        "split": split,
        **_execution_provenance(),
        "config_hash": None,
        "model_hash": None,
        "seed": seed,
        "per_label_target": per_label,
        "maximum_files": maximum_files,
        "target_met": not underfilled,
        "underfilled_label_deficits": dict(sorted(underfilled.items())),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "existing_manifest_path": (
            str(existing_manifest_path) if existing_manifest_path is not None else None
        ),
        "existing_manifest_sha256": existing_hash,
        "existing_rows": len(existing_rows),
        "existing_label_counts": dict(sorted(existing_counts.items())),
        "excluded_existing_source_files": len(existing_sources),
        "selected_rows": len(selected_rows),
        "selected_source_files": len(selected_sources),
        "selected_unique_glitches": len({row["glitch_id"] for row in selected_rows}),
        "selected_unique_network_gps_blocks": len(
            {row["network_gps_block"] for row in selected_rows}
        ),
        "selected_label_counts": dict(sorted(selected_counts.items())),
        "combined_label_counts": dict(sorted(combined_counts.items())),
        "selected_runs": dict(
            sorted(Counter(str(row["observing_run"]) for row in selected_rows).items())
        ),
        "selected_ifos": dict(
            sorted(Counter(str(row["ifo"]) for row in selected_rows).items())
        ),
        "manifest_path": str(target),
        "manifest_sha256": file_sha256(target),
        "selected_sources_hash": canonical_hash(selected_sources, 64),
    }
    atomic_write_json(output / "gravityspy_source_selection_report.json", report)
    return report


def shard_gravityspy_strain_plan(
    manifest_path: str | Path,
    output_dir: str | Path,
    files_per_shard: int = 32,
    seed: int = 20260720,
) -> dict[str, Any]:
    if files_per_shard <= 0:
        raise ValueError("files per Gravity Spy shard must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Gravity Spy strain sharding requires a non-empty plan")
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["strain_source"]["hdf5_url"])].append(row)
    ordered_files = sorted(
        by_file,
        key=lambda url: canonical_hash({"seed": seed, "hdf5_url": url}, 64),
    )
    file_shards = {
        url: index // files_per_shard for index, url in enumerate(ordered_files)
    }
    sharded = []
    for row in rows:
        url = str(row["strain_source"]["hdf5_url"])
        sharded.append({**row, "strain_shard": file_shards[url]})
    sharded.sort(
        key=lambda row: (
            int(row["strain_shard"]),
            str(row["strain_source"]["hdf5_url"]),
            float(row["event_time"]),
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "gravityspy_strain_shards.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sharded)
    )
    rows_by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sharded:
        rows_by_shard[int(row["strain_shard"])].append(row)
    shard_summaries = []
    for shard, selected in sorted(rows_by_shard.items()):
        shard_summaries.append(
            {
                "shard": shard,
                "rows": len(selected),
                "unique_files": len(
                    {row["strain_source"]["hdf5_url"] for row in selected}
                ),
                "unique_glitches": len({row["glitch_id"] for row in selected}),
                "unique_network_gps_blocks": len(
                    {row["network_gps_block"] for row in selected}
                ),
                "labels": dict(
                    sorted(Counter(row["ml_label"] for row in selected).items())
                ),
                "runs": dict(
                    sorted(Counter(row["observing_run"] for row in selected).items())
                ),
                "ifos": dict(sorted(Counter(row["ifo"] for row in selected).items())),
            }
        )
    report = {
        "status": "bounded_gravityspy_strain_shards",
        "scientific_claim_allowed": False,
        **_execution_provenance(),
        "config_hash": None,
        "model_hash": None,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(target),
        "manifest_sha256": file_sha256(target),
        "seed": seed,
        "files_per_shard": files_per_shard,
        "rows": len(sharded),
        "unique_files": len(by_file),
        "shards": len(shard_summaries),
        "all_rows_preserved": len(sharded) == len(rows),
        "all_files_assigned_once": len(file_shards) == len(by_file),
        "shard_summaries": shard_summaries,
    }
    atomic_write_json(output / "gravityspy_strain_shard_report.json", report)
    return report


def materialize_gravityspy_strain_shard(
    manifest_path: str | Path,
    shard: int,
    config_path: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    output_duration: float = 8.0,
    download_workers: int = 8,
    chunk_samples: int = 1_048_576,
) -> dict[str, Any]:
    """Download, verify and materialize one bounded Gravity Spy strain shard."""
    if shard < 0 or output_duration <= 0 or download_workers <= 0 or chunk_samples <= 0:
        raise ValueError("invalid Gravity Spy shard materialization settings")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in all_rows if int(row["strain_shard"]) == shard]
    if not rows:
        raise ValueError(f"Gravity Spy strain shard {shard} is empty or absent")
    config = load_yaml(config_path)
    section_name = next(
        (
            name
            for name in ("physical_training", "numeric_training")
            if name in config
        ),
        None,
    )
    if section_name is None:
        raise ValueError("Gravity Spy materialization needs a training configuration")
    settings = config[section_name]
    tensor = settings["tensor"]
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    target_sample_rate = int(settings["target_sample_rate"])
    if output_duration >= float(rows[0]["context_duration"]):
        raise ValueError("output duration must be shorter than the whitening context")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    run_identity = {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "source_manifest_sha256": file_sha256(manifest_path),
        "config_hash": canonical_hash(config),
        "shard": shard,
        "output_duration": output_duration,
        "download_workers": download_workers,
        "chunk_samples": chunk_samples,
    }
    state_path = output / "materialization_state.json"
    partial_path = output / "materialization_partial.json"
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    verified_sources: dict[str, dict[str, Any]] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing Gravity Spy shard state belongs to a different run")
        if partial_path.is_file():
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = list(partial.get("records", []))
            rejected = list(partial.get("rejected", []))
            verified_sources = dict(partial.get("verified_sources", {}))
    completed_ids: set[str] = set()
    for record in completed:
        glitch_id = str(record["glitch_id"])
        if glitch_id in completed_ids or file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Invalid resumable Gravity Spy sample {glitch_id}")
        completed_ids.add(glitch_id)
    rejected_ids = {str(row["glitch_id"]) for row in rejected}
    if len(rejected_ids) != len(rejected) or completed_ids & rejected_ids:
        raise ValueError("Invalid resumable Gravity Spy rejection inventory")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["strain_source"]["hdf5_url"])].append(row)
    for url, source_rows in sorted(by_source.items()):
        source = source_rows[0]["strain_source"]
        filename = Path(url.split("?", 1)[0]).name
        download = download_resumable(
            url,
            cache / str(source_rows[0]["observing_run"]) / filename,
            workers=download_workers,
        )
        source_verification = verified_sources.get(url)
        if source_verification is None:
            source_verification = verify_hdf5_against_detail(
                download["path"], _api_json(str(source["detail_url"])), chunk_samples
            )
            if not source_verification["passed"]:
                raise RuntimeError(f"Full-file verification failed for {url}")
            verified_sources[url] = {
                **source_verification,
                "detail_url": source["detail_url"],
            }
        elif source_verification["sha256"] != file_sha256(download["path"]):
            raise ValueError(f"Cached source changed after verification: {url}")
        for row in source_rows:
            glitch_id = str(row["glitch_id"])
            if glitch_id in completed_ids or glitch_id in rejected_ids:
                continue
            segment = read_hdf5_segment(
                download["path"], float(row["event_time"]), float(row["context_duration"])
            )
            if not np.isfinite(segment["strain"]).all():
                rejected.append(
                    {"glitch_id": glitch_id, "reason": "nonfinite_strain_context"}
                )
                rejected_ids.add(glitch_id)
                atomic_write_json(
                    partial_path,
                    {
                        "run_identity": run_identity,
                        "verified_sources": verified_sources,
                        "records": completed,
                        "rejected": rejected,
                    },
                )
                continue
            data_quality = np.asarray(segment["quality"].get("DQmask", []), dtype=np.int64)
            if data_quality.size == 0 or not np.all(data_quality & 1):
                rejected.append(
                    {"glitch_id": glitch_id, "reason": "data_quality_bit_missing"}
                )
                rejected_ids.add(glitch_id)
                atomic_write_json(
                    partial_path,
                    {
                        "run_identity": run_identity,
                        "verified_sources": verified_sources,
                        "records": completed,
                        "rejected": rejected,
                    },
                )
                continue
            context = _fft_downsample(
                segment["strain"], int(segment["sample_rate"]), target_sample_rate
            )
            whitened_context = _whiten(context)
            output_samples = int(round(output_duration * target_sample_rate))
            center = whitened_context.size // 2
            start = center - output_samples // 2
            whitened = whitened_context[start : start + output_samples]
            raw = context[start : start + output_samples]
            single_power = multiresolution_power(
                whitened[None, :],
                target_sample_rate,
                q_values,
                int(tensor["frequency_bins"]),
                int(tensor["time_bins"]),
                float(tensor["fmin"]),
                float(tensor["fmax"]),
            )
            features = np.zeros(
                (len(model_ifos), *single_power.shape[1:]), dtype=np.float32
            )
            ifo_index = model_ifos.index(str(row["ifo"]))
            features[ifo_index] = _normalize_power(single_power)[0]
            glitch_mask = gravityspy_weak_mask(
                str(row["ifo"]),
                model_ifos,
                q_values,
                int(tensor["frequency_bins"]),
                int(tensor["time_bins"]),
                float(tensor["fmin"]),
                float(tensor["fmax"]),
                float(row["duration"]),
                float(row["peak_frequency"]),
                float(row["q_value"]),
                output_duration,
            )
            sample_path = output / "samples" / f"{canonical_hash(glitch_id, 24)}.npz"
            _atomic_savez(
                sample_path,
                {
                    "features": features.astype(np.float16),
                    "chirp_mask": np.zeros_like(glitch_mask),
                    "glitch_mask": glitch_mask,
                    "raw_strain": raw.astype(np.float32),
                    "whitened_strain": whitened.astype(np.float32),
                    "ifos": np.asarray(model_ifos),
                    "q_values": np.asarray(q_values, dtype=np.float32),
                    "sample_rate": np.asarray(target_sample_rate, dtype=np.int32),
                    "event_gps": np.asarray(row["event_time"], dtype=np.float64),
                },
            )
            record = {
                **row,
                "path": str(sample_path),
                "sha256": file_sha256(sample_path),
                "mask_provenance": "weak_gravityspy_duration_peak_frequency_q_geometry_v1",
                "human_pixel_mask": False,
                "data_quality": {
                    "seconds": int(data_quality.size),
                    "dqmask_min": int(data_quality.min()),
                    "dqmask_max": int(data_quality.max()),
                    "injmask_values": sorted(
                        int(value)
                        for value in np.unique(segment["quality"].get("Injmask", []))
                    ),
                },
            }
            completed.append(record)
            completed_ids.add(glitch_id)
            atomic_write_json(
                partial_path,
                {
                    "run_identity": run_identity,
                    "verified_sources": verified_sources,
                    "records": completed,
                    "rejected": rejected,
                },
            )
            atomic_write_json(
                state_path,
                {
                    "status": "in_progress",
                    "run_identity": run_identity,
                    "completed_rows": len(completed),
                    "rejected_rows": len(rejected),
                    "requested_rows": len(rows),
                    "verified_files": len(verified_sources),
                    "requested_files": len(by_source),
                },
            )
    if len(completed_ids) + len(rejected_ids) != len(rows):
        raise RuntimeError("Gravity Spy shard rows were not fully accounted")
    completed.sort(key=lambda row: str(row["glitch_id"]))
    manifest = output / "gravityspy_numeric_manifest.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in completed)
    )
    report = {
        "status": "verified_gravityspy_numeric_weak_masks",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "metadata-derived masks are weak supervision and require pixel-mask audit; source "
            "files remain cached until verified retention or controlled eviction is implemented"
        ),
        "run_identity": run_identity,
        **_execution_provenance(),
        "config_hash": run_identity["config_hash"],
        "model_hash": None,
        "seed": None,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "rows": len(completed),
        "requested_rows": len(rows),
        "rejected_rows": len(rejected),
        "rejection_reason_counts": dict(
            sorted(Counter(row["reason"] for row in rejected).items())
        ),
        "rejection_examples": rejected[:20],
        "unique_glitches": len(completed_ids),
        "verified_files": len(verified_sources),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "tensor_shape": [
            len(model_ifos),
            len(q_values),
            int(tensor["frequency_bins"]),
            int(tensor["time_bins"]),
        ],
        "mask_provenance": "weak_gravityspy_duration_peak_frequency_q_geometry_v1",
        "human_pixel_masks": 0,
        "source_cache_evicted": False,
    }
    report_path = output / "gravityspy_numeric_report.json"
    atomic_write_json(report_path, report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed_rows": len(completed),
            "rejected_rows": len(rejected),
            "requested_rows": len(rows),
            "verified_files": len(verified_sources),
            "requested_files": len(by_source),
            "report_sha256": file_sha256(report_path),
        },
    )
    return report


def merge_gravityspy_numeric_manifests(
    report_paths: Iterable[str | Path],
    output_dir: str | Path,
    expected_split: str,
) -> dict[str, Any]:
    """Hash-verify and merge completed numeric shards from one frozen split."""
    if expected_split not in {"train", "val", "test"}:
        raise ValueError("expected Gravity Spy split must be train, val or test")
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise ValueError("at least one Gravity Spy numeric report is required")
    rows = []
    source_reports = []
    seen_glitches: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "verified_gravityspy_numeric_weak_masks":
            raise ValueError(f"Gravity Spy numeric report is incomplete: {path}")
        manifest = Path(report["manifest_path"])
        if file_sha256(manifest) != report["manifest_sha256"]:
            raise ValueError(f"Gravity Spy numeric manifest hash mismatch: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        if len(source_rows) != int(report["rows"]):
            raise ValueError(f"Gravity Spy numeric row count mismatch: {manifest}")
        for row in source_rows:
            glitch_id = str(row["glitch_id"])
            if glitch_id in seen_glitches:
                raise ValueError(f"Duplicate Gravity Spy glitch across shards: {glitch_id}")
            if row.get("split") != expected_split:
                raise ValueError(
                    f"Gravity Spy shard mixes split {row.get('split')} into {expected_split}"
                )
            if file_sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Gravity Spy numeric sample hash mismatch: {row['path']}")
            seen_glitches.add(glitch_id)
            rows.append(row)
        source_reports.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "manifest_sha256": report["manifest_sha256"],
                "rows": len(source_rows),
            }
        )
    rows.sort(key=lambda row: str(row["glitch_id"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"gravityspy_numeric_{expected_split}.jsonl"
    atomic_write_text(
        manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        "status": "verified_merged_gravityspy_numeric_split",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "weak masks require a frozen pixel-mask audit before segmentation claims"
        ),
        "split": expected_split,
        **_execution_provenance(),
        "config_hash": None,
        "model_hash": None,
        "seed": None,
        "source_reports": source_reports,
        "source_reports_hash": canonical_hash(source_reports, 64),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "rows": len(rows),
        "unique_glitch_ids": len(seen_glitches),
        "unique_network_gps_blocks": len({row["network_gps_block"] for row in rows}),
        "labels": dict(sorted(Counter(str(row["ml_label"]) for row in rows).items())),
        "runs": dict(sorted(Counter(str(row["observing_run"]) for row in rows).items())),
        "ifos": dict(sorted(Counter(str(row["ifo"]) for row in rows).items())),
        "human_pixel_masks": sum(bool(row.get("human_pixel_mask")) for row in rows),
        "weak_masks": sum(not bool(row.get("human_pixel_mask")) for row in rows),
    }
    atomic_write_json(output / "gravityspy_numeric_merge_report.json", result)
    return result


def merge_gravityspy_network_numeric_manifests(
    report_paths: Iterable[str | Path],
    output_dir: str | Path,
    expected_split: str,
) -> dict[str, Any]:
    """Hash-verify aligned-network shards without mixing single-IFO artifacts."""

    if expected_split not in {"train", "val", "test"}:
        raise ValueError("Expected network Gravity Spy split must be train, val or test")
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise ValueError("At least one network Gravity Spy report is required")
    rows: list[dict[str, Any]] = []
    seen_glitches: set[str] = set()
    source_reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "verified_gravityspy_aligned_network_numeric_weak_masks":
            raise ValueError(f"Network Gravity Spy report is incomplete: {path}")
        manifest = Path(report["manifest_path"])
        if file_sha256(manifest) != str(report["manifest_sha256"]):
            raise ValueError(f"Network Gravity Spy manifest hash mismatch: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        if len(source_rows) != int(report["rows"]):
            raise ValueError(f"Network Gravity Spy row count mismatch: {manifest}")
        for row in source_rows:
            glitch_id = str(row["glitch_id"])
            if glitch_id in seen_glitches:
                raise ValueError(f"Duplicate network Gravity Spy glitch: {glitch_id}")
            if row.get("split") != expected_split:
                raise ValueError("Network Gravity Spy shard mixes frozen splits")
            if not row.get("aligned_network_context"):
                raise ValueError("Network Gravity Spy row lacks aligned-context certification")
            if len(row.get("available_ifos", [])) < 2:
                raise ValueError("Network Gravity Spy row lacks a companion detector")
            if file_sha256(row["path"]) != str(row["sha256"]):
                raise ValueError(f"Network Gravity Spy sample hash mismatch: {row['path']}")
            seen_glitches.add(glitch_id)
            rows.append(row)
        source_reports.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "manifest_sha256": report["manifest_sha256"],
                "rows": len(source_rows),
                "shard": report.get("shard"),
            }
        )
    rows.sort(key=lambda row: str(row["glitch_id"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"gravityspy_network_numeric_{expected_split}.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        "status": "verified_merged_gravityspy_aligned_network_numeric_split",
        "scientific_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "scientific_blocker": (
            "weak masks require human audit and aligned contexts require frozen continuous "
            "background/coherence evaluation"
        ),
        "split": expected_split,
        "source_reports": source_reports,
        "source_reports_hash": canonical_hash(source_reports, 64),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "rows": len(rows),
        "unique_glitch_ids": len(seen_glitches),
        "unique_network_gps_blocks": len(
            {str(row["network_gps_block"]) for row in rows}
        ),
        "detector_subset_counts": dict(
            sorted(Counter("".join(row["available_ifos"]) for row in rows).items())
        ),
        "human_pixel_masks": sum(bool(row.get("human_pixel_mask")) for row in rows),
        "weak_masks": sum(not bool(row.get("human_pixel_mask")) for row in rows),
        **_execution_provenance(),
    }
    atomic_write_json(output / "gravityspy_network_numeric_merge_report.json", result)
    return result


def evict_gravityspy_verified_sources(
    materialization_report_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evict only fully verified, reproducible source files after sample validation."""
    report_path = Path(materialization_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    accepted_statuses = {
        "verified_gravityspy_numeric_weak_masks",
        "verified_gravityspy_aligned_network_numeric_weak_masks",
    }
    if report.get("status") not in accepted_statuses:
        raise ValueError("Gravity Spy materialization is not complete")
    manifest = Path(report["manifest_path"])
    if file_sha256(manifest) != report["manifest_sha256"]:
        raise ValueError("Gravity Spy materialized manifest hash mismatch before eviction")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != int(report["rows"]):
        raise ValueError("Gravity Spy materialized row count mismatch before eviction")
    for row in rows:
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Gravity Spy numeric sample changed before eviction: {row['path']}")
    partial_path = report_path.with_name("materialization_partial.json")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    if partial.get("run_identity") != report.get("run_identity"):
        raise ValueError("Gravity Spy partial state identity differs before eviction")
    verified_sources = dict(partial.get("verified_sources", {}))
    if not verified_sources or len(verified_sources) != int(report["verified_files"]):
        raise ValueError("Gravity Spy verified source inventory is incomplete")
    cache_root = Path(cache_dir).resolve()
    output = Path(output_path)
    identity = {
        "materialization_report_sha256": file_sha256(report_path),
        "manifest_sha256": report["manifest_sha256"],
        "cache_root": str(cache_root),
    }
    if output.is_file():
        state = json.loads(output.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("Existing Gravity Spy eviction state belongs to another run")
    else:
        state = {
            "status": "in_progress",
            "identity": identity,
            "recoverable_from": "official GWOSC hdf5_url recorded per source",
            "numeric_outputs_verified": len(rows),
            "sources": [],
        }
    completed = {str(item["hdf5_url"]): item for item in state["sources"]}
    for url, verification in sorted(verified_sources.items()):
        if url in completed and completed[url].get("evicted"):
            if Path(completed[url]["path"]).exists():
                raise ValueError("Previously evicted Gravity Spy source unexpectedly reappeared")
            continue
        source = Path(verification["path"]).resolve()
        if not source.is_relative_to(cache_root):
            raise ValueError(f"Refusing to evict source outside declared cache: {source}")
        if not source.is_file() or file_sha256(source) != verification["sha256"]:
            raise ValueError(f"Gravity Spy source changed or disappeared before eviction: {source}")
        entry = {
            "hdf5_url": url,
            "path": str(source),
            "sha256": verification["sha256"],
            "bytes": source.stat().st_size,
            "evicted": False,
        }
        state["sources"].append(entry)
        atomic_write_json(output, state)
        source.unlink()
        entry["evicted"] = True
        atomic_write_json(output, state)
    state["status"] = "complete"
    state["evicted_files"] = len(state["sources"])
    state["evicted_bytes"] = sum(int(item["bytes"]) for item in state["sources"])
    atomic_write_json(output, state)
    return state


def _file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the publisher-provided checksum
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def zenodo_files(record_id: int) -> dict[str, dict[str, Any]]:
    payload = _api_json(f"{ZENODO_API}/{record_id}")
    return {
        str(item["key"]): {
            "key": str(item["key"]),
            "bytes": int(item["size"]),
            "checksum": str(item["checksum"]),
            "url": str(item["links"]["self"]),
        }
        for item in payload.get("files", [])
    }


def _infer_run(filename: str) -> str:
    for run in ("O3a", "O3b", "O2", "O1"):
        if run in filename:
            return run
    raise ValueError(f"Cannot infer observing run from {filename}")


def index_gravityspy_csv(
    path: str | Path,
    source_file: str,
    minimum_confidence: float,
    per_label: int,
    seed: int,
    excluded_labels: Iterable[str] = DEFAULT_EXCLUDED_LABELS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between zero and one")
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    excluded = set(excluded_labels)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_count += 1
            label = str(row["ml_label"])
            confidence = float(row["ml_confidence"])
            if label in excluded or confidence < minimum_confidence:
                continue
            event_time = float(row["event_time"])
            observing_run = _infer_run(source_file)
            candidates[label].append(
                {
                    "gravityspy_id": str(row["gravityspy_id"]),
                    "glitch_id": f"gravityspy:{row['gravityspy_id']}",
                    "ifo": str(row["ifo"]),
                    "observing_run": observing_run,
                    "event_time": event_time,
                    "gps_block": f"{row['ifo']}:{int(event_time // 64) * 64}:64",
                    "network_gps_block": (
                        f"{observing_run}:{int(event_time // 64) * 64}:64"
                    ),
                    "duration": float(row["duration"]),
                    "peak_frequency": float(row["peak_frequency"]),
                    "snr": float(row["snr"]),
                    "q_value": float(row["q_value"]),
                    "ml_label": label,
                    "ml_confidence": confidence,
                    "omega_scan_urls": [str(row[f"url{index}"]) for index in range(1, 5)],
                    "source_file": source_file,
                }
            )

    selected = []
    for label in sorted(candidates):
        rows = sorted(candidates[label], key=lambda item: item["gravityspy_id"])
        random.Random(f"{seed}:{label}").shuffle(rows)
        selected.extend(rows[:per_label])
    selected.sort(key=lambda item: (item["ml_label"], item["gravityspy_id"]))
    report = {
        "source_file": source_file,
        "raw_rows": raw_count,
        "eligible_rows": sum(len(items) for items in candidates.values()),
        "selected_rows": len(selected),
        "minimum_confidence": minimum_confidence,
        "per_label": per_label,
        "excluded_labels": sorted(excluded),
        "eligible_label_counts": dict(sorted((key, len(value)) for key, value in candidates.items())),
        "selected_label_counts": dict(sorted(Counter(item["ml_label"] for item in selected).items())),
        "unique_glitch_ids": len({item["glitch_id"] for item in selected}),
        "unique_gps_blocks": len({item["gps_block"] for item in selected}),
    }
    return selected, report


def run_gravityspy_index(
    record_id: int,
    filenames: Iterable[str],
    cache_dir: str | Path,
    output_dir: str | Path,
    minimum_confidence: float = 0.9,
    per_label: int = 100,
    seed: int = 20260719,
    download_workers: int = 8,
) -> dict[str, Any]:
    files = zenodo_files(record_id)
    cache = Path(cache_dir)
    output = Path(output_dir)
    all_rows = []
    source_reports = []
    for filename in filenames:
        if filename not in files:
            raise ValueError(f"{filename} is not present in Zenodo record {record_id}")
        source = files[filename]
        download = download_resumable(
            source["url"], cache / filename, workers=download_workers
        )
        checksum_type, expected_checksum = source["checksum"].split(":", 1)
        if checksum_type != "md5":
            raise ValueError(f"Unsupported Zenodo checksum: {source['checksum']}")
        actual_checksum = _file_md5(download["path"])
        if actual_checksum != expected_checksum:
            raise IOError(f"Zenodo checksum mismatch for {filename}")
        rows, report = index_gravityspy_csv(
            download["path"], filename, minimum_confidence, per_label, seed
        )
        all_rows.extend(rows)
        source_reports.append(
            {
                **report,
                "download": download,
                "publisher_checksum": source["checksum"],
            }
        )

    unique_rows = {row["glitch_id"]: row for row in all_rows}
    selected = sorted(unique_rows.values(), key=lambda item: (item["ml_label"], item["glitch_id"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "gravityspy_anchors.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected),
    )
    report = {
        "record_id": record_id,
        "record_url": f"https://zenodo.org/records/{record_id}",
        "user_agent": USER_AGENT,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "selected_rows": len(selected),
        "unique_glitch_ids": len(unique_rows),
        "unique_gps_blocks": len({item["gps_block"] for item in selected}),
        "label_counts": dict(sorted(Counter(item["ml_label"] for item in selected).items())),
        "sources": source_reports,
    }
    atomic_write_json(output / "gravityspy_index_report.json", report)
    return report


def split_gravityspy_anchors(
    manifest_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 20260720,
) -> dict[str, Any]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Gravity Spy validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Gravity Spy validation and test fractions must sum to less than one")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Gravity Spy anchor manifest cannot be empty")
    seen_glitches = set()
    split_rows = []
    for original in rows:
        row = dict(original)
        glitch_id = str(row["glitch_id"])
        if glitch_id in seen_glitches:
            raise ValueError(f"Duplicate Gravity Spy glitch ID: {glitch_id}")
        seen_glitches.add(glitch_id)
        network_block = str(
            row.get(
                "network_gps_block",
                f"{row['observing_run']}:{int(float(row['event_time']) // 64) * 64}:64",
            )
        )
        uniform = int(canonical_hash(f"{seed}:{network_block}", 16), 16) / 16**16
        if uniform < test_fraction:
            split = "test"
        elif uniform < test_fraction + validation_fraction:
            split = "val"
        else:
            split = "train"
        row["network_gps_block"] = network_block
        row["split"] = split
        split_rows.append(row)
    block_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        block_splits[str(row["network_gps_block"])].add(str(row["split"]))
    leaking_blocks = sorted(block for block, splits in block_splits.items() if len(splits) != 1)
    if leaking_blocks:
        raise ValueError(f"Network GPS blocks cross splits: {leaking_blocks[:10]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_manifests = {}
    split_counts = {}
    for split in ("train", "val", "test"):
        selected = sorted(
            (row for row in split_rows if row["split"] == split),
            key=lambda row: (str(row["observing_run"]), float(row["event_time"]), str(row["ifo"])),
        )
        if not selected:
            raise ValueError(f"Gravity Spy split {split} is empty")
        target = output / f"gravityspy_{split}.jsonl"
        atomic_write_text(
            target,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        )
        split_manifests[split] = {
            "path": str(target),
            "sha256": file_sha256(target),
        }
        split_counts[split] = {
            "rows": len(selected),
            "unique_glitches": len({row["glitch_id"] for row in selected}),
            "unique_network_gps_blocks": len(
                {row["network_gps_block"] for row in selected}
            ),
            "labels": dict(sorted(Counter(row["ml_label"] for row in selected).items())),
            "runs": dict(sorted(Counter(row["observing_run"] for row in selected).items())),
            "ifos": dict(sorted(Counter(row["ifo"] for row in selected).items())),
        }
    split_sets = {
        split: {
            "glitches": {str(row["glitch_id"]) for row in split_rows if row["split"] == split},
            "network_blocks": {
                str(row["network_gps_block"])
                for row in split_rows
                if row["split"] == split
            },
        }
        for split in ("train", "val", "test")
    }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = {
            field: len(split_sets[left][field] & split_sets[right][field])
            for field in ("glitches", "network_blocks")
        }
    report = {
        "status": "group_safe_gravityspy_split",
        "passed": all(
            count == 0
            for pair in overlaps.values()
            for count in pair.values()
        ),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "seed": seed,
        "fractions": {
            "train": 1.0 - validation_fraction - test_fraction,
            "val": validation_fraction,
            "test": test_fraction,
        },
        "rows": len(split_rows),
        "unique_network_gps_blocks": len(block_splits),
        "split_counts": split_counts,
        "cross_split_overlaps": overlaps,
        "manifests": split_manifests,
    }
    atomic_write_json(output / "gravityspy_split_report.json", report)
    return report
