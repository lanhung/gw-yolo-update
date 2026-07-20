from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .gwosc import API_ROOT, USER_AGENT, _api_json, _api_results, download_resumable
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_EXCLUDED_LABELS = ("Chirp", "No_Glitch", "None_of_the_Above")


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
    shard_summaries = []
    for shard in sorted(set(file_shards.values())):
        selected = [row for row in sharded if row["strain_shard"] == shard]
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
