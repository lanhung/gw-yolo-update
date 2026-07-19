from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .gwosc import USER_AGENT, _api_json, download_resumable
from .io import atomic_write_json, atomic_write_text, file_sha256


ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_EXCLUDED_LABELS = ("Chirp", "No_Glitch", "None_of_the_Above")


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
            candidates[label].append(
                {
                    "gravityspy_id": str(row["gravityspy_id"]),
                    "glitch_id": f"gravityspy:{row['gravityspy_id']}",
                    "ifo": str(row["ifo"]),
                    "observing_run": _infer_run(source_file),
                    "event_time": event_time,
                    "gps_block": f"{row['ifo']}:{int(event_time // 64) * 64}:64",
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
