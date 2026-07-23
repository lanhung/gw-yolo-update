from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance


_REQUIRED_FROZEN_ARTIFACTS = {
    "config",
    "model",
    "threshold_calibration",
    "ood_policy",
}
_LOCKED_SUITE_OUTPUT_KEYS = {
    "raw_candidate_search",
    "mask_candidate_search",
    "paired_raw_mask_search",
    "locked_ood_transfer",
    "dingo_batch",
    "amplfi_batch",
    "paired_pe_portfolio",
    "catalog_diagnostic",
    "suite_receipt",
}
_LOCKED_SUITE_INPUT_KEYS = {
    "raw_test_time_slide_report",
    "mask_test_time_slide_report",
    "raw_test_background_manifest",
    "mask_test_background_manifest",
    "raw_test_injection_ranking_report",
    "mask_test_injection_ranking_report",
    "locked_ood_score_manifest",
    "locked_ood_score_report",
    "locked_ood_source_manifest",
    "dingo_locked_source_batch_report",
    "amplfi_locked_source_batch_report",
    "catalog_source_manifest",
    "catalog_candidate_manifest",
    "catalog_candidate_report",
    "catalog_prediction_manifest",
    "catalog_prediction_report",
}
_LOCKED_SUITE_REQUIRED_FROZEN_ARTIFACTS = {
    "config",
    "model",
    "threshold_calibration",
    "ood_policy",
    "raw_candidate_calibration",
    "mask_candidate_calibration",
    "validation_raw_mask_comparison",
    "validation_ood_report",
    "validation_pe_promotion",
    "catalog_metadata",
    "locked_execution_plan",
}
_LOCKED_STREAM_SHARD_ARTIFACT_KEYS = {
    "mask_candidate_rows",
    "ood_score_rows",
    "raw_candidate_rows",
}


def freeze_locked_o4b_streaming_execution_plan(
    suite_plan_path: str | Path,
    corpus_freeze_path: str | Path,
    availability_manifest_path: str | Path,
    availability_report_path: str | Path,
    inventory_manifest_path: str | Path,
    inventory_report_path: str | Path,
    work_root: str | Path,
    shard_manifest_path: str | Path,
    output_path: str | Path,
    code_commit: str,
    blocks_per_shard: int = 1,
    minimum_free_kb: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Freeze the post-access O4b streaming order without reading test strain.

    The plan binds every score-blind availability block to its single predeclared
    injection and to an immutable shard/work-directory identity.  It deliberately
    contains no scores, DQ-driven replacement policy or result-dependent stopping
    rule.  The resulting report is intended to be included as
    ``locked_execution_plan`` in the one-time access receipt.
    """

    if blocks_per_shard < 1 or minimum_free_kb < 1024 * 1024:
        raise ValueError("locked streaming resource limits are invalid")
    if not code_commit.strip():
        raise ValueError("locked streaming execution requires an exact code commit")

    suite_file = Path(suite_plan_path).resolve()
    freeze_file = Path(corpus_freeze_path).resolve()
    availability_manifest = Path(availability_manifest_path).resolve()
    availability_report_file = Path(availability_report_path).resolve()
    inventory_manifest = Path(inventory_manifest_path).resolve()
    inventory_report_file = Path(inventory_report_path).resolve()
    shard_manifest = Path(shard_manifest_path).resolve()
    target = Path(output_path).resolve()
    work = Path(work_root).resolve()
    required_files = (
        suite_file,
        freeze_file,
        availability_manifest,
        availability_report_file,
        inventory_manifest,
        inventory_report_file,
    )
    if any(not path.is_file() for path in required_files):
        raise FileNotFoundError("locked streaming plan inputs are absent")

    suite = json.loads(suite_file.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    availability_report = json.loads(
        availability_report_file.read_text(encoding="utf-8")
    )
    inventory_report = json.loads(inventory_report_file.read_text(encoding="utf-8"))
    access_log = Path(str(freeze.get("access_log_path", ""))).resolve()
    suite_root = Path(str(suite.get("output_root", ""))).resolve()
    if (
        suite.get("status") != "frozen_locked_evaluation_suite_plan"
        or suite.get("passed") is not True
        or suite.get("locked_corpus_opened") is not False
        or suite.get("test_rows_read") != 0
        or suite.get("code_commit") != code_commit
        or suite.get("corpus_label") != "GWTC-5.0_O4b_locked_suite_v2"
        or freeze.get("status") != "locked_evaluation_corpus_unopened"
        or freeze.get("evaluation_opened") is not False
        or freeze.get("candidate_scores_inspected") is not False
        or freeze.get("corpus_label") != suite.get("corpus_label")
        or Path(str(freeze.get("manifest_path", ""))).resolve()
        != inventory_manifest
        or freeze.get("manifest_sha256") != file_sha256(inventory_manifest)
        or access_log.exists()
    ):
        raise ValueError("locked suite/corpus is not at the unopened execution boundary")
    if work == suite_root or suite_root not in work.parents:
        raise ValueError("locked streaming work root must be a child of the suite output root")
    declared_paths = {
        Path(str(path)).resolve()
        for inventory in (suite.get("inputs", {}), suite.get("outputs", {}))
        for path in inventory.values()
    }
    if work in declared_paths or shard_manifest in declared_paths or target in declared_paths:
        raise ValueError("locked streaming control paths collide with suite artifacts")
    if work.exists():
        raise FileExistsError("locked streaming work root already exists")

    if (
        availability_report.get("status")
        != "score_blind_gwtc5_o4b_availability_inventory"
        or availability_report.get("passed") is not True
        or availability_report.get("manifest_sha256")
        != file_sha256(availability_manifest)
        or Path(str(availability_report.get("manifest_path", ""))).resolve()
        != availability_manifest
        or Path(str(availability_report.get("access_log_path", ""))).resolve()
        != access_log
        or availability_report.get("candidate_scores_inspected") is not False
        or int(availability_report.get("test_strain_rows_read", -1)) != 0
        or inventory_report.get("status")
        != "score_blind_gwtc5_locked_injection_inventory"
        or inventory_report.get("passed") is not True
        or inventory_report.get("manifest_sha256") != file_sha256(inventory_manifest)
        or Path(str(inventory_report.get("manifest_path", ""))).resolve()
        != inventory_manifest
        or inventory_report.get("availability_manifest_sha256")
        != file_sha256(availability_manifest)
        or Path(str(inventory_report.get("access_log_path", ""))).resolve()
        != access_log
        or inventory_report.get("post_access_dq_replacement_allowed") is not False
        or inventory_report.get("candidate_scores_inspected") is not False
        or int(inventory_report.get("test_strain_rows_read", -1)) != 0
    ):
        raise ValueError("locked score-blind inventory evidence failed replay")

    availability_rows = _load_jsonl(availability_manifest)
    injection_rows = _load_jsonl(inventory_manifest)
    by_availability: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(availability_rows):
        identity = str(row.get("availability_id", ""))
        sources = row.get("sources")
        if not identity or identity in by_availability or not isinstance(sources, dict):
            raise ValueError(f"locked availability identity is invalid at row {index}")
        available_ifos = [str(value) for value in row.get("available_ifos", [])]
        if available_ifos != sorted(sources) or len(available_ifos) < 2:
            raise ValueError(f"locked availability sources are incomplete at row {index}")
        for ifo, source in sources.items():
            if (
                ifo not in {"H1", "L1", "V1"}
                or not isinstance(source, dict)
                or str(source.get("detector")) != ifo
                or not str(source.get("hdf5_url", "")).startswith("https://gwosc.org/")
                or not str(source.get("detail_url", "")).startswith("https://gwosc.org/")
            ):
                raise ValueError(f"locked availability source is invalid at row {index}")
        by_availability[identity] = row

    ordered_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    injection_ids: set[str] = set()
    for index, injection in enumerate(injection_rows):
        availability_id = str(injection.get("availability_id", ""))
        injection_id = str(injection.get("injection_id", ""))
        availability = by_availability.get(availability_id)
        if (
            availability is None
            or not injection_id
            or injection_id in injection_ids
            or injection.get("split") != "test"
            or injection.get("observing_run") != "O4b"
            or str(injection.get("gps_block")) != str(availability.get("gps_block"))
            or list(injection.get("ifos", [])) != list(availability.get("available_ifos", []))
        ):
            raise ValueError(f"locked injection/availability join failed at row {index}")
        injection_ids.add(injection_id)
        ordered_rows.append((injection, availability))
    if len(ordered_rows) != len(by_availability):
        raise ValueError("locked streaming requires one injection per availability block")

    shard_rows = []
    for shard_index, start in enumerate(range(0, len(ordered_rows), blocks_per_shard)):
        batch = ordered_rows[start : start + blocks_per_shard]
        sources = []
        for injection, availability in batch:
            for ifo in availability["available_ifos"]:
                source = availability["sources"][ifo]
                sources.append(
                    {
                        "availability_id": availability["availability_id"],
                        "ifo": ifo,
                        "gps_start": source["gps_start"],
                        "duration": source["duration"],
                        "hdf5_url": source["hdf5_url"],
                        "detail_url": source["detail_url"],
                    }
                )
        shard_rows.append(
            {
                "schema": "locked_o4b_stream_shard_v1",
                "shard_index": shard_index,
                "row_start": start,
                "row_stop_exclusive": start + len(batch),
                "work_dir": str(work / f"shard-{shard_index:05d}"),
                "availability_ids": [str(value[1]["availability_id"]) for value in batch],
                "injection_ids": [str(value[0]["injection_id"]) for value in batch],
                "waveform_ids": [str(value[0]["waveform_id"]) for value in batch],
                "gps_blocks": [str(value[0]["gps_block"]) for value in batch],
                "source_files": sources,
                "source_cache_dir": str(work / f"shard-{shard_index:05d}" / "sources"),
                "source_download_report_path": str(
                    work
                    / f"shard-{shard_index:05d}"
                    / "locked_source_download_report.json"
                ),
                "source_eviction_report_path": str(
                    work
                    / f"shard-{shard_index:05d}"
                    / "locked_source_eviction_report.json"
                ),
                "artifact_paths": {
                    label: str(
                        work / f"shard-{shard_index:05d}" / f"{label}.jsonl"
                    )
                    for label in sorted(_LOCKED_STREAM_SHARD_ARTIFACT_KEYS)
                },
                "receipt_path": str(
                    work / f"shard-{shard_index:05d}" / "shard_receipt.json"
                ),
                "post_access_dq_replacement_allowed": False,
                "result_dependent_stopping_allowed": False,
                "source_eviction_required_after_verified_reduction": True,
            }
        )

    identity = {
        "suite_plan_path": str(suite_file),
        "suite_plan_sha256": file_sha256(suite_file),
        "corpus_freeze_path": str(freeze_file),
        "corpus_freeze_sha256": file_sha256(freeze_file),
        "availability_manifest_path": str(availability_manifest),
        "availability_manifest_sha256": file_sha256(availability_manifest),
        "availability_report_path": str(availability_report_file),
        "availability_report_sha256": file_sha256(availability_report_file),
        "inventory_manifest_path": str(inventory_manifest),
        "inventory_manifest_sha256": file_sha256(inventory_manifest),
        "inventory_report_path": str(inventory_report_file),
        "inventory_report_sha256": file_sha256(inventory_report_file),
        "work_root": str(work),
        "shard_manifest_path": str(shard_manifest),
        "blocks_per_shard": blocks_per_shard,
        "minimum_free_kb": minimum_free_kb,
        "code_commit": code_commit,
    }
    if target.is_file():
        completed = json.loads(target.read_text(encoding="utf-8"))
        if completed.get("freeze_identity") != identity:
            raise ValueError("existing locked streaming execution plan has another identity")
        if (
            not shard_manifest.is_file()
            or completed.get("shard_manifest_sha256") != file_sha256(shard_manifest)
        ):
            raise ValueError("locked streaming shard manifest changed after freezing")
        return completed
    if target.exists() or shard_manifest.exists():
        raise FileExistsError("partial locked streaming execution plan exists")
    atomic_write_text(
        shard_manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shard_rows),
    )
    report = {
        "status": "frozen_locked_o4b_streaming_execution_plan",
        "passed": True,
        "scientific_claim_allowed": False,
        "evaluation_opened": False,
        "test_strain_rows_read": 0,
        "candidate_scores_inspected": False,
        "corpus_label": suite["corpus_label"],
        "freeze_identity": identity,
        "access_log_path": str(access_log),
        "access_log_exists": False,
        "rows": len(ordered_rows),
        "unique_injections": len(injection_ids),
        "unique_gps_blocks": len({str(row[0]["gps_block"]) for row in ordered_rows}),
        "shards": len(shard_rows),
        "blocks_per_shard": blocks_per_shard,
        "maximum_concurrent_shards": 1,
        "minimum_free_kb": minimum_free_kb,
        "post_access_dq_replacement_allowed": False,
        "result_dependent_stopping_allowed": False,
        "source_eviction_required_after_verified_reduction": True,
        "shard_manifest_path": str(shard_manifest),
        "shard_manifest_sha256": file_sha256(shard_manifest),
        "receipt_manifest_path": str(work / "streaming-shard-receipts.jsonl"),
        "receipt_merge_report_path": str(
            work / "streaming-shard-receipt-merge-report.json"
        ),
        "completion_audit_path": str(work / "streaming-completion-audit.json"),
        "code_commit": code_commit,
        **execution_provenance(),
    }
    report["runtime_provenance"] = {
        "runtime_code_commit": report.pop("code_commit"),
        "exact_command": report.pop("exact_command"),
        "environment": report.pop("environment"),
    }
    report["code_commit"] = code_commit
    atomic_write_json(target, report)
    return report


def download_locked_o4b_streaming_shard_sources(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    shard_index: int,
    code_commit: str,
    download_workers: int = 4,
    chunk_samples: int = 1_048_576,
) -> dict[str, Any]:
    """Download one predeclared O4b shard only after irreversible suite access.

    Generic development downloaders continue to reject O4b. This gate derives
    every URL and output path from the pre-access streaming plan, permits only
    one active shard, enforces the frozen storage floor, and verifies each HDF5
    source against its GWOSC metadata before publishing the shard report.
    """

    if (
        shard_index < 0
        or download_workers < 1
        or chunk_samples < 1
        or not code_commit.strip()
    ):
        raise ValueError("locked shard download settings are invalid")
    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    if not plan_file.is_file() or not access_file.is_file():
        raise FileNotFoundError("locked shard plan/access input is absent")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    frozen_plan = access.get("frozen_artifacts", {}).get("locked_execution_plan", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    if (
        plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or plan.get("maximum_concurrent_shards") != 1
        or plan.get("result_dependent_stopping_allowed") is not False
        or plan.get("post_access_dq_replacement_allowed") is not False
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("corpus_label") != plan.get("corpus_label")
        or access.get("code_commit") != plan.get("code_commit")
        or plan.get("code_commit") != code_commit
        or Path(str(plan.get("access_log_path", ""))).resolve() != access_file
        or frozen_plan.get("path") != str(plan_file)
        or frozen_plan.get("sha256") != file_sha256(plan_file)
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
    ):
        raise ValueError("locked shard download access/plan binding failed replay")
    shards = _load_jsonl(shard_manifest)
    if shard_index >= len(shards) or len(shards) != int(plan.get("shards", -1)):
        raise ValueError("locked shard index is outside the frozen schedule")
    shard = shards[shard_index]
    if int(shard.get("shard_index", -1)) != shard_index:
        raise ValueError("locked shard manifest order changed after freezing")

    work_dir = Path(str(shard.get("work_dir", ""))).resolve()
    work_root = Path(str(plan.get("freeze_identity", {}).get("work_root", ""))).resolve()
    source_dir = Path(str(shard.get("source_cache_dir", ""))).resolve()
    report_path = Path(str(shard.get("source_download_report_path", ""))).resolve()
    if (
        work_root not in work_dir.parents
        or work_dir not in source_dir.parents
        or work_dir not in report_path.parents
        or work_dir.name != f"shard-{shard_index:05d}"
    ):
        raise ValueError("locked shard paths are not children of the frozen work root")
    identity = {
        "execution_plan_path": str(plan_file),
        "execution_plan_sha256": file_sha256(plan_file),
        "access_log_path": str(access_file),
        "access_log_sha256": file_sha256(access_file),
        "shard_manifest_sha256": file_sha256(shard_manifest),
        "shard_index": shard_index,
        "source_files_sha256": canonical_hash(shard["source_files"], length=64),
        "download_workers": download_workers,
        "chunk_samples": chunk_samples,
        "code_commit": plan["code_commit"],
    }
    if report_path.is_file():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        completed_files = completed.get("files")
        if (
            completed.get("status") != "verified_locked_o4b_shard_sources"
            or completed.get("passed") is not True
            or completed.get("run_identity") != identity
            or not isinstance(completed_files, list)
            or len(completed_files) != len(shard["source_files"])
            or completed.get("verified_files") != len(shard["source_files"])
            or any(
                row.get("source_index") != source_index
                or row.get("availability_id") != source["availability_id"]
                or row.get("detector") != source["ifo"]
                or row.get("gps_start") != source["gps_start"]
                or row.get("duration") != source["duration"]
                or row.get("hdf5_url") != source["hdf5_url"]
                or row.get("detail_url") != source["detail_url"]
                or Path(str(row.get("path", ""))).resolve()
                != (
                    source_dir / f"source-{source_index:03d}-{source['ifo']}.hdf5"
                ).resolve()
                or not Path(str(row.get("path", ""))).is_file()
                or row.get("sha256") != file_sha256(row["path"])
                or row.get("verification", {}).get("passed") is not True
                for source_index, (source, row) in enumerate(
                    zip(shard["source_files"], completed_files)
                )
            )
        ):
            raise ValueError("existing locked shard source report failed replay")
        return completed
    if report_path.exists():
        raise FileExistsError("partial locked shard source report exists")

    minimum_free_bytes = int(plan["minimum_free_kb"]) * 1024
    work_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_root).free < minimum_free_bytes:
        raise RuntimeError("locked shard download storage guard is not satisfied")
    lease_path = work_root / ".active-shard.lock"
    try:
        descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("another locked O4b shard is already active") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "shard_index": shard_index,
                    "execution_plan_sha256": identity["execution_plan_sha256"],
                    "access_log_sha256": identity["access_log_sha256"],
                    "pid": os.getpid(),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        from .gwosc import _api_json, download_resumable, verify_hdf5_against_detail

        source_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for source_index, source in enumerate(shard["source_files"]):
            if shutil.disk_usage(work_root).free < minimum_free_bytes:
                raise RuntimeError(
                    "locked shard download storage guard failed during execution"
                )
            detector = str(source["ifo"])
            destination = source_dir / f"source-{source_index:03d}-{detector}.hdf5"
            download = download_resumable(
                str(source["hdf5_url"]),
                destination,
                workers=download_workers,
            )
            verification = verify_hdf5_against_detail(
                download["path"],
                _api_json(str(source["detail_url"])),
                chunk_samples,
            )
            if verification.get("passed") is not True:
                raise RuntimeError(
                    f"locked O4b source verification failed: {shard_index}/{source_index}"
                )
            files.append(
                {
                    "source_index": source_index,
                    "availability_id": source["availability_id"],
                    "detector": detector,
                    "gps_start": source["gps_start"],
                    "duration": source["duration"],
                    "hdf5_url": source["hdf5_url"],
                    "detail_url": source["detail_url"],
                    "path": str(destination),
                    "sha256": download["sha256"],
                    "bytes": download["bytes"],
                    "downloaded": download["downloaded"],
                    "verification": verification,
                }
            )
        result = {
            "status": "verified_locked_o4b_shard_sources",
            "passed": True,
            "scientific_claim_allowed": False,
            "candidate_scores_inspected": False,
            "test_rows_processed": 0,
            "shard_index": shard_index,
            "availability_ids": shard["availability_ids"],
            "injection_ids": shard["injection_ids"],
            "gps_blocks": shard["gps_blocks"],
            "files": files,
            "verified_files": len(files),
            "run_identity": identity,
            "code_commit": plan["code_commit"],
            **execution_provenance(),
        }
        result["runtime_provenance"] = {
            "runtime_code_commit": result.pop("code_commit"),
            "exact_command": result.pop("exact_command"),
            "environment": result.pop("environment"),
        }
        result["code_commit"] = plan["code_commit"]
        atomic_write_json(report_path, result)
        return result
    finally:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass


def finalize_locked_o4b_streaming_shard(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    shard_index: int,
    code_commit: str,
) -> dict[str, Any]:
    """Hash reduced shard products, evict verified strain, and seal its receipt."""

    if shard_index < 0 or not code_commit.strip():
        raise ValueError("locked shard index must be non-negative")
    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    frozen_plan = access.get("frozen_artifacts", {}).get("locked_execution_plan", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    if (
        plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("code_commit") != plan.get("code_commit")
        or plan.get("code_commit") != code_commit
        or Path(str(plan.get("access_log_path", ""))).resolve() != access_file
        or frozen_plan.get("path") != str(plan_file)
        or frozen_plan.get("sha256") != file_sha256(plan_file)
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
    ):
        raise ValueError("locked shard finalization access/plan binding failed replay")
    shards = _load_jsonl(shard_manifest)
    if shard_index >= len(shards):
        raise ValueError("locked shard index is outside the frozen schedule")
    shard = shards[shard_index]
    if int(shard.get("shard_index", -1)) != shard_index:
        raise ValueError("locked shard order changed after freezing")
    work_dir = Path(str(shard["work_dir"])).resolve()
    source_dir = Path(str(shard["source_cache_dir"])).resolve()
    source_report_path = Path(str(shard["source_download_report_path"])).resolve()
    eviction_path = Path(str(shard["source_eviction_report_path"])).resolve()
    receipt_path = Path(str(shard["receipt_path"])).resolve()
    if any(
        work_dir not in path.parents
        for path in (source_dir, source_report_path, eviction_path, receipt_path)
    ):
        raise ValueError("locked shard finalization paths escaped the frozen work dir")
    if not source_report_path.is_file():
        raise FileNotFoundError("locked shard source download report is absent")

    expected_artifacts = shard.get("artifact_paths")
    if (
        not isinstance(expected_artifacts, dict)
        or set(expected_artifacts) != _LOCKED_STREAM_SHARD_ARTIFACT_KEYS
    ):
        raise ValueError("locked shard artifact paths were not frozen")
    artifact_entries = {}
    for label in sorted(_LOCKED_STREAM_SHARD_ARTIFACT_KEYS):
        path = Path(str(expected_artifacts[label])).resolve()
        if work_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(f"locked shard artifact is absent: {label}")
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"locked shard artifact has invalid JSON: {label}/{line_number}"
                    ) from error
                if (
                    not isinstance(row, dict)
                    or row.get("shard_index") != shard_index
                    or (
                        row.get("injection_id") is not None
                        and str(row["injection_id"]) not in shard["injection_ids"]
                    )
                    or (
                        row.get("availability_id") is not None
                        and str(row["availability_id"]) not in shard["availability_ids"]
                    )
                ):
                    raise ValueError(
                        f"locked shard artifact row failed identity replay: "
                        f"{label}/{line_number}"
                    )
                rows.append(row)
        artifact_entries[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(rows),
        }

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_files = source_report.get("files")
    if (
        source_report.get("status") != "verified_locked_o4b_shard_sources"
        or source_report.get("passed") is not True
        or source_report.get("shard_index") != shard_index
        or source_report.get("run_identity", {}).get("execution_plan_sha256")
        != file_sha256(plan_file)
        or source_report.get("run_identity", {}).get("access_log_sha256")
        != file_sha256(access_file)
        or not isinstance(source_files, list)
        or len(source_files) != len(shard["source_files"])
    ):
        raise ValueError("locked shard source report failed finalization replay")
    targets = []
    for source_index, (source, observed) in enumerate(
        zip(shard["source_files"], source_files)
    ):
        path = Path(str(observed.get("path", ""))).resolve()
        if (
            observed.get("source_index") != source_index
            or observed.get("availability_id") != source["availability_id"]
            or observed.get("detector") != source["ifo"]
            or source_dir not in path.parents
            or not str(observed.get("sha256", ""))
            or isinstance(observed.get("bytes"), bool)
            or not isinstance(observed.get("bytes"), int)
            or observed["bytes"] < 0
            or observed.get("verification", {}).get("passed") is not True
        ):
            raise ValueError("locked shard source inventory failed finalization replay")
        targets.append(
            {
                "path": str(path),
                "sha256": observed["sha256"],
                "bytes": observed["bytes"],
            }
        )

    intent_path = eviction_path.with_suffix(eviction_path.suffix + ".intent.json")
    eviction_identity = {
        "execution_plan_sha256": file_sha256(plan_file),
        "access_log_sha256": file_sha256(access_file),
        "shard_index": shard_index,
        "source_download_report_sha256": file_sha256(source_report_path),
        "source_files_sha256": canonical_hash(shard["source_files"], length=64),
        "artifacts": artifact_entries,
        "targets": targets,
    }
    if eviction_path.is_file():
        eviction = json.loads(eviction_path.read_text(encoding="utf-8"))
        if (
            eviction.get("status") != "verified_locked_source_eviction"
            or eviction.get("passed") is not True
            or eviction.get("eviction_identity") != eviction_identity
            or eviction.get("source_files_removed") != len(targets)
            or eviction.get("source_files_retained") != 0
            or any(Path(row["path"]).exists() for row in targets)
        ):
            raise ValueError("existing locked shard source eviction failed replay")
    else:
        if intent_path.is_file():
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if (
                intent.get("status") != "validated_locked_source_eviction_intent"
                or intent.get("eviction_identity") != eviction_identity
            ):
                raise ValueError("locked shard source eviction intent changed")
        else:
            for row in targets:
                path = Path(row["path"])
                if not path.is_file() or file_sha256(path) != row["sha256"]:
                    raise ValueError("locked shard source hash changed before eviction")
            atomic_write_json(
                intent_path,
                {
                    "status": "validated_locked_source_eviction_intent",
                    "eviction_identity": eviction_identity,
                },
            )
        for row in targets:
            path = Path(row["path"])
            if path.exists():
                if not path.is_file() or file_sha256(path) != row["sha256"]:
                    raise ValueError("locked shard source hash changed during eviction")
                path.unlink()
        eviction = {
            "status": "verified_locked_source_eviction",
            "passed": True,
            "recoverable": True,
            "recovery": (
                "re-run locked-o4b-streaming-shard-download using the same "
                "hash-bound execution plan and access log"
            ),
            "eviction_identity": eviction_identity,
            "source_files_removed": len(targets),
            "source_bytes_removed": sum(row["bytes"] for row in targets),
            "source_files_retained": 0,
            "source_files_sha256": canonical_hash(shard["source_files"], length=64),
            "intent_path": str(intent_path),
            "intent_sha256": file_sha256(intent_path),
            "code_commit": plan["code_commit"],
            **execution_provenance(),
        }
        eviction["runtime_provenance"] = {
            "runtime_code_commit": eviction.pop("code_commit"),
            "exact_command": eviction.pop("exact_command"),
            "environment": eviction.pop("environment"),
        }
        eviction["code_commit"] = plan["code_commit"]
        atomic_write_json(eviction_path, eviction)

    receipt = {
        "status": "completed_locked_o4b_stream_shard",
        "passed": True,
        **{
            key: shard[key]
            for key in (
                "shard_index",
                "row_start",
                "row_stop_exclusive",
                "availability_ids",
                "injection_ids",
                "waveform_ids",
                "gps_blocks",
            )
        },
        "test_rows_processed": int(shard["row_stop_exclusive"])
        - int(shard["row_start"]),
        "result_dependent_stopping_used": False,
        "post_access_dq_replacement_used": False,
        "negative_and_null_results_retained": True,
        "streaming_plan_sha256": file_sha256(plan_file),
        "access_log_sha256": file_sha256(access_file),
        "artifacts": artifact_entries,
        "source_download_report": {
            "path": str(source_report_path),
            "sha256": file_sha256(source_report_path),
        },
        "source_eviction": {
            "status": eviction["status"],
            "passed": eviction["passed"],
            "path": str(eviction_path),
            "sha256": file_sha256(eviction_path),
            "source_files_removed": eviction["source_files_removed"],
            "source_files_retained": eviction["source_files_retained"],
            "source_files_sha256": eviction["source_files_sha256"],
        },
        "code_commit": plan["code_commit"],
        **execution_provenance(),
    }
    receipt["runtime_provenance"] = {
        "runtime_code_commit": receipt.pop("code_commit"),
        "exact_command": receipt.pop("exact_command"),
        "environment": receipt.pop("environment"),
    }
    receipt["code_commit"] = plan["code_commit"]
    if receipt_path.is_file():
        completed = json.loads(receipt_path.read_text(encoding="utf-8"))
        comparable = dict(completed)
        comparable.pop("runtime_provenance", None)
        expected = dict(receipt)
        expected.pop("runtime_provenance", None)
        if comparable != expected:
            raise ValueError("existing locked shard receipt changed")
        return completed
    atomic_write_json(receipt_path, receipt)
    return receipt


def merge_locked_o4b_streaming_shard_receipts(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    code_commit: str,
) -> dict[str, Any]:
    """Merge every predeclared shard receipt in frozen order without selection."""

    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    frozen_plan = access.get("frozen_artifacts", {}).get("locked_execution_plan", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    receipt_manifest = Path(str(plan.get("receipt_manifest_path", ""))).resolve()
    merge_report = Path(str(plan.get("receipt_merge_report_path", ""))).resolve()
    if (
        plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("code_commit") != plan.get("code_commit")
        or plan.get("code_commit") != code_commit
        or frozen_plan.get("path") != str(plan_file)
        or frozen_plan.get("sha256") != file_sha256(plan_file)
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
    ):
        raise ValueError("locked shard receipt merge access/plan binding failed replay")
    shards = _load_jsonl(shard_manifest)
    receipts = []
    for expected_index, shard in enumerate(shards):
        receipt_path = Path(str(shard.get("receipt_path", ""))).resolve()
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"locked shard receipt is absent: {expected_index}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            int(shard.get("shard_index", -1)) != expected_index
            or receipt.get("status") != "completed_locked_o4b_stream_shard"
            or receipt.get("passed") is not True
            or receipt.get("shard_index") != expected_index
            or receipt.get("streaming_plan_sha256") != file_sha256(plan_file)
            or receipt.get("access_log_sha256") != file_sha256(access_file)
        ):
            raise ValueError(f"locked shard receipt failed merge replay: {expected_index}")
        receipts.append(receipt)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts)
    if receipt_manifest.is_file():
        if receipt_manifest.read_text(encoding="utf-8") != payload:
            raise ValueError("existing locked shard receipt manifest changed")
    else:
        atomic_write_text(receipt_manifest, payload)
    result = {
        "status": "merged_locked_o4b_streaming_shard_receipts",
        "passed": True,
        "scientific_claim_allowed": False,
        "all_predeclared_shards_present": True,
        "negative_and_null_results_retained": True,
        "execution_plan_sha256": file_sha256(plan_file),
        "access_log_sha256": file_sha256(access_file),
        "receipt_manifest_path": str(receipt_manifest),
        "receipt_manifest_sha256": file_sha256(receipt_manifest),
        "completed_shards": len(receipts),
        "rows": sum(int(row["test_rows_processed"]) for row in receipts),
        "code_commit": plan["code_commit"],
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = plan["code_commit"]
    if merge_report.is_file():
        completed = json.loads(merge_report.read_text(encoding="utf-8"))
        if (
            completed.get("execution_plan_sha256")
            != result["execution_plan_sha256"]
            or completed.get("access_log_sha256") != result["access_log_sha256"]
            or completed.get("receipt_manifest_sha256")
            != result["receipt_manifest_sha256"]
        ):
            raise ValueError("existing locked shard receipt merge report changed")
        return completed
    atomic_write_json(merge_report, result)
    return result


def audit_locked_o4b_streaming_completion(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    receipt_manifest_path: str | Path,
    output_path: str | Path,
    code_commit: str,
) -> dict[str, Any]:
    """Fail closed unless every frozen O4b shard was processed and retained.

    The data-plane worker writes one receipt row per frozen shard. This reducer
    does not choose shards or inspect endpoint values: it replays the exact
    pre-access order, hashes every raw/mask/OOD artifact, verifies source
    eviction, and publishes a complete artifact inventory for downstream locked
    endpoint reducers. Empty shard artifacts are allowed because null results
    must be retained.
    """

    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    receipts_file = Path(receipt_manifest_path).resolve()
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("locked streaming completion audits are immutable")
    if not plan_file.is_file() or not access_file.is_file() or not receipts_file.is_file():
        raise FileNotFoundError("locked streaming completion inputs are absent")

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    frozen_plan = access.get("frozen_artifacts", {}).get("locked_execution_plan", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    receipt_merge_report = Path(
        str(plan.get("receipt_merge_report_path", ""))
    ).resolve()
    if (
        plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or plan.get("evaluation_opened") is not False
        or plan.get("candidate_scores_inspected") is not False
        or plan.get("result_dependent_stopping_allowed") is not False
        or plan.get("post_access_dq_replacement_allowed") is not False
        or plan.get("maximum_concurrent_shards") != 1
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("corpus_label") != plan.get("corpus_label")
        or access.get("code_commit") != plan.get("code_commit")
        or plan.get("code_commit") != code_commit
        or Path(str(plan.get("access_log_path", ""))).resolve() != access_file
        or frozen_plan.get("path") != str(plan_file)
        or frozen_plan.get("sha256") != file_sha256(plan_file)
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
        or receipts_file
        != Path(str(plan.get("receipt_manifest_path", ""))).resolve()
        or target != Path(str(plan.get("completion_audit_path", ""))).resolve()
        or not receipt_merge_report.is_file()
    ):
        raise ValueError("locked streaming access/plan binding failed replay")
    merged = json.loads(receipt_merge_report.read_text(encoding="utf-8"))
    if (
        merged.get("status") != "merged_locked_o4b_streaming_shard_receipts"
        or merged.get("passed") is not True
        or merged.get("execution_plan_sha256") != file_sha256(plan_file)
        or merged.get("access_log_sha256") != file_sha256(access_file)
        or merged.get("receipt_manifest_path") != str(receipts_file)
        or merged.get("receipt_manifest_sha256") != file_sha256(receipts_file)
        or merged.get("completed_shards") != plan.get("shards")
    ):
        raise ValueError("locked streaming receipt merge failed replay")

    shards = _load_jsonl(shard_manifest)
    receipts = _load_jsonl(receipts_file)
    if (
        len(shards) != int(plan.get("shards", -1))
        or len(receipts) != len(shards)
        or int(plan.get("rows", -1))
        != sum(int(row["row_stop_exclusive"]) - int(row["row_start"]) for row in shards)
    ):
        raise ValueError("locked streaming receipts do not cover every frozen shard")

    artifact_inventory: dict[str, list[dict[str, Any]]] = {
        key: [] for key in sorted(_LOCKED_STREAM_SHARD_ARTIFACT_KEYS)
    }
    processed_injections: list[str] = []
    processed_availability: list[str] = []
    for expected_index, (shard, receipt) in enumerate(zip(shards, receipts)):
        work_dir = Path(str(shard.get("work_dir", ""))).resolve()
        expected_rows = int(shard["row_stop_exclusive"]) - int(shard["row_start"])
        identity_fields = (
            "shard_index",
            "row_start",
            "row_stop_exclusive",
            "availability_ids",
            "injection_ids",
            "waveform_ids",
            "gps_blocks",
        )
        if (
            int(shard.get("shard_index", -1)) != expected_index
            or int(receipt.get("shard_index", -1)) != expected_index
            or receipt.get("status") != "completed_locked_o4b_stream_shard"
            or receipt.get("passed") is not True
            or receipt.get("test_rows_processed") != expected_rows
            or receipt.get("result_dependent_stopping_used") is not False
            or receipt.get("post_access_dq_replacement_used") is not False
            or receipt.get("negative_and_null_results_retained") is not True
            or any(receipt.get(field) != shard.get(field) for field in identity_fields)
            or receipt.get("streaming_plan_sha256") != file_sha256(plan_file)
            or receipt.get("access_log_sha256") != file_sha256(access_file)
        ):
            raise ValueError(
                f"locked streaming shard receipt failed identity replay: {expected_index}"
            )

        artifacts = receipt.get("artifacts")
        expected_artifacts = shard.get("artifact_paths")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != _LOCKED_STREAM_SHARD_ARTIFACT_KEYS
            or not isinstance(expected_artifacts, dict)
            or set(expected_artifacts) != _LOCKED_STREAM_SHARD_ARTIFACT_KEYS
        ):
            raise ValueError(
                f"locked streaming shard artifact inventory is incomplete: {expected_index}"
            )
        for label in sorted(_LOCKED_STREAM_SHARD_ARTIFACT_KEYS):
            entry = artifacts[label]
            if not isinstance(entry, dict):
                raise ValueError(f"locked streaming shard artifact is invalid: {label}")
            artifact_path = Path(str(entry.get("path", ""))).resolve()
            rows = entry.get("rows")
            if (
                not artifact_path.is_file()
                or artifact_path != Path(str(expected_artifacts[label])).resolve()
                or work_dir not in artifact_path.parents
                or entry.get("sha256") != file_sha256(artifact_path)
                or isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 0
            ):
                raise ValueError(
                    f"locked streaming shard artifact failed replay: "
                    f"{expected_index}/{label}"
                )
            artifact_inventory[label].append(
                {
                    "shard_index": expected_index,
                    "path": str(artifact_path),
                    "sha256": entry["sha256"],
                    "rows": rows,
                }
            )

        source_report_path = Path(
            str(shard.get("source_download_report_path", ""))
        ).resolve()
        source_report = receipt.get("source_download_report")
        if (
            not isinstance(source_report, dict)
            or source_report_path != Path(str(source_report.get("path", ""))).resolve()
            or not source_report_path.is_file()
            or source_report.get("sha256") != file_sha256(source_report_path)
        ):
            raise ValueError(
                f"locked streaming shard source report failed replay: {expected_index}"
            )
        eviction = receipt.get("source_eviction")
        expected_sources = len(shard.get("source_files", []))
        if (
            not isinstance(eviction, dict)
            or eviction.get("status") != "verified_locked_source_eviction"
            or eviction.get("passed") is not True
            or eviction.get("source_files_removed") != expected_sources
            or eviction.get("source_files_retained") != 0
            or eviction.get("source_files_sha256")
            != canonical_hash(shard.get("source_files", []), length=64)
        ):
            raise ValueError(
                f"locked streaming shard source eviction failed replay: {expected_index}"
            )
        processed_injections.extend(map(str, shard["injection_ids"]))
        processed_availability.extend(map(str, shard["availability_ids"]))

    if (
        len(processed_injections) != int(plan["rows"])
        or len(set(processed_injections)) != len(processed_injections)
        or len(set(processed_availability)) != len(processed_availability)
    ):
        raise ValueError("locked streaming completion repeats or omits physical rows")

    result = {
        "status": "completed_locked_o4b_streaming_execution_audit",
        "passed": True,
        "scientific_claim_allowed": False,
        "all_predeclared_shards_reduced": True,
        "negative_and_null_results_retained": True,
        "result_dependent_stopping_used": False,
        "post_access_dq_replacement_used": False,
        "expected_shards": len(shards),
        "completed_shards": len(receipts),
        "failed_shards": [],
        "rows": len(processed_injections),
        "unique_injections": len(set(processed_injections)),
        "unique_availability_blocks": len(set(processed_availability)),
        "execution_plan": {
            "path": str(plan_file),
            "sha256": file_sha256(plan_file),
        },
        "access_log": {
            "path": str(access_file),
            "sha256": file_sha256(access_file),
        },
        "receipt_manifest": {
            "path": str(receipts_file),
            "sha256": file_sha256(receipts_file),
        },
        "artifact_inventory": artifact_inventory,
        "code_commit": plan["code_commit"],
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = plan["code_commit"]
    atomic_write_json(target, result)
    return result


def freeze_locked_evaluation_suite_plan(
    validation_evidence_report_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    code_commit: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze every final output and endpoint before one-time locked-corpus access."""

    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("Locked evaluation suite plans are immutable")
    evidence_path = Path(validation_evidence_report_path).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "publication_evidence_ready"
        or evidence.get("publication_ready") is not True
        or evidence.get("phase") != "validation_freeze"
        or evidence.get("scientific_claim_allowed") is not False
        or evidence.get("summary", {}).get("required_pending") != 0
        or evidence.get("summary", {}).get("required_failed") != 0
        or evidence.get("summary", {}).get("required_passed")
        != evidence.get("summary", {}).get("required_total")
    ):
        raise ValueError("Locked suite requires a complete validation-freeze evidence audit")
    config = load_yaml(config_path)
    settings = config.get("locked_evaluation_suite")
    if not isinstance(settings, dict) or settings.get("schema") != "locked_suite_v2":
        raise ValueError("Configuration requires locked_evaluation_suite schema v2")
    if str(settings.get("required_split")) != "test":
        raise ValueError("Locked evaluation suite must use the test split")
    if settings.get("observing_runs") != ["O4b"]:
        raise ValueError("Locked evaluation suite must remain restricted to O4b")
    if settings.get("catalog_release") != "GWTC-5.0":
        raise ValueError("Locked evaluation suite must predeclare GWTC-5.0")
    required_frozen_artifacts = settings.get("required_frozen_artifacts")
    if (
        not isinstance(required_frozen_artifacts, list)
        or len(required_frozen_artifacts)
        != len(_LOCKED_SUITE_REQUIRED_FROZEN_ARTIFACTS)
        or set(str(value) for value in required_frozen_artifacts)
        != _LOCKED_SUITE_REQUIRED_FROZEN_ARTIFACTS
    ):
        raise ValueError("Locked suite frozen-artifact inventory is incomplete")
    if not code_commit.strip():
        raise ValueError("Locked evaluation suite requires an exact code commit")
    outputs = settings.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != _LOCKED_SUITE_OUTPUT_KEYS:
        raise ValueError("Locked evaluation suite output inventory is incomplete")
    inputs = settings.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _LOCKED_SUITE_INPUT_KEYS:
        raise ValueError("Locked evaluation suite input inventory is incomplete")
    root = Path(output_root).resolve()

    def resolve_inventory(values: dict[str, Any], inventory: str) -> dict[str, str]:
        resolved_values = {}
        for key, relative_value in sorted(values.items()):
            relative = Path(str(relative_value))
            if relative.is_absolute() or ".." in relative.parts or relative.name == "":
                raise ValueError(f"Locked suite {inventory} must be a safe relative path: {key}")
            resolved = (root / relative).resolve()
            if root not in resolved.parents or resolved.exists():
                raise ValueError(
                    f"Locked suite {inventory} exists or escapes its root: {key}"
                )
            resolved_values[key] = str(resolved)
        return resolved_values

    resolved_outputs = resolve_inventory(outputs, "output")
    resolved_inputs = resolve_inventory(inputs, "input")
    all_paths = [*resolved_outputs.values(), *resolved_inputs.values()]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Locked evaluation suite input/output paths must be unique")
    endpoints = settings.get("endpoints")
    if not isinstance(endpoints, dict):
        raise ValueError("Locked evaluation suite requires predeclared endpoints")
    numeric_minima = {
        "target_far_per_year": 0.0,
        "minimum_test_live_time_years": 0.0,
        "minimum_test_injections": 0,
        "minimum_injection_gps_blocks": 1,
        "minimum_paired_pe_injections": 0,
        "minimum_locked_ood_rows": 0,
        "minimum_background_gps_blocks": 1,
        "minimum_background_shifts": 0,
        "bootstrap_replicates": 9999,
    }
    for field, lower in numeric_minima.items():
        value = endpoints.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= lower:
            raise ValueError(f"Locked evaluation endpoint is invalid: {field}")
    bootstrap_seed = endpoints.get("bootstrap_seed")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("Locked evaluation endpoint is invalid: bootstrap_seed")
    credible_level = endpoints.get("pe_credible_level")
    if (
        isinstance(credible_level, bool)
        or not isinstance(credible_level, (int, float))
        or not 0 < float(credible_level) < 1
    ):
        raise ValueError("Locked evaluation endpoint is invalid: pe_credible_level")
    if endpoints.get("primary_search_metric") != "paired_delta_recovered_vt_at_common_far":
        raise ValueError("Locked suite primary endpoint must be paired fixed-FAR recovered VT")
    if endpoints.get("threshold_policy") != "validation_frozen_no_test_retuning":
        raise ValueError("Locked suite must prohibit test threshold retuning")
    if (
        endpoints.get("background_dependence_uncertainty")
        != "physical_block_x_block_x_offset_pigeonhole_v1"
    ):
        raise ValueError("Locked suite must predeclare clustered background uncertainty")
    if (
        endpoints.get("uncertainty")
        != "gps_block_then_paired_injection_hierarchical_bootstrap_v1"
    ):
        raise ValueError("Locked suite must predeclare physical injection uncertainty")
    if endpoints.get("catalog_search_arm") not in {
        "raw_candidate_search",
        "mask_candidate_search",
    }:
        raise ValueError("Locked suite catalog search arm is invalid")
    result = {
        "status": "frozen_locked_evaluation_suite_plan",
        "passed": True,
        "scientific_claim_allowed": False,
        "locked_corpus_opened": False,
        "test_rows_read": 0,
        "candidate_scores_inspected": False,
        "schema": settings["schema"],
        "corpus_label": str(settings.get("corpus_label")),
        "required_split": "test",
        "observing_runs": ["O4b"],
        "catalog_release": "GWTC-5.0",
        "required_frozen_artifacts": sorted(
            _LOCKED_SUITE_REQUIRED_FROZEN_ARTIFACTS
        ),
        "code_commit": code_commit,
        "output_root": str(root),
        "outputs": resolved_outputs,
        "inputs": resolved_inputs,
        "endpoints": endpoints,
        "validation_evidence": {
            "path": str(evidence_path),
            "sha256": file_sha256(evidence_path),
        },
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": file_sha256(config_path),
            "canonical_hash": canonical_hash(config, 64),
        },
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = code_commit
    atomic_write_json(target, result)
    return result


def validate_locked_evaluation_suite_access(
    plan_path: str | Path,
    access_log_path: str | Path,
    output_key: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Replay the suite plan and one-time access receipt for one final output."""

    plan_file = Path(plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "frozen_locked_evaluation_suite_plan"
        or plan.get("passed") is not True
        or plan.get("locked_corpus_opened") is not False
        or plan.get("test_rows_read") != 0
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("test_metrics") is not None
        or access.get("code_commit") != plan.get("code_commit")
        or access.get("corpus_label") != plan.get("corpus_label")
    ):
        raise ValueError("Locked suite plan or one-time access receipt is invalid")
    frozen = access.get("frozen_artifacts", {}).get("locked_suite_plan", {})
    frozen_artifacts = access.get("frozen_artifacts", {})
    if (
        Path(str(frozen.get("path", ""))).resolve() != plan_file
        or frozen.get("sha256") != file_sha256(plan_file)
        or access.get("predeclared_evaluation_output")
        != plan.get("outputs", {}).get("suite_receipt")
        or not set(plan.get("required_frozen_artifacts", {})).issubset(
            frozen_artifacts
        )
    ):
        raise ValueError("One-time access receipt does not bind the frozen suite plan")
    expected = plan.get("outputs", {}).get(output_key)
    if expected is None or Path(str(expected)).resolve() != Path(output_path).resolve():
        raise ValueError("Locked evaluator output is not predeclared by the suite plan")
    return {
        "plan_path": str(plan_file),
        "plan_sha256": file_sha256(plan_file),
        "access_log_path": str(access_file),
        "access_log_sha256": file_sha256(access_file),
        "output_key": output_key,
        "output_path": str(Path(output_path).resolve()),
        "code_commit": plan["code_commit"],
        "corpus_label": plan["corpus_label"],
        "endpoints": plan["endpoints"],
        "inputs": plan["inputs"],
        "frozen_artifacts": frozen_artifacts,
    }


def validate_locked_evaluation_suite_input(
    plan_path: str | Path,
    input_key: str,
    input_path: str | Path,
) -> dict[str, Any]:
    """Require a locked intermediate artifact to use its predeclared suite path."""

    plan_file = Path(plan_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    expected = plan.get("inputs", {}).get(input_key)
    if (
        plan.get("status") != "frozen_locked_evaluation_suite_plan"
        or plan.get("passed") is not True
        or plan.get("locked_corpus_opened") is not False
        or expected is None
        or Path(str(expected)).resolve() != Path(input_path).resolve()
    ):
        raise ValueError("Locked evaluator input is not predeclared by the suite plan")
    return {
        "plan_path": str(plan_file),
        "plan_sha256": file_sha256(plan_file),
        "input_key": input_key,
        "input_path": str(Path(input_path).resolve()),
        "code_commit": plan["code_commit"],
        "corpus_label": plan["corpus_label"],
    }


def finalize_locked_evaluation_suite_receipt(
    plan_path: str | Path,
    access_log_path: str | Path,
    streaming_completion_audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash every predeclared locked output into one immutable completion receipt."""

    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("locked evaluation suite receipts are immutable")
    suite_binding = validate_locked_evaluation_suite_access(
        plan_path, access_log_path, "suite_receipt", target
    )
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    access_file = Path(access_log_path).resolve()
    streaming_file = Path(streaming_completion_audit_path).resolve()
    if not streaming_file.is_file():
        raise FileNotFoundError("locked streaming completion audit is absent")
    streaming = json.loads(streaming_file.read_text(encoding="utf-8"))
    frozen_execution = suite_binding["frozen_artifacts"].get("locked_execution_plan", {})
    if (
        streaming.get("status")
        != "completed_locked_o4b_streaming_execution_audit"
        or streaming.get("passed") is not True
        or streaming.get("all_predeclared_shards_reduced") is not True
        or streaming.get("negative_and_null_results_retained") is not True
        or streaming.get("result_dependent_stopping_used") is not False
        or streaming.get("post_access_dq_replacement_used") is not False
        or streaming.get("failed_shards") != []
        or streaming.get("completed_shards") != streaming.get("expected_shards")
        or streaming.get("code_commit") != suite_binding["code_commit"]
        or streaming.get("execution_plan", {}).get("path")
        != frozen_execution.get("path")
        or streaming.get("execution_plan", {}).get("sha256")
        != frozen_execution.get("sha256")
        or streaming.get("access_log", {}).get("path") != str(access_file)
        or streaming.get("access_log", {}).get("sha256") != file_sha256(access_file)
    ):
        raise ValueError("locked suite lacks a complete all-shard streaming audit")
    expected_statuses = {
        "raw_candidate_search": "locked_candidate_search_evaluation",
        "mask_candidate_search": "locked_candidate_search_evaluation",
        "paired_raw_mask_search": (
            "locked_paired_raw_mask_candidate_search_comparison"
        ),
        "locked_ood_transfer": "locked_detector_set_ood_transfer_evaluation",
        "dingo_batch": "locked_dingo_paired_pe_batch_complete",
        "amplfi_batch": "locked_amplfi_paired_pe_batch_complete",
        "paired_pe_portfolio": "locked_paired_pe_robustness_portfolio_complete",
        "catalog_diagnostic": "locked_gwtc5_catalog_diagnostic",
    }
    expected_inputs = {
        "raw_candidate_search": {
            "time_slide": "raw_test_time_slide_report",
            "background_manifest": "raw_test_background_manifest",
            "injection_ranking": "raw_test_injection_ranking_report",
        },
        "mask_candidate_search": {
            "time_slide": "mask_test_time_slide_report",
            "background_manifest": "mask_test_background_manifest",
            "injection_ranking": "mask_test_injection_ranking_report",
        },
        "locked_ood_transfer": {
            "source_manifest": "locked_ood_source_manifest",
            "score_manifest": "locked_ood_score_manifest",
            "score_report": "locked_ood_score_report",
        },
        "dingo_batch": {"single": "dingo_locked_source_batch_report"},
        "amplfi_batch": {"single": "amplfi_locked_source_batch_report"},
        "catalog_diagnostic": {
            "catalog_source_manifest": "catalog_source_manifest",
            "catalog_candidate_manifest": "catalog_candidate_manifest",
            "catalog_candidate_report": "catalog_candidate_report",
            "catalog_prediction_manifest": "catalog_prediction_manifest",
            "catalog_prediction_report": "catalog_prediction_report",
        },
    }
    outputs = {}
    endpoint_outcomes = {}
    for key, expected_status in expected_statuses.items():
        path = Path(plan["outputs"][key]).resolve()
        binding = validate_locked_evaluation_suite_access(
            plan_path, access_log_path, key, path
        )
        if not path.is_file():
            raise FileNotFoundError(f"predeclared locked suite output is missing: {key}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"locked suite output is not readable JSON: {key}") from error
        if (
            not isinstance(report, dict)
            or report.get("status") != expected_status
            or report.get("locked_suite_access") != binding
        ):
            raise ValueError(f"locked suite output failed plan replay: {key}")
        if key in expected_inputs:
            replayed_inputs = {
                alias: validate_locked_evaluation_suite_input(
                    plan_path,
                    input_key,
                    plan["inputs"][input_key],
                )
                for alias, input_key in expected_inputs[key].items()
            }
            observed_inputs = report.get("locked_suite_inputs")
            if "single" in replayed_inputs:
                observed_inputs = {"single": report.get("locked_suite_input")}
            if observed_inputs != replayed_inputs:
                raise ValueError(f"locked suite output lacks its frozen input binding: {key}")
        outputs[key] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "status": expected_status,
        }
        endpoint_outcomes[key] = {
            field: report[field]
            for field in (
                "candidate_endpoint_gates_passed",
                "endpoint_complete",
                "promote_to_paper",
                "primary_endpoint_result",
            )
            if field in report
        }
    result = {
        "status": "completed_locked_evaluation_suite_receipt",
        "passed": True,
        "scientific_claim_allowed": False,
        "all_predeclared_outputs_present": len(outputs) == len(expected_statuses),
        "negative_and_null_results_retained": True,
        "protocol": (
            "hash every predeclared output without filtering on endpoint direction or "
            "statistical significance"
        ),
        "outputs": outputs,
        "endpoint_outcomes": endpoint_outcomes,
        "streaming_completion_audit": {
            "path": str(streaming_file),
            "sha256": file_sha256(streaming_file),
            "completed_shards": streaming["completed_shards"],
            "rows": streaming["rows"],
        },
        "locked_suite_access": suite_binding,
        "code_commit": suite_binding["code_commit"],
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = suite_binding["code_commit"]
    result["exact_command"] = result["runtime_provenance"]["exact_command"]
    result["environment"] = result["runtime_provenance"]["environment"]
    atomic_write_json(target, result)
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"evaluation manifest cannot be empty: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"evaluation manifest must contain JSON objects: {path}")
    return rows


def _exclusive_atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete JSON file exactly once using an atomic hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"evaluation corpus was already opened: {path}"
            ) from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def open_evaluation_corpus_once(
    freeze_report_path: str | Path,
    code_commit: str,
    frozen_artifacts: dict[str, str | Path],
    comparison_manifests: tuple[str | Path, ...],
    evaluation_output_path: str | Path,
    evaluation_command: str,
    overlap_fields: tuple[str, ...] = (
        "injection_id",
        "waveform_id",
        "gps_block",
        "glitch_id",
    ),
) -> dict[str, Any]:
    """Irreversibly record the first authorized access to a locked corpus.

    This command does not calculate test metrics. It is the one-time gate immediately
    before score extraction and records every frozen analysis dependency by SHA-256.
    """
    freeze_path = Path(freeze_report_path).resolve()
    with freeze_path.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "locked_evaluation_corpus_unopened":
        raise ValueError("evaluation freeze report is not an unopened corpus contract")
    if not code_commit.strip() or not evaluation_command.strip():
        raise ValueError("code commit and exact evaluation command must be frozen")
    if not comparison_manifests or not overlap_fields:
        raise ValueError("comparison manifests and overlap fields are required")
    missing_artifacts = sorted(_REQUIRED_FROZEN_ARTIFACTS - set(frozen_artifacts))
    if missing_artifacts:
        raise ValueError(f"missing frozen evaluation artifacts: {missing_artifacts}")

    # A suite-bound opening is irreversible, so reject an incomplete final
    # dependency inventory before the exclusive access receipt is written.
    if "locked_suite_plan" in frozen_artifacts:
        suite_plan_path = Path(frozen_artifacts["locked_suite_plan"]).resolve()
        suite_plan = json.loads(suite_plan_path.read_text(encoding="utf-8"))
        required_suite_artifacts = set(
            map(str, suite_plan.get("required_frozen_artifacts", []))
        ) | {"locked_suite_plan"}
        missing_suite_artifacts = sorted(required_suite_artifacts - set(frozen_artifacts))
        if (
            suite_plan.get("status") != "frozen_locked_evaluation_suite_plan"
            or suite_plan.get("passed") is not True
            or suite_plan.get("code_commit") != code_commit
            or suite_plan.get("corpus_label") != freeze.get("corpus_label")
            or missing_suite_artifacts
        ):
            raise ValueError(
                "locked suite opening lacks its complete frozen artifact inventory: "
                f"{missing_suite_artifacts}"
            )
        execution_path = Path(frozen_artifacts["locked_execution_plan"]).resolve()
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if (
            execution.get("status")
            != "frozen_locked_o4b_streaming_execution_plan"
            or execution.get("passed") is not True
            or execution.get("evaluation_opened") is not False
            or execution.get("candidate_scores_inspected") is not False
            or execution.get("test_strain_rows_read") != 0
            or execution.get("code_commit") != code_commit
            or execution.get("corpus_label") != freeze.get("corpus_label")
            or execution.get("freeze_identity", {}).get("suite_plan_sha256")
            != file_sha256(suite_plan_path)
            or execution.get("freeze_identity", {}).get("corpus_freeze_sha256")
            != file_sha256(freeze_path)
            or Path(str(execution.get("access_log_path", ""))).resolve()
            != Path(str(freeze.get("access_log_path", ""))).resolve()
        ):
            raise ValueError("locked streaming execution plan failed pre-access replay")

    manifest = Path(freeze["manifest_path"]).resolve()
    if file_sha256(manifest) != freeze["manifest_sha256"]:
        raise ValueError("locked evaluation manifest changed after freezing")
    test_rows = _load_jsonl(manifest)
    expected_split = str(freeze["expected_split"])
    if any(str(row.get("split")) != expected_split for row in test_rows):
        raise ValueError("locked evaluation manifest split changed after freezing")

    artifact_hashes: dict[str, dict[str, str]] = {}
    for label, raw_path in sorted(frozen_artifacts.items()):
        if not label.strip():
            raise ValueError("frozen artifact labels cannot be empty")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact does not exist: {path}")
        artifact_hashes[label] = {"path": str(path), "sha256": file_sha256(path)}

    test_values = {
        field: {str(row[field]) for row in test_rows if row.get(field) is not None}
        for field in overlap_fields
    }
    manifest_audits = []
    for raw_path in comparison_manifests:
        path = Path(raw_path).resolve()
        rows = _load_jsonl(path)
        compared: dict[str, dict[str, int]] = {}
        overlaps: dict[str, list[str]] = {}
        for field in overlap_fields:
            other = {str(row[field]) for row in rows if row.get(field) is not None}
            if not test_values[field] or not other:
                continue
            shared = sorted(test_values[field] & other)
            compared[field] = {
                "locked_unique": len(test_values[field]),
                "comparison_unique": len(other),
                "overlap": len(shared),
            }
            if shared:
                overlaps[field] = shared[:20]
        if not compared:
            raise ValueError(
                f"comparison manifest has no auditable identity field: {path}"
            )
        if overlaps:
            raise ValueError(f"locked evaluation group overlap in {path}: {overlaps}")
        manifest_audits.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "fields": compared,
                "passed": True,
            }
        )

    access_log = Path(freeze["access_log_path"]).resolve()
    evaluation_output = Path(evaluation_output_path).resolve()
    if evaluation_output.exists():
        raise FileExistsError(
            f"predeclared evaluation output already exists: {evaluation_output}"
        )
    if evaluation_output == access_log:
        raise ValueError("evaluation output and access log must differ")
    report = {
        "status": "locked_evaluation_corpus_opened_once",
        "scientific_claim_allowed": False,
        "evaluation_opened": True,
        "test_metrics": None,
        "freeze_report_path": str(freeze_path),
        "freeze_report_sha256": file_sha256(freeze_path),
        "corpus_label": freeze["corpus_label"],
        "manifest_path": str(manifest),
        "manifest_sha256": freeze["manifest_sha256"],
        "rows": len(test_rows),
        "code_commit": code_commit,
        "frozen_artifacts": artifact_hashes,
        "comparison_manifest_audits": manifest_audits,
        "overlap_fields": list(overlap_fields),
        "predeclared_evaluation_output": str(evaluation_output),
        "predeclared_evaluation_command": evaluation_command,
        "protocol": (
            "irreversible one-time opening recorded immediately before locked score "
            "extraction; this receipt alone is not a scientific result"
        ),
        **execution_provenance(),
    }
    # Preserve the explicitly frozen code identity instead of allowing the runtime
    # environment variable to silently replace it.
    report["opening_command_provenance"] = {
        "runtime_code_commit": report.pop("code_commit"),
        "exact_command": report.pop("exact_command"),
        "environment": report.pop("environment"),
    }
    report["code_commit"] = code_commit
    _exclusive_atomic_json(access_log, report)
    return report


def freeze_evaluation_corpus(
    manifest_path: str | Path,
    output_path: str | Path,
    access_log_path: str | Path,
    corpus_label: str,
    expected_split: str = "test",
    minimum_rows: int = 1,
    group_fields: tuple[str, ...] = (
        "injection_id",
        "waveform_id",
        "gps_block",
        "source_family",
    ),
) -> dict[str, Any]:
    """Write an immutable, unopened evaluation-corpus identity contract."""
    manifest = Path(manifest_path).resolve()
    target = Path(output_path).resolve()
    access_log = Path(access_log_path).resolve()
    if not manifest.is_file() or not corpus_label.strip():
        raise ValueError("evaluation corpus freeze requires a manifest and label")
    if minimum_rows < 1 or not expected_split or not group_fields:
        raise ValueError("evaluation corpus freeze settings are invalid")
    if target == access_log:
        raise ValueError("evaluation freeze report and access log must differ")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) < minimum_rows:
        raise ValueError("evaluation corpus is smaller than the declared minimum")
    if any(str(row.get("split")) != expected_split for row in rows):
        raise ValueError("evaluation corpus contains rows outside the locked split")
    missing = {
        field: [index for index, row in enumerate(rows) if field not in row][:10]
        for field in group_fields
    }
    missing = {field: indices for field, indices in missing.items() if indices}
    if missing:
        raise ValueError(f"evaluation corpus lacks frozen group fields: {missing}")
    for identity_field in ("injection_id", "waveform_id"):
        if identity_field in group_fields:
            values = [str(row[identity_field]) for row in rows]
            if len(set(values)) != len(values):
                raise ValueError(
                    f"evaluation corpus repeats physical identity {identity_field}"
                )
    group_counts = {
        field: len({str(row[field]) for row in rows}) for field in group_fields
    }
    value_counts = {
        field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for field in group_fields
        if field in {"source_family", "observing_run", "ifo", "detector_subset"}
    }
    identity = {
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "access_log_path": str(access_log),
        "corpus_label": corpus_label,
        "expected_split": expected_split,
        "minimum_rows": minimum_rows,
        "group_fields": list(group_fields),
    }
    if target.is_file():
        completed = json.loads(target.read_text(encoding="utf-8"))
        if completed.get("freeze_identity") != identity:
            raise ValueError("existing evaluation freeze report has another identity")
        if file_sha256(manifest) != completed["manifest_sha256"]:
            raise ValueError("locked evaluation manifest changed after freezing")
        return completed
    if access_log.exists():
        raise FileExistsError("evaluation access log exists before corpus freezing")
    report = {
        "status": "locked_evaluation_corpus_unopened",
        "scientific_claim_allowed": False,
        "evaluation_opened": False,
        "test_metrics": None,
        "freeze_identity": identity,
        "corpus_label": corpus_label,
        "expected_split": expected_split,
        "rows": len(rows),
        "manifest_path": str(manifest),
        "manifest_sha256": identity["manifest_sha256"],
        "access_log_path": str(access_log),
        "access_log_exists": False,
        "group_fields": list(group_fields),
        "unique_group_counts": group_counts,
        "categorical_counts": value_counts,
        "opening_requirements": [
            "frozen code commit, config, model, threshold calibration and OOD policy hashes",
            "one-time locked evaluator that atomically writes the predeclared access log",
            "zero group overlap with every training/selection/calibration manifest",
        ],
        **execution_provenance(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def freeze_gwtc5_locked_corpus_contract(
    manifest_path: str | Path,
    inventory_report_path: str | Path,
    waveform_validation_report_path: str | Path,
    suite_config_path: str | Path,
    output_path: str | Path,
    access_log_path: str | Path,
) -> dict[str, Any]:
    """Freeze the exact score-blind GWTC-5/O4b test inventory without opening strain."""

    manifest = Path(manifest_path).resolve()
    inventory_report_file = Path(inventory_report_path).resolve()
    waveform_validation_file = Path(waveform_validation_report_path).resolve()
    config_path = Path(suite_config_path).resolve()
    target = Path(output_path).resolve()
    access_log = Path(access_log_path).resolve()
    if (
        not manifest.is_file()
        or not inventory_report_file.is_file()
        or not waveform_validation_file.is_file()
        or not config_path.is_file()
    ):
        raise FileNotFoundError("GWTC-5 corpus freeze inputs are absent")
    if target == access_log or access_log.exists():
        raise FileExistsError("GWTC-5 access log exists before corpus freezing")
    config = load_yaml(config_path)
    settings = config.get("locked_evaluation_suite")
    if (
        not isinstance(settings, dict)
        or settings.get("schema") != "locked_suite_v2"
        or settings.get("corpus_label") != "GWTC-5.0_O4b_locked_suite_v2"
        or settings.get("required_split") != "test"
        or settings.get("observing_runs") != ["O4b"]
        or settings.get("catalog_release") != "GWTC-5.0"
    ):
        raise ValueError("GWTC-5 corpus freeze requires the exact locked suite v2 identity")
    endpoints = settings.get("endpoints", {})
    minimum_rows = endpoints.get("minimum_test_injections")
    required_detector_subsets = settings.get("endpoints", {}).get(
        "required_detector_subsets"
    )
    required_source_families = settings.get("endpoints", {}).get(
        "required_source_families"
    )
    required_stress_strata = settings.get("endpoints", {}).get(
        "required_stress_strata"
    )
    if (
        isinstance(minimum_rows, bool)
        or not isinstance(minimum_rows, int)
        or minimum_rows < 3000
        or not isinstance(required_detector_subsets, list)
        or not required_detector_subsets
        or not isinstance(required_source_families, list)
        or not required_source_families
        or not isinstance(required_stress_strata, list)
        or not required_stress_strata
    ):
        raise ValueError("GWTC-5 suite lacks required frozen corpus strata")
    inventory_report = json.loads(inventory_report_file.read_text(encoding="utf-8"))
    population_config_path = Path(
        str(inventory_report.get("population_config_path", ""))
    ).resolve()
    availability_manifest_path = Path(
        str(inventory_report.get("availability_manifest_path", ""))
    ).resolve()
    availability_report_path = Path(
        str(inventory_report.get("availability_report_path", ""))
    ).resolve()
    if (
        inventory_report.get("status")
        != "score_blind_gwtc5_locked_injection_inventory"
        or inventory_report.get("passed") is not True
        or inventory_report.get("physical_stress_predicates_passed") is not True
        or inventory_report.get("candidate_catalog_queried") is not False
        or inventory_report.get("candidate_scores_inspected") is not False
        or inventory_report.get("event_level_parameters_inspected") is not False
        or int(inventory_report.get("test_strain_files_downloaded", -1)) != 0
        or int(inventory_report.get("test_strain_bytes_read", -1)) != 0
        or int(inventory_report.get("test_strain_rows_read", -1)) != 0
        or inventory_report.get("pre_access_vt_weights_assigned") is not False
        or inventory_report.get("post_access_dq_replacement_allowed") is not False
        or Path(str(inventory_report.get("manifest_path", ""))).resolve() != manifest
        or inventory_report.get("manifest_sha256") != file_sha256(manifest)
        or Path(str(inventory_report.get("suite_config_path", ""))).resolve()
        != config_path
        or inventory_report.get("suite_config_sha256") != file_sha256(config_path)
        or Path(str(inventory_report.get("access_log_path", ""))).resolve()
        != access_log
        or not population_config_path.is_file()
        or inventory_report.get("population_config_sha256")
        != file_sha256(population_config_path)
        or not availability_manifest_path.is_file()
        or inventory_report.get("availability_manifest_sha256")
        != file_sha256(availability_manifest_path)
        or not availability_report_path.is_file()
        or inventory_report.get("availability_report_sha256")
        != file_sha256(availability_report_path)
    ):
        raise ValueError("GWTC-5 corpus freeze requires a source-safe injection producer")
    rows = _load_jsonl(manifest)
    if len(rows) < minimum_rows:
        raise ValueError("GWTC-5 locked corpus is below the predeclared injection floor")
    required_fields = {
        "split",
        "injection_id",
        "waveform_id",
        "gps_block",
        "source_family",
        "observing_run",
        "detector_subset",
        "catalog_release",
        "stress_strata",
    }
    missing = [index for index, row in enumerate(rows) if required_fields - set(row)]
    if missing:
        raise ValueError(f"GWTC-5 locked corpus rows lack frozen strata: {missing[:10]}")
    if any(str(row["split"]) != "test" for row in rows):
        raise ValueError("GWTC-5 locked corpus mixes data outside the test split")
    if any(str(row["observing_run"]) != "O4b" for row in rows):
        raise ValueError("GWTC-5 locked corpus contains a non-O4b row")
    if any(str(row["catalog_release"]) != "GWTC-5.0" for row in rows):
        raise ValueError("GWTC-5 locked corpus contains another catalog release")
    for identity_field in ("injection_id", "waveform_id"):
        identities = [str(row[identity_field]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f"GWTC-5 locked corpus repeats {identity_field}")
    forbidden_fields = {
        "candidate_score",
        "ranking_statistic",
        "posterior_samples",
        "model_prediction",
        "selected_threshold",
        "test_metric",
        "recovered",
        "far",
        "ifar",
        "strain",
        "time_series",
        "q_transform",
        "features",
    }
    exposed = sorted(
        {
            field
            for row in rows
            for field in forbidden_fields
            if field in row
        }
    )
    if exposed:
        raise ValueError(f"GWTC-5 freeze manifest exposes selection/result fields: {exposed}")
    waveform_validation = json.loads(
        waveform_validation_file.read_text(encoding="utf-8")
    )
    waveform_runtime_receipt_file = Path(
        str(waveform_validation.get("runtime_receipt_path", ""))
    ).resolve()
    if not waveform_runtime_receipt_file.is_file():
        raise ValueError("GWTC-5 waveform runtime receipt is absent")
    waveform_runtime_receipt = json.loads(
        waveform_runtime_receipt_file.read_text(encoding="utf-8")
    )
    runtime_requirements_path = Path(
        str(waveform_runtime_receipt.get("requirements_path", ""))
    ).resolve()
    frozen_packages = waveform_runtime_receipt.get("pip_freeze")
    frozen_text = (
        "\n".join(map(str, frozen_packages)) + "\n"
        if isinstance(frozen_packages, list)
        else ""
    )
    expected_waveform_strata: Counter[str] = Counter()
    expected_approximants: set[str] = set()
    for row in rows:
        family = str(row["source_family"])
        primary = str(row.get("waveform_approximant", ""))
        if not primary:
            raise ValueError("GWTC-5 injection lacks a primary waveform approximant")
        expected_approximants.add(primary)
        expected_waveform_strata[f"{family}:primary:{primary}"] += 1
        alternative = row.get("alternative_waveform_approximant")
        if alternative:
            expected_approximants.add(str(alternative))
            expected_waveform_strata[f"{family}:alternative:{alternative}"] += 1
    observed_case_strata = waveform_validation.get("case_strata", {})
    if (
        waveform_validation.get("passed") is not True
        or waveform_validation.get("validation_scope")
        != "external_reference_waveform_equivalence"
        or waveform_validation.get("selection_mode") != "family_approximant"
        or waveform_validation.get("include_alternatives") is not True
        or Path(str(waveform_validation.get("recipe_manifest_path", ""))).resolve()
        != manifest
        or waveform_validation.get("recipe_manifest_sha256") != file_sha256(manifest)
        or set(waveform_validation.get("approximants", [])) != expected_approximants
        or set(observed_case_strata) != set(expected_waveform_strata)
        or any(int(value) < 3 for value in observed_case_strata.values())
        or int(waveform_validation.get("selected_cases", -1))
        != sum(int(value) for value in observed_case_strata.values())
        or not waveform_validation.get("versions", {}).get("pycbc")
        or not waveform_validation.get("versions", {}).get("lalsuite")
        or not waveform_validation.get("cases")
        or any(case.get("passed") is not True for case in waveform_validation["cases"])
        or waveform_validation.get("runtime_receipt_bound") is not True
        or waveform_validation.get("runtime_receipt_sha256")
        != file_sha256(waveform_runtime_receipt_file)
        or waveform_runtime_receipt.get("status")
        != "verified_isolated_waveform_runtime"
        or waveform_runtime_receipt.get("passed") is not True
        or waveform_runtime_receipt.get("pycbc_version")
        != waveform_validation.get("versions", {}).get("pycbc")
        or waveform_runtime_receipt.get("lalsuite_version")
        != waveform_validation.get("versions", {}).get("lalsuite")
        or waveform_runtime_receipt.get("code_commit")
        != waveform_validation.get("code_commit")
        or (
            os.environ.get("GWYOLO_CODE_COMMIT")
            and waveform_runtime_receipt.get("code_commit")
            != os.environ["GWYOLO_CODE_COMMIT"]
        )
        or Path(str(waveform_runtime_receipt.get("python_executable", ""))).resolve()
        != Path(
            str(waveform_validation.get("environment", {}).get("python_executable", ""))
        ).resolve()
        or not runtime_requirements_path.is_file()
        or waveform_runtime_receipt.get("requirements_sha256")
        != file_sha256(runtime_requirements_path)
        or waveform_validation.get("requirements_sha256")
        != waveform_runtime_receipt.get("requirements_sha256")
        or not frozen_text
        or waveform_runtime_receipt.get("pip_freeze_sha256")
        != hashlib.sha256(frozen_text.encode()).hexdigest()
        or waveform_validation.get("pip_freeze_sha256")
        != waveform_runtime_receipt.get("pip_freeze_sha256")
        or not expected_approximants
        <= set(waveform_runtime_receipt.get("approximants", {}))
    ):
        raise ValueError("GWTC-5 waveform runtime validation is incomplete")
    observed_detector_subsets = {str(row["detector_subset"]) for row in rows}
    observed_source_families = {str(row["source_family"]) for row in rows}
    observed_stress_strata: set[str] = set()
    for index, row in enumerate(rows):
        strata = row["stress_strata"]
        if (
            not isinstance(strata, list)
            or not strata
            or any(not isinstance(value, str) or not value for value in strata)
        ):
            raise ValueError(f"GWTC-5 stress strata are invalid at row {index}")
        observed_stress_strata.update(strata)
    detector_coverage = set(map(str, required_detector_subsets)) <= observed_detector_subsets
    family_coverage = set(map(str, required_source_families)) <= observed_source_families
    stress_coverage = set(map(str, required_stress_strata)) <= observed_stress_strata
    if not detector_coverage or not family_coverage or not stress_coverage:
        raise ValueError("GWTC-5 locked corpus does not cover every predeclared stratum")
    from .locked_injections import audit_gwtc5_locked_injection_rows

    availability_rows = _load_jsonl(availability_manifest_path)
    population_settings = load_yaml(population_config_path).get(
        "gwtc5_locked_injection_population"
    )
    if not isinstance(population_settings, dict):
        raise ValueError("GWTC-5 population config is invalid")
    physical_audit = audit_gwtc5_locked_injection_rows(
        rows,
        availability_rows,
        settings,
        population_settings,
    )
    if physical_audit != inventory_report.get("audit"):
        raise ValueError("GWTC-5 physical injection audit differs from producer evidence")

    group_fields = [
        "injection_id",
        "waveform_id",
        "gps_block",
        "source_family",
        "observing_run",
        "detector_subset",
    ]
    identity = {
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "inventory_report_path": str(inventory_report_file),
        "inventory_report_sha256": file_sha256(inventory_report_file),
        "waveform_validation_report_path": str(waveform_validation_file),
        "waveform_validation_report_sha256": file_sha256(waveform_validation_file),
        "waveform_runtime_receipt_path": str(waveform_runtime_receipt_file),
        "waveform_runtime_receipt_sha256": file_sha256(
            waveform_runtime_receipt_file
        ),
        "suite_config_path": str(config_path),
        "suite_config_sha256": file_sha256(config_path),
        "access_log_path": str(access_log),
        "corpus_label": settings["corpus_label"],
        "expected_split": "test",
        "minimum_rows": minimum_rows,
        "group_fields": group_fields,
    }
    if target.is_file():
        completed = json.loads(target.read_text(encoding="utf-8"))
        if completed.get("freeze_identity") != identity:
            raise ValueError("existing GWTC-5 freeze report has another identity")
        return completed
    report = {
        "status": "locked_evaluation_corpus_unopened",
        "locked_suite_schema": "locked_suite_v2",
        "corpus_label": settings["corpus_label"],
        "catalog_release": "GWTC-5.0",
        "observing_runs": ["O4b"],
        "expected_split": "test",
        "scientific_claim_allowed": False,
        "evaluation_opened": False,
        "test_metrics": None,
        "candidate_scores_inspected": False,
        "test_strain_rows_read": 0,
        "test_manifest_metadata_rows_read": len(rows),
        "selection_or_result_fields_present": False,
        "inventory_producer_bound": True,
        "physical_stress_predicates_passed": True,
        "waveform_runtime_validation_bound": True,
        "one_injection_per_frozen_gps_block": True,
        "pre_access_vt_weights_assigned": False,
        "post_access_dq_replacement_allowed": False,
        "freeze_identity": identity,
        "rows": len(rows),
        "minimum_test_injections": minimum_rows,
        "manifest_path": str(manifest),
        "manifest_sha256": identity["manifest_sha256"],
        "inventory_report_path": str(inventory_report_file),
        "inventory_report_sha256": identity["inventory_report_sha256"],
        "waveform_validation_report_path": str(waveform_validation_file),
        "waveform_validation_report_sha256": identity[
            "waveform_validation_report_sha256"
        ],
        "waveform_runtime_receipt_path": str(waveform_runtime_receipt_file),
        "waveform_runtime_receipt_sha256": identity[
            "waveform_runtime_receipt_sha256"
        ],
        "availability_manifest_path": str(availability_manifest_path),
        "availability_manifest_sha256": file_sha256(availability_manifest_path),
        "availability_report_path": str(availability_report_path),
        "availability_report_sha256": file_sha256(availability_report_path),
        "population_config_path": str(population_config_path),
        "population_config_sha256": file_sha256(population_config_path),
        "suite_config_path": str(config_path),
        "suite_config_sha256": identity["suite_config_sha256"],
        "access_log_path": str(access_log),
        "access_log_exists": False,
        "required_detector_subsets_covered": detector_coverage,
        "required_source_families_covered": family_coverage,
        "required_stress_strata_covered": stress_coverage,
        "observed_detector_subsets": sorted(observed_detector_subsets),
        "observed_source_families": sorted(observed_source_families),
        "observed_stress_strata": sorted(observed_stress_strata),
        "physical_inventory_audit": physical_audit,
        "unique_group_counts": {
            field: len({str(row[field]) for row in rows}) for field in group_fields
        },
        "manifest_fields_used_for_freeze": sorted(required_fields),
        "opening_requirements": [
            "validation evidence ledger at 10/10",
            "frozen suite plan and analysis artifact hashes",
            "exclusive one-time access log immediately before strain scoring",
        ],
        **execution_provenance(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report
