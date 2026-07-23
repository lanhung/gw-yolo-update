from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    load_yaml,
)
from .pe import (
    PAIRED_PE_LATENCY_SCOPE_V1,
    posterior_sky_area_equal_solid_angle,
    sky_area_estimator_identity,
    validate_paired_pe_latency,
)
from .runtime import execution_provenance


def _parse_block(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) != 3 or fields[0] != "gps":
        raise ValueError(f"invalid GPS block identity: {value}")
    start, duration = int(fields[1]), int(fields[2])
    if duration <= 0:
        raise ValueError(f"invalid GPS block duration: {value}")
    return start, duration


def _contiguous_runs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    windows = sorted((int(row["gps_start"]), int(row["gps_end"])) for row in rows)
    if any(end <= start for start, end in windows):
        raise ValueError("background window has non-positive duration")
    runs: list[list[int]] = []
    for start, end in windows:
        if not runs or start > runs[-1][1]:
            runs.append([start, end])
        elif start == runs[-1][1]:
            runs[-1][1] = end
        else:
            raise ValueError("background windows overlap within a GPS block")
    return [(start, end) for start, end in runs]


def _read_source_segment(
    path: Path,
    start: int,
    end: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("AMPLFI background export requires h5py") from error
    with h5py.File(path, "r") as handle:
        dataset = handle["strain/Strain"]
        spacing = float(dataset.attrs["Xspacing"])
        source_start = float(dataset.attrs["Xstart"])
        source_rate = int(round(1.0 / spacing))
        first = int(round((start - source_start) * source_rate))
        last = int(round((end - source_start) * source_rate))
        if first < 0 or last > dataset.shape[0]:
            raise ValueError(f"requested GPS run is outside source file: {path}")
        values = np.asarray(dataset[first:last], dtype=np.float64)
    if source_rate == target_sample_rate:
        return values, source_rate
    if source_rate % target_sample_rate != 0:
        raise ValueError("AMPLFI export currently requires an integer downsample ratio")
    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise RuntimeError("downsampled AMPLFI export requires scipy.signal.resample_poly") from error
    ratio = source_rate // target_sample_rate
    values = resample_poly(values, up=1, down=ratio, window=("kaiser", 8.6))
    expected = (end - start) * target_sample_rate
    if values.size != expected:
        raise ValueError("resampled AMPLFI background has an unexpected length")
    return np.asarray(values, dtype=np.float64), source_rate


def export_amplfi_group_safe_background(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    target_sample_rate: int = 2048,
    minimum_segment_seconds: int = 16,
    required_ifos: tuple[str, ...] = ("H1", "L1"),
) -> dict[str, Any]:
    if target_sample_rate <= 0 or minimum_segment_seconds <= 0:
        raise ValueError("target sample rate and minimum segment duration must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("AMPLFI background export manifest is empty")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    block_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported AMPLFI background split: {split}")
        if tuple(row.get("ifos", [])) != required_ifos:
            continue
        block = str(row.get("gps_block"))
        _parse_block(block)
        pair_id = str(row.get("pair_id"))
        if not pair_id:
            raise ValueError("AMPLFI background row lacks pair_id")
        block_splits[block].add(split)
        groups[(split, block, pair_id)].append(row)
    leaked = sorted(block for block, splits in block_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"GPS blocks cross AMPLFI export splits: {leaked[:5]}")
    if not groups:
        raise ValueError("manifest contains no rows for the required IFO set")

    source_hashes: dict[str, str] = {}
    output_root = Path(output_dir).resolve()
    records = []
    split_durations: dict[str, float] = defaultdict(float)
    for (split, block, pair_id), group_rows in sorted(groups.items()):
        block_start, block_duration = _parse_block(block)
        for start, end in _contiguous_runs(group_rows):
            if start < block_start or end > block_start + block_duration:
                raise ValueError("background run escapes its declared GPS block")
            duration = end - start
            if duration < minimum_segment_seconds:
                continue
            arrays = {}
            source_records = {}
            observed_source_rates = set()
            source_files = group_rows[0].get("source_files")
            if not isinstance(source_files, dict):
                raise ValueError("AMPLFI background row lacks source_files")
            for ifo in required_ifos:
                identity = source_files.get(ifo)
                if not isinstance(identity, dict):
                    raise ValueError(f"AMPLFI background lacks {ifo} source identity")
                path = Path(str(identity.get("path", ""))).resolve()
                expected_hash = str(identity.get("sha256", ""))
                if not path.is_file():
                    raise FileNotFoundError(f"AMPLFI source strain is absent: {path}")
                observed_hash = source_hashes.setdefault(str(path), file_sha256(path))
                if observed_hash != expected_hash:
                    raise ValueError(f"AMPLFI source strain hash mismatch: {path}")
                values, source_rate = _read_source_segment(
                    path, start, end, target_sample_rate
                )
                arrays[ifo] = values
                observed_source_rates.add(source_rate)
                source_records[ifo] = {"path": str(path), "sha256": observed_hash}
            if len(observed_source_rates) != 1:
                raise ValueError("AMPLFI source IFO sample rates differ")
            split_dir = "validation" if split == "val" else split
            target_dir = output_root / split_dir / "background"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"gwyolo-{start}-{duration}.hdf5"
            if target.exists():
                raise FileExistsError(f"refusing to overwrite AMPLFI background: {target}")
            try:
                import h5py
            except ImportError as error:
                raise RuntimeError("AMPLFI background export requires h5py") from error
            temporary = target.with_suffix(target.suffix + ".part")
            if temporary.exists():
                raise FileExistsError(f"stale AMPLFI background temporary exists: {temporary}")
            with h5py.File(temporary, "w") as handle:
                handle.attrs.update(
                    {
                        "gps_start": start,
                        "duration_seconds": duration,
                        "gps_block": block,
                        "pair_id": pair_id,
                        "split": split,
                        "target_sample_rate_hz": target_sample_rate,
                        "source_sample_rate_hz": observed_source_rates.pop(),
                    }
                )
                for ifo in required_ifos:
                    dataset = handle.create_dataset(
                        ifo,
                        data=arrays[ifo],
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )
                    dataset.attrs["dx"] = 1.0 / target_sample_rate
                    dataset.attrs["x0"] = start
            temporary.replace(target)
            split_durations[split] += duration
            records.append(
                {
                    "path": str(target),
                    "sha256": file_sha256(target),
                    "split": split,
                    "gps_block": block,
                    "pair_id": pair_id,
                    "gps_start": start,
                    "duration_seconds": duration,
                    "source_files": source_records,
                }
            )
    if not records:
        raise ValueError("no contiguous AMPLFI background runs meet the minimum duration")
    return {
        "status": "group_safe_amplfi_background",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "output_dir": str(output_root),
        "required_ifos": list(required_ifos),
        "target_sample_rate_hz": target_sample_rate,
        "minimum_segment_seconds": minimum_segment_seconds,
        "resampling": "scipy.signal.resample_poly_kaiser_beta_8.6_when_needed",
        "source_file_count": len(source_hashes),
        "output_file_count": len(records),
        "split_file_counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val", "test")
        },
        "split_duration_seconds": {
            split: split_durations.get(split, 0.0) for split in ("train", "val", "test")
        },
        "cross_split_gps_block_overlap": 0,
        "files": records,
        **execution_provenance(),
    }


def run_amplfi_group_safe_background_export(
    manifest_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    target_sample_rate: int = 2048,
    minimum_segment_seconds: int = 16,
) -> dict[str, Any]:
    report = export_amplfi_group_safe_background(
        manifest_path,
        output_dir,
        target_sample_rate=target_sample_rate,
        minimum_segment_seconds=minimum_segment_seconds,
    )
    atomic_write_json(report_path, report)
    return report


def audit_amplfi_background_capacity(
    manifest_path: str | Path,
    policy_path: str | Path,
) -> dict[str, Any]:
    """Audit group-safe train/validation noise capacity without opening strain arrays."""

    policy = load_yaml(policy_path)
    if policy.get("schema_version") != 1:
        raise ValueError("AMPLFI background capacity policy schema_version must be 1")
    required_ifos = tuple(str(value) for value in policy.get("required_ifos", []))
    if not required_ifos:
        raise ValueError("AMPLFI background capacity policy requires detector identities")
    minimum_segment = int(policy.get("minimum_contiguous_segment_seconds", 0))
    minimum_duration = policy.get("minimum_duration_seconds", {})
    minimum_blocks = policy.get("minimum_gps_blocks", {})
    if (
        minimum_segment <= 0
        or not isinstance(minimum_duration, dict)
        or not isinstance(minimum_blocks, dict)
        or any(int(minimum_duration.get(split, 0)) <= 0 for split in ("train", "val"))
        or any(int(minimum_blocks.get(split, 0)) <= 0 for split in ("train", "val"))
    ):
        raise ValueError("AMPLFI background capacity thresholds must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("AMPLFI background capacity manifest is empty")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    block_splits: dict[str, set[str]] = defaultdict(set)
    rows_by_split: dict[str, int] = defaultdict(int)
    excluded_detector_rows = 0
    for row in rows:
        split = str(row.get("split"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported AMPLFI background split: {split}")
        block = str(row.get("gps_block"))
        _parse_block(block)
        block_splits[block].add(split)
        rows_by_split[split] += 1
        if tuple(row.get("ifos", [])) != required_ifos:
            excluded_detector_rows += 1
            continue
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError("AMPLFI background capacity row lacks pair_id")
        if split in {"train", "val"}:
            groups[(split, block, pair_id)].append(row)
    leaked = sorted(block for block, splits in block_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"GPS blocks cross AMPLFI capacity splits: {leaked[:5]}")

    durations: dict[str, float] = defaultdict(float)
    eligible_runs: dict[str, int] = defaultdict(int)
    blocks: dict[str, set[str]] = defaultdict(set)
    pairs: dict[str, set[str]] = defaultdict(set)
    for (split, block, pair_id), group_rows in sorted(groups.items()):
        for start, end in _contiguous_runs(group_rows):
            duration = end - start
            if duration < minimum_segment:
                continue
            durations[split] += duration
            eligible_runs[split] += 1
            blocks[split].add(block)
            pairs[split].add(pair_id)
    checks = {}
    for split in ("train", "val"):
        checks[split] = {
            "duration_seconds": durations.get(split, 0.0),
            "minimum_duration_seconds": int(minimum_duration[split]),
            "duration_passed": durations.get(split, 0.0) >= int(minimum_duration[split]),
            "gps_blocks": len(blocks[split]),
            "minimum_gps_blocks": int(minimum_blocks[split]),
            "gps_blocks_passed": len(blocks[split]) >= int(minimum_blocks[split]),
            "source_pairs": len(pairs[split]),
            "eligible_contiguous_runs": eligible_runs.get(split, 0),
        }
    passed = all(
        value["duration_passed"] and value["gps_blocks_passed"]
        for value in checks.values()
    )
    return {
        "status": (
            "amplfi_background_capacity_ready"
            if passed
            else "amplfi_background_capacity_insufficient"
        ),
        "passed": passed,
        "scientific_claim_allowed": False,
        "scientific_blocker": "capacity audit is not AMPLFI training or posterior evidence",
        "strain_arrays_read": 0,
        "test_strain_rows_read": 0,
        "test_metadata_rows_excluded": rows_by_split.get("test", 0),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "policy_path": str(Path(policy_path).resolve()),
        "policy_sha256": file_sha256(policy_path),
        "required_ifos": list(required_ifos),
        "minimum_contiguous_segment_seconds": minimum_segment,
        "input_rows_by_split": {
            split: rows_by_split.get(split, 0) for split in ("train", "val", "test")
        },
        "excluded_detector_rows": excluded_detector_rows,
        "cross_split_gps_block_overlap": 0,
        "checks": checks,
        **execution_provenance(),
    }


def run_amplfi_background_capacity_audit(
    manifest_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = audit_amplfi_background_capacity(manifest_path, policy_path)
    atomic_write_json(output_path, report)
    if not report["passed"]:
        raise RuntimeError(f"AMPLFI background capacity is insufficient; inspect {output_path}")
    return report


def merge_amplfi_streamed_background_extension(
    base_merge_report_path: str | Path,
    extension_plan_path: str | Path,
    shard_directories: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge a source-disjoint streamed extension with a verified base bank."""

    base_path = Path(base_merge_report_path).resolve()
    plan_path = Path(extension_plan_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    roots = [Path(value).resolve() for value in shard_directories]
    if (
        base.get("status") != "verified_streamed_amplfi_background_bank"
        or base.get("passed") is not True
        or base.get("recoverable") is not True
        or int(base.get("test_strain_rows_read", -1)) != 0
        or int(base.get("test_rows_exported", -1)) != 0
        or plan.get("status") != "development_acquisition_plan"
        or plan.get("selection_rule") != "stratified_exclusion_complement_v1"
        or plan.get("candidate_scores_inspected") is not False
        or plan.get("test_data_opened") is not False
        or plan.get("locked_evaluation_data") is not False
        or tuple(plan.get("detectors", [])) != ("H1", "L1")
        or str(plan.get("run")) != "O4a"
        or not roots
        or len(roots) != len(set(roots))
    ):
        raise ValueError("AMPLFI extension inputs are not score-blind streamed development data")
    base_manifest = Path(str(base.get("background_manifest_path", ""))).resolve()
    if base.get("background_manifest_sha256") != file_sha256(base_manifest):
        raise ValueError("AMPLFI base streamed manifest changed")
    parent_hash = str(base.get("parent_plan_sha256", ""))
    exclusion_hashes = {
        str(row.get("sha256", "")) for row in plan.get("exclusion_plans", [])
    }
    if not parent_hash or parent_hash not in exclusion_hashes:
        raise ValueError("AMPLFI extension plan does not exclude its streamed base plan")

    rows = [
        json.loads(line)
        for line in base_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("AMPLFI base streamed manifest is empty")
    base_pair_ids = {str(row.get("pair_id", "")) for row in rows}
    base_blocks = {str(row.get("gps_block", "")) for row in rows}
    if "" in base_pair_ids or "" in base_blocks:
        raise ValueError("AMPLFI base rows lack physical group identities")
    selected_plan_ids = {str(row["pair_id"]) for row in plan["pairs"]}
    selected_plan_gps = {
        str(row["pair_id"]): int(row["gps_start"]) for row in plan["pairs"]
    }
    extension_pair_ids: set[str] = set()
    extension_blocks: set[str] = set()
    shard_records = []
    exported_files: dict[str, str] = {}
    for root in roots:
        paths = {
            "plan": root / "acquisition_plan.json",
            "batch": root / "download/batch_download_report.json",
            "background": root / "background/background_plan_report.json",
            "export": root / "amplfi_export_report.json",
            "eviction": root / "source_eviction_report.json",
        }
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"AMPLFI extension shard is incomplete: {root}")
        values = {
            key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in paths.items()
        }
        shard = values["plan"]
        batch = values["batch"]
        background = values["background"]
        export = values["export"]
        eviction = values["eviction"]
        background_manifest = Path(str(background.get("manifest_path", ""))).resolve()
        shard_ids = {str(row["pair_id"]) for row in shard.get("pairs", [])}
        if (
            shard.get("status") != "development_acquisition_plan"
            or shard.get("locked_evaluation_data") is not False
            or shard.get("parent_plan_sha256") != file_sha256(plan_path)
            or not shard_ids
            or not shard_ids <= selected_plan_ids
            or batch.get("status") != "verified_development_strain_batch"
            or batch.get("passed") is not True
            or batch.get("plan_sha256") != file_sha256(paths["plan"])
            or background.get("status")
            != "verified_multi_segment_development_background"
            or background.get("passed") is not True
            or background.get("split_strategy") != "hash_threshold_v1"
            or int(background.get("splits", {}).get("test", {}).get("windows", -1))
            != 0
            or background.get("source_batch_report_sha256s")
            != [file_sha256(paths["batch"])]
            or background.get("manifest_sha256")
            != file_sha256(background_manifest)
            or export.get("status") != "group_safe_amplfi_background"
            or export.get("manifest_sha256") != background["manifest_sha256"]
            or int(export.get("split_file_counts", {}).get("test", -1)) != 0
            or eviction.get("status")
            != "verified_exported_amplfi_source_eviction"
            or eviction.get("recoverable") is not True
            or eviction.get("amplfi_export_report_sha256")
            != file_sha256(paths["export"])
        ):
            raise ValueError(f"AMPLFI extension shard failed identity replay: {root}")
        if extension_pair_ids & shard_ids:
            raise ValueError("AMPLFI extension repeats a source pair across shards")
        extension_pair_ids.update(shard_ids)
        shard_rows = [
            json.loads(line)
            for line in background_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in shard_rows:
            pair_id = str(row.get("pair_id", ""))
            block = str(row.get("gps_block", ""))
            if (
                row.get("split") not in {"train", "val"}
                or pair_id not in shard_ids
                or not block
                or not (
                    selected_plan_gps[pair_id]
                    <= int(row["gps_start"])
                    < selected_plan_gps[pair_id] + 4096
                )
            ):
                raise ValueError("AMPLFI extension background row is foreign or test data")
            extension_blocks.add(block)
        for item in export.get("files", []):
            artifact = Path(str(item.get("path", ""))).resolve()
            expected = str(item.get("sha256", ""))
            if file_sha256(artifact) != expected:
                raise ValueError(f"AMPLFI extension export changed: {artifact}")
            prior = exported_files.setdefault(str(artifact), expected)
            if prior != expected:
                raise ValueError("AMPLFI extension export paths conflict")
        rows.extend(shard_rows)
        shard_records.append(
            {
                key: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for key, path in paths.items()
            }
        )
    if (
        extension_pair_ids & base_pair_ids
        or extension_blocks & base_blocks
        or not extension_pair_ids
    ):
        raise ValueError("AMPLFI extension is not physically disjoint from the base")

    window_ids = [str(row.get("window_id", "")) for row in rows]
    block_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        block_splits[str(row["gps_block"])].add(str(row["split"]))
    if (
        any(not value for value in window_ids)
        or len(window_ids) != len(set(window_ids))
        or any(len(splits) != 1 for splits in block_splits.values())
    ):
        raise ValueError("AMPLFI extended manifest repeats windows or crosses splits")
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            int(row["gps_start"]),
            str(row["window_id"]),
        )
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "amplfi_background_train_val.jsonl"
    report_path = output / "amplfi_background_stream_extension_merge.json"
    atomic_write_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    result = {
        "status": "verified_extended_streamed_amplfi_background_bank",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_scores_inspected": False,
        "test_strain_rows_read": 0,
        "test_rows_exported": 0,
        "base_merge_report_path": str(base_path),
        "base_merge_report_sha256": file_sha256(base_path),
        "base_parent_plan_sha256": parent_hash,
        "extension_plan_path": str(plan_path),
        "extension_plan_sha256": file_sha256(plan_path),
        "extension_shards": shard_records,
        "extension_source_pairs": len(extension_pair_ids),
        "extension_pair_ids_hash": canonical_hash(
            sorted(extension_pair_ids), 64
        ),
        "extension_gps_blocks": len(extension_blocks),
        "extension_exported_files": len(exported_files),
        "background_windows": len(rows),
        "unique_gps_blocks": len(block_splits),
        "cross_split_gps_block_overlap": 0,
        "background_manifest_path": str(manifest),
        "background_manifest_sha256": file_sha256(manifest),
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def freeze_amplfi_training_stage_config(
    base_config_path: str | Path,
    stage_policy_path: str | Path,
    stage: str,
    output_config_path: str | Path,
    output_report_path: str | Path,
) -> dict[str, Any]:
    """Freeze one predeclared AMPLFI compute budget into a resolved config."""

    base_path = Path(base_config_path).resolve()
    policy_path = Path(stage_policy_path).resolve()
    policy = load_yaml(policy_path)
    config = load_yaml(base_path)
    if policy.get("schema_version") != 1:
        raise ValueError("AMPLFI stage policy schema_version must be 1")
    if policy.get("expected_base_config_sha256") != file_sha256(base_path):
        raise ValueError("AMPLFI stage policy does not bind the base training config")
    stages = policy.get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        raise ValueError(f"AMPLFI training stage is not predeclared: {stage}")
    selected = stages[stage]
    if not isinstance(selected, dict):
        raise ValueError("AMPLFI training stage must be a mapping")
    values = {
        "trainer.max_epochs": int(selected.get("max_epochs", 0)),
        "data.init_args.batches_per_epoch": int(selected.get("batches_per_epoch", 0)),
        "data.init_args.batch_size": int(selected.get("batch_size", 0)),
        "data.init_args.min_valid_duration": float(
            selected.get("min_valid_duration", 0)
        ),
        "data.init_args.waveform_sampler.init_args.num_fit_params": int(
            selected.get("num_fit_params", 0)
        ),
        "data.init_args.waveform_sampler.init_args.num_val_waveforms": int(
            selected.get("num_val_waveforms", 0)
        ),
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("AMPLFI training stage budgets must be positive")
    for dotted, value in values.items():
        target: dict[str, Any] = config
        fields = dotted.split(".")
        for field in fields[:-1]:
            child = target.get(field)
            if not isinstance(child, dict):
                raise ValueError(f"AMPLFI base config lacks stage field: {dotted}")
            target = child
        target[fields[-1]] = value
    logger_args = (
        config.get("trainer", {}).get("logger", {}).get("init_args", {})
    )
    if not isinstance(logger_args, dict):
        raise ValueError("AMPLFI base config lacks CSV logger init_args")
    logger_args["version"] = f"gwyolo_{stage}"
    max_epochs = int(values["trainer.max_epochs"])
    batches = int(values["data.init_args.batches_per_epoch"])
    batch_size = int(values["data.init_args.batch_size"])
    resolved = Path(output_config_path).resolve()
    report_target = Path(output_report_path).resolve()
    if resolved.exists() or report_target.exists():
        raise FileExistsError("AMPLFI resolved stage config/report are immutable")
    atomic_write_text(resolved, yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "frozen_amplfi_training_stage_config",
        "stage": stage,
        "publication_candidate": bool(selected.get("publication_candidate", False)),
        "base_config_path": str(base_path),
        "base_config_sha256": file_sha256(base_path),
        "stage_policy_path": str(policy_path),
        "stage_policy_sha256": file_sha256(policy_path),
        "resolved_config_path": str(resolved),
        "resolved_config_sha256": file_sha256(resolved),
        "overrides": values,
        "logger_version": logger_args["version"],
        "compute_budget": {
            "epochs": max_epochs,
            "updates": max_epochs * batches,
            "online_waveform_examples": max_epochs * batches * batch_size,
            "validation_waveforms_per_evaluation": int(
                values["data.init_args.waveform_sampler.init_args.num_val_waveforms"]
            ),
        },
        "test_rows_read": 0,
        "scientific_claim_allowed": False,
        **execution_provenance(),
    }
    atomic_write_json(report_target, report)
    return report


def audit_amplfi_common_prior_projection(
    canonical_prior_path: str | Path,
    amplfi_prior_path: str | Path,
    amplfi_training_config_path: str | Path,
) -> dict[str, Any]:
    canonical = load_yaml(canonical_prior_path)
    native_prior = load_yaml(amplfi_prior_path)
    training = load_yaml(amplfi_training_config_path)
    failures: list[str] = []
    if canonical.get("schema_version") != 1 or canonical.get("population") != "BBH":
        failures.append("canonical prior must be schema v1 BBH")
    distributions = canonical.get("distributions")
    nuisance = canonical.get("nuisance_distributions")
    if not isinstance(distributions, dict) or not isinstance(nuisance, dict):
        failures.append("canonical prior distributions are malformed")
        distributions = distributions if isinstance(distributions, dict) else {}
        nuisance = nuisance if isinstance(nuisance, dict) else {}
    native = native_prior.get("init_args", {}).get("priors", {})
    data = training.get("data", {}).get("init_args", {})
    if not isinstance(native, dict) or not isinstance(data, dict):
        failures.append("AMPLFI prior or training data configuration is malformed")
        native = native if isinstance(native, dict) else {}
        data = data if isinstance(data, dict) else {}

    mappings = {
        "chirp_mass": (distributions.get("chirp_mass"), native.get("chirp_mass")),
        "mass_ratio": (distributions.get("mass_ratio"), native.get("mass_ratio")),
        "luminosity_distance": (
            distributions.get("luminosity_distance"),
            native.get("distance"),
        ),
        "theta_jn": (distributions.get("theta_jn"), native.get("inclination")),
        "phase": (nuisance.get("phase"), native.get("phic")),
        "a_1": (nuisance.get("a_1"), native.get("a_1")),
        "a_2": (nuisance.get("a_2"), native.get("a_2")),
        "tilt_1": (nuisance.get("tilt_1"), native.get("tilt_1")),
        "tilt_2": (nuisance.get("tilt_2"), native.get("tilt_2")),
        "phi_jl": (nuisance.get("phi_jl"), native.get("phi_jl")),
        "phi_12": (nuisance.get("phi_12"), native.get("phi_12")),
        "ra": (distributions.get("ra"), data.get("phi")),
        "dec": (distributions.get("dec"), data.get("dec")),
        "psi": (distributions.get("psi"), data.get("psi")),
    }
    expected_classes = {
        "uniform": "torch.distributions.Uniform",
        "uniform_periodic": "torch.distributions.Uniform",
        "sine": "ml4gw.distributions.Sine",
        "cosine": "ml4gw.distributions.Cosine",
    }
    checks = {}
    for canonical_name, (expected, observed) in mappings.items():
        if not isinstance(expected, dict) or not isinstance(observed, dict):
            failures.append(f"prior projection is missing {canonical_name}")
            continue
        family = str(expected.get("family"))
        class_path = str(observed.get("class_path"))
        expected_class = expected_classes.get(family)
        if expected_class is None or class_path != expected_class:
            failures.append(f"prior family/class mismatch for {canonical_name}")
        arguments = observed.get("init_args") or {}
        if family != "cosine":
            low = arguments.get("low")
            high = arguments.get("high")
            try:
                bounds_match = float(low) == float(expected.get("minimum")) and float(
                    high
                ) == float(expected.get("maximum"))
            except (TypeError, ValueError):
                bounds_match = False
            if not bounds_match:
                failures.append(f"prior bounds mismatch for {canonical_name}")
        checks[canonical_name] = {
            "canonical_family": family,
            "native_class": class_path,
            "canonical_bounds": [expected.get("minimum"), expected.get("maximum")],
            "native_bounds": [arguments.get("low"), arguments.get("high")],
        }
    if data.get("ifos") != ["H1", "L1"]:
        failures.append("AMPLFI common training detector set must be H1/L1")
    if data.get("sample_rate") != 2048 or data.get("kernel_length") != 3:
        failures.append("AMPLFI native sample rate/kernel length differs from frozen contract")
    return {
        "status": "passed" if not failures else "failed",
        "publication_ready": not failures,
        "canonical_prior_path": str(Path(canonical_prior_path).resolve()),
        "canonical_prior_sha256": file_sha256(canonical_prior_path),
        "amplfi_prior_path": str(Path(amplfi_prior_path).resolve()),
        "amplfi_prior_sha256": file_sha256(amplfi_prior_path),
        "amplfi_training_config_path": str(Path(amplfi_training_config_path).resolve()),
        "amplfi_training_config_sha256": file_sha256(amplfi_training_config_path),
        "checks": checks,
        "failures": failures,
        **execution_provenance(),
    }


def run_amplfi_common_prior_audit(
    canonical_prior_path: str | Path,
    amplfi_prior_path: str | Path,
    amplfi_training_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = audit_amplfi_common_prior_projection(
        canonical_prior_path,
        amplfi_prior_path,
        amplfi_training_config_path,
    )
    atomic_write_json(output_path, report)
    if not report["publication_ready"]:
        raise RuntimeError(f"AMPLFI common prior projection failed; inspect {output_path}")
    return report


try:
    from amplfi.train.data.datasets import FlowDataset as _FlowDataset
except ImportError:
    _FlowDataset = object  # type: ignore[assignment,misc]


class GroupSafeFlowDataset(_FlowDataset):  # type: ignore[misc,valid-type]
    """AMPLFI FlowDataset using explicit train/validation background directories."""

    def train_val_split(self) -> tuple[list[str], list[str]]:
        if _FlowDataset is object:
            raise RuntimeError("GroupSafeFlowDataset requires the AMPLFI package")
        train = sorted((self.data_dir / "train" / "background").glob("*.hdf5"))
        validation = sorted(
            (self.data_dir / "validation" / "background").glob("*.hdf5")
        )
        if not train or not validation:
            raise ValueError("group-safe AMPLFI train and validation directories must be non-empty")
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("GroupSafeFlowDataset requires h5py") from error

        def identities(paths: list[Path]) -> tuple[set[str], float]:
            blocks = set()
            duration = 0.0
            for path in paths:
                with h5py.File(path, "r") as handle:
                    block = handle.attrs.get("gps_block")
                    seconds = handle.attrs.get("duration_seconds")
                if not block or seconds is None:
                    raise ValueError(f"group-safe AMPLFI background lacks identity: {path}")
                blocks.add(str(block))
                duration += float(seconds)
            return blocks, duration

        train_blocks, _ = identities(train)
        validation_blocks, validation_duration = identities(validation)
        if train_blocks & validation_blocks:
            raise ValueError("group-safe AMPLFI GPS blocks overlap across train/validation")
        if validation_duration < float(self.hparams.min_valid_duration):
            raise ValueError(
                "group-safe AMPLFI validation duration is below min_valid_duration"
            )
        return [str(path) for path in train], [str(path) for path in validation]


AMPLFI_PE_CONDITIONS = ("clean", "contaminated", "mask_conditioned")


def _load_amplfi_native_rows(
    path: str | Path, required_split: str
) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("AMPLFI native conditioning manifest is empty")
    if any(row.get("backend") != "AMPLFI" for row in rows):
        raise ValueError("AMPLFI batch received a non-AMPLFI conditioning row")
    if any(str(row.get("split")) != required_split for row in rows):
        raise ValueError("AMPLFI batch native manifest contains another split")
    by_injection: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_injection[str(row["injection_id"])].add(str(row["condition"]))
    if any(values != set(AMPLFI_PE_CONDITIONS) for values in by_injection.values()):
        raise ValueError("AMPLFI batch requires three conditions for every injection")
    if len(rows) != len(AMPLFI_PE_CONDITIONS) * len(by_injection):
        raise ValueError("AMPLFI batch native manifest repeats an injection condition")
    return rows


def _validated_amplfi_report(
    path: Path,
    event_sha256: str,
    model_sha256: str,
    model_config_sha256: str,
    native_prior_sha256: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "real_amplfi_flow_posterior_complete",
        "backend": "AMPLFI",
        "event_sha256": event_sha256,
        "model_sha256": model_sha256,
        "model_config_sha256": model_config_sha256,
        "native_prior_sha256": native_prior_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing AMPLFI posterior report belongs to another run")
    for prefix in ("posterior", "native_result"):
        artifact = Path(report[f"{prefix}_path"])
        if not artifact.is_file() or file_sha256(artifact) != report[f"{prefix}_sha256"]:
            raise ValueError(f"Existing AMPLFI {prefix} artifact hash mismatch")
    validate_paired_pe_latency(report)
    return report


def run_amplfi_common_batch(
    native_manifest: str | Path,
    model_metadata_path: str | Path,
    native_prior_path: str | Path,
    python_executable: str | Path,
    runner_script: str | Path,
    output_dir: str | Path,
    required_split: str,
    num_samples: int = 10000,
    sample_batch_size: int = 1000,
    device: str = "cuda",
    seed: int = 20260721,
) -> dict[str, Any]:
    """Run a validation-selected AMPLFI checkpoint on matched PE conditions."""

    if required_split not in {"val", "test"}:
        raise ValueError("AMPLFI batch is restricted to val or test")
    if num_samples <= 0 or sample_batch_size <= 0:
        raise ValueError("AMPLFI batch sampling settings must be positive")
    metadata_path = Path(model_metadata_path).resolve()
    metadata = load_yaml(metadata_path)
    if metadata.get("backend") != "AMPLFI" or metadata.get(
        "selection_split"
    ) != "validation":
        raise ValueError("AMPLFI batch requires validation-selected model metadata")
    model = Path(metadata["model_path"]).resolve()
    if file_sha256(model) != str(metadata["model_sha256"]):
        raise ValueError("AMPLFI model hash differs from standardized metadata")
    artifacts = metadata.get("artifacts", {})
    training_identity = artifacts.get("training_config", {})
    training_data_identity = artifacts.get("training_data_manifest", {})
    selection_identity = artifacts.get("selection_report", {})
    conditioning_identity = artifacts.get("native_conditioning_config", {})
    analysis_prior_identity = artifacts.get("analysis_prior", {})
    native_prior_identity = artifacts.get("native_prior", {})
    prior_projection_identity = artifacts.get("prior_projection_report", {})
    model_config = Path(str(training_identity.get("path", ""))).resolve()
    if (
        not model_config.is_file()
        or file_sha256(model_config) != training_identity.get("sha256")
    ):
        raise ValueError("AMPLFI training configuration hash differs from metadata")
    native_prior = Path(native_prior_path).resolve()
    native_prior_sha = file_sha256(native_prior)
    verified_prior_artifacts = {
        "training_data_manifest": training_data_identity,
        "selection_report": selection_identity,
        "analysis_prior": analysis_prior_identity,
        "native_prior": native_prior_identity,
        "prior_projection_report": prior_projection_identity,
    }
    for label, identity in verified_prior_artifacts.items():
        artifact = Path(str(identity.get("path", ""))).resolve()
        if (
            not artifact.is_file()
            or file_sha256(artifact) != identity.get("sha256")
        ):
            raise ValueError(f"AMPLFI {label} hash differs from model metadata")
    if native_prior_sha != native_prior_identity.get("sha256"):
        raise ValueError("AMPLFI runtime native prior differs from model metadata")
    projection_path = Path(str(prior_projection_identity["path"])).resolve()
    projection = load_yaml(projection_path)
    if (
        projection.get("status") != "passed"
        or projection.get("publication_ready") is not True
        or projection.get("failures") not in (None, [])
        or projection.get("canonical_prior_sha256")
        != analysis_prior_identity.get("sha256")
        or projection.get("amplfi_prior_sha256")
        != native_prior_identity.get("sha256")
        or projection.get("amplfi_training_config_sha256")
        != training_identity.get("sha256")
    ):
        raise ValueError("AMPLFI prior projection differs from model metadata")
    selection = load_yaml(selection_identity["path"])
    if (
        selection.get("status") != "validation_selected_checkpoint"
        or selection.get("publication_eligible") is not True
        or selection.get("selection_split") != "validation"
        or selection.get("selected_checkpoint_sha256") != metadata["model_sha256"]
        or selection.get("selection_metric") != metadata.get("selection_metric")
    ):
        raise ValueError("AMPLFI validation selection report differs from metadata")
    python = Path(python_executable).resolve()
    runner = Path(runner_script).resolve()
    if not python.is_file() or not runner.is_file():
        raise FileNotFoundError("AMPLFI pinned interpreter or runner script is absent")
    source_input = metadata.get("source_input", {})
    if (
        source_input.get("ifos") != ["H1", "L1"]
        or not source_input.get("common_asd_required")
        or float(source_input.get("sample_rate_hz", 0)) <= 0
        or float(source_input.get("duration_seconds", 0)) <= 0
        or float(source_input.get("post_trigger_seconds", 0)) <= 0
    ):
        raise ValueError("AMPLFI model metadata lacks the common H1/L1 ASD contract")
    rows = _load_amplfi_native_rows(native_manifest, required_split)
    if any(
        row.get("input_ifos") != source_input["ifos"]
        or not np.isclose(
            float(row.get("input_sample_rate_hz", 0)),
            float(source_input["sample_rate_hz"]),
        )
        or not np.isclose(
            float(row.get("input_duration_seconds", 0)),
            float(source_input["duration_seconds"]),
        )
        or not np.isclose(
            float(row.get("input_post_trigger_seconds", 0)),
            float(source_input["post_trigger_seconds"]),
        )
        for row in rows
    ):
        raise ValueError("AMPLFI native rows differ from the model common-source contract")
    expected_conditioning_sha = str(conditioning_identity.get("sha256", ""))
    if any(
        str(row.get("native_conditioning_config_sha256"))
        != expected_conditioning_sha
        for row in rows
    ):
        raise ValueError("AMPLFI native rows use a conditioning config outside model metadata")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": "amplfi_common_batch_v1",
        "native_manifest_sha256": file_sha256(native_manifest),
        "model_metadata_sha256": file_sha256(metadata_path),
        "model_sha256": metadata["model_sha256"],
        "model_config_sha256": training_identity["sha256"],
        "native_prior_sha256": native_prior_sha,
        "analysis_prior_sha256": analysis_prior_identity["sha256"],
        "training_data_manifest_sha256": training_data_identity["sha256"],
        "selection_report_sha256": selection_identity["sha256"],
        "prior_projection_report_sha256": prior_projection_identity["sha256"],
        "python_executable": str(python),
        "runner_sha256": file_sha256(runner),
        "required_split": required_split,
        "num_samples": num_samples,
        "sample_batch_size": sample_batch_size,
        "device": device,
        "seed": seed,
    }
    state_path = output / "amplfi_batch_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing AMPLFI batch output belongs to another run")
    else:
        atomic_write_json(
            state_path, {"status": "in_progress", "run_identity": run_identity, "completed": 0}
        )

    result_rows = []
    for index, row in enumerate(rows, start=1):
        event = Path(row["native_conditioning_path"]).resolve()
        if file_sha256(event) != str(row["native_conditioning_sha256"]):
            raise ValueError("AMPLFI native conditioning artifact hash mismatch")
        event_output = output / "events" / str(row["injection_id"]) / str(
            row["condition"]
        )
        posterior = event_output / "posterior.npz"
        native_result = event_output / "amplfi_result.hdf5"
        report_path = event_output / "amplfi_inference_report.json"
        log_path = event_output / "amplfi_inference.log"
        if report_path.is_file():
            report = _validated_amplfi_report(
                report_path,
                row["native_conditioning_sha256"],
                metadata["model_sha256"],
                training_identity["sha256"],
                native_prior_sha,
            )
        else:
            event_output.mkdir(parents=True, exist_ok=True)
            command = [
                str(python),
                str(runner),
                "--event",
                str(event),
                "--model",
                str(model),
                "--model-config",
                str(model_config),
                "--native-prior",
                str(native_prior),
                "--posterior-output",
                str(posterior),
                "--result-output",
                str(native_result),
                "--report-output",
                str(report_path),
                "--expected-event-sha256",
                row["native_conditioning_sha256"],
                "--expected-model-sha256",
                metadata["model_sha256"],
                "--expected-model-config-sha256",
                training_identity["sha256"],
                "--expected-native-prior-sha256",
                native_prior_sha,
                "--num-samples",
                str(num_samples),
                "--sample-batch-size",
                str(sample_batch_size),
                "--device",
                device,
                "--seed",
                str(seed + index - 1),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            atomic_write_text(
                log_path,
                "command: "
                + json.dumps(command)
                + "\nstdout:\n"
                + completed.stdout
                + "\nstderr:\n"
                + completed.stderr,
            )
            if completed.returncode != 0:
                atomic_write_json(
                    event_output / "amplfi_inference_failure.json",
                    {
                        "status": "failed",
                        "returncode": completed.returncode,
                        "log_path": str(log_path),
                        "log_sha256": file_sha256(log_path),
                        "event_sha256": row["native_conditioning_sha256"],
                        "model_sha256": metadata["model_sha256"],
                        "model_config_sha256": training_identity["sha256"],
                        "native_prior_sha256": native_prior_sha,
                    },
                )
                raise RuntimeError(f"AMPLFI inference failed; inspect {log_path}")
            report = _validated_amplfi_report(
                report_path,
                row["native_conditioning_sha256"],
                metadata["model_sha256"],
                training_identity["sha256"],
                native_prior_sha,
            )
        with np.load(report["posterior_path"], allow_pickle=False) as posterior:
            if "ra" not in posterior.files or "dec" not in posterior.files:
                raise ValueError("AMPLFI posterior lacks RA/Dec sky samples")
            sky_area = posterior_sky_area_equal_solid_angle(
                posterior["ra"], posterior["dec"]
            )
        latency_components = validate_paired_pe_latency(report)
        result_rows.append(
            {
                **row,
                "backend": "AMPLFI",
                "posterior_path": report["posterior_path"],
                "posterior_sha256": report["posterior_sha256"],
                "latency_seconds": report["latency_seconds"],
                "effective_sample_size": report["effective_sample_size"],
                "sky_area_90_deg2": sky_area["area_deg2"],
                "sky_area_estimator": sky_area_estimator_identity(sky_area),
                "sky_area_diagnostics": {
                    field: sky_area[field]
                    for field in ("sample_count", "occupied_pixels", "credible_pixels")
                },
                "backend_version": report["backend_version"],
                "backend_model_hash": report["model_sha256"],
                "prior_hash": row["common_prior_sha256"],
                "native_prior_path": report["native_prior_path"],
                "native_prior_sha256": report["native_prior_sha256"],
                "waveform_approximant": metadata["analysis_waveform_approximant"],
                "detector_set": row["input_ifos"],
                "calibration_version": "none_software_injection_o4a_strain",
                "source_event_hash": row["source_event_hash"],
                "hardware": {
                    "hostname": report["environment"]["hostname"],
                    "gpu": report["environment"]["gpu"],
                },
                "latency_scope": PAIRED_PE_LATENCY_SCOPE_V1,
                "backend_native_latency_scope": report["latency_scope"],
                "backend_native_latency_components_seconds": latency_components,
            }
        )
        atomic_write_json(
            state_path,
            {"status": "in_progress", "run_identity": run_identity, "completed": index},
        )
    manifest = output / "amplfi_posterior_manifest.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    report = {
        "status": "real_amplfi_common_batch_complete",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "matched DINGO results and paired robustness evaluation are still required"
        ),
        "rows": len(result_rows),
        "paired_injections": len({row["injection_id"] for row in result_rows}),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(output / "amplfi_batch_report.json", report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(result_rows),
            "manifest_sha256": report["manifest_sha256"],
        },
    )
    return report
