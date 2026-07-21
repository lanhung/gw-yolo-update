from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, file_sha256, load_yaml
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
