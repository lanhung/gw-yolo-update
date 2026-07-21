from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance


CONDITIONS = ("clean", "contaminated", "mask_conditioned")


def _load_rows(path: str | Path, required_split: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("PE native conditioning received an empty source manifest")
    if any(str(row.get("split")) != required_split for row in rows):
        raise ValueError("PE native conditioning source manifest contains another split")
    keys = [(str(row["injection_id"]), str(row["condition"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("PE native conditioning source manifest repeats a condition")
    by_injection: dict[str, set[str]] = defaultdict(set)
    for injection_id, condition in keys:
        if condition not in CONDITIONS:
            raise ValueError(f"Unsupported PE source condition: {condition}")
        by_injection[injection_id].add(condition)
    if any(values != set(CONDITIONS) for values in by_injection.values()):
        raise ValueError("Every PE injection must have all three source conditions")
    return rows


def _load_source(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["analysis_input_path"]).resolve()
    if file_sha256(path) != str(row["analysis_input_sha256"]):
        raise ValueError("Common PE source artifact hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "strain",
            "asd",
            "asd_frequencies",
            "ifos",
            "sample_rate",
            "gps_start",
            "geocent_time",
            "post_trigger_seconds",
            "condition",
            "injection_id",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"Common PE source artifact lacks fields: {missing}")
        result = {
            "strain": np.asarray(arrays["strain"], dtype=np.float64),
            "asd": np.asarray(arrays["asd"], dtype=np.float64),
            "asd_frequencies": np.asarray(arrays["asd_frequencies"], dtype=np.float64),
            "ifos": tuple(str(value) for value in arrays["ifos"].tolist()),
            "sample_rate": int(arrays["sample_rate"]),
            "gps_start": float(arrays["gps_start"]),
            "geocent_time": float(arrays["geocent_time"]),
            "post_trigger_seconds": float(arrays["post_trigger_seconds"]),
            "condition": str(arrays["condition"].item()),
            "injection_id": str(arrays["injection_id"].item()),
        }
    if result["condition"] != str(row["condition"]) or result["injection_id"] != str(
        row["injection_id"]
    ):
        raise ValueError("Common PE source artifact identity differs from its manifest row")
    if result["strain"].ndim != 2 or not np.isfinite(result["strain"]).all():
        raise ValueError("Common PE source strain is invalid")
    if (
        result["asd"].shape[0] != result["strain"].shape[0]
        or result["asd"].ndim != 2
        or result["asd_frequencies"].shape != (result["asd"].shape[1],)
        or not np.isfinite(result["asd"]).all()
        or np.any(result["asd"] <= 0)
    ):
        raise ValueError("Common PE source ASD is invalid")
    observed_asd_hash = hashlib.sha256(
        np.asarray(result["asd"], dtype="<f8").tobytes(order="C")
    ).hexdigest()
    if observed_asd_hash != str(row["common_asd_sha256"]):
        raise ValueError("Common PE source ASD hash differs from its manifest row")
    return result


def _tukey_window(samples: int, alpha: float) -> np.ndarray:
    if samples <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Invalid Tukey window settings")
    if alpha == 0:
        return np.ones(samples, dtype=np.float64)
    if alpha == 1:
        return np.hanning(samples)
    indices = np.arange(samples, dtype=np.float64)
    normalized = indices / (samples - 1)
    window = np.ones(samples, dtype=np.float64)
    left = normalized < alpha / 2
    right = normalized >= 1 - alpha / 2
    window[left] = 0.5 * (1 + np.cos(np.pi * (2 * normalized[left] / alpha - 1)))
    window[right] = 0.5 * (
        1 + np.cos(np.pi * (2 * normalized[right] / alpha - 2 / alpha + 1))
    )
    return window


def _atomic_hdf5(path: Path, writer: Any) -> None:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("PE native conditioning requires h5py") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".hdf5", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            writer(handle)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _condition_dingo(
    source: dict[str, Any], config: dict[str, Any], target: Path, row: dict[str, Any]
) -> dict[str, Any]:
    ifos = tuple(str(value) for value in config["ifos"])
    sample_rate = int(config["source_sample_rate_hz"])
    duration = float(config["source_duration_seconds"])
    post_trigger = float(config["source_post_trigger_seconds"])
    frequency = config["frequency_domain"]
    f_min = float(frequency["minimum_frequency_hz"])
    f_max = float(frequency["maximum_frequency_hz"])
    delta_f = float(frequency["delta_frequency_hz"])
    roll_off = float(config["window"]["roll_off_seconds"])
    expected_samples = int(round(sample_rate * duration))
    if source["ifos"] != ifos or source["sample_rate"] != sample_rate:
        raise ValueError("DINGO source detector/sample-rate contract mismatch")
    if source["strain"].shape != (len(ifos), expected_samples):
        raise ValueError("DINGO source duration contract mismatch")
    if not np.isclose(source["post_trigger_seconds"], post_trigger, atol=1e-12):
        raise ValueError("DINGO source post-trigger contract mismatch")
    if not np.isclose(1.0 / duration, delta_f, atol=1e-12):
        raise ValueError("DINGO delta-frequency contract differs from source duration")
    alpha = 2.0 * roll_off / duration
    window = _tukey_window(expected_samples, alpha)
    frequencies = np.fft.rfftfreq(expected_samples, d=1.0 / sample_rate)
    keep = frequencies <= f_max + 1e-12
    frequencies = frequencies[keep]
    waveform = np.fft.rfft(source["strain"] * window[None, :], axis=-1)
    waveform = waveform[:, keep] / sample_rate
    waveform *= np.exp(-2j * np.pi * frequencies * post_trigger)[None, :]
    expected_frequencies = np.arange(frequencies.size, dtype=np.float64) * delta_f
    if not np.allclose(frequencies, expected_frequencies, rtol=0.0, atol=1e-10):
        raise ValueError("DINGO native frequency grid is inconsistent")
    if not np.array_equal(source["asd_frequencies"], expected_frequencies):
        raise ValueError("DINGO common ASD grid differs from the native model grid")
    asd = source["asd"].copy()
    waveform[:, frequencies < f_min] = 0.0
    asd[:, frequencies < f_min] = float(config["asd"]["below_minimum_frequency_value"])
    settings = {
        "time_event": source["geocent_time"],
        "time_buffer": post_trigger,
        "detectors": list(ifos),
        "f_s": sample_rate,
        "T": duration,
        "window_type": "tukey",
        "roll_off": roll_off,
        "minimum_frequency": {ifo: f_min for ifo in ifos},
        "maximum_frequency": {ifo: f_max for ifo in ifos},
        "gwyolo_source_sha256": row["analysis_input_sha256"],
        "gwyolo_common_asd_sha256": row["common_asd_sha256"],
        "gwyolo_condition": row["condition"],
        "gwyolo_injection_id": row["injection_id"],
    }

    def write(handle: Any) -> None:
        handle.attrs["dataset_type"] = "event_dataset"
        handle.attrs["settings"] = repr(settings)
        handle.create_dataset("version", data=np.bytes_("gwyolo-dingo-conditioning-v1"))
        data = handle.create_group("data")
        waveforms = data.create_group("waveform")
        asds = data.create_group("asds")
        for index, ifo in enumerate(ifos):
            waveforms.create_dataset(ifo, data=waveform[index].astype(np.complex128))
            asds.create_dataset(ifo, data=asd[index].astype(np.float64))

    _atomic_hdf5(target, write)
    return {
        "native_sample_rate_hz": sample_rate,
        "native_duration_seconds": duration,
        "native_post_trigger_seconds": post_trigger,
        "native_frequency_bins": int(frequencies.size),
        "native_frequency_range_hz": [0.0, f_max],
        "native_delta_frequency_hz": delta_f,
        "window_alpha": alpha,
        "event_dataset_settings": settings,
    }


def _condition_amplfi(
    source: dict[str, Any], config: dict[str, Any], target: Path, row: dict[str, Any]
) -> dict[str, Any]:
    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise RuntimeError("AMPLFI native conditioning requires scipy") from error
    ifos = tuple(str(value) for value in config["ifos"])
    source_rate = int(config["source_sample_rate_hz"])
    native_rate = int(config["native_sample_rate_hz"])
    duration = float(config["source_duration_seconds"])
    post_trigger = float(config["source_post_trigger_seconds"])
    if source["ifos"] != ifos or source["sample_rate"] != source_rate:
        raise ValueError("AMPLFI source detector/sample-rate contract mismatch")
    if not np.isclose(source["post_trigger_seconds"], post_trigger, atol=1e-12):
        raise ValueError("AMPLFI source post-trigger contract mismatch")
    if source_rate % native_rate:
        raise ValueError("AMPLFI source-to-native sample rates require integer decimation")
    window_config = config["resampling"]["window"]
    if list(window_config) != ["kaiser", 8.6]:
        raise ValueError("AMPLFI native conditioning resampling window is unsupported")
    strain = resample_poly(
        source["strain"],
        up=1,
        down=source_rate // native_rate,
        axis=-1,
        window=("kaiser", 8.6),
        padtype="constant",
    )
    expected_samples = int(round(duration * native_rate))
    if strain.shape != (len(ifos), expected_samples) or not np.isfinite(strain).all():
        raise ValueError("AMPLFI native strain has an invalid shape or content")

    def write(handle: Any) -> None:
        handle.attrs["schema"] = "gwyolo-amplfi-common-source-v1"
        handle.attrs["source_sha256"] = row["analysis_input_sha256"]
        handle.attrs["common_asd_sha256"] = row["common_asd_sha256"]
        handle.attrs["condition"] = row["condition"]
        handle.attrs["injection_id"] = row["injection_id"]
        handle.attrs["gps_start"] = source["gps_start"]
        handle.attrs["geocent_time"] = source["geocent_time"]
        handle.attrs["post_trigger_seconds"] = post_trigger
        handle.attrs["sample_rate_hz"] = native_rate
        handle.create_dataset("strain", data=strain.astype(np.float32), compression="gzip")
        handle.create_dataset("ifos", data=np.asarray(ifos, dtype="S2"))
        handle.create_dataset("asd", data=source["asd"].astype(np.float64))
        handle.create_dataset(
            "asd_frequencies", data=source["asd_frequencies"].astype(np.float64)
        )

    _atomic_hdf5(target, write)
    return {
        "native_sample_rate_hz": native_rate,
        "native_duration_seconds": duration,
        "native_post_trigger_seconds": post_trigger,
        "native_samples_per_ifo": expected_samples,
        "native_kernel_seconds": float(config["native_kernel_seconds"]),
        "native_whitening_duration_seconds": float(
            config["native_whitening_duration_seconds"]
        ),
        "native_highpass_hz": float(config["native_highpass_hz"]),
        "runtime_whitening_must_not_reestimate_psd": bool(
            config["asd"]["runtime_whitening_must_not_reestimate_psd"]
        ),
    }


def materialize_native_pe_conditioning(
    source_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    required_split: str,
) -> dict[str, Any]:
    if required_split not in {"val", "test"}:
        raise ValueError("PE native conditioning is restricted to val or test")
    config = load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("PE native conditioning config schema_version must be 1")
    backend = str(config.get("backend", "")).upper()
    if backend not in {"DINGO", "AMPLFI"}:
        raise ValueError("PE native conditioning backend must be DINGO or AMPLFI")
    if config.get("asd", {}).get("condition_invariant_required") is not True:
        raise ValueError("PE native conditioning must require a condition-invariant common ASD")
    rows = _load_rows(source_manifest, required_split)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": "native_pe_conditioning_v1",
        "backend": backend,
        "source_manifest_sha256": file_sha256(source_manifest),
        "config_sha256": file_sha256(config_path),
        "required_split": required_split,
        "source_rows_hash": canonical_hash(
            [
                [row["injection_id"], row["condition"], row["analysis_input_sha256"]]
                for row in rows
            ],
            64,
        ),
    }
    state_path = output / "native_conditioning_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing native PE conditioning output belongs to another run")
    else:
        atomic_write_json(
            state_path, {"status": "in_progress", "run_identity": run_identity, "completed": 0}
        )

    result_rows = []
    asd_by_injection: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        source = _load_source(row)
        asd_by_injection[str(row["injection_id"])].add(str(row["common_asd_sha256"]))
        target = (
            output
            / "artifacts"
            / str(row["condition"])
            / f"{row['injection_id']}.hdf5"
        )
        condition = _condition_dingo if backend == "DINGO" else _condition_amplfi
        if target.exists():
            verification = target.with_name(f".{target.name}.resume-verification.hdf5")
            if verification.exists():
                raise FileExistsError(f"Stale native-conditioning verification exists: {verification}")
            native = condition(source, config, verification, row)
            try:
                if file_sha256(verification) != file_sha256(target):
                    raise ValueError(
                        "Existing native conditioning artifact differs from deterministic rebuild"
                    )
            finally:
                verification.unlink(missing_ok=True)
        else:
            native = condition(source, config, target, row)
        result_rows.append(
            {
                **row,
                "backend": backend,
                "native_conditioning_path": str(target),
                "native_conditioning_sha256": file_sha256(target),
                "native_conditioning_config_path": str(Path(config_path).resolve()),
                "native_conditioning_config_sha256": run_identity["config_sha256"],
                **native,
            }
        )
        atomic_write_json(
            state_path,
            {"status": "in_progress", "run_identity": run_identity, "completed": index},
        )
    if any(len(values) != 1 for values in asd_by_injection.values()):
        raise ValueError("A PE injection uses different common ASDs across conditions")
    manifest = output / f"{backend.lower()}_native_conditioning.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    report = {
        "status": "native_pe_conditioning_materialized",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "the locked backend must load these artifacts and produce real posterior samples"
        ),
        "backend": backend,
        "rows": len(result_rows),
        "paired_injections": len(asd_by_injection),
        "condition_counts": dict(Counter(row["condition"] for row in result_rows)),
        "condition_invariant_common_asd": True,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(output / "native_conditioning_report.json", report)
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


def read_dingo_event_settings(path: str | Path) -> dict[str, Any]:
    """Small dependency-free test helper matching DingoDataset attribute parsing."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("DINGO event inspection requires h5py") from error
    with h5py.File(path, "r") as handle:
        settings = ast.literal_eval(str(handle.attrs["settings"]))
    if not isinstance(settings, dict):
        raise ValueError("DINGO event settings are not a mapping")
    return settings
