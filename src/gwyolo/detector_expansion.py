from __future__ import annotations

import json
import os
import platform
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .waveforms import (
    PyCBCWaveformBackend,
    _atomic_save_npz,
    optimal_snr_stratum,
    pack_scaled_float16_signal,
    place_waveform_samples,
    validate_recipe_identities,
)


def _stored_signal(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["materialized_path"]).resolve()
    if file_sha256(path) != str(row["materialized_sha256"]):
        raise ValueError(f"source materialized hash mismatch: {row['injection_id']}")
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "ifos",
            "sample_rate",
            "context_gps_start",
            "analysis_gps_start",
            "analysis_start_index",
            "analysis_stop_index",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"source injection artifact lacks fields: {missing}")
        if "signal" in arrays:
            signal = np.asarray(arrays["signal"], dtype=np.float64)
        elif "signal_scaled" in arrays and "signal_peak_scale" in arrays:
            signal = np.asarray(arrays["signal_scaled"], dtype=np.float64) * np.asarray(
                arrays["signal_peak_scale"], dtype=np.float64
            )[:, None]
        else:
            raise ValueError("source injection artifact lacks a supported signal")
        result = {
            "path": path,
            "ifos": [str(value) for value in arrays["ifos"].tolist()],
            "sample_rate": int(arrays["sample_rate"]),
            "context_gps_start": float(arrays["context_gps_start"]),
            "analysis_gps_start": float(arrays["analysis_gps_start"]),
            "analysis_start_index": int(arrays["analysis_start_index"]),
            "analysis_stop_index": int(arrays["analysis_stop_index"]),
            "signal": signal,
        }
    if (
        signal.ndim != 2
        or signal.shape[0] != len(result["ifos"])
        or len(set(result["ifos"])) != len(result["ifos"])
        or result["sample_rate"] <= 0
        or not 0
        <= result["analysis_start_index"]
        < result["analysis_stop_index"]
        <= signal.shape[1]
        or not np.isfinite(signal).all()
    ):
        raise ValueError("source injection signal contract is invalid")
    return result


def _signal_equivalence(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("detector projection equivalence arrays are invalid")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    scale = max(left_norm, np.finfo(np.float64).tiny)
    overlap = (
        1.0
        if left_norm == 0 and right_norm == 0
        else (
            0.0
            if left_norm == 0 or right_norm == 0
            else float(abs(np.vdot(left, right)) / (left_norm * right_norm))
        )
    )
    return {
        "normalized_overlap": overlap,
        "relative_l2_error": float(np.linalg.norm(left - right) / scale),
        "reference_norm": left_norm,
        "candidate_norm": right_norm,
    }


def _pycbc_reference_psd_snr(
    signal: np.ndarray,
    sample_rate: int,
    _ifo: str,
    psd_model: str,
    low_frequency: float,
    high_frequency: float,
) -> float:
    try:
        from pycbc import psd
        from pycbc.filter import sigma
        from pycbc.types import TimeSeries
    except ImportError as exc:
        raise RuntimeError("reference-PSD SNR annotation requires PyCBC") from exc
    model = getattr(psd, psd_model, None)
    if not callable(model):
        raise ValueError(f"unknown PyCBC reference PSD model: {psd_model}")
    series = TimeSeries(np.asarray(signal, dtype=np.float64), delta_t=1.0 / sample_rate)
    frequency = series.to_frequencyseries()
    spectrum = model(len(frequency), frequency.delta_f, low_frequency)
    value = float(
        sigma(
            series,
            psd=spectrum,
            low_frequency_cutoff=low_frequency,
            high_frequency_cutoff=min(high_frequency, sample_rate / 2.0 - 1.0),
        )
    )
    if not np.isfinite(value) or value < 0:
        raise ValueError("reference-PSD optimal SNR is invalid")
    return value


def expand_materialized_injection_detector_set(
    manifest_path: str | Path,
    config_path: str | Path,
    backend_validation_report_path: str | Path,
    output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
    *,
    backend: Any | None = None,
    snr_calculator: Callable[[np.ndarray, int, str, str, float, float], float]
    | None = None,
) -> dict[str, Any]:
    """Reproject a fixed waveform population onto a larger detector set."""

    if split not in {"train", "val"}:
        raise ValueError("detector-set expansion is limited to development train/val data")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    identity = validate_recipe_identities(source_rows)
    rows = [row for row in source_rows if row.get("split") == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("detector-set expansion limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("detector-set expansion selected no rows")

    config = load_yaml(config_path)
    settings = config.get("detector_set_expansion")
    if not isinstance(settings, dict):
        raise ValueError("configuration requires detector_set_expansion")
    target_ifos = tuple(str(value) for value in settings["target_ifos"])
    psd_models = {
        str(key): str(value)
        for key, value in settings["reference_psd_models_by_ifo"].items()
    }
    low_frequency = float(settings["low_frequency_hz"])
    high_frequency = float(settings["high_frequency_hz"])
    minimum_overlap = float(settings["minimum_common_ifo_normalized_overlap"])
    maximum_relative_error = float(settings["maximum_common_ifo_relative_l2_error"])
    if (
        len(target_ifos) < 2
        or len(set(target_ifos)) != len(target_ifos)
        or set(psd_models) != set(target_ifos)
        or low_frequency <= 0
        or high_frequency <= low_frequency
        or not 0 < minimum_overlap <= 1
        or maximum_relative_error <= 0
    ):
        raise ValueError("detector-set expansion configuration is invalid")

    validation_path = Path(backend_validation_report_path).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("passed") is not True
        or validation.get("validation_scope")
        != "external_reference_waveform_equivalence"
    ):
        raise ValueError("waveform backend validation report did not pass")

    waveform_backend = backend if backend is not None else PyCBCWaveformBackend()
    calculate_snr = snr_calculator or _pycbc_reference_psd_snr
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "manifest_sha256": file_sha256(manifest_path),
        "config_sha256": file_sha256(config_path),
        "backend_validation_report_sha256": file_sha256(validation_path),
        "selected_injection_ids_hash": canonical_hash(
            [str(row["injection_id"]) for row in rows], 64
        ),
        "split": split,
        "limit": limit,
        "target_ifos": list(target_ifos),
        "reference_psd_models_by_ifo": psd_models,
        "low_frequency_hz": low_frequency,
        "high_frequency_hz": high_frequency,
        "backend": waveform_backend.metadata,
        "storage_mode": "signal_scaled_float16",
    }
    state_path = output / "detector_set_expansion_state.json"
    partial_path = output / "expanded_injections.partial.jsonl"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("existing detector-set expansion belongs to another run")
    completed: list[dict[str, Any]] = []
    if partial_path.is_file():
        with partial_path.open("r", encoding="utf-8") as handle:
            completed = [json.loads(line) for line in handle if line.strip()]
    requested_ids = [str(row["injection_id"]) for row in rows]
    if [str(row["injection_id"]) for row in completed] != requested_ids[: len(completed)]:
        raise ValueError("partial detector expansion is not a requested prefix")
    for row in completed:
        if file_sha256(row["materialized_path"]) != str(row["materialized_sha256"]):
            raise ValueError("partial expanded injection hash mismatch")

    started = time.time()
    common_equivalence: list[dict[str, Any]] = []
    for index, row in enumerate(rows[len(completed) :], start=len(completed) + 1):
        source = _stored_signal(row)
        if not set(source["ifos"]) < set(target_ifos):
            raise ValueError("target detector set must strictly expand every source row")
        projected, signal_summary = waveform_backend.generate(
            row, list(target_ifos), source["sample_rate"]
        )
        generated = np.stack(
            [
                place_waveform_samples(
                    source["context_gps_start"],
                    source["sample_rate"],
                    source["signal"].shape[1],
                    projected[ifo][0],
                    projected[ifo][1],
                )
                for ifo in target_ifos
            ]
        )
        per_ifo_equivalence = {}
        for ifo in source["ifos"]:
            metrics = _signal_equivalence(
                source["signal"][source["ifos"].index(ifo)],
                generated[target_ifos.index(ifo)],
            )
            metrics["passed"] = bool(
                metrics["normalized_overlap"] >= minimum_overlap
                and metrics["relative_l2_error"] <= maximum_relative_error
            )
            if not metrics["passed"]:
                raise ValueError(
                    f"regenerated common-IFO projection differs for "
                    f"{row['injection_id']} {ifo}"
                )
            per_ifo_equivalence[ifo] = metrics
            common_equivalence.append(
                {
                    "injection_id": str(row["injection_id"]),
                    "ifo": ifo,
                    **metrics,
                }
            )
        start = source["analysis_start_index"]
        stop = source["analysis_stop_index"]
        by_ifo = {
            ifo: float(
                calculate_snr(
                    generated[target_ifos.index(ifo), start:stop],
                    source["sample_rate"],
                    ifo,
                    psd_models[ifo],
                    low_frequency,
                    high_frequency,
                )
            )
            for ifo in target_ifos
        }
        if any(not np.isfinite(value) or value < 0 for value in by_ifo.values()):
            raise ValueError("expanded detector-set SNR is invalid")
        network_snr = float(np.sqrt(np.sum(np.square(list(by_ifo.values())))))
        packed, peaks, reconstruction = pack_scaled_float16_signal(generated)
        artifact = output / "arrays" / f"{row['injection_id']}.npz"
        _atomic_save_npz(
            artifact,
            ifos=np.asarray(target_ifos),
            sample_rate=np.asarray(source["sample_rate"], dtype=np.int64),
            context_gps_start=np.asarray(
                source["context_gps_start"], dtype=np.float64
            ),
            analysis_gps_start=np.asarray(
                source["analysis_gps_start"], dtype=np.float64
            ),
            analysis_start_index=np.asarray(start, dtype=np.int64),
            analysis_stop_index=np.asarray(stop, dtype=np.int64),
            signal_scaled=packed,
            signal_peak_scale=peaks,
        )
        completed.append(
            {
                **row,
                "ifos": list(target_ifos),
                "materialized_path": str(artifact.resolve()),
                "materialized_sha256": file_sha256(artifact),
                "source_materialized_path": str(source["path"]),
                "source_materialized_sha256": str(row["materialized_sha256"]),
                "source_ifos": source["ifos"],
                "storage_mode": "signal_scaled_float16",
                "signal_dtype": "scaled_float16_with_float64_ifo_peak",
                "signal_reconstruction": reconstruction,
                "signal_summary": signal_summary,
                "common_ifo_projection_equivalence": per_ifo_equivalence,
                "optimal_snr_by_ifo": by_ifo,
                "network_optimal_snr": network_snr,
                "optimal_snr_stratum": optimal_snr_stratum(network_snr),
                "optimal_snr_definition": (
                    "PyCBC sigma on the analysis-window physical projection with "
                    "frozen LALSimulation O4 reference PSD by detector"
                ),
                "reference_psd_models_by_ifo": psd_models,
                "detector_set_expansion_role": "robustness_ablation_not_sample_scaling",
                "signal_projection_background_independent": True,
            }
        )
        if index % 10 == 0 or index == len(rows):
            atomic_write_text(
                partial_path,
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in completed),
            )
            atomic_write_json(
                state_path,
                {
                    "status": "in_progress",
                    "run_identity": run_identity,
                    "completed": len(completed),
                    "requested": len(rows),
                },
            )

    manifest = output / "expanded_injections.jsonl"
    atomic_write_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in completed),
    )
    overlap_values = [row["normalized_overlap"] for row in common_equivalence]
    error_values = [row["relative_l2_error"] for row in common_equivalence]
    report = {
        "status": "verified_physical_detector_set_expansion",
        "passed": True,
        "scientific_claim_allowed": False,
        "same_distribution_data_scaling_claim_allowed": False,
        "detector_set_robustness_ablation_ready": True,
        "scientific_blocker": (
            "reference-PSD detector expansion is a training robustness ablation; "
            "empirical-noise O4 transfer, continuous-background FAR/IFAR/<VT> and "
            "locked evaluation remain required"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "selected_split": split,
        "rows": len(completed),
        "unique_injection_ids": len({str(row["injection_id"]) for row in completed}),
        "unique_waveform_ids": len({str(row["waveform_id"]) for row in completed}),
        "unique_gps_blocks": len({str(row["gps_block"]) for row in completed}),
        "source_identity_audit": identity,
        "source_detector_sets": dict(
            sorted(
                Counter(
                    "+".join(sorted(str(value) for value in row["source_ifos"]))
                    for row in completed
                ).items()
            )
        ),
        "target_detector_sets": dict(
            sorted(
                Counter(
                    "+".join(sorted(str(value) for value in row["ifos"]))
                    for row in completed
                ).items()
            )
        ),
        "reference_psd_models_by_ifo": psd_models,
        "common_ifo_projection_equivalence": {
            "comparisons": len(common_equivalence),
            "minimum_normalized_overlap": float(min(overlap_values)),
            "maximum_relative_l2_error": float(max(error_values)),
            "required_minimum_normalized_overlap": minimum_overlap,
            "required_maximum_relative_l2_error": maximum_relative_error,
            "passed": True,
        },
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "run_identity": run_identity,
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "detector_set_expansion_report.json", report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(completed),
            "requested": len(rows),
            "manifest_sha256": report["manifest_sha256"],
        },
    )
    return report
