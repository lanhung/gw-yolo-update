from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .gwosc import _fft_downsample, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, file_sha256


def place_waveform_samples(
    segment_start: float,
    sample_rate: int,
    segment_samples: int,
    waveform_start: float,
    waveform: np.ndarray,
    interpolation_half_width: int = 16,
) -> np.ndarray:
    """Place an absolute-time waveform using integer copy or Lanczos sinc interpolation."""
    if sample_rate <= 0 or segment_samples <= 0:
        raise ValueError("sample rate and segment size must be positive")
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("waveform contains non-finite values")
    offset_float = (float(waveform_start) - float(segment_start)) * sample_rate
    offset = int(round(offset_float))
    output = np.zeros(segment_samples, dtype=np.float64)
    if abs(offset_float - offset) > 1e-8:
        if interpolation_half_width < 2:
            raise ValueError("interpolation half width must be at least two")
        target_indices = np.arange(segment_samples, dtype=np.float64)
        source_coordinates = target_indices - offset_float
        centers = np.floor(source_coordinates).astype(np.int64)
        taps = np.arange(
            -interpolation_half_width + 1, interpolation_half_width + 1, dtype=np.int64
        )
        source_indices = centers[:, None] + taps[None, :]
        distances = source_coordinates[:, None] - source_indices
        weights = np.sinc(distances) * np.sinc(distances / interpolation_half_width)
        valid = (source_indices >= 0) & (source_indices < values.size)
        clipped_indices = np.clip(source_indices, 0, max(values.size - 1, 0))
        output = np.sum(values[clipped_indices] * weights * valid, axis=1)
        return output
    destination_start = max(offset, 0)
    destination_stop = min(offset + values.size, segment_samples)
    if destination_stop <= destination_start:
        return output
    source_start = destination_start - offset
    source_stop = source_start + destination_stop - destination_start
    output[destination_start:destination_stop] = values[source_start:source_stop]
    return output


def validate_recipe_identities(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise ValueError("Injection recipe manifest cannot be empty")
    for field in ("injection_id", "waveform_id"):
        counts = Counter(str(row[field]) for row in rows)
        duplicates = sorted(key for key, count in counts.items() if count != 1)
        if duplicates:
            raise ValueError(f"Duplicate {field} values: {duplicates[:10]}")
    split_waveforms: dict[str, set[str]] = {}
    split_backgrounds: dict[str, set[str]] = {}
    for row in rows:
        split = str(row["split"])
        split_waveforms.setdefault(split, set()).add(str(row["waveform_id"]))
        split_backgrounds.setdefault(split, set()).add(str(row["gps_block"]))
    for index, left in enumerate(sorted(split_waveforms)):
        for right in sorted(split_waveforms)[index + 1 :]:
            if split_waveforms[left] & split_waveforms[right]:
                raise ValueError(f"Waveform leakage between {left} and {right}")
            if split_backgrounds[left] & split_backgrounds[right]:
                raise ValueError(f"GPS-block leakage between {left} and {right}")
    return {
        "unique_injection_ids": len(rows),
        "unique_waveform_ids": len(rows),
        "unique_gps_blocks": len({str(row["gps_block"]) for row in rows}),
    }


class PyCBCWaveformBackend:
    def __init__(self) -> None:
        try:
            from pycbc.detector import Detector
            from pycbc.waveform import get_td_waveform
        except ImportError as exc:
            raise RuntimeError(
                "Physical injection materialization requires the optional PyCBC/LALSuite backend"
            ) from exc
        self._detector = Detector
        self._get_td_waveform = get_td_waveform
        try:
            pycbc_version = version("pycbc")
        except PackageNotFoundError:
            pycbc_version = "unknown"
        try:
            lalsuite_version = version("lalsuite")
        except PackageNotFoundError:
            lalsuite_version = "unknown"
        self.metadata = {
            "backend": "pycbc_lalsimulation",
            "pycbc_version": pycbc_version,
            "lalsuite_version": lalsuite_version,
        }

    def generate(
        self, recipe: dict[str, Any], ifos: list[str], sample_rate: int
    ) -> tuple[dict[str, tuple[float, np.ndarray]], dict[str, Any]]:
        required = (
            "mass_1_detector_msun",
            "mass_2_detector_msun",
            "luminosity_distance_mpc",
            "waveform_approximant",
            "f_lower_hz",
        )
        missing = [field for field in required if field not in recipe]
        if missing:
            raise ValueError(f"Recipe lacks detector-frame waveform fields: {missing}")
        hp, hc = self._get_td_waveform(
            approximant=str(recipe["waveform_approximant"]),
            mass1=float(recipe["mass_1_detector_msun"]),
            mass2=float(recipe["mass_2_detector_msun"]),
            spin1z=float(recipe["spin_1z"]),
            spin2z=float(recipe["spin_2z"]),
            lambda1=float(recipe.get("lambda_1", 0.0)),
            lambda2=float(recipe.get("lambda_2", 0.0)),
            inclination=float(recipe["inclination"]),
            coa_phase=float(recipe["coalescence_phase"]),
            distance=float(recipe["luminosity_distance_mpc"]),
            delta_t=1.0 / sample_rate,
            f_lower=float(recipe["f_lower_hz"]),
        )
        coalescence_gps = float(recipe["gps_time"])
        hp.start_time += coalescence_gps
        hc.start_time += coalescence_gps
        projected = {}
        signal_summary = {}
        for ifo in ifos:
            strain = self._detector(ifo).project_wave(
                hp,
                hc,
                float(recipe["right_ascension"]),
                float(recipe["declination"]),
                float(recipe["polarization"]),
                method="lal",
                reference_time=coalescence_gps,
            )
            values = np.asarray(strain, dtype=np.float64)
            projected[ifo] = (float(strain.start_time), values)
            signal_summary[ifo] = {
                "waveform_start_gps": float(strain.start_time),
                "waveform_samples": int(values.size),
                "waveform_peak_absolute_strain": float(np.max(np.abs(values))),
            }
        return projected, signal_summary


def _read_background(
    row: dict[str, Any], target_sample_rate: int, context_duration: float
) -> tuple[list[str], np.ndarray, float]:
    analysis_start = float(row["gps_start"])
    analysis_duration = float(row["duration"])
    if context_duration < analysis_duration:
        raise ValueError("context duration cannot be shorter than the analysis window")
    center = analysis_start + analysis_duration / 2.0
    segment_start = center - context_duration / 2.0
    detector_noise = []
    ifos = [str(ifo) for ifo in row["ifos"]]
    for ifo in ifos:
        source = row["source_files"][ifo]
        segment = read_hdf5_segment(source["path"], center, context_duration)
        values = np.asarray(segment["strain"], dtype=np.float64)
        values = _fft_downsample(values, int(segment["sample_rate"]), target_sample_rate)
        expected = int(round(context_duration * target_sample_rate))
        if values.size != expected:
            raise ValueError(f"Background segment for {ifo} has {values.size}, expected {expected}")
        detector_noise.append(values)
    return ifos, np.stack(detector_noise), segment_start


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
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


def materialize_recipe(
    recipe: dict[str, Any],
    background: dict[str, Any],
    backend: Any,
    sample_rate: int,
    output_path: str | Path,
    context_duration: float = 64.0,
) -> dict[str, Any]:
    if str(recipe["background_window_id"]) != str(background["window_id"]):
        raise ValueError("Recipe/background window identity mismatch")
    if str(recipe["gps_block"]) != str(background["gps_block"]):
        raise ValueError("Recipe/background GPS-block identity mismatch")
    if str(recipe["split"]) != str(background["split"]):
        raise ValueError("Recipe/background split mismatch")
    ifos, noise, context_start = _read_background(
        background, sample_rate, context_duration
    )
    projected, signal_summary = backend.generate(recipe, ifos, sample_rate)
    signal = np.stack(
        [
            place_waveform_samples(
                context_start,
                sample_rate,
                noise.shape[1],
                projected[ifo][0],
                projected[ifo][1],
            )
            for ifo in ifos
        ]
    )
    if not np.any(signal):
        raise ValueError(f"Projected waveform misses background window for {recipe['injection_id']}")
    mixture = noise + signal
    analysis_start_index = int(
        round((float(background["gps_start"]) - context_start) * sample_rate)
    )
    analysis_stop_index = analysis_start_index + int(
        round(float(background["duration"]) * sample_rate)
    )
    if analysis_start_index < 0 or analysis_stop_index > noise.shape[1]:
        raise ValueError("Analysis window falls outside materialized context")
    target = Path(output_path)
    _atomic_save_npz(
        target,
        noise=noise,
        signal=signal,
        strain=mixture,
        ifos=np.asarray(ifos),
        sample_rate=np.asarray(sample_rate, dtype=np.int64),
        context_gps_start=np.asarray(context_start, dtype=np.float64),
        analysis_gps_start=np.asarray(background["gps_start"], dtype=np.float64),
        analysis_start_index=np.asarray(analysis_start_index, dtype=np.int64),
        analysis_stop_index=np.asarray(analysis_stop_index, dtype=np.int64),
    )
    return {
        **recipe,
        "materialized_path": str(target),
        "materialized_sha256": file_sha256(target),
        "sample_rate": sample_rate,
        "samples_per_ifo": int(noise.shape[1]),
        "context_duration": context_duration,
        "analysis_start_index": analysis_start_index,
        "analysis_stop_index": analysis_stop_index,
        "signal_summary": signal_summary,
        "time_alignment": "integer_copy_or_lanczos_sinc_half_width_16",
        "background_source_sha256": {
            ifo: str(background["source_files"][ifo]["sha256"]) for ifo in ifos
        },
    }


def run_injection_materialization(
    recipe_manifest: str | Path,
    background_manifest: str | Path,
    output_dir: str | Path,
    sample_rate: int = 2048,
    split: str | None = None,
    limit: int | None = None,
    backend_validation_report: str | Path | None = None,
    context_duration: float = 64.0,
) -> dict[str, Any]:
    with Path(recipe_manifest).open("r", encoding="utf-8") as handle:
        recipes = [json.loads(line) for line in handle if line.strip()]
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        background_rows = [json.loads(line) for line in handle if line.strip()]
    identity_audit = validate_recipe_identities(recipes)
    selected = [row for row in recipes if split is None or row["split"] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No recipes selected for materialization")
    backgrounds = {str(row["window_id"]): row for row in background_rows}
    missing = sorted(
        {str(row["background_window_id"]) for row in selected} - set(backgrounds)
    )
    if missing:
        raise ValueError(f"Recipes reference missing background windows: {missing[:10]}")
    validation = None
    if backend_validation_report is not None:
        with Path(backend_validation_report).open("r", encoding="utf-8") as handle:
            validation = json.load(handle)
        if not validation.get("passed", False):
            raise ValueError("Waveform backend validation report did not pass")
        if validation.get("validation_scope") != "external_reference_waveform_equivalence":
            raise ValueError(
                "A claim-validating report must declare external_reference_waveform_equivalence"
            )
    backend = PyCBCWaveformBackend()
    output = Path(output_dir)
    materialized = []
    for recipe in selected:
        artifact_path = output / "arrays" / f"{recipe['injection_id']}.npz"
        materialized.append(
            materialize_recipe(
                recipe,
                backgrounds[str(recipe["background_window_id"])],
                backend,
                sample_rate,
                artifact_path,
                context_duration,
            )
        )
    manifest_path = output / "materialized_injections.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized),
    )
    report = {
        "status": (
            "materialized_externally_validated_backend"
            if validation
            else "integration_only_unvalidated_backend"
        ),
        "waveform_materialization_validated": validation is not None,
        "sensitivity_claim_allowed": False,
        "sensitivity_claim_blocker": "requires frozen thresholds, independent background exposure and locked injections",
        "selected_split": split,
        "selected_recipes": len(selected),
        "sample_rate": sample_rate,
        "context_duration": context_duration,
        "identity_audit": identity_audit,
        "backend": backend.metadata,
        "recipe_manifest_sha256": file_sha256(recipe_manifest),
        "background_manifest_sha256": file_sha256(background_manifest),
        "backend_validation_report_sha256": (
            file_sha256(backend_validation_report) if backend_validation_report else None
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(output / "materialization_report.json", report)
    return report
