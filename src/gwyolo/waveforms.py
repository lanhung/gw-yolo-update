from __future__ import annotations

import json
import os
import platform
import shlex
import sys
import tempfile
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .gwosc import _fft_downsample, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


def optimal_snr_stratum(network_snr: float) -> str:
    if not np.isfinite(network_snr) or network_snr < 0:
        raise ValueError("network optimal SNR must be finite and non-negative")
    for upper, label in ((4.0, "snr_lt_4"), (8.0, "snr_4_8"), (15.0, "snr_8_15"), (30.0, "snr_15_30")):
        if network_snr < upper:
            return label
    return "snr_ge_30"


def annotate_materialized_optimal_snr(
    manifest_path: str | Path,
    output_dir: str | Path,
    low_frequency: float = 20.0,
    high_frequency: float = 500.0,
    psd_segment_seconds: float = 8.0,
    psd_stride_seconds: float = 4.0,
) -> dict[str, Any]:
    """Add empirical-noise optimal SNR to materialized physical injections."""
    try:
        from pycbc.filter import sigma
        from pycbc.psd import interpolate, welch
        from pycbc.types import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Optimal-SNR annotation requires PyCBC") from exc
    if low_frequency <= 0 or high_frequency <= low_frequency:
        raise ValueError("SNR frequency bounds are invalid")
    if psd_segment_seconds <= 0 or psd_stride_seconds <= 0:
        raise ValueError("PSD segment and stride must be positive")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Materialized manifest cannot be empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "manifest_sha256": file_sha256(manifest_path),
        "low_frequency": low_frequency,
        "high_frequency": high_frequency,
        "psd_segment_seconds": psd_segment_seconds,
        "psd_stride_seconds": psd_stride_seconds,
        "signal_support": "recorded_analysis_window_only",
        "pycbc_version": version("pycbc"),
        "lalsuite_version": version("lalsuite"),
    }
    state_path = output / "snr_annotation_state.json"
    partial_path = output / "materialized_injections_snr.partial.jsonl"
    completed = []
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("run_identity") != run_identity:
            raise ValueError("Existing SNR annotation belongs to a different run")
        if partial_path.is_file():
            with partial_path.open("r", encoding="utf-8") as handle:
                completed = [json.loads(line) for line in handle if line.strip()]
    elif partial_path.is_file():
        raise ValueError("Partial SNR manifest exists without run identity")
    requested_ids = [str(row["injection_id"]) for row in rows]
    if [str(row["injection_id"]) for row in completed] != requested_ids[: len(completed)]:
        raise ValueError("Partial SNR annotation is not a prefix of the requested manifest")
    verified_background_hashes: dict[str, str] = {}
    started = time.time()
    for index, row in enumerate(rows[len(completed) :], start=len(completed) + 1):
        context = load_materialized_context(row, verified_background_hashes)
        sample_rate = int(context["sample_rate"])
        segment_samples = int(round(psd_segment_seconds * sample_rate))
        stride_samples = int(round(psd_stride_seconds * sample_rate))
        if segment_samples > context["noise"].shape[1]:
            raise ValueError("PSD segment is longer than materialized context")
        by_ifo = {}
        analysis_start = int(context["analysis_start_index"])
        analysis_stop = int(context["analysis_stop_index"])
        for ifo, noise, signal_values in zip(
            context["ifos"], context["noise"], context["signal"]
        ):
            noise_series = TimeSeries(noise, delta_t=1.0 / sample_rate)
            signal_series = TimeSeries(
                signal_values[analysis_start:analysis_stop], delta_t=1.0 / sample_rate
            )
            signal_frequency = signal_series.to_frequencyseries()
            psd = interpolate(
                welch(
                    noise_series,
                    seg_len=segment_samples,
                    seg_stride=stride_samples,
                    avg_method="median",
                ),
                signal_frequency.delta_f,
            )
            ifo_snr = float(
                sigma(
                    signal_series,
                    psd=psd,
                    low_frequency_cutoff=low_frequency,
                    high_frequency_cutoff=min(high_frequency, sample_rate / 2.0 - 1.0),
                )
            )
            if not np.isfinite(ifo_snr) or ifo_snr < 0:
                raise ValueError(f"Invalid optimal SNR for {row['injection_id']} {ifo}")
            by_ifo[str(ifo)] = ifo_snr
        network_snr = float(np.sqrt(np.sum(np.square(list(by_ifo.values())))))
        completed.append(
            {
                **row,
                "optimal_snr_by_ifo": by_ifo,
                "network_optimal_snr": network_snr,
                "optimal_snr_stratum": optimal_snr_stratum(network_snr),
                "optimal_snr_definition": (
                    "PyCBC sigma on the recorded analysis-window signal with a median-Welch "
                    "empirical noise PSD estimated from the full context"
                ),
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
    target = output / "materialized_injections_snr.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(item, sort_keys=True) + "\n" for item in completed)
    )
    report = {
        "status": "empirical_noise_optimal_snr_annotation",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "SNR annotation supports curriculum and stratification but does not replace locked "
            "injection recovery at frozen FAR"
        ),
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": file_sha256(manifest_path),
        "output_manifest_path": str(target),
        "output_manifest_sha256": file_sha256(target),
        "rows": len(completed),
        "split_counts": dict(sorted(Counter(row["split"] for row in completed).items())),
        "family_counts": dict(
            sorted(Counter(row["source_family"] for row in completed).items())
        ),
        "snr_stratum_counts": dict(
            sorted(Counter(row["optimal_snr_stratum"] for row in completed).items())
        ),
        "network_snr_quantiles": {
            str(quantile): float(
                np.quantile([row["network_optimal_snr"] for row in completed], quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
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
    atomic_write_json(output / "snr_annotation_report.json", report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(completed),
            "requested": len(rows),
            "manifest_sha256": report["output_manifest_sha256"],
        },
    )
    return report


def waveform_equivalence_metrics(
    wrapper: np.ndarray,
    reference: np.ndarray,
    wrapper_epoch: float,
    reference_epoch: float,
) -> dict[str, Any]:
    wrapper_values = np.asarray(wrapper, dtype=np.complex128).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.complex128).reshape(-1)
    same_length = wrapper_values.size == reference_values.size
    compared = min(wrapper_values.size, reference_values.size)
    if compared == 0:
        raise ValueError("Waveform comparison cannot be empty")
    left = wrapper_values[:compared]
    right = reference_values[:compared]
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Waveform comparison contains non-finite values")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Waveform comparison cannot use a zero-norm waveform")
    return {
        "wrapper_samples": int(wrapper_values.size),
        "reference_samples": int(reference_values.size),
        "same_length": same_length,
        "normalized_complex_overlap": float(
            abs(np.vdot(left, right)) / (left_norm * right_norm)
        ),
        "relative_l2_error": float(np.linalg.norm(left - right) / right_norm),
        "amplitude_norm_ratio": left_norm / right_norm,
        "epoch_difference_seconds": abs(float(wrapper_epoch) - float(reference_epoch)),
    }


def validate_waveform_backend(
    recipe_manifest: str | Path,
    output_path: str | Path,
    sample_rate: int = 2048,
    reference_duration: float = 128.0,
    per_family: int = 5,
    minimum_overlap: float = 0.999999,
    maximum_relative_error: float = 1e-3,
    maximum_epoch_error_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Validate PyCBC parameter routing against the direct LALSimulation FD API."""
    if sample_rate <= 0 or reference_duration <= 0 or per_family <= 0:
        raise ValueError("sample rate, reference duration and per-family count must be positive")
    try:
        import lal
        import lalsimulation
        from pycbc.waveform import get_fd_waveform
    except ImportError as exc:
        raise RuntimeError("Waveform validation requires PyCBC and LALSuite") from exc

    with Path(recipe_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Waveform validation recipe manifest cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_family"]), []).append(row)
    selected = []
    for family in sorted(grouped):
        selected.extend(
            sorted(grouped[family], key=lambda row: str(row["injection_id"]))[:per_family]
        )
    delta_f = 1.0 / reference_duration
    f_max = sample_rate / 2.0
    cases = []
    for recipe in selected:
        approximant = str(recipe["waveform_approximant"])
        pycbc_plus, pycbc_cross = get_fd_waveform(
            approximant=approximant,
            mass1=float(recipe["mass_1_detector_msun"]),
            mass2=float(recipe["mass_2_detector_msun"]),
            spin1z=float(recipe["spin_1z"]),
            spin2z=float(recipe["spin_2z"]),
            lambda1=float(recipe.get("lambda_1", 0.0)),
            lambda2=float(recipe.get("lambda_2", 0.0)),
            inclination=float(recipe["inclination"]),
            coa_phase=float(recipe["coalescence_phase"]),
            distance=float(recipe["luminosity_distance_mpc"]),
            delta_f=delta_f,
            f_lower=float(recipe["f_lower_hz"]),
            f_final=f_max,
        )
        parameters = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
            parameters, float(recipe.get("lambda_1", 0.0))
        )
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
            parameters, float(recipe.get("lambda_2", 0.0))
        )
        reference_plus, reference_cross = lalsimulation.SimInspiralChooseFDWaveform(
            float(recipe["mass_1_detector_msun"]) * lal.MSUN_SI,
            float(recipe["mass_2_detector_msun"]) * lal.MSUN_SI,
            0.0,
            0.0,
            float(recipe["spin_1z"]),
            0.0,
            0.0,
            float(recipe["spin_2z"]),
            float(recipe["luminosity_distance_mpc"]) * 1e6 * lal.PC_SI,
            float(recipe["inclination"]),
            float(recipe["coalescence_phase"]),
            0.0,
            0.0,
            0.0,
            delta_f,
            float(recipe["f_lower_hz"]),
            f_max,
            0.0,
            parameters,
            lalsimulation.GetApproximantFromString(approximant),
        )
        polarizations = {
            "plus": waveform_equivalence_metrics(
                pycbc_plus.numpy(),
                reference_plus.data.data,
                float(pycbc_plus.start_time),
                float(reference_plus.epoch),
            ),
            "cross": waveform_equivalence_metrics(
                pycbc_cross.numpy(),
                reference_cross.data.data,
                float(pycbc_cross.start_time),
                float(reference_cross.epoch),
            ),
        }
        passed = all(
            metrics["same_length"]
            and metrics["normalized_complex_overlap"] >= minimum_overlap
            and metrics["relative_l2_error"] <= maximum_relative_error
            and abs(metrics["amplitude_norm_ratio"] - 1.0) <= maximum_relative_error
            and metrics["epoch_difference_seconds"] <= maximum_epoch_error_seconds
            for metrics in polarizations.values()
        )
        cases.append(
            {
                "injection_id": recipe["injection_id"],
                "source_family": recipe["source_family"],
                "approximant": approximant,
                "polarizations": polarizations,
                "passed": passed,
            }
        )
    passed = bool(cases) and all(case["passed"] for case in cases)
    report = {
        "passed": passed,
        "validation_scope": "external_reference_waveform_equivalence",
        "wrapper_backend": "pycbc_get_fd_waveform",
        "reference_backend": "direct_lalsimulation_SimInspiralChooseFDWaveform",
        "limitation": (
            "This validates parameter routing, complex strain, amplitude and epoch against the "
            "direct LALSimulation API; detector projection and astrophysical population validity "
            "remain separate gates."
        ),
        "recipe_manifest_path": str(recipe_manifest),
        "recipe_manifest_sha256": file_sha256(recipe_manifest),
        "families": sorted(grouped),
        "approximants": sorted({str(row["waveform_approximant"]) for row in selected}),
        "selected_cases": len(cases),
        "per_family": per_family,
        "sample_rate": sample_rate,
        "reference_duration": reference_duration,
        "thresholds": {
            "minimum_overlap": minimum_overlap,
            "maximum_relative_error": maximum_relative_error,
            "maximum_epoch_error_seconds": maximum_epoch_error_seconds,
        },
        "versions": {"pycbc": version("pycbc"), "lalsuite": version("lalsuite")},
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "cases": cases,
    }
    atomic_write_json(output_path, report)
    if not passed:
        raise RuntimeError(f"Waveform backend validation failed; see {output_path}")
    return report


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


def pack_scaled_float16_signal(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Pack physical strain without float16 underflow and certify reconstruction."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scaled signal storage expects finite [IFO, time] strain")
    peaks = np.max(np.abs(values), axis=1)
    normalized = np.divide(
        values,
        peaks[:, None],
        out=np.zeros_like(values),
        where=peaks[:, None] > 0,
    )
    packed = normalized.astype(np.float16)
    reconstructed = packed.astype(np.float64) * peaks[:, None]
    denominator = float(np.linalg.norm(values))
    relative_l2 = float(
        np.linalg.norm(reconstructed - values) / max(denominator, 1e-300)
    )
    overlap_denominator = float(np.linalg.norm(reconstructed)) * denominator
    overlap = (
        float(np.vdot(reconstructed, values).real / overlap_denominator)
        if overlap_denominator > 0
        else 1.0
    )
    metrics = {
        "relative_l2_error": relative_l2,
        "normalized_overlap": overlap,
        "maximum_absolute_normalized_error": float(
            np.max(np.abs(packed.astype(np.float64) - normalized))
        ),
    }
    if relative_l2 > 1e-3 or overlap < 0.999999:
        raise ValueError(f"scaled float16 signal reconstruction gate failed: {metrics}")
    return packed, peaks.astype(np.float64), metrics


def load_materialized_context(
    row: dict[str, Any], verified_background_hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    artifact = Path(row["materialized_path"])
    if file_sha256(artifact) != str(row["materialized_sha256"]):
        raise ValueError("materialized array hash mismatch")
    with np.load(artifact, allow_pickle=False) as arrays:
        source_rate = int(arrays["sample_rate"])
        ifos = [str(value) for value in arrays["ifos"].tolist()]
        context_start = float(arrays["context_gps_start"])
        analysis_start = float(arrays["analysis_gps_start"])
        analysis_start_index = int(arrays["analysis_start_index"])
        analysis_stop_index = int(arrays["analysis_stop_index"])
        if "signal" in arrays:
            signal = np.asarray(arrays["signal"], dtype=np.float64)
        elif "signal_scaled" in arrays and "signal_peak_scale" in arrays:
            signal = np.asarray(arrays["signal_scaled"], dtype=np.float64) * np.asarray(
                arrays["signal_peak_scale"], dtype=np.float64
            )[:, None]
        else:
            raise ValueError("materialized artifact lacks a supported signal representation")
        stored_noise = (
            np.asarray(arrays["noise"], dtype=np.float64) if "noise" in arrays else None
        )
        stored_mixture = (
            np.asarray(arrays["strain"], dtype=np.float64) if "strain" in arrays else None
        )
    if stored_noise is None:
        verified = verified_background_hashes if verified_background_hashes is not None else {}
        context_duration = signal.shape[1] / source_rate
        context_center = context_start + context_duration / 2.0
        detector_noise = []
        for ifo in ifos:
            source = row["background_source_files"][ifo]
            source_path = str(source["path"])
            expected_hash = str(source["sha256"])
            actual_hash = verified.get(source_path)
            if actual_hash is None:
                actual_hash = file_sha256(source_path)
                verified[source_path] = actual_hash
            if actual_hash != expected_hash:
                raise ValueError(f"background source hash mismatch for {ifo}")
            segment = read_hdf5_segment(source_path, context_center, context_duration)
            detector_noise.append(
                _fft_downsample(
                    np.asarray(segment["strain"], dtype=np.float64),
                    int(segment["sample_rate"]),
                    source_rate,
                )
            )
        noise = np.stack(detector_noise)
    else:
        noise = stored_noise
    if noise.shape != signal.shape:
        raise ValueError("reconstructed background shape differs from signal")
    mixture = noise + signal if stored_mixture is None else stored_mixture
    if mixture.shape != signal.shape:
        raise ValueError("stored mixture shape differs from signal")
    return {
        "artifact": artifact,
        "sample_rate": source_rate,
        "ifos": ifos,
        "context_gps_start": context_start,
        "analysis_gps_start": analysis_start,
        "analysis_start_index": analysis_start_index,
        "analysis_stop_index": analysis_stop_index,
        "noise": noise,
        "signal": signal,
        "mixture": mixture,
    }


def materialize_recipe(
    recipe: dict[str, Any],
    background: dict[str, Any],
    backend: Any,
    sample_rate: int,
    output_path: str | Path,
    context_duration: float = 64.0,
    storage_mode: str = "signal_only",
) -> dict[str, Any]:
    if storage_mode not in {"signal_only", "signal_scaled_float16", "full"}:
        raise ValueError("unsupported materialized storage mode")
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
    stored_arrays = {
        "ifos": np.asarray(ifos),
        "sample_rate": np.asarray(sample_rate, dtype=np.int64),
        "context_gps_start": np.asarray(context_start, dtype=np.float64),
        "analysis_gps_start": np.asarray(background["gps_start"], dtype=np.float64),
        "analysis_start_index": np.asarray(analysis_start_index, dtype=np.int64),
        "analysis_stop_index": np.asarray(analysis_stop_index, dtype=np.int64),
    }
    reconstruction = None
    if storage_mode == "signal_scaled_float16":
        packed, peaks, reconstruction = pack_scaled_float16_signal(signal)
        stored_arrays.update({"signal_scaled": packed, "signal_peak_scale": peaks})
        signal_dtype = "scaled_float16_with_float64_ifo_peak"
    else:
        stored_arrays["signal"] = signal.astype(np.float32)
        signal_dtype = "float32"
    if storage_mode == "full":
        stored_arrays.update(
            {"noise": noise.astype(np.float32), "strain": mixture.astype(np.float32)}
        )
    _atomic_save_npz(target, **stored_arrays)
    return {
        **recipe,
        "materialized_path": str(target),
        "materialized_sha256": file_sha256(target),
        "sample_rate": sample_rate,
        "samples_per_ifo": int(noise.shape[1]),
        "context_duration": context_duration,
        "analysis_start_index": analysis_start_index,
        "analysis_stop_index": analysis_stop_index,
        "storage_mode": storage_mode,
        "signal_dtype": signal_dtype,
        "signal_reconstruction": reconstruction,
        "signal_summary": signal_summary,
        "time_alignment": "integer_copy_or_lanczos_sinc_half_width_16",
        "background_source_files": {
            ifo: {
                "path": str(background["source_files"][ifo]["path"]),
                "sha256": str(background["source_files"][ifo]["sha256"]),
            }
            for ifo in ifos
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
    storage_mode: str = "signal_only",
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
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "recipe_manifest_sha256": file_sha256(recipe_manifest),
        "background_manifest_sha256": file_sha256(background_manifest),
        "selected_injection_ids_hash": canonical_hash(
            [str(row["injection_id"]) for row in selected], 64
        ),
        "sample_rate": sample_rate,
        "context_duration": context_duration,
        "storage_mode": storage_mode,
        "signal_dtype": (
            "scaled_float16_with_float64_ifo_peak"
            if storage_mode == "signal_scaled_float16"
            else "float32"
        ),
        "backend": backend.metadata,
        "backend_validation_report_sha256": (
            file_sha256(backend_validation_report) if backend_validation_report else None
        ),
    }
    state_path = output / "materialization_state.json"
    partial_path = output / "materialized_injections.partial.jsonl"
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as handle:
            prior_state = json.load(handle)
        if prior_state.get("run_identity") != run_identity:
            raise ValueError("Existing materialization state belongs to a different run")
    materialized = []
    if partial_path.is_file():
        with partial_path.open("r", encoding="utf-8") as handle:
            materialized = [json.loads(line) for line in handle if line.strip()]
    selected_ids = {str(row["injection_id"]) for row in selected}
    completed_by_id = {}
    for completed in materialized:
        injection_id = str(completed["injection_id"])
        if injection_id not in selected_ids or injection_id in completed_by_id:
            raise ValueError("Partial materialization contains unexpected or duplicate injection")
        if file_sha256(completed["materialized_path"]) != str(
            completed["materialized_sha256"]
        ):
            raise ValueError(f"Partial materialized hash mismatch for {injection_id}")
        completed_by_id[injection_id] = completed
    materialized = []
    for index, recipe in enumerate(selected, start=1):
        injection_id = str(recipe["injection_id"])
        if injection_id in completed_by_id:
            materialized.append(completed_by_id[injection_id])
            continue
        artifact_path = output / "arrays" / f"{recipe['injection_id']}.npz"
        materialized.append(
            materialize_recipe(
                recipe,
                backgrounds[str(recipe["background_window_id"])],
                backend,
                sample_rate,
                artifact_path,
                context_duration,
                storage_mode,
            )
        )
        if index % 10 == 0 or index == len(selected):
            atomic_write_text(
                partial_path,
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized),
            )
            atomic_write_json(
                state_path,
                {
                    "status": "in_progress",
                    "run_identity": run_identity,
                    "completed": len(materialized),
                    "requested": len(selected),
                },
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
        "storage_mode": storage_mode,
        "signal_dtype": run_identity["signal_dtype"],
        "materialized_bytes": sum(
            Path(row["materialized_path"]).stat().st_size for row in materialized
        ),
        "maximum_signal_reconstruction_relative_l2": max(
            (
                float(row["signal_reconstruction"]["relative_l2_error"])
                for row in materialized
                if row.get("signal_reconstruction") is not None
            ),
            default=None,
        ),
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
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(materialized),
            "requested": len(selected),
            "manifest_sha256": report["manifest_sha256"],
        },
    )
    return report
