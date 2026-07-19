from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .io import atomic_write_json, file_sha256


API_ROOT = "https://gwosc.org/api/v2"
USER_AGENT = "GW-YOLO-research/0.1"


def _api_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object from {url}")
    return value


def resolve_event(event: str) -> dict[str, Any]:
    event_record = _api_json(f"{API_ROOT}/events/{event}")
    versions = event_record.get("versions", [])
    if not versions:
        raise ValueError(f"GWOSC returned no versions for {event}")
    preferred = next(
        (item for item in versions if item.get("catalog") == "O4_Discovery_Papers"),
        versions[0],
    )
    detail = _api_json(str(preferred["detail_url"]))
    return {
        "event": event,
        "gps": float(detail["gps"]),
        "run": str(detail["run"]),
        "version": int(detail["version"]),
        "catalog": str(detail["catalog"]),
        "detectors": [str(item) for item in detail.get("detectors", [])],
    }


def event_strain_files(
    event: str,
    detectors: Iterable[str] | None = None,
    sample_rate_khz: int = 4,
) -> list[dict[str, Any]]:
    wanted = set(detectors or [])
    payload = _api_json(f"{API_ROOT}/events/{event}/strain-files")
    records = []
    for item in payload.get("results", []):
        if int(item["sample_rate_kHz"]) != sample_rate_khz:
            continue
        if wanted and str(item["detector"]) not in wanted:
            continue
        records.append(
            {
                "detector": str(item["detector"]),
                "sample_rate": sample_rate_khz * 1024,
                "gps_start": int(item["gps_start"]),
                "hdf5_url": str(item["hdf5_url"]),
            }
        )
    records.sort(key=lambda item: item["detector"])
    if wanted - {record["detector"] for record in records}:
        raise ValueError(f"Missing GWOSC strain for detectors: {sorted(wanted - {record['detector'] for record in records})}")
    if not records:
        raise ValueError(f"No {sample_rate_khz} kHz strain files found for {event}")
    return records


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value else None


def download_resumable(url: str, destination: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = _remote_size(url)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": file_sha256(target),
            "downloaded": False,
        }

    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and response.status == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            offset = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise IOError(f"Incomplete download for {url}: {actual_size} != {expected_size}")
    os.replace(partial, target)
    return {
        "path": str(target),
        "bytes": actual_size,
        "sha256": file_sha256(target),
        "downloaded": True,
    }


def _hdf_scalar(handle: Any, path: str) -> Any:
    value = handle[path][()]
    return value.item() if hasattr(value, "item") else value


def read_hdf5_segment(path: str | Path, gps_center: float, duration: float) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Reading GWOSC HDF5 requires the optional 'h5py' dependency") from exc

    with h5py.File(path, "r") as handle:
        gps_start = float(_hdf_scalar(handle, "meta/GPSstart"))
        dataset = handle["strain/Strain"]
        spacing = float(dataset.attrs["Xspacing"])
        sample_rate = int(round(1.0 / spacing))
        start = int(round((gps_center - duration / 2 - gps_start) * sample_rate))
        stop = start + int(round(duration * sample_rate))
        if start < 0 or stop > dataset.shape[0]:
            raise ValueError(f"Requested [{start}:{stop}] outside strain file with {dataset.shape[0]} samples")
        strain = np.asarray(dataset[start:stop], dtype=np.float64)
        quality: dict[str, np.ndarray] = {}
        for key in ("DQmask", "Injmask"):
            dataset_path = f"quality/simple/{key}"
            if dataset_path in handle:
                second_start = int(np.floor(gps_center - duration / 2 - gps_start))
                second_stop = int(np.ceil(gps_center + duration / 2 - gps_start))
                quality[key] = np.asarray(handle[dataset_path][second_start:second_stop])
    return {
        "strain": strain,
        "sample_rate": sample_rate,
        "gps_start": gps_center - duration / 2,
        "quality": quality,
    }


def _fft_downsample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.copy()
    if source_rate % target_rate:
        raise ValueError("source sample rate must be an integer multiple of target rate")
    ratio = source_rate // target_rate
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / source_rate)
    spectrum[frequencies >= target_rate * 0.45] = 0
    filtered = np.fft.irfft(spectrum, n=signal.size)
    return filtered[::ratio]


def _whiten(signal: np.ndarray, smoothing_bins: int = 129) -> np.ndarray:
    centered = signal - np.median(signal)
    spectrum = np.fft.rfft(centered)
    raw_psd = np.abs(spectrum) ** 2
    width = min(smoothing_bins, max(3, raw_psd.size // 16 * 2 + 1))
    kernel = np.ones(width, dtype=np.float64) / width
    psd = np.convolve(raw_psd, kernel, mode="same")
    floor = max(float(np.median(psd)) * 1e-6, np.finfo(np.float64).tiny)
    whitened = np.fft.irfft(spectrum / np.sqrt(np.maximum(psd, floor)), n=signal.size)
    return (whitened / max(float(np.std(whitened)), 1e-12)).astype(np.float32)


def run_gwosc_pilot(
    event: str,
    cache_dir: str | Path,
    output_dir: str | Path,
    detectors: Iterable[str] | None = None,
    context_duration: float = 64.0,
    output_duration: float = 8.0,
    target_sample_rate: int = 1024,
    allow_locked_evaluation_data: bool = False,
) -> dict[str, Any]:
    event_record = resolve_event(event)
    if str(event_record["run"]).lower().startswith("o4b") and not allow_locked_evaluation_data:
        raise ValueError("O4b is locked evaluation data; pass explicit unlock only for a frozen evaluation")
    wanted = list(detectors or event_record["detectors"])
    files = event_strain_files(event, wanted, sample_rate_khz=4)
    cache = Path(cache_dir)
    output = Path(output_dir)
    downloads = []
    raw_segments = []
    quality = {}
    for record in files:
        filename = Path(record["hdf5_url"]).name
        download = download_resumable(record["hdf5_url"], cache / filename)
        downloads.append({**record, **download})
        segment = read_hdf5_segment(download["path"], event_record["gps"], context_duration)
        resampled = _fft_downsample(segment["strain"], segment["sample_rate"], target_sample_rate)
        raw_segments.append(resampled)
        quality[record["detector"]] = {
            key: value.astype(int).tolist() for key, value in segment["quality"].items()
        }

    raw = np.stack(raw_segments).astype(np.float32)
    whitened_context = np.stack([_whiten(item) for item in raw])
    output_samples = int(round(output_duration * target_sample_rate))
    context_center = whitened_context.shape[1] // 2
    selection = slice(context_center - output_samples // 2, context_center + output_samples // 2)
    whitened = whitened_context[:, selection]
    raw_selected = raw[:, selection]
    q_values = (4.0, 8.0, 16.0)
    power = multiresolution_power(
        whitened,
        target_sample_rate,
        q_values,
        frequency_bins=96,
        time_bins=96,
        fmin=16.0,
        fmax=500.0,
    )

    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / f"{event}_real_o4a.npz"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{tensor_path.name}.", suffix=".npz", dir=output)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            features=_normalize_power(power),
            whitened_strain=whitened,
            raw_strain=raw_selected,
            ifos=np.asarray([record["detector"] for record in files]),
            q_values=np.asarray(q_values, dtype=np.float32),
            sample_rate=np.asarray(target_sample_rate, dtype=np.int32),
            event_gps=np.asarray(event_record["gps"], dtype=np.float64),
        )
        os.replace(temporary, tensor_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    report = {
        "event": event_record,
        "development_only": True,
        "detectors": [record["detector"] for record in files],
        "context_duration": context_duration,
        "output_duration": output_duration,
        "target_sample_rate": target_sample_rate,
        "tensor_path": str(tensor_path),
        "tensor_sha256": file_sha256(tensor_path),
        "tensor_shape": list(power.shape),
        "quality": quality,
        "downloads": downloads,
        "preprocessing": "FFT anti-alias downsample; context PSD whitening; Q-conditioned STFT",
    }
    atomic_write_json(output / f"{event}_report.json", report)
    return report
