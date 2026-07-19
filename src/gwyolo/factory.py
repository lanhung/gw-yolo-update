from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .provenance import SCENE_TYPES, SPLITS, SceneRecipe, audit_provenance, write_recipe_manifest


def _allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("scene count cannot be negative")
    unknown = set(fractions) - set(SCENE_TYPES)
    if unknown:
        raise ValueError(f"Unknown scene types: {sorted(unknown)}")
    weight_sum = sum(float(value) for value in fractions.values())
    if weight_sum <= 0:
        raise ValueError("scene-type fractions must have positive total weight")
    exact = {key: total * float(fractions.get(key, 0.0)) / weight_sum for key in SCENE_TYPES}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(SCENE_TYPES, key=lambda key: (exact[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def plan_recipes(config: dict[str, Any]) -> list[SceneRecipe]:
    data = config["data_factory"]
    ifos = tuple(str(item) for item in data["ifos"])
    q_values = tuple(float(item) for item in data["q_values"])
    duration = float(data["duration"])
    sample_rate = int(data["sample_rate"])
    observing_run = str(data["observing_run"])
    base_seed = int(data.get("seed", 0))
    scene_mix = {str(key): float(value) for key, value in data["scene_mix"].items()}
    gps_bases = data.get("gps_bases", {"train": 1_360_000_000, "val": 1_370_000_000, "test": 1_380_000_000})
    source_families = tuple(str(item) for item in data.get("source_families", ["BBH", "BNS", "NSBH"]))
    snr_range = tuple(float(item) for item in data.get("snr_range", [5.0, 30.0]))
    if len(snr_range) != 2 or snr_range[0] <= 0 or snr_range[1] < snr_range[0]:
        raise ValueError("snr_range must be [positive_min, max]")

    recipes: list[SceneRecipe] = []
    for split_index, split in enumerate(SPLITS):
        total = int(data["split_counts"].get(split, 0))
        counts = _allocate_counts(total, scene_mix)
        kinds = [kind for kind in SCENE_TYPES for _ in range(counts[kind])]
        rng = np.random.default_rng(base_seed + split_index * 1_000_003)
        rng.shuffle(kinds)
        for index, kind in enumerate(kinds):
            has_chirp = kind in {"chirp_only", "overlap"}
            has_glitch = kind in {"noise_only", "overlap"}
            suffix = f"{split}-{index:08d}"
            target_snr = float(rng.uniform(*snr_range)) if has_chirp else None
            recipes.append(
                SceneRecipe(
                    split=split,
                    scene_type=kind,
                    observing_run=observing_run,
                    gps_start=int(gps_bases[split]) + int(math.ceil(duration)) * index,
                    duration=duration,
                    sample_rate=sample_rate,
                    ifos=ifos,
                    q_values=q_values,
                    seed=base_seed + split_index * 1_000_003 + index,
                    waveform_id=f"waveform-{suffix}" if has_chirp else None,
                    injection_id=f"injection-{suffix}" if has_chirp else None,
                    glitch_id=f"glitch-{suffix}" if has_glitch else None,
                    glitch_ifo=str(rng.choice(ifos)) if has_glitch else None,
                    source_family=str(rng.choice(source_families)) if has_chirp else None,
                    target_snr=target_snr,
                )
            )
    return recipes


def _colored_noise(rng: np.random.Generator, size: int, sample_rate: int) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size)
    frequencies = np.fft.rfftfreq(size, 1.0 / sample_rate)
    shaping = np.ones_like(frequencies)
    positive = frequencies > 0
    shaping[positive] = np.sqrt(1.0 + (35.0 / np.maximum(frequencies[positive], 1.0)) ** 4)
    shaping *= np.sqrt(1.0 + (frequencies / 350.0) ** 2)
    colored = np.fft.irfft(np.fft.rfft(white) * shaping, n=size)
    colored -= np.mean(colored)
    return colored / max(float(np.std(colored)), 1e-12)


def _chirp_track(
    size: int,
    sample_rate: int,
    family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.zeros(size, dtype=np.float64)
    family_ranges = {
        "BBH": (20.0, 220.0, 0.7, 1.8),
        "BNS": (25.0, 480.0, 1.5, 3.5),
        "NSBH": (20.0, 350.0, 1.0, 2.6),
    }
    low, high, min_duration, max_duration = family_ranges.get(family, family_ranges["BBH"])
    chirp_duration = min(float(rng.uniform(min_duration, max_duration)), size / sample_rate * 0.85)
    count = max(16, int(chirp_duration * sample_rate))
    end = min(size - 1, int(size * rng.uniform(0.58, 0.82)))
    start = max(0, end - count)
    count = end - start
    if count < 2:
        return signal
    local_time = np.arange(count, dtype=np.float64) / sample_rate
    progress = np.linspace(0.0, 1.0, count, endpoint=False)
    f0 = float(rng.uniform(low, low * 1.5))
    f1 = min(float(rng.uniform(high * 0.7, high)), sample_rate * 0.45)
    frequency = f0 + (f1 - f0) * progress ** float(rng.uniform(1.8, 3.2))
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate + float(rng.uniform(0, 2 * np.pi))
    envelope = np.sin(np.pi * progress) ** 2 * (0.25 + 0.75 * progress)
    signal[start:end] = envelope * np.sin(phase)
    _ = local_time
    return signal


def _glitch_track(size: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(size, dtype=np.float64) / sample_rate
    center = float(rng.uniform(0.2, 0.8)) * size / sample_rate
    kind = str(rng.choice(["sine_gaussian", "blip", "scattering"]))
    if kind == "sine_gaussian":
        frequency = float(rng.uniform(30.0, min(450.0, sample_rate * 0.4)))
        width = float(rng.uniform(0.015, 0.12))
        return np.exp(-0.5 * ((time - center) / width) ** 2) * np.sin(
            2 * np.pi * frequency * (time - center)
        )
    if kind == "blip":
        width = float(rng.uniform(0.008, 0.035))
        carrier = float(rng.uniform(50.0, min(300.0, sample_rate * 0.35)))
        x = (time - center) / width
        return (1.0 - 2.0 * x**2) * np.exp(-x**2) * np.cos(2 * np.pi * carrier * (time - center))
    width = float(rng.uniform(0.2, 0.6))
    x = time - center
    frequency = float(rng.uniform(20.0, 55.0)) + 45.0 * np.abs(x)
    return np.exp(-0.5 * (x / width) ** 2) * np.sin(2 * np.pi * frequency * x)


def _shift(signal: np.ndarray, samples: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    if samples == 0:
        shifted[:] = signal
    elif samples > 0 and samples < signal.size:
        shifted[samples:] = signal[:-samples]
    elif samples < 0 and -samples < signal.size:
        shifted[:samples] = signal[-samples:]
    return shifted


def _stft_power(
    signal: np.ndarray,
    sample_rate: int,
    window_size: int,
    frequency_bins: int,
    time_bins: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    window_size = min(max(16, window_size), signal.size)
    nfft = 1 << int(math.ceil(math.log2(window_size)))
    hop = max(1, (signal.size - window_size) // max(time_bins - 1, 1))
    starts = list(range(0, signal.size - window_size + 1, hop))
    if not starts:
        starts = [0]
    window = np.hanning(window_size)
    columns = []
    for start in starts:
        frame = signal[start : start + window_size] * window
        columns.append(np.abs(np.fft.rfft(frame, n=nfft)) ** 2)
    power = np.stack(columns, axis=1)
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    target_frequencies = np.linspace(fmin, min(fmax, sample_rate / 2), frequency_bins)
    frequency_resampled = np.stack(
        [np.interp(target_frequencies, frequencies, power[:, column]) for column in range(power.shape[1])],
        axis=1,
    )
    old_time = np.linspace(0.0, 1.0, frequency_resampled.shape[1])
    new_time = np.linspace(0.0, 1.0, time_bins)
    return np.stack(
        [np.interp(new_time, old_time, row) for row in frequency_resampled], axis=0
    ).astype(np.float32)


def multiresolution_power(
    strain: np.ndarray,
    sample_rate: int,
    q_values: tuple[float, ...],
    frequency_bins: int,
    time_bins: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Return an IFO x Q x frequency x time numeric tensor.

    This dependency-free backend uses Q-conditioned STFT resolutions. A true constant-Q
    GWPy backend can consume the same recipes without changing physical split identity.
    """

    if strain.ndim != 2:
        raise ValueError("strain must have shape [ifo, time]")
    planes = []
    for ifo_signal in strain:
        ifo_planes = []
        for q_value in q_values:
            window_seconds = float(np.clip(q_value / 32.0, 0.0625, 1.0))
            ifo_planes.append(
                _stft_power(
                    ifo_signal,
                    sample_rate,
                    int(round(window_seconds * sample_rate)),
                    frequency_bins,
                    time_bins,
                    fmin,
                    fmax,
                )
            )
        planes.append(np.stack(ifo_planes))
    return np.stack(planes).astype(np.float32)


def _normalize_power(power: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(power, 0.0))
    flattened = logged.reshape(*logged.shape[:2], -1)
    median = np.median(flattened, axis=-1)[..., None, None]
    q25 = np.percentile(flattened, 25, axis=-1)[..., None, None]
    q75 = np.percentile(flattened, 75, axis=-1)[..., None, None]
    return np.clip((logged - median) / np.maximum(q75 - q25, 1e-6), -8.0, 16.0).astype(np.float32)


def _component_mask(power: np.ndarray) -> np.ndarray:
    flattened = power.reshape(*power.shape[:2], -1)
    peaks = np.max(flattened, axis=-1)[..., None, None]
    return (power >= np.maximum(peaks * 0.08, 1e-12)).astype(np.uint8)


def synthesize_scene(recipe: SceneRecipe, tensor_config: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(recipe.seed)
    size = int(round(recipe.duration * recipe.sample_rate))
    noise = np.stack([_colored_noise(rng, size, recipe.sample_rate) for _ in recipe.ifos])
    chirp = np.zeros_like(noise)
    glitch = np.zeros_like(noise)

    if recipe.injection_id:
        base_chirp = _chirp_track(size, recipe.sample_rate, str(recipe.source_family), rng)
        norm = max(float(np.linalg.norm(base_chirp)), 1e-12)
        base_chirp *= float(recipe.target_snr) / norm
        for index in range(len(recipe.ifos)):
            delay = int(rng.integers(-max(1, recipe.sample_rate // 100), max(2, recipe.sample_rate // 100)))
            antenna = float(rng.uniform(0.35, 1.0)) * (-1.0 if rng.random() < 0.25 else 1.0)
            chirp[index] = antenna * _shift(base_chirp, delay)
    if recipe.glitch_id:
        glitch_index = recipe.ifos.index(str(recipe.glitch_ifo))
        base_glitch = _glitch_track(size, recipe.sample_rate, rng)
        glitch[glitch_index] = base_glitch * float(rng.uniform(3.0, 12.0))

    mixture = noise + chirp + glitch
    kwargs = {
        "sample_rate": recipe.sample_rate,
        "q_values": recipe.q_values,
        "frequency_bins": int(tensor_config.get("frequency_bins", 96)),
        "time_bins": int(tensor_config.get("time_bins", 96)),
        "fmin": float(tensor_config.get("fmin", 16.0)),
        "fmax": float(tensor_config.get("fmax", 512.0)),
    }
    mixture_power = multiresolution_power(mixture, **kwargs)
    chirp_power = multiresolution_power(chirp, **kwargs)
    glitch_power = multiresolution_power(glitch, **kwargs)
    return {
        "features": _normalize_power(mixture_power),
        "chirp_mask": _component_mask(chirp_power) if recipe.injection_id else np.zeros_like(chirp_power, dtype=np.uint8),
        "glitch_mask": _component_mask(glitch_power) if recipe.glitch_id else np.zeros_like(glitch_power, dtype=np.uint8),
        "strain": mixture.astype(np.float32),
        "clean_strain": (noise + chirp).astype(np.float32),
        "chirp_strain": chirp.astype(np.float32),
        "glitch_strain": glitch.astype(np.float32),
        "sample_rate": np.asarray(recipe.sample_rate, dtype=np.int32),
        "ifos": np.asarray(recipe.ifos),
        "q_values": np.asarray(recipe.q_values, dtype=np.float32),
    }


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def run_data_factory(config_path: str | Path, output_dir: str | Path, limit: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    recipes = plan_recipes(config)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        recipes = recipes[:limit]
    audit = audit_provenance(recipes)
    if not audit["passed"]:
        raise ValueError(f"Physical provenance audit failed: {audit}")

    output = Path(output_dir)
    manifest_path = output / "recipes.jsonl"
    write_recipe_manifest(manifest_path, recipes)
    records = []
    factory_config = config["data_factory"]
    tensor_config = factory_config.get("tensor", {})
    materialization = str(factory_config.get("materialization", "full"))
    if materialization not in {"full", "tensor", "recipe_only"}:
        raise ValueError("materialization must be full, tensor, or recipe_only")
    if materialization != "recipe_only":
        for recipe in recipes:
            sample_path = output / recipe.split / f"{recipe.scene_id}.npz"
            arrays = synthesize_scene(recipe, tensor_config)
            if materialization == "tensor":
                arrays = {
                    key: value
                    for key, value in arrays.items()
                    if key in {"features", "chirp_mask", "glitch_mask", "sample_rate", "ifos", "q_values"}
                }
                arrays["features"] = arrays["features"].astype(np.float16)
            _atomic_savez(sample_path, arrays)
            records.append(
                {
                    "scene_id": recipe.scene_id,
                    "split": recipe.split,
                    "scene_type": recipe.scene_type,
                    "path": str(sample_path),
                    "sha256": file_sha256(sample_path),
                }
            )

    tensor_shape = None
    if records:
        with np.load(records[0]["path"], allow_pickle=False) as first:
            tensor_shape = list(first["features"].shape)
    sample_bytes = sum(Path(record["path"]).stat().st_size for record in records)
    per_scene_bytes = sample_bytes / len(records) if records else None
    manifest_bytes_per_scene = manifest_path.stat().st_size / max(len(recipes), 1)
    projection_counts = (10_000, 200_000, 500_000, 2_000_000)
    report = {
        "backend": "numpy_q_conditioned_stft_v1",
        "materialization": materialization,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "output_dir": str(output),
        "planned_scenes": len(recipes),
        "generated_scenes": len(records),
        "tensor_shape": tensor_shape,
        "scene_types": dict(sorted(Counter(record["scene_type"] for record in records).items())),
        "provenance_audit": audit,
        "storage": {
            "materialized_bytes": sample_bytes,
            "bytes_per_materialized_scene": per_scene_bytes,
            "manifest_bytes_per_scene": manifest_bytes_per_scene,
            "projected_materialized_bytes": {
                str(count): int(per_scene_bytes * count) if per_scene_bytes is not None else None
                for count in projection_counts
            },
            "projected_recipe_manifest_bytes": {
                str(count): int(manifest_bytes_per_scene * count) for count in projection_counts
            },
        },
        "records": records,
    }
    atomic_write_json(output / "factory_report.json", report)
    return report
