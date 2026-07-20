from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .physical_training import (
    gate_component_by_ifo_snr,
    relative_component_mask,
    scale_component_for_transform,
)
from .runtime import execution_provenance
from .waveforms import _atomic_save_npz, load_materialized_context


OVERLAP_ARTIFACT_VERSION = "gravityspy-physical-overlap-v2-network-aware"
OVERLAP_LEAKAGE_FIELDS = (
    "mixture_id",
    "injection_id",
    "waveform_id",
    "glitch_id",
    "injection_gps_block",
    "gps_block",
    "network_gps_block",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _require_unique(rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str) -> None:
    rows = list(rows)
    for field in fields:
        values = [str(row[field]) for row in rows]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError(f"{label} contains duplicate {field}: {duplicates[:5]}")


def _supported_ifos(row: dict[str, Any]) -> tuple[str, ...]:
    ifos = row.get("ifos")
    if ifos is None and row.get("optimal_snr_by_ifo"):
        ifos = list(row["optimal_snr_by_ifo"])
    if not isinstance(ifos, (list, tuple)) or not ifos:
        context = load_materialized_context(row)
        ifos = context["ifos"]
    return tuple(str(ifo) for ifo in ifos)


def _glitch_available_ifos(row: dict[str, Any]) -> tuple[str, ...]:
    values = row.get("available_ifos")
    if values is None:
        values = [row["ifo"]]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Gravity Spy detector availability is empty or invalid")
    available = tuple(str(ifo) for ifo in values)
    if len(available) != len(set(available)) or str(row["ifo"]) not in available:
        raise ValueError("Gravity Spy available_ifos must uniquely include the event IFO")
    return available


def pair_overlap_rows(
    glitch_rows: list[dict[str, Any]],
    injection_rows: list[dict[str, Any]],
    split: str,
    seed: int,
    limit: int | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build a deterministic one-use pairing without changing either split identity."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported overlap split: {split}")
    glitches = [row for row in glitch_rows if row.get("split") == split]
    injections = [row for row in injection_rows if row.get("split") == split]
    if not glitches or not injections:
        raise ValueError(f"Overlap pairing requires non-empty {split} inputs")
    _require_unique(glitches, ("glitch_id",), "glitch manifest")
    _require_unique(injections, ("injection_id", "waveform_id"), "injection manifest")
    if limit is not None and limit <= 0:
        raise ValueError("Overlap limit must be positive")
    requested = min(len(glitches), len(injections)) if limit is None else limit
    if requested > min(len(glitches), len(injections)):
        raise ValueError("Overlap limit exceeds the available unique physical groups")

    rng = np.random.default_rng(seed)
    glitch_order = rng.permutation(len(glitches)).tolist()
    injection_order = rng.permutation(len(injections)).tolist()
    remaining = set(injection_order)
    supported = {index: set(_supported_ifos(injections[index])) for index in injection_order}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for glitch_index in glitch_order:
        glitch = glitches[glitch_index]
        required_ifos = set(_glitch_available_ifos(glitch))
        match = next(
            (
                index
                for index in injection_order
                if index in remaining and required_ifos <= supported[index]
            ),
            None,
        )
        if match is None:
            continue
        remaining.remove(match)
        pairs.append((glitch, injections[match]))
        if len(pairs) == requested:
            break
    if len(pairs) != requested:
        raise ValueError(
            f"Only {len(pairs)} of {requested} requested pairs have a shared detector"
        )
    return pairs


def _load_gravityspy_sample(
    row: dict[str, Any],
    model_ifos: tuple[str, ...],
    q_values: tuple[float, ...],
    target_sample_rate: int,
    frequency_bins: int,
    time_bins: int,
) -> dict[str, Any]:
    path = Path(row["path"])
    if file_sha256(path) != str(row["sha256"]):
        raise ValueError(f"Gravity Spy sample hash mismatch: {row['glitch_id']}")
    with np.load(path, allow_pickle=False) as arrays:
        raw = np.asarray(arrays["raw_strain"], dtype=np.float64)
        glitch_mask = np.asarray(arrays["glitch_mask"], dtype=np.float32)
        stored_ifos = tuple(str(value) for value in arrays["ifos"].tolist())
        stored_q = tuple(float(value) for value in arrays["q_values"].tolist())
        stored_rate = int(arrays["sample_rate"])
        stored_availability = (
            np.asarray(arrays["detector_availability"], dtype=np.uint8)
            if "detector_availability" in arrays
            else None
        )
    expected_mask = (len(model_ifos), len(q_values), frequency_bins, time_bins)
    if raw.ndim == 1:
        expanded = np.zeros((len(model_ifos), raw.size), dtype=np.float64)
        expanded[model_ifos.index(str(row["ifo"]))] = raw
        raw = expanded
    if raw.ndim != 2 or raw.shape[0] != len(model_ifos) or not np.isfinite(raw).all():
        raise ValueError(f"Gravity Spy raw strain is invalid: {row['glitch_id']}")
    if stored_ifos != model_ifos or not np.allclose(stored_q, q_values, rtol=0, atol=1e-6):
        raise ValueError("Gravity Spy detector/Q contract differs from overlap configuration")
    if stored_rate != target_sample_rate:
        raise ValueError("Gravity Spy sample rate differs from overlap configuration")
    if glitch_mask.shape != expected_mask or not np.isfinite(glitch_mask).all():
        raise ValueError("Gravity Spy weak-mask shape/content is invalid")
    if np.any((glitch_mask != 0) & (glitch_mask != 1)):
        raise ValueError("Gravity Spy weak mask must be binary")
    if str(row["ifo"]) not in model_ifos:
        raise ValueError("Gravity Spy event IFO is absent from model_ifos")
    available_ifos = _glitch_available_ifos(row)
    if any(ifo not in model_ifos for ifo in available_ifos):
        raise ValueError("Gravity Spy available detector is absent from model_ifos")
    expected_availability = np.asarray(
        [int(ifo in available_ifos) for ifo in model_ifos], dtype=np.uint8
    )
    if stored_availability is not None and not np.array_equal(
        stored_availability, expected_availability
    ):
        raise ValueError("Stored Gravity Spy detector availability differs from manifest")
    if np.any(raw[expected_availability == 0] != 0):
        raise ValueError("Unavailable Gravity Spy detector strain must be exactly zero")
    return {
        "raw": raw,
        "glitch_mask": glitch_mask,
        "availability": expected_availability,
        "available_ifos": available_ifos,
    }


def _active_injection_signals(
    row: dict[str, Any],
    active_ifos: tuple[str, ...],
    model_ifos: tuple[str, ...],
    target_sample_rate: int,
    output_samples: int,
    minimum_ifo_mask_snr: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    context = load_materialized_context(row)
    source_ifos = list(context["ifos"])
    missing = sorted(set(active_ifos) - set(source_ifos))
    if missing:
        raise ValueError(f"Injection {row['injection_id']} does not support {missing}")
    scale = float(row.get("training_signal_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training_signal_scale must be finite and positive")
    signal = np.asarray(context["signal"], dtype=np.float64) * scale
    target_signal = signal
    if minimum_ifo_mask_snr is not None:
        if "optimal_snr_by_ifo" not in row:
            raise ValueError("Visibility-gated overlap masks require optimal_snr_by_ifo")
        target_signal = gate_component_by_ifo_snr(
            signal,
            source_ifos,
            row["optimal_snr_by_ifo"],
            minimum_ifo_mask_snr,
            signal_scale=scale,
        )
    start = int(context["analysis_start_index"])
    stop = int(context["analysis_stop_index"])
    physical = np.zeros((len(model_ifos), output_samples), dtype=np.float64)
    target = np.zeros_like(physical)
    for ifo in active_ifos:
        source_index = source_ifos.index(ifo)
        physical_ifo = _fft_downsample(
            signal[source_index, start:stop],
            int(context["sample_rate"]),
            target_sample_rate,
        )
        target_ifo = _fft_downsample(
            target_signal[source_index, start:stop],
            int(context["sample_rate"]),
            target_sample_rate,
        )
        if physical_ifo.shape != (output_samples,) or target_ifo.shape != (output_samples,):
            raise ValueError(
                f"Injection analysis duration differs from Gravity Spy context: "
                f"{physical_ifo.size}/{target_ifo.size} != {output_samples}"
            )
        model_index = model_ifos.index(ifo)
        physical[model_index] = physical_ifo
        target[model_index] = target_ifo
    if not np.isfinite(physical).all() or not np.isfinite(target).all():
        raise ValueError("Injection signal contains non-finite samples")
    return physical, target


def materialize_physical_overlaps(
    gravityspy_manifest: str | Path,
    injection_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    split: str,
    seed: int = 20260720,
    limit: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    settings = config.get("overlap_factory")
    if not isinstance(settings, dict):
        raise ValueError("Overlap configuration requires overlap_factory")
    tensor = settings.get("tensor")
    if not isinstance(tensor, dict):
        raise ValueError("overlap_factory requires tensor settings")
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    target_rate = int(settings["target_sample_rate"])
    if not model_ifos or not q_values or target_rate <= 0:
        raise ValueError("Overlap detector, Q, and sample-rate settings are invalid")
    frequency_bins = int(tensor["frequency_bins"])
    time_bins = int(tensor["time_bins"])
    minimum_snr = tensor.get("minimum_ifo_mask_snr")
    minimum_snr = None if minimum_snr is None else float(minimum_snr)
    if str(tensor.get("whitening", "self")) != "self":
        raise ValueError("Real-glitch overlap v1 supports self whitening only")
    if str(tensor.get("target_whitening", "morphology")) != "morphology":
        raise ValueError("Real-glitch overlap v1 supports morphology targets only")

    glitch_rows = _read_jsonl(gravityspy_manifest)
    injection_rows = _read_jsonl(injection_manifest)
    pairs = pair_overlap_rows(glitch_rows, injection_rows, split, seed, limit)
    output = Path(output_dir)
    records: list[dict[str, Any]] = []
    for glitch, injection in pairs:
        ifo = str(glitch["ifo"])
        gravity = _load_gravityspy_sample(
            glitch, model_ifos, q_values, target_rate, frequency_bins, time_bins
        )
        raw_glitch = gravity["raw"]
        signal_active, target_signal_active = _active_injection_signals(
            injection,
            gravity["available_ifos"],
            model_ifos,
            target_rate,
            raw_glitch.shape[1],
            minimum_snr,
        )
        availability = gravity["availability"]
        glitch_strain = raw_glitch
        signal_strain = signal_active
        target_signal_strain = target_signal_active
        mixture_strain = glitch_strain + signal_strain
        whitened = np.zeros_like(mixture_strain)
        for detector_index in np.flatnonzero(availability):
            whitened[detector_index] = _whiten(mixture_strain[detector_index])
        feature_power = multiresolution_power(
            whitened,
            target_rate,
            q_values,
            frequency_bins,
            time_bins,
            float(tensor["fmin"]),
            float(tensor["fmax"]),
        )
        signal_power = multiresolution_power(
            scale_component_for_transform(target_signal_strain),
            target_rate,
            q_values,
            frequency_bins,
            time_bins,
            float(tensor["fmin"]),
            float(tensor["fmax"]),
        )
        features = _normalize_power(feature_power)
        chirp_mask = relative_component_mask(
            signal_power, float(tensor.get("mask_fraction", 0.08))
        )
        if np.any(features[availability == 0] != 0):
            raise RuntimeError("Unavailable detector planes must remain exactly zero")
        identity = {
            "version": OVERLAP_ARTIFACT_VERSION,
            "split": split,
            "glitch_id": str(glitch["glitch_id"]),
            "injection_id": str(injection["injection_id"]),
            "waveform_id": str(injection["waveform_id"]),
            "glitch_sha256": str(glitch["sha256"]),
            "injection_sha256": str(injection["materialized_sha256"]),
            "config_sha256": file_sha256(config_path),
        }
        mixture_id = f"overlap-{canonical_hash(identity, 24)}"
        sample_path = output / "samples" / f"{mixture_id}.npz"
        _atomic_save_npz(
            sample_path,
            features=features.astype(np.float16),
            chirp_mask=chirp_mask.astype(np.uint8),
            glitch_mask=gravity["glitch_mask"].astype(np.uint8),
            raw_glitch_strain=glitch_strain.astype(np.float32),
            signal_strain=signal_strain.astype(np.float64),
            target_signal_strain=target_signal_strain.astype(np.float64),
            mixture_strain=mixture_strain.astype(np.float32),
            detector_availability=availability,
            ifos=np.asarray(model_ifos),
            q_values=np.asarray(q_values, dtype=np.float32),
            sample_rate=np.asarray(target_rate, dtype=np.int32),
            event_gps=np.asarray(glitch["event_time"], dtype=np.float64),
        )
        record = {
            "mixture_id": mixture_id,
            "scene_type": "physical_chirp_real_glitch_overlap",
            "split": split,
            "path": str(sample_path),
            "sha256": file_sha256(sample_path),
            "injection_id": str(injection["injection_id"]),
            "waveform_id": str(injection["waveform_id"]),
            "source_family": injection.get("source_family"),
            "injection_gps_block": str(injection["gps_block"]),
            "glitch_id": str(glitch["glitch_id"]),
            "gps_block": str(glitch["network_gps_block"]),
            "network_gps_block": str(glitch["network_gps_block"]),
            "glitch_ifo": ifo,
            "ifo": ifo,
            "observing_run": glitch.get("observing_run"),
            "ml_label": glitch.get("ml_label"),
            "detector_availability": availability.tolist(),
            "available_ifos": list(gravity["available_ifos"]),
            "mask_provenance": glitch.get("mask_provenance"),
            "human_pixel_mask": bool(glitch.get("human_pixel_mask", False)),
            "glitch_artifact_sha256": str(glitch["sha256"]),
            "injection_materialized_sha256": str(injection["materialized_sha256"]),
            "training_signal_scale": float(injection.get("training_signal_scale", 1.0)),
            "optimal_snr_by_ifo": injection.get("optimal_snr_by_ifo"),
            "artifact_version": OVERLAP_ARTIFACT_VERSION,
        }
        records.append(record)

    _require_unique(
        records,
        ("mixture_id", "injection_id", "waveform_id", "glitch_id"),
        "overlap output",
    )
    manifest = output / f"physical_overlap_{split}_manifest.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    aligned_network_rows = sum(len(row["available_ifos"]) >= 2 for row in records)
    single_ifo_rows = len(records) - aligned_network_rows
    report = {
        "status": "verified_real_glitch_physical_overlap_training_data",
        "scientific_claim_allowed": False,
        "search_claim_allowed": False,
        "network_coherence_claim_allowed": False,
        "reason": (
            "Detector availability is preserved per source artifact; weak-mask audit, frozen "
            "continuous-background evaluation and physical-lag coherence gates remain required"
        ),
        "artifact_version": OVERLAP_ARTIFACT_VERSION,
        "split": split,
        "rows": len(records),
        "unique_physical_counts": {
            "mixtures": len(records),
            "injections": len({row["injection_id"] for row in records}),
            "waveforms": len({row["waveform_id"] for row in records}),
            "glitches": len({row["glitch_id"] for row in records}),
            "glitch_gps_blocks": len({row["network_gps_block"] for row in records}),
        },
        "rendered_image_count": 0,
        "aligned_network_rows": aligned_network_rows,
        "single_ifo_rows": single_ifo_rows,
        "event_ifo_counts": dict(Counter(row["ifo"] for row in records)),
        "detector_subset_counts": dict(
            Counter("".join(row["available_ifos"]) for row in records)
        ),
        "weak_masks": sum(not row["human_pixel_mask"] for row in records),
        "human_pixel_masks": sum(row["human_pixel_mask"] for row in records),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "gravityspy_manifest_sha256": file_sha256(gravityspy_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        "config_sha256": file_sha256(config_path),
        "seed": seed,
        "required_next_gates": [
            "joint_cross_split_overlap_audit",
            "human_weak_mask_audit",
            "continuous_background_far_ifar_vt",
        ]
        + (["aligned_companion_ifo_materialization"] if single_ifo_rows else [])
        + (["physical_lag_coherence_validation"] if aligned_network_rows else []),
        **execution_provenance(),
    }
    atomic_write_json(output / "physical_overlap_report.json", report)
    return report


def audit_physical_overlap_manifests(
    manifests: list[str | Path], output_path: str | Path
) -> dict[str, Any]:
    if len(manifests) < 2:
        raise ValueError("A joint overlap audit requires at least two split manifests")
    by_split: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for manifest in manifests:
        rows = _read_jsonl(manifest)
        splits = {str(row["split"]) for row in rows}
        if len(splits) != 1:
            raise ValueError(f"Overlap manifest mixes splits: {manifest}")
        split = next(iter(splits))
        if split in by_split:
            raise ValueError(f"Duplicate overlap manifest for split {split}")
        _require_unique(
            rows,
            ("mixture_id", "injection_id", "waveform_id", "glitch_id"),
            f"{split} overlap manifest",
        )
        by_split[split] = rows
        hashes[split] = file_sha256(manifest)
    overlaps: dict[str, dict[str, list[str]]] = {}
    split_names = sorted(by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            pair = f"{left}__{right}"
            overlaps[pair] = {}
            for field in OVERLAP_LEAKAGE_FIELDS:
                left_values = {str(row[field]) for row in by_split[left]}
                right_values = {str(row[field]) for row in by_split[right]}
                overlaps[pair][field] = sorted(left_values & right_values)
    if any(values for pair in overlaps.values() for values in pair.values()):
        raise ValueError(f"Physical overlap split leakage: {overlaps}")
    report = {
        "status": "passed_physical_overlap_group_audit",
        "passed": True,
        "manifest_sha256_by_split": hashes,
        "rows_by_split": {split: len(rows) for split, rows in by_split.items()},
        "cross_split_overlaps": overlaps,
        **execution_provenance(),
    }
    atomic_write_json(output_path, report)
    return report
