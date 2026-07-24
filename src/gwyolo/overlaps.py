from __future__ import annotations

import hashlib
import json
import re
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


OVERLAP_ARTIFACT_VERSION = (
    "gravityspy-physical-overlap-v3-automatic-component-masks"
)
OVERLAP_LEAKAGE_FIELDS = (
    "mixture_id",
    "injection_id",
    "waveform_id",
    "glitch_id",
    "injection_gps_block",
    "gps_block",
    "network_gps_block",
)


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's exact dtype, shape, and contiguous stored bytes."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(value) for value in contiguous.shape).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


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
    """Build a maximum-cardinality detector-compatible one-use pairing."""

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
    requested = limit
    if requested is not None and requested > min(len(glitches), len(injections)):
        raise ValueError("Overlap limit exceeds the available unique physical groups")

    rng = np.random.default_rng(seed)
    glitch_groups: dict[tuple[str, ...], list[int]] = {}
    for index in rng.permutation(len(glitches)).tolist():
        key = tuple(sorted(_glitch_available_ifos(glitches[index])))
        glitch_groups.setdefault(key, []).append(index)
    injection_groups: dict[tuple[str, ...], list[int]] = {}
    for index in rng.permutation(len(injections)).tolist():
        key = tuple(sorted(_supported_ifos(injections[index])))
        injection_groups.setdefault(key, []).append(index)

    # The detector universe has only seven non-empty H1/L1/V1 subsets.  A
    # category-level integral max flow is therefore exact and avoids an O(N^2)
    # physical-row graph while preventing a greedy broad-detector assignment
    # from starving a more constrained glitch category.
    source = ("source",)
    sink = ("sink",)
    glitch_nodes = {key: ("glitch", *key) for key in sorted(glitch_groups)}
    injection_nodes = {key: ("injection", *key) for key in sorted(injection_groups)}
    residual: dict[tuple[str, ...], dict[tuple[str, ...], int]] = {}

    def add_edge(left: tuple[str, ...], right: tuple[str, ...], capacity: int) -> None:
        residual.setdefault(left, {})[right] = capacity
        residual.setdefault(right, {}).setdefault(left, 0)

    for key, node in glitch_nodes.items():
        add_edge(source, node, len(glitch_groups[key]))
    compatible_edges: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    maximum_capacity = min(len(glitches), len(injections))
    for glitch_key, glitch_node in glitch_nodes.items():
        for injection_key, injection_node in injection_nodes.items():
            if set(glitch_key) <= set(injection_key):
                add_edge(glitch_node, injection_node, maximum_capacity)
                compatible_edges.append((glitch_key, injection_key))
    for key, node in injection_nodes.items():
        add_edge(node, sink, len(injection_groups[key]))

    while True:
        parent: dict[tuple[str, ...], tuple[str, ...] | None] = {source: None}
        queue = [source]
        for node in queue:
            for neighbor in sorted(residual.get(node, {})):
                if residual[node][neighbor] > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
            if sink in parent:
                break
        if sink not in parent:
            break
        increment = maximum_capacity
        node = sink
        while parent[node] is not None:
            prior = parent[node]
            increment = min(increment, residual[prior][node])
            node = prior
        node = sink
        while parent[node] is not None:
            prior = parent[node]
            residual[prior][node] -= increment
            residual[node][prior] += increment
            node = prior

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    glitch_cursor = {key: 0 for key in glitch_groups}
    injection_cursor = {key: 0 for key in injection_groups}
    for glitch_key, injection_key in compatible_edges:
        matched = residual[injection_nodes[injection_key]][glitch_nodes[glitch_key]]
        for _ in range(matched):
            glitch_index = glitch_groups[glitch_key][glitch_cursor[glitch_key]]
            injection_index = injection_groups[injection_key][
                injection_cursor[injection_key]
            ]
            glitch_cursor[glitch_key] += 1
            injection_cursor[injection_key] += 1
            pairs.append((glitches[glitch_index], injections[injection_index]))
    if pairs:
        order = rng.permutation(len(pairs)).tolist()
        pairs = [pairs[index] for index in order]
    if requested is not None and len(pairs) < requested:
        raise ValueError(
            f"Only {len(pairs)} of {requested} requested pairs have a shared detector"
        )
    if not pairs:
        raise ValueError("No detector-compatible physical overlap pairs are available")
    return pairs if requested is None else pairs[:requested]


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
    artifact = Path(row["materialized_path"])
    if file_sha256(artifact) != str(row["materialized_sha256"]):
        raise ValueError("materialized array hash mismatch")
    with np.load(artifact, allow_pickle=False) as arrays:
        source_rate = int(arrays["sample_rate"])
        source_ifos = [str(value) for value in arrays["ifos"].tolist()]
        start = int(arrays["analysis_start_index"])
        stop = int(arrays["analysis_stop_index"])
        if "signal" in arrays:
            source_signal = np.asarray(arrays["signal"], dtype=np.float64)
        elif "signal_scaled" in arrays and "signal_peak_scale" in arrays:
            source_signal = np.asarray(
                arrays["signal_scaled"], dtype=np.float64
            ) * np.asarray(arrays["signal_peak_scale"], dtype=np.float64)[:, None]
        else:
            raise ValueError(
                "materialized artifact lacks a supported signal representation"
            )
    if (
        source_signal.ndim != 2
        or source_signal.shape[0] != len(source_ifos)
        or not 0 <= start < stop <= source_signal.shape[1]
        or source_rate <= 0
        or not np.isfinite(source_signal).all()
    ):
        raise ValueError("materialized injection signal contract is invalid")
    missing = sorted(set(active_ifos) - set(source_ifos))
    if missing:
        raise ValueError(f"Injection {row['injection_id']} does not support {missing}")
    scale = float(row.get("training_signal_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training_signal_scale must be finite and positive")
    signal = source_signal * scale
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
    physical = np.zeros((len(model_ifos), output_samples), dtype=np.float64)
    target = np.zeros_like(physical)
    for ifo in active_ifos:
        source_index = source_ifos.index(ifo)
        physical_ifo = _fft_downsample(
            signal[source_index, start:stop],
            source_rate,
            target_sample_rate,
        )
        target_ifo = _fft_downsample(
            target_signal[source_index, start:stop],
            source_rate,
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
    gravityspy_corpus_audit: str | Path | None = None,
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
    glitch_mask_source = str(
        tensor.get(
            "glitch_mask_source",
            "isolated_real_glitch_component_power_v1",
        )
    )
    glitch_mask_fraction = float(
        tensor.get("glitch_mask_fraction", tensor.get("mask_fraction", 0.08))
    )
    if (
        glitch_mask_source != "isolated_real_glitch_component_power_v1"
        or not 0 < glitch_mask_fraction < 1
        or tensor.get("manual_annotation_required", False) is not False
    ):
        raise ValueError("automatic real-glitch mask policy is invalid")

    corpus_audit_sha256 = None
    if gravityspy_corpus_audit is not None:
        audit_path = Path(gravityspy_corpus_audit)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status")
            != "verified_group_safe_gravityspy_aligned_network_corpus"
            or not audit.get("passed")
        ):
            raise ValueError("Gravity Spy network corpus audit did not pass")
        expected_key = "train_manifest_sha256" if split == "train" else "validation_manifest_sha256"
        if str(audit.get(expected_key)) != file_sha256(gravityspy_manifest):
            raise ValueError("Gravity Spy corpus audit does not bind this split manifest")
        overlaps = audit.get("split_audit", {}).get("cross_split_overlaps", {})
        if not overlaps or any(overlaps.values()):
            raise ValueError("Gravity Spy corpus audit lacks a zero-overlap certificate")
        corpus_audit_sha256 = file_sha256(audit_path)

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
        # Persist and transform the same quantized component so an independent
        # audit can reproduce the pseudo-mask bit-for-bit from the NPZ alone.
        stored_glitch_strain = raw_glitch.astype(np.float32)
        glitch_strain = stored_glitch_strain.astype(np.float64)
        signal_strain = signal_active.astype(np.float64)
        target_signal_strain = target_signal_active.astype(np.float64)
        stored_mixture_strain = (glitch_strain + signal_strain).astype(np.float32)
        mixture_strain = stored_mixture_strain.astype(np.float64)
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
        whitened_glitch = np.zeros_like(glitch_strain)
        for detector_index in np.flatnonzero(availability):
            whitened_glitch[detector_index] = _whiten(
                glitch_strain[detector_index]
            )
        glitch_power = multiresolution_power(
            whitened_glitch,
            target_rate,
            q_values,
            frequency_bins,
            time_bins,
            float(tensor["fmin"]),
            float(tensor["fmax"]),
        )
        automatic_glitch_mask = relative_component_mask(
            glitch_power, glitch_mask_fraction
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
            "gravityspy_corpus_audit_sha256": corpus_audit_sha256,
        }
        mixture_id = f"overlap-{canonical_hash(identity, 24)}"
        sample_path = output / "samples" / f"{mixture_id}.npz"
        _atomic_save_npz(
            sample_path,
            features=features.astype(np.float16),
            chirp_mask=chirp_mask.astype(np.uint8),
            glitch_mask=automatic_glitch_mask.astype(np.uint8),
            legacy_metadata_glitch_mask=gravity["glitch_mask"].astype(np.uint8),
            raw_glitch_strain=stored_glitch_strain,
            signal_strain=signal_strain,
            target_signal_strain=target_signal_strain,
            mixture_strain=stored_mixture_strain,
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
            "mask_provenance": glitch_mask_source,
            "mask_fraction": glitch_mask_fraction,
            "automatic_pseudo_mask": True,
            "human_pixel_mask": False,
            "legacy_metadata_mask_provenance": glitch.get("mask_provenance"),
            "glitch_artifact_path": str(Path(glitch["path"]).resolve()),
            "glitch_artifact_sha256": str(glitch["sha256"]),
            "injection_materialized_path": str(
                Path(injection["materialized_path"]).resolve()
            ),
            "injection_materialized_sha256": str(injection["materialized_sha256"]),
            "raw_glitch_component_sha256": array_sha256(stored_glitch_strain),
            "signal_component_sha256": array_sha256(signal_strain),
            "target_signal_component_sha256": array_sha256(target_signal_strain),
            "mixture_component_sha256": array_sha256(stored_mixture_strain),
            "training_signal_scale": float(injection.get("training_signal_scale", 1.0)),
            "optimal_snr_by_ifo": injection.get("optimal_snr_by_ifo"),
            "artifact_version": OVERLAP_ARTIFACT_VERSION,
            "gravityspy_corpus_audit_sha256": corpus_audit_sha256,
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
            "Detector availability is preserved per source artifact; deterministic automatic "
            "mask replay, frozen continuous-background evaluation and physical-lag coherence "
            "gates remain required"
        ),
        "artifact_version": OVERLAP_ARTIFACT_VERSION,
        "split": split,
        "rows": len(records),
        "pairing_policy": (
            "maximum_cardinality_detector_subset_flow_v1"
            if limit is None
            else "strict_predeclared_pair_limit_v1"
        ),
        "predeclared_pair_limit": limit,
        "eligible_input_rows": {
            "glitches": sum(row.get("split") == split for row in glitch_rows),
            "injections": sum(row.get("split") == split for row in injection_rows),
        },
        "unpaired_input_rows": {
            "glitches": sum(row.get("split") == split for row in glitch_rows)
            - len(records),
            "injections": sum(row.get("split") == split for row in injection_rows)
            - len(records),
        },
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
        "weak_masks": 0,
        "automatic_pseudo_masks": sum(
            bool(row["automatic_pseudo_mask"]) for row in records
        ),
        "human_pixel_masks": sum(row["human_pixel_mask"] for row in records),
        "manual_annotation_required": False,
        "automatic_mask_policy": {
            "source": glitch_mask_source,
            "fraction": glitch_mask_fraction,
            "whitening": "per_available_ifo_self_whitening",
            "transform": "fresh_multi_q_numeric_power",
            "threshold_selection": "frozen_fraction_of_per_plane_peak",
            "human_ground_truth_claimed": False,
        },
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "gravityspy_manifest_sha256": file_sha256(gravityspy_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        "config_sha256": file_sha256(config_path),
        "gravityspy_corpus_audit_path": (
            str(gravityspy_corpus_audit) if gravityspy_corpus_audit is not None else None
        ),
        "gravityspy_corpus_audit_sha256": corpus_audit_sha256,
        "seed": seed,
        "required_next_gates": [
            "joint_cross_split_overlap_audit",
            "automatic_mask_policy_replay",
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


def freeze_physical_overlap_scaling_subsets(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    gravityspy_corpus_audit: str | Path,
    scales: list[int],
    output_dir: str | Path,
    seed: int = 20260728,
    include_full: bool = False,
) -> dict[str, Any]:
    """Freeze nested physical-group subsets for overlap data-scaling controls."""

    train_rows = _read_jsonl(train_manifest)
    validation_rows = _read_jsonl(validation_manifest)
    corpus_audit_path = Path(gravityspy_corpus_audit)
    corpus_audit = json.loads(corpus_audit_path.read_text(encoding="utf-8"))
    corpus_audit_sha256 = file_sha256(corpus_audit_path)
    if (
        corpus_audit.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or corpus_audit.get("passed") is not True
    ):
        raise ValueError("Gravity Spy corpus audit did not pass")
    split_audit = audit_physical_overlap_manifests(
        [train_manifest, validation_manifest],
        Path(output_dir) / "train_validation_group_audit.json",
    )
    if set(split_audit["rows_by_split"]) != {"train", "val"}:
        raise ValueError("Overlap scaling requires exactly train and validation manifests")
    if not scales:
        raise ValueError("Overlap scaling requires at least one declared scale")
    requested = [int(value) for value in scales]
    if any(value <= 0 for value in requested) or requested != sorted(set(requested)):
        raise ValueError("Overlap scales must be unique, positive and strictly increasing")
    if include_full and len(train_rows) not in requested:
        requested.append(len(train_rows))
        requested.sort()
    if requested[-1] > len(train_rows):
        raise ValueError("Overlap scale exceeds the available unique physical groups")

    required_fields = (
        "mixture_id",
        "injection_id",
        "waveform_id",
        "glitch_id",
        "injection_gps_block",
        "network_gps_block",
        "path",
        "sha256",
    )
    for label, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            missing = [field for field in required_fields if field not in row]
            if missing:
                raise ValueError(f"{label} overlap row is missing fields: {missing}")
            artifact = Path(str(row["path"]))
            if not artifact.is_file() or file_sha256(artifact) != row["sha256"]:
                raise ValueError(f"{label} overlap artifact hash mismatch")
            if row.get("gravityspy_corpus_audit_sha256") != corpus_audit_sha256:
                raise ValueError(f"{label} overlap row differs from the corpus audit")

    def rank(row: dict[str, Any]) -> str:
        return canonical_hash(
            {
                "schema": "physical_overlap_scale_rank_v1",
                "seed": seed,
                "mixture_id": row["mixture_id"],
                "injection_id": row["injection_id"],
                "waveform_id": row["waveform_id"],
                "glitch_id": row["glitch_id"],
                "injection_gps_block": row["injection_gps_block"],
                "network_gps_block": row["network_gps_block"],
            }
        )

    ranked = sorted(train_rows, key=lambda row: (rank(row), str(row["mixture_id"])))
    output = Path(output_dir)
    subsets = []
    previous_ids: set[str] = set()
    for scale in requested:
        rows = ranked[:scale]
        mixture_ids = {str(row["mixture_id"]) for row in rows}
        if not previous_ids <= mixture_ids:
            raise RuntimeError("Overlap scaling subsets are not nested")
        manifest = output / f"scale-{scale}" / "physical_overlap_train_manifest.jsonl"
        atomic_write_text(
            manifest,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
        counts = {
            field: len({str(row[field]) for row in rows})
            for field in (
                "mixture_id",
                "injection_id",
                "waveform_id",
                "glitch_id",
                "injection_gps_block",
                "network_gps_block",
            )
        }
        if any(value != scale for value in counts.values()):
            raise ValueError("Overlap scaling subset reuses a physical group")
        subsets.append(
            {
                "scale": scale,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": file_sha256(manifest),
                "unique_physical_counts": counts,
                "glitch_family_counts": dict(
                    sorted(Counter(str(row.get("ml_label", "unknown")) for row in rows).items())
                ),
                "source_family_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("source_family", "unknown")) for row in rows
                        ).items()
                    )
                ),
                "detector_subset_counts": dict(
                    sorted(
                        Counter(
                            "".join(str(value) for value in row.get("available_ifos", []))
                            for row in rows
                        ).items()
                    )
                ),
                "parent_prefix_mixture_id_hash": canonical_hash(
                    [str(row["mixture_id"]) for row in rows]
                ),
            }
        )
        previous_ids = mixture_ids

    report = {
        "status": "frozen_group_safe_physical_overlap_scaling_subsets",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "subsets alone are not a scaling result; run fixed-epoch and fixed-update "
            "controls on the same frozen validation endpoint"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "rank_schema": "physical_overlap_scale_rank_v1",
        "seed": seed,
        "available_train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "include_full": include_full,
        "scales": requested,
        "subsets": subsets,
        "train_manifest_path": str(Path(train_manifest).resolve()),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_path": str(Path(validation_manifest).resolve()),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "gravityspy_corpus_audit": {
            "path": str(corpus_audit_path.resolve()),
            "sha256": corpus_audit_sha256,
        },
        "train_validation_group_audit": {
            "path": str((output / "train_validation_group_audit.json").resolve()),
            "sha256": file_sha256(output / "train_validation_group_audit.json"),
        },
        "required_training_controls": ["fixed_epochs", "fixed_optimizer_updates"],
        **execution_provenance(),
    }
    atomic_write_json(output / "physical_overlap_scaling_subsets.json", report)
    return report


def audit_physical_overlap_expansion_capacity(
    hard_endpoint_report_path: str | Path | None,
    current_overlap_manifest_path: str | Path,
    candidate_glitch_manifest_path: str | Path,
    candidate_injection_manifest_path: str | Path,
    gravityspy_corpus_audit_path: str | Path,
    output_path: str | Path,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Determine whether an authorized next scale has real independent-source capacity."""

    hard_path = (
        None
        if hard_endpoint_report_path is None
        else Path(hard_endpoint_report_path).resolve()
    )
    if hard_path is None:
        authorized = False
        next_scale = None
    else:
        hard = json.loads(hard_path.read_text(encoding="utf-8"))
        if (
            hard.get("status")
            != "completed_group_safe_physical_overlap_data_scaling_curve"
            or hard.get("passed") is not True
            or hard.get("test_rows_read") != 0
            or hard.get("test_evaluation") is not None
            or hard.get("hard_endpoint_binding", {}).get("passed") is not True
        ):
            raise ValueError("Physical-overlap hard endpoint failed replay")
        authorized = hard.get("scale_promotion_authorized") is True
        next_scale = hard.get("authorized_next_physical_scale")
        if authorized != (isinstance(next_scale, int) and next_scale > 0):
            raise ValueError(
                "Physical-overlap hard endpoint has an inconsistent authorization"
            )

    current_path = Path(current_overlap_manifest_path).resolve()
    glitch_path = Path(candidate_glitch_manifest_path).resolve()
    injection_path = Path(candidate_injection_manifest_path).resolve()
    audit_path = Path(gravityspy_corpus_audit_path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or audit.get("passed") is not True
        or audit.get("train_manifest_sha256") != file_sha256(glitch_path)
        or any(
            audit.get("split_audit", {}).get("cross_split_overlaps", {}).values()
        )
    ):
        raise ValueError("Candidate Gravity Spy corpus failed its group-safe audit")

    current = _read_jsonl(current_path)
    glitches = _read_jsonl(glitch_path)
    injections = _read_jsonl(injection_path)
    _require_unique(
        current,
        ("mixture_id", "injection_id", "waveform_id", "glitch_id"),
        "current overlap manifest",
    )
    if any(row.get("split") != "train" for row in current):
        raise ValueError("Current overlap expansion source is not train-only")
    candidate_glitch_ids = {
        str(row["glitch_id"]) for row in glitches if row.get("split") == "train"
    }
    candidate_injection_ids = {
        str(row["injection_id"])
        for row in injections
        if row.get("split") == "train"
    }
    if (
        {str(row["glitch_id"]) for row in current} - candidate_glitch_ids
        or {str(row["injection_id"]) for row in current} - candidate_injection_ids
    ):
        raise ValueError("Current overlap rows are not contained in candidate sources")

    all_pairs = pair_overlap_rows(glitches, injections, "train", seed)
    current_subsets = {
        tuple(sorted(str(value) for value in row["available_ifos"]))
        for row in current
    }
    same_distribution_glitches = [
        row
        for row in glitches
        if row.get("split") == "train"
        and tuple(sorted(_glitch_available_ifos(row))) in current_subsets
    ]
    same_pairs = pair_overlap_rows(
        same_distribution_glitches,
        injections,
        "train",
        seed,
    )
    current_count = len(current)
    maximum_same_distribution = len(same_pairs)
    maximum_all_detector_sets = len(all_pairs)
    if (
        maximum_same_distribution < current_count
        or maximum_all_detector_sets < maximum_same_distribution
    ):
        raise ValueError("Candidate expansion capacity is smaller than the current corpus")

    def subset_counts(rows: list[dict[str, Any]], glitch: bool) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    "+".join(
                        sorted(
                            _glitch_available_ifos(row)
                            if glitch
                            else _supported_ifos(row)
                        )
                    )
                    for row in rows
                    if row.get("split") == "train"
                ).items()
            )
        )

    if not authorized:
        mode = "not_authorized_by_hard_endpoint"
        same_ready = False
        all_ready = False
        training_authorized = False
        minimum_new_sources = 0
    else:
        same_ready = maximum_same_distribution >= next_scale
        all_ready = maximum_all_detector_sets >= next_scale
        minimum_new_sources = max(0, next_scale - maximum_all_detector_sets)
        if same_ready:
            mode = "same_distribution_capacity_ready"
            training_authorized = True
        elif all_ready:
            mode = "detector_set_expansion_requires_separate_ablation"
            training_authorized = False
        else:
            mode = "new_physical_sources_required"
            training_authorized = False

    result = {
        "status": "audited_physical_overlap_expansion_capacity",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "capacity is an authorization preflight, not a trained scaling result, "
            "continuous-background search result or locked evaluation"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "hard_endpoint_authorized": authorized,
        "authorized_next_physical_scale": next_scale if authorized else None,
        "current_physical_groups": current_count,
        "maximum_same_distribution_physical_groups": maximum_same_distribution,
        "maximum_all_detector_set_physical_groups": maximum_all_detector_sets,
        "same_distribution_capacity_ready": same_ready,
        "all_detector_set_capacity_ready": all_ready,
        "next_scale_training_authorized": training_authorized,
        "expansion_mode": mode,
        "minimum_new_detector_compatible_physical_groups": minimum_new_sources,
        "detector_set_capacity": {
            "current_overlap": dict(
                sorted(
                    Counter(
                        "+".join(sorted(str(value) for value in row["available_ifos"]))
                        for row in current
                    ).items()
                )
            ),
            "candidate_glitches": subset_counts(glitches, True),
            "candidate_injections": subset_counts(injections, False),
        },
        "required_followup": (
            ["wait for the frozen validation hard-endpoint decision; do not train"]
            if mode == "not_authorized_by_hard_endpoint"
            else (
                [
                    "acquire new unique glitch IDs and GPS blocks compatible with the "
                    "current detector-set distribution",
                    "pair with disjoint unique waveform/injection IDs",
                    "rerun the joint train/validation leakage audit before training",
                ]
                if mode == "new_physical_sources_required"
                else (
                    [
                        "treat detector-set expansion as a separate predeclared robustness "
                        "ablation, not same-distribution data scaling"
                    ]
                    if mode == "detector_set_expansion_requires_separate_ablation"
                    else []
                )
            )
        ),
        "inputs": {
            "hard_endpoint": (
                None
                if hard_path is None
                else {
                    "path": str(hard_path),
                    "sha256": file_sha256(hard_path),
                }
            ),
            "current_overlap_manifest": {
                "path": str(current_path),
                "sha256": file_sha256(current_path),
            },
            "candidate_glitch_manifest": {
                "path": str(glitch_path),
                "sha256": file_sha256(glitch_path),
            },
            "candidate_injection_manifest": {
                "path": str(injection_path),
                "sha256": file_sha256(injection_path),
            },
            "gravityspy_corpus_audit": {
                "path": str(audit_path),
                "sha256": file_sha256(audit_path),
            },
        },
        "pairing_policy": "maximum_cardinality_detector_subset_flow_v1",
        "seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def freeze_physical_overlap_scaling_hard_subset(
    validation_manifest: str | Path,
    gravityspy_corpus_audit: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze a score-blind validation hard subset before scaling results exist."""

    output = Path(output_dir)
    provenance = execution_provenance()
    commit = provenance.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        raise ValueError("Scaling hard-subset freeze requires full commit provenance")
    report_path = output / "physical_overlap_scaling_hard_subset_report.json"
    manifest_path = output / "physical_overlap_scaling_hard_subset.jsonl"
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError("Physical overlap scaling hard-subset outputs are immutable")
    config = load_yaml(config_path)
    settings = config.get("physical_overlap_scaling_hard_subset")
    if not isinstance(settings, dict) or settings.get("schema") != (
        "physical_overlap_scaling_hard_subset_v1"
    ):
        raise ValueError("Unsupported physical overlap scaling hard-subset config")
    if settings.get("policy") != "score_blind_validation_metadata_v1":
        raise ValueError("Hard-subset policy must be score blind")
    required_strata = [str(value) for value in settings.get("required_strata", [])]
    allowed_strata = {
        "low_network_snr",
        "missing_detector",
        "o3b_transfer",
        "rare_glitch_family",
    }
    if (
        not required_strata
        or len(set(required_strata)) != len(required_strata)
        or set(required_strata) != allowed_strata
    ):
        raise ValueError("Hard-subset config must declare all four frozen strata")
    minimum_rows = int(settings.get("minimum_rows_per_stratum", 0))
    minimum_glitches = int(settings.get("minimum_unique_glitches_per_stratum", 0))
    low_snr_max = float(settings.get("low_network_snr_max", 0.0))
    full_detector_count = int(settings.get("full_detector_count", 0))
    transfer_runs = {str(value) for value in settings.get("transfer_observing_runs", [])}
    rare_fraction = float(settings.get("rare_glitch_family_max_fraction", 0.0))
    if (
        minimum_rows < 25
        or minimum_glitches < 25
        or low_snr_max <= 0
        or full_detector_count < 2
        or not transfer_runs
        or not 0 < rare_fraction < 0.5
    ):
        raise ValueError("Hard-subset thresholds do not meet publication minima")

    corpus_path = Path(gravityspy_corpus_audit).resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_sha = file_sha256(corpus_path)
    if (
        corpus.get("status") != "verified_group_safe_gravityspy_aligned_network_corpus"
        or corpus.get("passed") is not True
    ):
        raise ValueError("Hard-subset Gravity Spy corpus audit did not pass")
    rows = _read_jsonl(validation_manifest)
    required_split = str(settings.get("required_split"))
    if required_split != "val" or any(row.get("split") != required_split for row in rows):
        raise ValueError("Hard subset must use validation rows only")
    required_fields = (
        "mixture_id",
        "injection_id",
        "waveform_id",
        "glitch_id",
        "network_gps_block",
        "observing_run",
        "ml_label",
        "available_ifos",
        "optimal_snr_by_ifo",
        "path",
        "sha256",
        "gravityspy_corpus_audit_sha256",
    )
    for row in rows:
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Hard-subset validation row lacks metadata: {missing}")
        artifact = Path(str(row["path"]))
        if not artifact.is_file() or file_sha256(artifact) != row["sha256"]:
            raise ValueError("Hard-subset validation artifact hash mismatch")
        if row["gravityspy_corpus_audit_sha256"] != corpus_sha:
            raise ValueError("Hard-subset row differs from the corpus audit")
    _require_unique(
        rows,
        ("mixture_id", "injection_id", "waveform_id", "glitch_id"),
        "hard-subset validation bank",
    )

    family_counts = Counter(str(row["ml_label"]) for row in rows)
    rare_families = {
        family
        for family, count in family_counts.items()
        if count / len(rows) <= rare_fraction
    }
    if not rare_families:
        raise ValueError("Hard-subset policy found no predeclared rare glitch families")

    selected = []
    stratum_counts: Counter[str] = Counter()
    stratum_glitches: dict[str, set[str]] = {
        stratum: set() for stratum in required_strata
    }
    for row in rows:
        available_ifos = row["available_ifos"]
        if not isinstance(available_ifos, list) or not available_ifos:
            raise ValueError("Hard-subset detector availability is invalid")
        observed_ifos = [str(value) for value in available_ifos]
        if len(set(observed_ifos)) != len(observed_ifos):
            raise ValueError("Hard-subset detector availability contains duplicates")
        raw_snr = row["optimal_snr_by_ifo"]
        if not isinstance(raw_snr, dict) or not raw_snr:
            raise ValueError("Hard-subset row lacks per-IFO optimal SNR")
        snr_values = []
        if any(ifo not in raw_snr for ifo in observed_ifos):
            raise ValueError("Hard-subset observed detector lacks optimal SNR")
        for ifo in observed_ifos:
            value = raw_snr[ifo]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Hard-subset per-IFO SNR must be numeric")
            snr_values.append(float(value))
        network_snr = float(np.sqrt(np.sum(np.square(snr_values))))
        if not np.isfinite(network_snr):
            raise ValueError("Hard-subset network SNR must be finite")
        strata = []
        if network_snr <= low_snr_max:
            strata.append("low_network_snr")
        if len(observed_ifos) < full_detector_count:
            strata.append("missing_detector")
        if str(row["observing_run"]) in transfer_runs:
            strata.append("o3b_transfer")
        if str(row["ml_label"]) in rare_families:
            strata.append("rare_glitch_family")
        if not strata:
            continue
        record = dict(row)
        record["hard_subset_strata"] = sorted(strata)
        record["hard_subset_network_snr"] = network_snr
        selected.append(record)
        for stratum in strata:
            stratum_counts[stratum] += 1
            stratum_glitches[stratum].add(str(row["glitch_id"]))

    failures = {
        stratum: {
            "rows": int(stratum_counts[stratum]),
            "unique_glitches": len(stratum_glitches[stratum]),
        }
        for stratum in required_strata
        if stratum_counts[stratum] < minimum_rows
        or len(stratum_glitches[stratum]) < minimum_glitches
    }
    if failures:
        raise ValueError(f"Hard-subset strata are undersized: {failures}")
    selected.sort(key=lambda row: str(row["mixture_id"]))
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
    )
    report = {
        "status": "frozen_score_blind_physical_overlap_scaling_hard_subset",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_scores_inspected": False,
        "model_outputs_inspected": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "policy": str(settings["policy"]),
        "required_split": required_split,
        "rows": len(selected),
        "validation_rows_considered": len(rows),
        "required_strata": required_strata,
        "strata": {
            stratum: {
                "rows": int(stratum_counts[stratum]),
                "unique_glitches": len(stratum_glitches[stratum]),
            }
            for stratum in required_strata
        },
        "rare_glitch_families": sorted(rare_families),
        "hard_subset_manifest_path": str(manifest_path.resolve()),
        "hard_subset_manifest_sha256": file_sha256(manifest_path),
        "validation_manifest_path": str(Path(validation_manifest).resolve()),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "gravityspy_corpus_audit": {
            "path": str(corpus_path),
            "sha256": corpus_sha,
        },
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": file_sha256(config_path),
        },
        "selection_thresholds": {
            "minimum_rows_per_stratum": minimum_rows,
            "minimum_unique_glitches_per_stratum": minimum_glitches,
            "low_network_snr_max": low_snr_max,
            "full_detector_count": full_detector_count,
            "transfer_observing_runs": sorted(transfer_runs),
            "rare_glitch_family_max_fraction": rare_fraction,
        },
        **provenance,
    }
    atomic_write_json(report_path, report)
    return report


def _fft_upsample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("FFT upsampling requires finite one-dimensional strain")
    if source_rate == target_rate:
        return signal.copy()
    if source_rate <= 0 or target_rate % source_rate:
        raise ValueError("Target sample rate must be an integer multiple of source rate")
    target_size = signal.size * (target_rate // source_rate)
    source_spectrum = np.fft.rfft(signal)
    target_spectrum = np.zeros(target_size // 2 + 1, dtype=np.complex128)
    target_spectrum[: source_spectrum.size] = source_spectrum
    # For an even-length source the final rFFT bin is an unpaired Nyquist
    # coefficient. At the larger target length it gains a conjugate partner,
    # so split it to preserve the original samples and amplitude.
    if signal.size % 2 == 0:
        target_spectrum[source_spectrum.size - 1] *= 0.5
    return np.fft.irfft(target_spectrum, n=target_size) * (target_size / signal.size)


def build_contaminated_injection_overrides(
    overlap_manifest: str | Path,
    injection_manifest: str | Path,
    output_dir: str | Path,
    required_split: str,
) -> dict[str, Any]:
    """Expose real-glitch overlap strain through the standard injection scorer contract."""

    if required_split not in {"train", "val", "test"}:
        raise ValueError("Contaminated override split must be train, val or test")
    overlaps = _read_jsonl(overlap_manifest)
    injections = _read_jsonl(injection_manifest)
    if any(row.get("split") != required_split for row in overlaps):
        raise ValueError("Overlap manifest contains a different split")
    if any(row.get("split") != required_split for row in injections):
        raise ValueError("Injection manifest contains a different split")
    injection_by_id = {str(row["injection_id"]): row for row in injections}
    if len(injection_by_id) != len(injections):
        raise ValueError("Injection manifest contains duplicate injection IDs")
    overlap_ids = [str(row["injection_id"]) for row in overlaps]
    if len(overlap_ids) != len(set(overlap_ids)):
        raise ValueError("Overlap manifest reuses an injection ID")
    missing = sorted(set(overlap_ids) - set(injection_by_id))
    if missing:
        raise ValueError(f"Overlap rows lack source injection metadata: {missing[:10]}")

    output = Path(output_dir)
    records = []
    verified_background_hashes: dict[str, str] = {}
    for overlap in overlaps:
        injection = injection_by_id[str(overlap["injection_id"])]
        if str(overlap["injection_materialized_sha256"]) != str(
            injection["materialized_sha256"]
        ):
            raise ValueError("Overlap and injection materialized hashes differ")
        signal_scale = float(overlap.get("training_signal_scale", 1.0))
        if required_split != "train" and not np.isclose(
            signal_scale, 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Validation/test contaminated overrides cannot rescale injections")
        artifact = Path(overlap["path"])
        if file_sha256(artifact) != str(overlap["sha256"]):
            raise ValueError(f"Overlap artifact hash mismatch: {overlap['mixture_id']}")
        with np.load(artifact, allow_pickle=False) as arrays:
            mixture = np.asarray(arrays["mixture_strain"], dtype=np.float64)
            overlap_ifos = [str(value) for value in arrays["ifos"].tolist()]
            overlap_rate = int(arrays["sample_rate"])
            availability = np.asarray(arrays["detector_availability"], dtype=np.uint8)
        context = load_materialized_context(injection, verified_background_hashes)
        source_ifos = list(context["ifos"])
        source_rate = int(context["sample_rate"])
        start = int(context["analysis_start_index"])
        stop = int(context["analysis_stop_index"])
        expected_samples = stop - start
        selected = []
        for ifo in source_ifos:
            if ifo not in overlap_ifos:
                raise ValueError(f"Overlap artifact lacks source detector {ifo}")
            overlap_index = overlap_ifos.index(ifo)
            if availability[overlap_index] != 1:
                raise ValueError(f"Overlap artifact marks source detector {ifo} unavailable")
            values = _fft_upsample(mixture[overlap_index], overlap_rate, source_rate)
            if values.shape != (expected_samples,):
                raise ValueError("Overlap and source injection analysis durations differ")
            selected.append(values)
        analysis = np.stack(selected)
        override_path = output / "arrays" / f"{overlap['mixture_id']}.npz"
        _atomic_save_npz(
            override_path,
            analysis_strain=analysis,
            ifos=np.asarray(source_ifos),
            sample_rate=np.asarray(source_rate, dtype=np.int64),
            analysis_gps_start=np.asarray(
                context["analysis_gps_start"], dtype=np.float64
            ),
        )
        records.append(
            {
                **injection,
                "analysis_override_path": str(override_path),
                "analysis_override_sha256": file_sha256(override_path),
                "analysis_override_kind": "real_glitch_contaminated",
                "overlap_mixture_id": overlap["mixture_id"],
                "overlap_artifact_sha256": overlap["sha256"],
                "glitch_id": overlap["glitch_id"],
                "glitch_gps_block": overlap["network_gps_block"],
                "glitch_ifo": overlap["glitch_ifo"],
                "glitch_label": overlap.get("ml_label"),
                "glitch_mask_provenance": overlap.get("mask_provenance"),
            }
        )
    manifest = output / f"contaminated_injection_{required_split}.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    clean_manifest = output / f"paired_clean_injection_{required_split}.jsonl"
    paired_clean = [injection_by_id[str(row["injection_id"])] for row in overlaps]
    atomic_write_text(
        clean_manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paired_clean),
    )
    report = {
        "status": "verified_real_glitch_contaminated_injection_overrides",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "contaminated and mask-conditioned scores require validation-only threshold "
            "calibration, clean non-inferiority and continuous-background evaluation"
        ),
        "split": required_split,
        "rows": len(records),
        "unique_injection_ids": len({row["injection_id"] for row in records}),
        "unique_waveform_ids": len({row["waveform_id"] for row in records}),
        "unique_glitch_ids": len({row["glitch_id"] for row in records}),
        "unique_injection_gps_blocks": len({row["gps_block"] for row in records}),
        "unique_glitch_gps_blocks": len(
            {row["glitch_gps_block"] for row in records}
        ),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "paired_clean_manifest_path": str(clean_manifest),
        "paired_clean_manifest_sha256": file_sha256(clean_manifest),
        "overlap_manifest_sha256": file_sha256(overlap_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        **execution_provenance(),
    }
    atomic_write_json(output / "contaminated_injection_report.json", report)
    return report
