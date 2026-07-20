from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _parabolic_offset(profile: np.ndarray, index: int) -> float:
    if index <= 0 or index >= profile.size - 1:
        return 0.0
    left, center, right = (float(profile[index - 1]), float(profile[index]), float(profile[index + 1]))
    denominator = left - 2.0 * center + right
    if denominator >= 0 or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def extract_temporal_clusters(
    chirp_probability: np.ndarray,
    glitch_probability: np.ndarray,
    ifos: Iterable[str],
    gps_start: float,
    duration: float,
    chirp_threshold: float,
    minimum_bins: int = 1,
) -> list[dict[str, Any]]:
    chirp = np.asarray(chirp_probability, dtype=np.float64)
    glitch = np.asarray(glitch_probability, dtype=np.float64)
    ifo_names = [str(ifo) for ifo in ifos]
    if chirp.shape != glitch.shape or chirp.ndim != 4:
        raise ValueError("probabilities must share shape [IFO, Q, frequency, time]")
    if chirp.shape[0] != len(ifo_names) or chirp.shape[-1] < 2:
        raise ValueError("probability shape does not match IFOs or has too few time bins")
    if not np.isfinite(chirp).all() or not np.isfinite(glitch).all():
        raise ValueError("probabilities must be finite")
    if not 0 <= chirp_threshold <= 1 or minimum_bins <= 0 or duration <= 0:
        raise ValueError("invalid threshold, minimum bin count, or duration")
    chirp_profiles = np.max(chirp, axis=(1, 2))
    glitch_profiles = np.max(glitch, axis=(1, 2))
    time_bins = chirp.shape[-1]
    bin_width = duration / time_bins
    output = []
    for ifo_index, ifo in enumerate(ifo_names):
        profile = chirp_profiles[ifo_index]
        for start, stop in _active_runs(profile >= chirp_threshold):
            if stop - start < minimum_bins:
                continue
            local_peak = int(np.argmax(profile[start:stop]))
            peak_index = start + local_peak
            sub_bin_offset = _parabolic_offset(profile, peak_index)
            output.append(
                {
                    "ifo": ifo,
                    "start_bin": start,
                    "stop_bin_exclusive": stop,
                    "peak_bin": peak_index,
                    "sub_bin_offset": sub_bin_offset,
                    "gps_start": gps_start + start * bin_width,
                    "gps_end": gps_start + stop * bin_width,
                    "gps_peak": gps_start + (peak_index + 0.5 + sub_bin_offset) * bin_width,
                    "chirp_score": float(profile[peak_index]),
                    "glitch_score_at_peak": float(glitch_profiles[ifo_index, peak_index]),
                    "chirp_glitch_margin": float(
                        profile[peak_index] - glitch_profiles[ifo_index, peak_index]
                    ),
                    "cluster_bins": stop - start,
                    "time_bins": time_bins,
                    "bin_width_seconds": bin_width,
                    "timing_uncertainty_floor_seconds": bin_width / 2,
                    "timing_refinement": "three-bin parabolic interpolation",
                }
            )
    return output


def run_candidate_extraction(
    trigger_manifest: str | Path,
    output_dir: str | Path,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
) -> dict[str, Any]:
    with Path(trigger_manifest).open("r", encoding="utf-8") as handle:
        trigger_rows = [json.loads(line) for line in handle if line.strip()]
    if not trigger_rows:
        raise ValueError("Trigger manifest cannot be empty")
    output_rows = []
    bin_widths = set()
    for row in trigger_rows:
        path = row.get("probability_path")
        expected_sha = row.get("probability_sha256")
        if not path or not expected_sha or file_sha256(path) != expected_sha:
            raise ValueError(f"Missing or invalid probability artifact for {row.get('window_id')}")
        with np.load(path, allow_pickle=False) as payload:
            ifos = [str(item) for item in payload["ifos"].tolist()]
            clusters = extract_temporal_clusters(
                payload["chirp_probability"],
                payload["glitch_probability"],
                ifos,
                float(row["gps_start"]),
                float(row["gps_end"]) - float(row["gps_start"]),
                chirp_threshold,
                minimum_bins,
            )
        valid_ifos = set(row["valid_ifos"])
        for cluster_index, cluster in enumerate(clusters):
            if cluster["ifo"] not in valid_ifos:
                continue
            bin_widths.add(float(cluster["bin_width_seconds"]))
            output_rows.append(
                {
                    "candidate_id": f"candidate-{canonical_hash({'window': row['window_id'], 'ifo': cluster['ifo'], 'index': cluster_index}, 24)}",
                    "window_id": row["window_id"],
                    "split": row["split"],
                    "gps_block": row["gps_block"],
                    **cluster,
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "single_ifo_candidates.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    maximum_bin_width = max(bin_widths) if bin_widths else None
    report = {
        "status": "subwindow_cluster_integration_only",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "publication coincidence requires a validated <=10 ms timing representation, "
            "clustered time slides and adequate independent exposure"
        ),
        "trigger_manifest_path": str(trigger_manifest),
        "trigger_manifest_sha256": file_sha256(trigger_manifest),
        "input_windows": len(trigger_rows),
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "candidates": len(output_rows),
        "candidate_counts_by_ifo": dict(
            sorted(Counter(row["ifo"] for row in output_rows).items())
        ),
        "maximum_bin_width_seconds": maximum_bin_width,
        "publication_timing_gate_passed": (
            maximum_bin_width is not None and maximum_bin_width <= 0.01
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(output / "candidate_extraction_report.json", report)
    return report
