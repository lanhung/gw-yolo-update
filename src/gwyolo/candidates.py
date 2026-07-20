from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .trigger import network_ranking


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


def _cluster_network_rows(
    rows: list[dict[str, Any]], cluster_window_seconds: float
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["gps_peak"]))
    groups = [[ordered[0]]]
    for row in ordered[1:]:
        if float(row["gps_peak"]) - float(groups[-1][-1]["gps_peak"]) <= cluster_window_seconds:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [max(group, key=lambda row: float(row["ranking_score"])) for group in groups]


def build_candidate_time_slides(
    candidates: Iterable[dict[str, Any]],
    background_windows: Iterable[dict[str, Any]],
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    coincidence_window_seconds: float,
    cluster_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reference_ifo == shifted_ifo:
        raise ValueError("reference and shifted IFOs must differ")
    if min(slide_count, step_seconds, coincidence_window_seconds, cluster_window_seconds) <= 0:
        raise ValueError("slide, step, coincidence, and cluster values must be positive")
    windows = [row for row in background_windows if str(row["split"]) == split]
    if not windows:
        raise ValueError(f"No background windows for split {split}")
    durations = {float(row["gps_end"]) - float(row["gps_start"]) for row in windows}
    if len(durations) != 1:
        raise ValueError("Candidate slides require one common window duration")
    duration = next(iter(durations))
    if step_seconds < duration:
        raise ValueError("Candidate slide step must be at least one window duration")
    by_start = {int(round(float(row["gps_start"]) * 1e9)): row for row in windows}
    if len(by_start) != len(windows):
        raise ValueError("Background windows have duplicate GPS starts")
    candidates_by_window: dict[str, dict[str, list[dict[str, Any]]]] = {}
    maximum_bin_width = 0.0
    window_ids = {str(row["window_id"]) for row in windows}
    for row in candidates:
        if str(row["split"]) != split:
            continue
        window_id = str(row["window_id"])
        if window_id not in window_ids:
            raise ValueError(f"Candidate references an unknown {split} window: {window_id}")
        ifo = str(row["ifo"])
        candidates_by_window.setdefault(window_id, {}).setdefault(ifo, []).append(row)
        maximum_bin_width = max(maximum_bin_width, float(row["bin_width_seconds"]))
    output = []
    slide_exposure = []
    for slide_index in range(1, slide_count + 1):
        offset = slide_index * step_seconds
        offset_key = int(round(offset * 1e9))
        intervals = []
        raw = []
        paired_windows = 0
        for reference in sorted(windows, key=lambda row: float(row["gps_start"])):
            reference_key = int(round(float(reference["gps_start"]) * 1e9))
            shifted = by_start.get(reference_key + offset_key)
            if shifted is None:
                continue
            paired_windows += 1
            intervals.append((float(reference["gps_start"]), float(reference["gps_end"])))
            reference_candidates = candidates_by_window.get(str(reference["window_id"]), {}).get(
                reference_ifo, []
            )
            shifted_candidates = candidates_by_window.get(str(shifted["window_id"]), {}).get(
                shifted_ifo, []
            )
            for first in reference_candidates:
                for second in shifted_candidates:
                    shifted_time = float(second["gps_peak"]) - offset
                    separation = abs(float(first["gps_peak"]) - shifted_time)
                    if separation > coincidence_window_seconds:
                        continue
                    chirp_scores = {
                        reference_ifo: float(first["chirp_score"]),
                        shifted_ifo: float(second["chirp_score"]),
                    }
                    glitch_scores = {
                        reference_ifo: float(first["glitch_score_at_peak"]),
                        shifted_ifo: float(second["glitch_score_at_peak"]),
                    }
                    raw.append(
                        {
                            "candidate_id": f"slide-candidate-{canonical_hash({'slide': slide_index, 'first': first['candidate_id'], 'second': second['candidate_id']}, 24)}",
                            "slide_id": f"slide-{slide_index:06d}",
                            "slide_index": slide_index,
                            "split": split,
                            "gps_peak": (float(first["gps_peak"]) + shifted_time) / 2,
                            "peak_separation_seconds": separation,
                            "offset_seconds": {reference_ifo: 0.0, shifted_ifo: offset},
                            "source_candidate_ids": {
                                reference_ifo: first["candidate_id"],
                                shifted_ifo: second["candidate_id"],
                            },
                            "source_window_ids": {
                                reference_ifo: reference["window_id"],
                                shifted_ifo: shifted["window_id"],
                            },
                            "source_gps_blocks": {
                                reference_ifo: reference["gps_block"],
                                shifted_ifo: shifted["gps_block"],
                            },
                            "chirp_scores": chirp_scores,
                            "glitch_scores": glitch_scores,
                            **network_ranking(
                                chirp_scores,
                                glitch_scores,
                                [reference_ifo, shifted_ifo],
                            ),
                        }
                    )
        clustered = _cluster_network_rows(raw, cluster_window_seconds)
        output.extend(clustered)
        exposure = _union_duration(intervals)
        slide_exposure.append(
            {
                "slide_index": slide_index,
                "offset_seconds": offset,
                "paired_windows": paired_windows,
                "raw_coincidences": len(raw),
                "clustered_candidates": len(clustered),
                "live_time_seconds": exposure,
            }
        )
    exposure_seconds = sum(row["live_time_seconds"] for row in slide_exposure)
    report = {
        "status": "subwindow_clustered_time_slide_integration_only",
        "scientific_claim_allowed": False,
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "slide_count": slide_count,
        "step_seconds": step_seconds,
        "coincidence_window_seconds": coincidence_window_seconds,
        "cluster_window_seconds": cluster_window_seconds,
        "input_windows": len(windows),
        "input_candidates": sum(
            len(rows) for by_ifo in candidates_by_window.values() for rows in by_ifo.values()
        ),
        "background_rows": len(output),
        "slide_exposure": slide_exposure,
        "equivalent_live_time_seconds": exposure_seconds,
        "equivalent_live_time_years": exposure_seconds / SECONDS_PER_YEAR,
        "maximum_bin_width_seconds": maximum_bin_width or None,
        "publication_timing_gate_passed": (
            maximum_bin_width > 0
            and maximum_bin_width <= 0.01
            and coincidence_window_seconds <= 0.01
        ),
    }
    return output, report


def run_candidate_time_slides(
    candidates_path: str | Path,
    background_manifest: str | Path,
    output_dir: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    coincidence_window_seconds: float,
    cluster_window_seconds: float,
) -> dict[str, Any]:
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        windows = [json.loads(line) for line in handle if line.strip()]
    rows, report = build_candidate_time_slides(
        candidates,
        windows,
        split,
        reference_ifo,
        shifted_ifo,
        slide_count,
        step_seconds,
        coincidence_window_seconds,
        cluster_window_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{split}_candidate_time_slide_background.jsonl"
    atomic_write_text(
        manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        **report,
        "candidate_manifest_sha256": file_sha256(candidates_path),
        "background_manifest_sha256": file_sha256(background_manifest),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(output / f"{split}_candidate_time_slide_report.json", result)
    return result
