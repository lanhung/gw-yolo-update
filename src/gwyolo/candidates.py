from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .metrics import wilson_interval
from .runtime import execution_provenance
from .trigger import network_ranking


def _available_ifos(row: dict[str, Any]) -> set[str]:
    """Return the explicit detector set for a background window.

    Background plans use ``ifos`` while scored trigger rows use ``valid_ifos``.  A
    time-slide exposure is meaningful only when the detector contributing that
    side of the coincidence was actually observing.
    """

    values = row.get("valid_ifos", row.get("ifos"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"Window {row.get('window_id')} lacks an explicit detector set")
    ifos = {str(value) for value in values}
    if len(ifos) != len(values):
        raise ValueError(f"Window {row.get('window_id')} repeats a detector")
    return ifos


def _scoring_provenance(
    trigger_manifest: str | Path, report_filename: str
) -> dict[str, Any]:
    """Verify the scorer report adjacent to a trigger manifest when available."""

    manifest = Path(trigger_manifest)
    report_path = manifest.parent / report_filename
    if not report_path.is_file():
        return {"available": False, "blocker": f"missing {report_path}"}
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if str(report.get("triggers_sha256")) != file_sha256(manifest):
        raise ValueError("adjacent score report does not bind the trigger manifest")
    required = ("checkpoint_sha256", "config_sha256", "code_commit")
    missing = [field for field in required if not report.get(field)]
    if missing:
        raise ValueError(f"adjacent score report lacks provenance: {missing}")
    return {
        "available": True,
        "score_report_path": str(report_path),
        "score_report_sha256": file_sha256(report_path),
        "checkpoint_sha256": str(report["checkpoint_sha256"]),
        "config_sha256": str(report["config_sha256"]),
        "code_commit": str(report["code_commit"]),
        "source_manifest_sha256": str(report["manifest_sha256"]),
        "trigger_manifest_sha256": str(report["triggers_sha256"]),
    }


def _candidate_extraction_provenance(candidate_manifest: str | Path) -> dict[str, Any]:
    manifest = Path(candidate_manifest)
    names = (
        "candidate_extraction_report.json",
        "injection_candidate_extraction_report.json",
    )
    reports = [manifest.parent / name for name in names if (manifest.parent / name).is_file()]
    if not reports:
        return {"available": False, "blocker": "missing adjacent candidate extraction report"}
    if len(reports) != 1:
        raise ValueError("candidate manifest has ambiguous adjacent extraction reports")
    report_path = reports[0]
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if str(report.get("manifest_sha256")) != file_sha256(manifest):
        raise ValueError("candidate extraction report does not bind its manifest")
    scoring = report.get("source_scoring_provenance", {})
    return {
        "available": bool(scoring.get("available")),
        "candidate_extraction_report_path": str(report_path),
        "candidate_extraction_report_sha256": file_sha256(report_path),
        "candidate_manifest_sha256": str(report["manifest_sha256"]),
        "scoring": scoring,
        "blocker": scoring.get("blocker") if not scoring.get("available") else None,
    }


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


def _apply_strain_timing_refinement(
    clusters: list[dict[str, Any]], scored_row: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach the exact timing method used by the continuous candidate path."""

    output = []
    for source in clusters:
        cluster = dict(source)
        timing = {
            "timing_method": "mask_profile_parabolic",
            "timing_resolution_seconds": float(cluster["bin_width_seconds"]),
            "timing_empirically_calibrated": False,
        }
        ifo = str(cluster["ifo"])
        refined = scored_row.get("strain_envelope_peak_times", {}).get(ifo)
        coarse = scored_row.get("peak_times", {}).get("chirp", {}).get(ifo)
        if refined is not None and coarse is not None:
            coarse_bin = int(coarse["time_bin"])
            if cluster["start_bin"] <= coarse_bin < cluster["stop_bin_exclusive"]:
                cluster["mask_profile_gps_peak"] = cluster["gps_peak"]
                cluster["gps_peak"] = float(refined["gps"])
                timing = {
                    "timing_method": "local_strain_envelope_in_global_chirp_roi",
                    "timing_resolution_seconds": float(
                        refined["sample_resolution_seconds"]
                    ),
                    "timing_smoothing_seconds": float(refined["smoothing_seconds"]),
                    "timing_empirically_calibrated": False,
                }
        output.append({**cluster, **timing})
    return output


def _local_envelope_timing_refinement(
    clusters: list[dict[str, Any]],
    whitened_strain: np.ndarray,
    ifos: list[str],
    analysis_gps_start: float,
    sample_rate: int,
    padding_seconds: float = 0.05,
    smoothing_seconds: float = 0.008,
) -> list[dict[str, Any]]:
    """Refine every mask cluster from its own local whitened-strain envelope."""

    strain = np.asarray(whitened_strain, dtype=np.float64)
    if (
        strain.ndim != 2
        or strain.shape[0] != len(ifos)
        or not np.isfinite(strain).all()
        or sample_rate <= 0
        or padding_seconds < 0
        or smoothing_seconds <= 0
    ):
        raise ValueError("local candidate timing strain contract is invalid")
    smoothing_samples = max(1, int(round(smoothing_seconds * sample_rate)))
    kernel = np.ones(smoothing_samples, dtype=np.float64) / smoothing_samples
    envelopes = []
    for values in strain:
        spectrum = np.fft.fft(values)
        multiplier = np.zeros(values.size, dtype=np.float64)
        multiplier[0] = 1.0
        if values.size % 2 == 0:
            multiplier[1 : values.size // 2] = 2.0
            multiplier[values.size // 2] = 1.0
        else:
            multiplier[1 : (values.size + 1) // 2] = 2.0
        envelope = np.abs(np.fft.ifft(spectrum * multiplier))
        envelopes.append(np.convolve(envelope, kernel, mode="same"))
    output = []
    padding_samples = int(round(padding_seconds * sample_rate))
    for source in clusters:
        row = dict(source)
        ifo_index = ifos.index(str(row["ifo"]))
        nominal_start = int(
            np.floor((float(row["gps_start"]) - analysis_gps_start) * sample_rate)
        )
        nominal_stop = int(
            np.ceil((float(row["gps_end"]) - analysis_gps_start) * sample_rate)
        )
        start = max(0, nominal_start - padding_samples)
        stop = min(strain.shape[1], nominal_stop + padding_samples)
        if stop <= start:
            raise ValueError("mask cluster lies outside saved whitened strain")
        peak_index = start + int(np.argmax(envelopes[ifo_index][start:stop]))
        row.update(
            {
                "mask_profile_gps_peak": row["gps_peak"],
                "gps_peak": analysis_gps_start + peak_index / sample_rate,
                "timing_method": "local_whitened_strain_envelope_per_mask_cluster_v1",
                "timing_resolution_seconds": 1.0 / sample_rate,
                "timing_smoothing_seconds": smoothing_samples / sample_rate,
                "timing_search_padding_seconds": padding_samples / sample_rate,
                "timing_search_start_gps": analysis_gps_start + start / sample_rate,
                "timing_search_end_gps": analysis_gps_start + stop / sample_rate,
                "timing_empirically_calibrated": False,
            }
        )
        output.append(row)
    return output


def candidate_proposal_coverage(
    injection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    padding_seconds: float,
) -> dict[str, Any]:
    """Measure all-instance proposal support without relabeling it as search recall."""

    if not injection_rows or padding_seconds < 0:
        raise ValueError("proposal coverage requires injections and non-negative padding")
    injections = {}
    for row in injection_rows:
        injection_id = str(row["injection_id"])
        if injection_id in injections:
            raise ValueError(f"duplicate proposal-audit injection: {injection_id}")
        injections[injection_id] = row
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    candidate_ids = set()
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate proposal candidate: {candidate_id}")
        candidate_ids.add(candidate_id)
        injection_id = str(row["injection_id"])
        if injection_id not in injections:
            raise ValueError(f"proposal candidate has unknown injection: {injection_id}")
        ifo = str(row["ifo"])
        arrivals = injections[injection_id].get("detector_arrival_gps", {})
        if ifo not in arrivals:
            raise ValueError(f"proposal candidate uses unavailable IFO: {injection_id}/{ifo}")
        start = float(row["gps_start"])
        stop = float(row["gps_end"])
        peak = float(row["gps_peak"])
        if not np.isfinite([start, stop, peak]).all() or stop <= start:
            raise ValueError(f"proposal candidate interval is invalid: {candidate_id}")
        candidates.setdefault((injection_id, ifo), []).append(row)

    audit_rows = []
    for injection_id, injection in injections.items():
        arrivals = {
            str(ifo): float(value)
            for ifo, value in injection.get("detector_arrival_gps", {}).items()
        }
        if len(arrivals) < 2:
            raise ValueError(f"proposal audit injection lacks network arrivals: {injection_id}")
        ifo_snr = {
            str(ifo): float(value)
            for ifo, value in injection.get("optimal_snr_by_ifo", {}).items()
        }
        for ifo, arrival in arrivals.items():
            rows = candidates.get((injection_id, ifo), [])
            interval_distances = []
            peak_errors = []
            containing_widths = []
            intervals = []
            for row in rows:
                start = float(row["gps_start"])
                stop = float(row["gps_end"])
                intervals.append((start, stop))
                interval_distances.append(
                    max(start - arrival, 0.0, arrival - stop)
                )
                peak_errors.append(abs(float(row["gps_peak"]) - arrival))
                if start <= arrival <= stop:
                    containing_widths.append(stop - start)
            nearest_interval = min(interval_distances) if interval_distances else None
            nearest_peak = min(peak_errors) if peak_errors else None
            analysis_duration = (
                int(injection["analysis_stop_index"])
                - int(injection["analysis_start_index"])
            ) / float(injection["sample_rate"])
            union_duration = _union_duration(intervals)
            if analysis_duration <= 0 or union_duration > analysis_duration + 1e-6:
                raise ValueError(
                    f"proposal intervals exceed analysis duration: {injection_id}/{ifo}"
                )
            audit_rows.append(
                {
                    "injection_id": injection_id,
                    "waveform_id": str(injection["waveform_id"]),
                    "ifo": ifo,
                    "source_family": str(injection["source_family"]),
                    "optimal_snr_stratum": str(
                        injection.get("optimal_snr_stratum", "unassigned")
                    ),
                    "ifo_optimal_snr": ifo_snr.get(ifo),
                    "proposal_count": len(rows),
                    "has_proposal": bool(rows),
                    "arrival_inside_proposal": nearest_interval == 0.0,
                    "arrival_inside_padded_proposal": (
                        nearest_interval is not None
                        and nearest_interval <= padding_seconds
                    ),
                    "nearest_interval_distance_seconds": nearest_interval,
                    "nearest_peak_error_seconds": nearest_peak,
                    "minimum_containing_proposal_width_seconds": (
                        min(containing_widths) if containing_widths else None
                    ),
                    "proposal_union_duration_seconds": union_duration,
                    "proposal_union_fraction_of_analysis": union_duration
                    / analysis_duration,
                }
            )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        if total == 0:
            raise ValueError("proposal coverage group is empty")
        any_count = sum(bool(row["has_proposal"]) for row in rows)
        interval_count = sum(bool(row["arrival_inside_proposal"]) for row in rows)
        padded_count = sum(
            bool(row["arrival_inside_padded_proposal"]) for row in rows
        )
        proposal_counts = np.asarray(
            [int(row["proposal_count"]) for row in rows], dtype=np.float64
        )
        peak_errors = np.asarray(
            [
                float(row["nearest_peak_error_seconds"])
                for row in rows
                if row["nearest_peak_error_seconds"] is not None
            ],
            dtype=np.float64,
        )
        union_fractions = np.asarray(
            [float(row["proposal_union_fraction_of_analysis"]) for row in rows],
            dtype=np.float64,
        )
        containing_widths = np.asarray(
            [
                float(row["minimum_containing_proposal_width_seconds"])
                for row in rows
                if row["minimum_containing_proposal_width_seconds"] is not None
            ],
            dtype=np.float64,
        )
        result = {
            "expected_detector_arrivals": total,
            "arrivals_with_any_proposal": any_count,
            "any_proposal_fraction": any_count / total,
            "any_proposal_wilson_95": list(wilson_interval(any_count, total)),
            "arrivals_inside_proposal": interval_count,
            "interval_coverage_fraction": interval_count / total,
            "interval_coverage_wilson_95": list(
                wilson_interval(interval_count, total)
            ),
            "arrivals_inside_padded_proposal": padded_count,
            "padded_coverage_fraction": padded_count / total,
            "padded_coverage_wilson_95": list(wilson_interval(padded_count, total)),
            "proposal_count_quantiles": {
                str(q): float(np.quantile(proposal_counts, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            },
            "proposal_union_fraction_of_analysis_quantiles": {
                str(q): float(np.quantile(union_fractions, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            },
        }
        if peak_errors.size:
            result["nearest_peak_error_seconds_quantiles_conditional_on_proposal"] = {
                str(q): float(np.quantile(peak_errors, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            }
        if containing_widths.size:
            result["minimum_containing_proposal_width_seconds_quantiles"] = {
                str(q): float(np.quantile(containing_widths, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            }
        return result

    groups: dict[str, list[dict[str, Any]]] = {"all": audit_rows}
    for row in audit_rows:
        for key in (
            f"family:{row['source_family']}",
            f"snr:{row['optimal_snr_stratum']}",
            f"ifo:{row['ifo']}",
        ):
            groups.setdefault(key, []).append(row)
    return {
        "padding_seconds": padding_seconds,
        "injections": len(injections),
        "candidates": len(candidate_rows),
        "audit_rows": audit_rows,
        "groups": {key: summarize(rows) for key, rows in sorted(groups.items())},
    }


def run_candidate_proposal_coverage_audit(
    injection_manifest: str | Path,
    candidate_manifest: str | Path,
    output_path: str | Path,
    padding_seconds: float = 0.5,
) -> dict[str, Any]:
    with Path(injection_manifest).open("r", encoding="utf-8") as handle:
        injections = [json.loads(line) for line in handle if line.strip()]
    with Path(candidate_manifest).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    coverage = candidate_proposal_coverage(injections, candidates, padding_seconds)
    report = {
        "status": "validation_only_all_instance_candidate_proposal_coverage",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "proposal support is not continuous-search recall and still requires frozen-threshold "
            "background, conditional timing and VT evaluation"
        ),
        "injection_manifest": str(injection_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "candidate_extraction_provenance": _candidate_extraction_provenance(
            candidate_manifest
        ),
        **coverage,
        **execution_provenance(),
    }
    destination = Path(output_path)
    if destination.is_file():
        raise ValueError("candidate proposal coverage output already exists")
    atomic_write_json(destination, report)
    return report


def _clusters_from_scored_row(
    row: dict[str, Any], chirp_threshold: float, minimum_bins: int
) -> list[dict[str, Any]]:
    path = row.get("probability_path")
    expected_sha = row.get("probability_sha256")
    if not path or not expected_sha or file_sha256(path) != expected_sha:
        raise ValueError(
            f"Missing or invalid probability artifact for "
            f"{row.get('window_id', row.get('injection_id'))}"
        )
    gps_start = row.get("analysis_gps_start", row.get("gps_start"))
    gps_end = row.get("analysis_gps_end", row.get("gps_end"))
    if gps_start is None or gps_end is None:
        chirp_peaks = row.get("peak_times", {}).get("chirp", {})
        if not chirp_peaks:
            raise ValueError("Scored row lacks an explicit analysis GPS interval")
        first = next(iter(chirp_peaks.values()))
        gps_start = float(first["gps"]) - float(first["offset_seconds"])
        gps_end = float(gps_start) + float(row["duration"])
    with np.load(path, allow_pickle=False) as payload:
        ifos = [str(item) for item in payload["ifos"].tolist()]
        clusters = extract_temporal_clusters(
            payload["chirp_probability"],
            payload["glitch_probability"],
            ifos,
            float(gps_start),
            float(gps_end) - float(gps_start),
            chirp_threshold,
            minimum_bins,
        )
        if "whitened_strain" in payload and "strain_sample_rate" in payload:
            return _local_envelope_timing_refinement(
                clusters,
                payload["whitened_strain"],
                ifos,
                float(gps_start),
                int(payload["strain_sample_rate"]),
            )
    return _apply_strain_timing_refinement(clusters, row)


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
        clusters = _clusters_from_scored_row(row, chirp_threshold, minimum_bins)
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
    source_scoring_provenance = _scoring_provenance(
        trigger_manifest, "trigger_score_report.json"
    )
    report = {
        "status": "subwindow_cluster_integration_only",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "publication coincidence requires a validated <=10 ms timing representation, "
            "clustered time slides and adequate independent exposure"
        ),
        "trigger_manifest_path": str(trigger_manifest),
        "trigger_manifest_sha256": file_sha256(trigger_manifest),
        "source_scoring_provenance": source_scoring_provenance,
        "input_windows": len(trigger_rows),
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "candidates": len(output_rows),
        "candidate_counts_by_ifo": dict(
            sorted(Counter(row["ifo"] for row in output_rows).items())
        ),
        "maximum_bin_width_seconds": maximum_bin_width,
        "timing_method_counts": dict(
            sorted(Counter(row["timing_method"] for row in output_rows).items())
        ),
        "publication_timing_gate_passed": False,
        "publication_timing_blocker": (
            "sample resolution alone is not an uncertainty calibration; calibrate the exact "
            "candidate timing method on validation injections before a coherence claim"
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / "candidate_extraction_report.json", report)
    return report


def calibrate_candidate_timing_rows(
    candidates: Iterable[dict[str, Any]],
    detector_arrivals: dict[str, dict[str, float]],
    association_window_seconds: float,
    uncertainty_quantile: float = 0.99,
    minimum_matches_per_method: int = 30,
    maximum_empirical_timing_uncertainty_seconds: float = 0.01,
) -> dict[str, Any]:
    """Calibrate the exact candidate timing method using validation injections only."""

    rows = list(candidates)
    if association_window_seconds <= 0 or not 0.5 <= uncertainty_quantile < 1.0:
        raise ValueError("timing association window or uncertainty quantile is invalid")
    if (
        minimum_matches_per_method <= 0
        or maximum_empirical_timing_uncertainty_seconds <= 0
    ):
        raise ValueError("timing calibration match and uncertainty limits must be positive")
    grouped_best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        injection_id = str(row["injection_id"])
        ifo = str(row["ifo"])
        target = detector_arrivals.get(injection_id, {}).get(ifo)
        if target is None:
            continue
        error = abs(float(row["gps_peak"]) - float(target))
        if error <= association_window_seconds:
            method = str(row["timing_method"])
            key = (method, injection_id, ifo)
            candidate = {**row, "absolute_error_seconds": error}
            previous = grouped_best.get(key)
            if previous is None or error < float(previous["absolute_error_seconds"]):
                grouped_best[key] = candidate
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (method, _, _), row in grouped_best.items():
        grouped.setdefault(method, []).append(row)
    methods = {}
    target_count = sum(len(values) for values in detector_arrivals.values())
    for method, matches in sorted(grouped.items()):
        errors = np.asarray(
            [float(row["absolute_error_seconds"]) for row in matches], dtype=np.float64
        )
        resolutions = np.asarray(
            [float(row["timing_resolution_seconds"]) for row in matches], dtype=np.float64
        )
        if not np.isfinite(errors).all() or not np.isfinite(resolutions).all():
            raise ValueError("timing calibration values must be finite")
        interval = wilson_interval(len(matches), target_count)
        empirical_uncertainty = float(np.quantile(errors, uncertainty_quantile))
        methods[method] = {
            "matches": len(matches),
            "eligible_detector_arrivals": target_count,
            "conditional_match_fraction": len(matches) / target_count,
            "conditional_match_wilson_95": list(interval),
            "association_window_seconds": association_window_seconds,
            "maximum_resolution_seconds": float(resolutions.max()),
            "absolute_error_seconds_quantiles": {
                str(q): float(np.quantile(errors, q))
                for q in (0.0, 0.5, 0.9, uncertainty_quantile, 1.0)
            },
            "empirical_timing_uncertainty_seconds": empirical_uncertainty,
            "maximum_allowed_empirical_timing_uncertainty_seconds": (
                maximum_empirical_timing_uncertainty_seconds
            ),
            "uncertainty_quantile": uncertainty_quantile,
            "minimum_matches_gate": len(matches) >= minimum_matches_per_method,
            "resolution_gate_10ms": float(resolutions.max()) <= 0.01,
            "empirical_uncertainty_gate": (
                empirical_uncertainty
                <= maximum_empirical_timing_uncertainty_seconds
            ),
            "calibration_gate_passed": (
                len(matches) >= minimum_matches_per_method
                and float(resolutions.max()) <= 0.01
                and empirical_uncertainty
                <= maximum_empirical_timing_uncertainty_seconds
            ),
        }
    return {
        "eligible_detector_arrivals": target_count,
        "input_candidates": len(rows),
        "methods": methods,
        "association_window_seconds": association_window_seconds,
        "uncertainty_quantile": uncertainty_quantile,
        "minimum_matches_per_method": minimum_matches_per_method,
        "maximum_empirical_timing_uncertainty_seconds": (
            maximum_empirical_timing_uncertainty_seconds
        ),
    }


def run_candidate_timing_calibration(
    injection_trigger_manifest: str | Path,
    output: str | Path,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
    association_window_seconds: float = 0.25,
    uncertainty_quantile: float = 0.99,
    minimum_matches_per_method: int = 30,
    maximum_empirical_timing_uncertainty_seconds: float = 0.01,
) -> dict[str, Any]:
    with Path(injection_trigger_manifest).open("r", encoding="utf-8") as handle:
        scored_rows = [json.loads(line) for line in handle if line.strip()]
    if not scored_rows or {str(row.get("split")) for row in scored_rows} != {"val"}:
        raise ValueError("candidate timing calibration requires validation injections only")
    candidates = []
    arrivals = {}
    for row in scored_rows:
        injection_id = str(row["injection_id"])
        valid_ifos = set(row["valid_ifos"])
        detector_targets = {
            str(ifo): float(value)
            for ifo, value in row.get("detector_arrival_gps", {}).items()
            if str(ifo) in valid_ifos
        }
        if not detector_targets:
            raise ValueError(f"Validation injection {injection_id} lacks detector arrival targets")
        arrivals[injection_id] = detector_targets
        for index, cluster in enumerate(
            _clusters_from_scored_row(row, chirp_threshold, minimum_bins)
        ):
            if cluster["ifo"] not in valid_ifos:
                continue
            candidates.append(
                {
                    "candidate_id": f"timing-{canonical_hash({'injection': injection_id, 'ifo': cluster['ifo'], 'index': index}, 24)}",
                    "injection_id": injection_id,
                    "split": "val",
                    **cluster,
                }
            )
    calibration = calibrate_candidate_timing_rows(
        candidates,
        arrivals,
        association_window_seconds,
        uncertainty_quantile,
        minimum_matches_per_method,
        maximum_empirical_timing_uncertainty_seconds,
    )
    source_scoring_provenance = _scoring_provenance(
        injection_trigger_manifest, "injection_score_report.json"
    )
    result = {
        "status": "validation_only_candidate_timing_calibration",
        "scientific_claim_allowed": False,
        "selection_data": "validation_injections_only",
        "test_evaluation": None,
        "injection_trigger_manifest_path": str(injection_trigger_manifest),
        "injection_trigger_manifest_sha256": file_sha256(injection_trigger_manifest),
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "source_scoring_provenance": source_scoring_provenance,
        **calibration,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def run_apply_candidate_timing_calibration(
    candidate_manifest: str | Path,
    calibration_report: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    with Path(calibration_report).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("status") != "validation_only_candidate_timing_calibration":
        raise ValueError("candidate timing calibration report has the wrong status")
    with Path(candidate_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report_sha = file_sha256(calibration_report)
    candidate_provenance = _candidate_extraction_provenance(candidate_manifest)
    calibration_scoring = calibration.get("source_scoring_provenance", {})
    candidate_scoring = candidate_provenance.get("scoring", {})
    provenance_matches = bool(
        calibration_scoring.get("available")
        and candidate_provenance.get("available")
        and all(
            str(calibration_scoring.get(field)) == str(candidate_scoring.get(field))
            for field in ("checkpoint_sha256", "config_sha256", "code_commit")
        )
    )
    calibrated = 0
    output_rows = []
    for source in rows:
        row = dict(source)
        method = calibration.get("methods", {}).get(str(row.get("timing_method")))
        if (
            method is not None
            and bool(method.get("calibration_gate_passed"))
            and provenance_matches
        ):
            if float(row["timing_resolution_seconds"]) > float(
                method["maximum_resolution_seconds"]
            ):
                raise ValueError("candidate timing resolution is outside calibration support")
            row.update(
                {
                    "timing_empirically_calibrated": True,
                    "empirical_timing_uncertainty_seconds": method[
                        "empirical_timing_uncertainty_seconds"
                    ],
                    "timing_uncertainty_quantile": method["uncertainty_quantile"],
                    "timing_calibration_report_sha256": report_sha,
                    "candidate_extraction_report_sha256": candidate_provenance[
                        "candidate_extraction_report_sha256"
                    ],
                    "candidate_checkpoint_sha256": candidate_scoring[
                        "checkpoint_sha256"
                    ],
                    "candidate_config_sha256": candidate_scoring["config_sha256"],
                    "candidate_code_commit": candidate_scoring["code_commit"],
                }
            )
            calibrated += 1
        output_rows.append(row)
    output_path = Path(output)
    atomic_write_text(
        output_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    result = {
        "status": "candidate_timing_calibration_applied",
        "input_candidates": len(rows),
        "calibrated_candidates": calibrated,
        "uncalibrated_candidates": len(rows) - calibrated,
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "calibration_report_sha256": report_sha,
        "candidate_extraction_provenance": candidate_provenance,
        "calibration_scoring_provenance": calibration_scoring,
        "scoring_provenance_matches": provenance_matches,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        **execution_provenance(),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    atomic_write_json(report_path, result)
    result["report_path"] = str(report_path)
    return result


def run_injection_candidate_extraction(
    injection_trigger_manifest: str | Path,
    output_dir: str | Path,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
) -> dict[str, Any]:
    """Preserve every single-IFO candidate from scored physical injections."""

    with Path(injection_trigger_manifest).open("r", encoding="utf-8") as handle:
        scored_rows = [json.loads(line) for line in handle if line.strip()]
    if not scored_rows:
        raise ValueError("injection trigger manifest cannot be empty")
    output_rows = []
    for row in scored_rows:
        valid_ifos = set(row["valid_ifos"])
        for index, cluster in enumerate(
            _clusters_from_scored_row(row, chirp_threshold, minimum_bins)
        ):
            if cluster["ifo"] not in valid_ifos:
                continue
            output_rows.append(
                {
                    "candidate_id": f"injection-candidate-{canonical_hash({'injection': row['injection_id'], 'ifo': cluster['ifo'], 'index': index}, 24)}",
                    "injection_id": row["injection_id"],
                    "waveform_id": row["waveform_id"],
                    "split": row["split"],
                    "source_family": row["source_family"],
                    "gps_block": row["gps_block"],
                    "injection_gps_time": row["gps_time"],
                    "detector_arrival_gps": row.get("detector_arrival_gps", {}),
                    "vt_weight": row["vt_weight"],
                    "vt_weight_unit": row["vt_weight_unit"],
                    **cluster,
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "single_ifo_injection_candidates.jsonl"
    atomic_write_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    source_scoring_provenance = _scoring_provenance(
        injection_trigger_manifest, "injection_score_report.json"
    )
    report = {
        "status": "single_ifo_physical_injection_candidates",
        "scientific_claim_allowed": False,
        "trigger_manifest_path": str(injection_trigger_manifest),
        "trigger_manifest_sha256": file_sha256(injection_trigger_manifest),
        "source_scoring_provenance": source_scoring_provenance,
        "input_injections": len(scored_rows),
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "candidates": len(output_rows),
        "candidate_counts_by_ifo": dict(
            sorted(Counter(row["ifo"] for row in output_rows).items())
        ),
        "timing_method_counts": dict(
            sorted(Counter(row["timing_method"] for row in output_rows).items())
        ),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        **execution_provenance(),
    }
    atomic_write_json(output / "injection_candidate_extraction_report.json", report)
    return report


def build_injection_candidate_rankings(
    injection_rows: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    split: str,
    reference_ifo: str,
    second_ifo: str,
    physical_delay_limit_seconds: float,
    empirical_timing_uncertainty_seconds: float,
    truth_association_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce all calibrated candidates to one physical network ranking per injection."""

    if reference_ifo == second_ifo:
        raise ValueError("injection candidate ranking requires two distinct detectors")
    if min(
        physical_delay_limit_seconds,
        truth_association_window_seconds,
    ) <= 0 or empirical_timing_uncertainty_seconds < 0:
        raise ValueError("injection candidate timing settings are invalid")
    parents = [row for row in injection_rows if str(row["split"]) == split]
    if not parents:
        raise ValueError(f"no scored injections for split {split}")
    by_id = {str(row["injection_id"]): row for row in parents}
    if len(by_id) != len(parents):
        raise ValueError("scored injection rows repeat injection IDs")
    candidate_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in candidates:
        if str(row["split"]) != split:
            continue
        injection_id = str(row["injection_id"])
        if injection_id not in by_id:
            raise ValueError(f"candidate references unknown {split} injection {injection_id}")
        candidate_map.setdefault(injection_id, {}).setdefault(str(row["ifo"]), []).append(row)
    allowed_separation = physical_delay_limit_seconds + 2.0 * empirical_timing_uncertainty_seconds
    outputs = []
    eligible = 0
    recovered_candidate_pairs = 0
    calibration_hashes = set()
    checkpoint_hashes = set()
    config_hashes = set()
    code_commits = set()
    for injection_id, parent in sorted(by_id.items()):
        valid_ifos = set(parent["valid_ifos"])
        arrivals = {
            str(ifo): float(value)
            for ifo, value in parent.get("detector_arrival_gps", {}).items()
        }
        required = {reference_ifo, second_ifo}
        if not required.issubset(valid_ifos) or not required.issubset(arrivals):
            continue
        eligible += 1
        by_ifo = candidate_map.get(injection_id, {})
        pairs = []
        for first in by_ifo.get(reference_ifo, []):
            for second in by_ifo.get(second_ifo, []):
                pair = (first, second)
                if not all(bool(row.get("timing_empirically_calibrated")) for row in pair):
                    continue
                if not all(
                    np.isclose(
                        float(row.get("empirical_timing_uncertainty_seconds", -1)),
                        empirical_timing_uncertainty_seconds,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for row in pair
                ):
                    continue
                if any(
                    abs(float(row["gps_peak"]) - arrivals[str(row["ifo"])])
                    > truth_association_window_seconds
                    for row in pair
                ):
                    continue
                separation = abs(float(first["gps_peak"]) - float(second["gps_peak"]))
                if separation > allowed_separation:
                    continue
                if any(not row.get("timing_calibration_report_sha256") for row in pair):
                    continue
                hashes = {
                    str(row["timing_calibration_report_sha256"]) for row in pair
                }
                if len(hashes) != 1:
                    continue
                if any(
                    not row.get("candidate_checkpoint_sha256")
                    or not row.get("candidate_config_sha256")
                    or not row.get("candidate_code_commit")
                    for row in pair
                ):
                    continue
                pair_checkpoints = {
                    str(row["candidate_checkpoint_sha256"]) for row in pair
                }
                pair_configs = {str(row["candidate_config_sha256"]) for row in pair}
                pair_commits = {str(row["candidate_code_commit"]) for row in pair}
                if not (
                    len(pair_checkpoints) == len(pair_configs) == len(pair_commits) == 1
                ):
                    continue
                calibration_hashes.update(hashes)
                checkpoint_hashes.update(pair_checkpoints)
                config_hashes.update(pair_configs)
                code_commits.update(pair_commits)
                chirp_scores = {
                    reference_ifo: float(first["chirp_score"]),
                    second_ifo: float(second["chirp_score"]),
                }
                glitch_scores = {
                    reference_ifo: float(first["glitch_score_at_peak"]),
                    second_ifo: float(second["glitch_score_at_peak"]),
                }
                pairs.append(
                    {
                        "source_candidate_ids": {
                            reference_ifo: first["candidate_id"],
                            second_ifo: second["candidate_id"],
                        },
                        "peak_separation_seconds": separation,
                        "chirp_scores": chirp_scores,
                        "glitch_scores": glitch_scores,
                        **network_ranking(
                            chirp_scores,
                            glitch_scores,
                            [reference_ifo, second_ifo],
                        ),
                    }
                )
        selected = max(pairs, key=lambda row: float(row["ranking_score"])) if pairs else None
        if selected is not None:
            recovered_candidate_pairs += 1
        outputs.append(
            {
                "injection_id": injection_id,
                "waveform_id": parent["waveform_id"],
                "split": split,
                "source_family": parent["source_family"],
                "stratum": parent.get("stratum", parent["source_family"]),
                "gps_block": parent["gps_block"],
                "gps_time": parent["gps_time"],
                "vt_weight": parent["vt_weight"],
                "vt_weight_unit": parent["vt_weight_unit"],
                "candidate_pair_found": selected is not None,
                "ranking_score": float(selected["ranking_score"]) if selected else 0.0,
                "network_candidate": selected,
            }
        )
    report = {
        "status": "physical_network_injection_candidate_rankings",
        "split": split,
        "input_injections": len(parents),
        "eligible_detector_set_injections": eligible,
        "ranked_injections": len(outputs),
        "candidate_pair_found": recovered_candidate_pairs,
        "excluded_missing_detector_or_arrival": len(parents) - eligible,
        "reference_ifo": reference_ifo,
        "second_ifo": second_ifo,
        "physical_delay_limit_seconds": physical_delay_limit_seconds,
        "empirical_timing_uncertainty_seconds": empirical_timing_uncertainty_seconds,
        "allowed_peak_separation_seconds": allowed_separation,
        "truth_association_window_seconds": truth_association_window_seconds,
        "timing_calibration_report_sha256": (
            next(iter(calibration_hashes)) if len(calibration_hashes) == 1 else None
        ),
        "timing_calibration_consistent": len(calibration_hashes) == 1,
        "candidate_checkpoint_sha256": (
            next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None
        ),
        "candidate_config_sha256": (
            next(iter(config_hashes)) if len(config_hashes) == 1 else None
        ),
        "candidate_code_commit": next(iter(code_commits)) if len(code_commits) == 1 else None,
        "candidate_scoring_provenance_consistent": (
            len(checkpoint_hashes) == len(config_hashes) == len(code_commits) == 1
        ),
    }
    return outputs, report


def run_injection_candidate_rankings(
    injection_trigger_manifest: str | Path,
    candidate_manifest: str | Path,
    output_dir: str | Path,
    split: str,
    reference_ifo: str,
    second_ifo: str,
    physical_delay_limit_seconds: float,
    empirical_timing_uncertainty_seconds: float,
    truth_association_window_seconds: float,
) -> dict[str, Any]:
    with Path(injection_trigger_manifest).open("r", encoding="utf-8") as handle:
        injections = [json.loads(line) for line in handle if line.strip()]
    with Path(candidate_manifest).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    rows, report = build_injection_candidate_rankings(
        injections,
        candidates,
        split,
        reference_ifo,
        second_ifo,
        physical_delay_limit_seconds,
        empirical_timing_uncertainty_seconds,
        truth_association_window_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{split}_network_injection_candidate_rankings.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        **report,
        "injection_trigger_manifest_sha256": file_sha256(injection_trigger_manifest),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        **execution_provenance(),
    }
    atomic_write_json(output / f"{split}_injection_candidate_ranking_report.json", result)
    return result


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
    physical_delay_limit_seconds: float | None = None,
    empirical_timing_uncertainty_seconds: float | None = None,
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
    availability = {str(row["window_id"]): _available_ifos(row) for row in windows}
    for row in candidates:
        if str(row["split"]) != split:
            continue
        window_id = str(row["window_id"])
        if window_id not in window_ids:
            raise ValueError(f"Candidate references an unknown {split} window: {window_id}")
        ifo = str(row["ifo"])
        if ifo not in availability[window_id]:
            raise ValueError(
                f"Candidate {row.get('candidate_id')} uses unavailable detector {ifo}"
            )
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
        skipped_unavailable_pairs = 0
        for reference in sorted(windows, key=lambda row: float(row["gps_start"])):
            reference_key = int(round(float(reference["gps_start"]) * 1e9))
            shifted = by_start.get(reference_key + offset_key)
            if shifted is None:
                continue
            if (
                reference_ifo not in availability[str(reference["window_id"])]
                or shifted_ifo not in availability[str(shifted["window_id"])]
            ):
                skipped_unavailable_pairs += 1
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
                "skipped_unavailable_pairs": skipped_unavailable_pairs,
                "raw_coincidences": len(raw),
                "clustered_candidates": len(clustered),
                "live_time_seconds": exposure,
            }
        )
    exposure_seconds = sum(row["live_time_seconds"] for row in slide_exposure)
    physics_gate_configured = (
        physical_delay_limit_seconds is not None
        and empirical_timing_uncertainty_seconds is not None
    )
    if physical_delay_limit_seconds is not None and physical_delay_limit_seconds <= 0:
        raise ValueError("physical delay limit must be positive")
    if (
        empirical_timing_uncertainty_seconds is not None
        and empirical_timing_uncertainty_seconds < 0
    ):
        raise ValueError("empirical timing uncertainty must be non-negative")
    expected_coincidence_window = (
        float(physical_delay_limit_seconds)
        + 2.0 * float(empirical_timing_uncertainty_seconds)
        if physics_gate_configured
        else None
    )
    coincidence_matches_physics = bool(
        expected_coincidence_window is not None
        and np.isclose(
            coincidence_window_seconds,
            expected_coincidence_window,
            rtol=0.0,
            atol=1e-12,
        )
    )
    candidate_timing_calibrated = bool(candidates_by_window) and all(
        bool(row.get("timing_empirically_calibrated", False))
        for by_ifo in candidates_by_window.values()
        for rows in by_ifo.values()
        for row in rows
        if row["ifo"] in {reference_ifo, shifted_ifo}
    )
    relevant_candidates = [
        row
        for by_ifo in candidates_by_window.values()
        for rows in by_ifo.values()
        for row in rows
        if row["ifo"] in {reference_ifo, shifted_ifo}
    ]
    calibrated_uncertainties = [
        float(row["empirical_timing_uncertainty_seconds"])
        for row in relevant_candidates
        if row.get("timing_empirically_calibrated", False)
        and row.get("empirical_timing_uncertainty_seconds") is not None
    ]
    uncertainty_matches_candidates = (
        bool(relevant_candidates)
        and len(calibrated_uncertainties) == len(relevant_candidates)
        and empirical_timing_uncertainty_seconds is not None
        and all(
            np.isclose(
                value,
                empirical_timing_uncertainty_seconds,
                rtol=0.0,
                atol=1e-12,
            )
            for value in calibrated_uncertainties
        )
    )
    calibration_hashes = sorted(
        {
            str(row["timing_calibration_report_sha256"])
            for row in relevant_candidates
            if row.get("timing_calibration_report_sha256")
        }
    )
    checkpoint_hashes = sorted(
        {
            str(row["candidate_checkpoint_sha256"])
            for row in relevant_candidates
            if row.get("candidate_checkpoint_sha256")
        }
    )
    config_hashes = sorted(
        {
            str(row["candidate_config_sha256"])
            for row in relevant_candidates
            if row.get("candidate_config_sha256")
        }
    )
    code_commits = sorted(
        {
            str(row["candidate_code_commit"])
            for row in relevant_candidates
            if row.get("candidate_code_commit")
        }
    )
    timing_resolutions = [
        float(row.get("timing_resolution_seconds", row["bin_width_seconds"]))
        for by_ifo in candidates_by_window.values()
        for rows in by_ifo.values()
        for row in rows
        if row["ifo"] in {reference_ifo, shifted_ifo}
    ]
    timing_resolution_gate = bool(timing_resolutions) and max(timing_resolutions) <= 0.01
    publication_timing_gate = (
        physics_gate_configured
        and coincidence_matches_physics
        and candidate_timing_calibrated
        and uncertainty_matches_candidates
        and len(calibration_hashes) == 1
        and len(checkpoint_hashes) == 1
        and len(config_hashes) == 1
        and len(code_commits) == 1
        and timing_resolution_gate
    )
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
        "physical_delay_limit_seconds": physical_delay_limit_seconds,
        "empirical_timing_uncertainty_seconds": empirical_timing_uncertainty_seconds,
        "expected_coincidence_window_seconds": expected_coincidence_window,
        "coincidence_window_matches_physics": coincidence_matches_physics,
        "input_windows": len(windows),
        "input_gps_blocks": sorted({str(row["gps_block"]) for row in windows}),
        "input_zero_lag_live_time_seconds": _union_duration(
            (float(row["gps_start"]), float(row["gps_end"])) for row in windows
        ),
        "input_candidates": sum(
            len(rows) for by_ifo in candidates_by_window.values() for rows in by_ifo.values()
        ),
        "background_rows": len(output),
        "slide_exposure": slide_exposure,
        "equivalent_live_time_seconds": exposure_seconds,
        "equivalent_live_time_years": exposure_seconds / SECONDS_PER_YEAR,
        "maximum_bin_width_seconds": maximum_bin_width or None,
        "maximum_timing_resolution_seconds": max(timing_resolutions)
        if timing_resolutions
        else None,
        "candidate_timing_empirically_calibrated": candidate_timing_calibrated,
        "timing_uncertainty_matches_candidates": uncertainty_matches_candidates,
        "timing_calibration_report_sha256": (
            calibration_hashes[0] if len(calibration_hashes) == 1 else None
        ),
        "candidate_checkpoint_sha256": (
            checkpoint_hashes[0] if len(checkpoint_hashes) == 1 else None
        ),
        "candidate_config_sha256": config_hashes[0] if len(config_hashes) == 1 else None,
        "candidate_code_commit": code_commits[0] if len(code_commits) == 1 else None,
        "publication_timing_gate_passed": publication_timing_gate,
        "scientific_blocker": (
            "timing gate passed, but a publication claim still requires provenance-linked "
            "validation freeze, independent locked-test background/injections and adequate "
            "equivalent live time"
            if publication_timing_gate and exposure_seconds > 0
            else "requires detector-duty-cycle-correct exposure, a predeclared physical delay "
            "plus validation-calibrated timing allowance, <=10 ms candidate resolution, and "
            "adequate independent background live time"
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
    physical_delay_limit_seconds: float | None = None,
    empirical_timing_uncertainty_seconds: float | None = None,
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
        physical_delay_limit_seconds,
        empirical_timing_uncertainty_seconds,
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
        **execution_provenance(),
    }
    atomic_write_json(output / f"{split}_candidate_time_slide_report.json", result)
    return result
