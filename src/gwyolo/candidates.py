from __future__ import annotations

import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration, parse_gps_block_identity
from .exposure import (
    CANDIDATE_BLOCK_PERMUTATION_METHOD,
    CANDIDATE_BLOCK_SELECTION_DATA,
    DETECTOR_SET_BLOCK_PERMUTATION_METHOD,
    DETECTOR_SET_BLOCK_SELECTION_DATA,
    candidate_block_schedule_identity,
    candidate_slide_schedule_identity,
    detector_set_block_schedule_identity,
    normalize_candidate_slide_indices,
)
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .runtime import execution_provenance
from .trigger import network_ranking


def _available_ifos(row: dict[str, Any]) -> set[str]:
    """Return the explicit detector set for a background window.

    Background plans use ``ifos`` while scored trigger rows use ``valid_ifos``.  A
    time-slide exposure is meaningful only when the detector contributing that
    side of the coincidence was actually observing.
    """

    values = row.get(
        "valid_ifos",
        row.get("ifos", row.get("available_ifos")),
    )
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
        "calibration_perturbation": report.get("calibration_perturbation"),
        "physical_time_domain_perturbation": bool(
            report.get("physical_time_domain_perturbation", False)
        ),
        "fresh_time_frequency_transform": bool(
            report.get("fresh_time_frequency_transform", False)
        ),
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
        "chirp_threshold": float(report["chirp_threshold"]),
        "minimum_bins": int(report["minimum_bins"]),
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


def select_candidate_proposal_threshold(
    audit_reports: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    """Select a validation proposal threshold only when coverage and support gates pass."""

    if not audit_reports:
        raise ValueError("proposal threshold selection requires audit reports")
    required_groups = tuple(str(value) for value in settings["required_groups"])
    if not required_groups or len(set(required_groups)) != len(required_groups):
        raise ValueError("proposal threshold required groups must be unique")
    common = None
    records = []
    thresholds = set()
    for report in audit_reports:
        if report.get("status") != "validation_only_all_instance_candidate_proposal_coverage":
            raise ValueError("proposal threshold input is not a validation coverage audit")
        provenance = report.get("candidate_extraction_provenance", {})
        scoring = provenance.get("scoring", {})
        identity = {
            "injection_manifest_sha256": str(report["injection_manifest_sha256"]),
            "padding_seconds": float(report["padding_seconds"]),
            "checkpoint_sha256": str(scoring.get("checkpoint_sha256")),
            "config_sha256": str(scoring.get("config_sha256")),
            "trigger_manifest_sha256": str(scoring.get("trigger_manifest_sha256")),
        }
        if not provenance.get("available") or any(
            value in {"", "None"} for value in identity.values()
        ):
            raise ValueError("proposal threshold audit lacks complete scoring provenance")
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("proposal threshold audits do not share one scoring identity")
        threshold = float(provenance["chirp_threshold"])
        if threshold in thresholds:
            raise ValueError(f"duplicate proposal threshold audit: {threshold}")
        thresholds.add(threshold)
        groups = report["groups"]
        missing = [key for key in required_groups if key not in groups]
        if missing:
            raise ValueError(f"proposal threshold audit lacks required groups: {missing}")
        all_group = groups["all"]
        coverage_checks = {
            key: float(groups[key]["padded_coverage_fraction"])
            >= float(settings["minimum_required_group_padded_coverage"])
            for key in required_groups
        }
        checks = {
            "all_padded_coverage": float(all_group["padded_coverage_fraction"])
            >= float(settings["minimum_all_padded_coverage"]),
            "required_group_padded_coverage": all(coverage_checks.values()),
            "median_union_fraction": float(
                all_group["proposal_union_fraction_of_analysis_quantiles"]["0.5"]
            )
            <= float(settings["maximum_median_union_fraction"]),
            "p90_union_fraction": float(
                all_group["proposal_union_fraction_of_analysis_quantiles"]["0.9"]
            )
            <= float(settings["maximum_p90_union_fraction"]),
            "median_containing_width": float(
                all_group["minimum_containing_proposal_width_seconds_quantiles"][
                    "0.5"
                ]
            )
            <= float(settings["maximum_median_containing_width_seconds"]),
        }
        records.append(
            {
                "chirp_threshold": threshold,
                "candidates": int(report["candidates"]),
                "audit_report_sha256": str(report["audit_report_sha256"]),
                "padded_coverage_fraction": float(
                    all_group["padded_coverage_fraction"]
                ),
                "median_union_fraction": float(
                    all_group["proposal_union_fraction_of_analysis_quantiles"]["0.5"]
                ),
                "p90_union_fraction": float(
                    all_group["proposal_union_fraction_of_analysis_quantiles"]["0.9"]
                ),
                "median_containing_width_seconds": float(
                    all_group[
                        "minimum_containing_proposal_width_seconds_quantiles"
                    ]["0.5"]
                ),
                "required_group_coverage_checks": coverage_checks,
                "checks": checks,
                "qualified": all(checks.values()),
            }
        )
    qualified = [record for record in records if record["qualified"]]
    selected = (
        min(
            qualified,
            key=lambda record: (
                record["median_union_fraction"],
                record["p90_union_fraction"],
                record["candidates"],
                -record["chirp_threshold"],
            ),
        )
        if qualified
        else None
    )
    return {
        "promotion_allowed": selected is not None,
        "selected": selected,
        "common_scoring_identity": common,
        "records": sorted(records, key=lambda record: record["chirp_threshold"]),
    }


def run_candidate_proposal_threshold_selection(
    config_path: str | Path,
    audit_report_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    settings = config["candidate_proposal_threshold_selection"]
    reports = []
    for path in audit_report_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        report["audit_report_sha256"] = file_sha256(path)
        provenance = report.get("candidate_extraction_provenance", {})
        if "chirp_threshold" not in provenance:
            extraction_path = provenance.get("candidate_extraction_report_path")
            if not extraction_path or not Path(extraction_path).is_file():
                raise ValueError("proposal audit lacks its candidate extraction report")
            with Path(extraction_path).open("r", encoding="utf-8") as handle:
                extraction = json.load(handle)
            provenance["chirp_threshold"] = float(extraction["chirp_threshold"])
            provenance["minimum_bins"] = int(extraction["minimum_bins"])
        reports.append(report)
    selection = select_candidate_proposal_threshold(reports, settings)
    report = {
        "status": "validation_only_candidate_proposal_threshold_selection",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "proposal threshold selection is not a search threshold and requires candidate-level "
            "timing, continuous background and frozen VT evaluation"
        ),
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "audit_reports": [str(path) for path in audit_report_paths],
        "audit_report_hashes": [file_sha256(path) for path in audit_report_paths],
        **selection,
        **execution_provenance(),
    }
    destination = Path(output_path)
    if destination.is_file():
        raise ValueError("candidate proposal threshold selection output already exists")
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
    scoring_compatibility_report: str | Path | None = None,
    calibration_perturbation_plan: str | Path | None = None,
    calibration_timing_compatibility_report: str | Path | None = None,
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
    calibration_commit = str(calibration_scoring.get("code_commit", ""))
    candidate_commit = str(candidate_scoring.get("code_commit", ""))
    cross_commit_compatibility = None
    calibration_timing_transfer = None
    if (
        calibration_scoring.get("available")
        and candidate_provenance.get("available")
        and calibration_commit != candidate_commit
    ):
        if scoring_compatibility_report is not None and any(
            value is not None
            for value in (
                calibration_perturbation_plan,
                calibration_timing_compatibility_report,
            )
        ):
            raise ValueError("choose generic or calibration timing compatibility, not both")
        if scoring_compatibility_report is not None:
            from .code_compatibility import validate_candidate_scoring_compatibility

            cross_commit_compatibility = validate_candidate_scoring_compatibility(
                scoring_compatibility_report,
                calibration_commit,
                candidate_commit,
            )
        elif (
            calibration_perturbation_plan is not None
            and calibration_timing_compatibility_report is not None
        ):
            from .code_compatibility import (
                validate_calibration_timing_transfer_compatibility,
            )

            plan_path = Path(calibration_perturbation_plan).resolve()
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            perturbation = candidate_scoring.get("calibration_perturbation")
            perturbation_role = (
                perturbation.get("role") if isinstance(perturbation, dict) else None
            )
            if (
                plan.get("status")
                != "frozen_validation_calibration_perturbation_plan"
                or plan.get("passed") is not True
                or plan.get("test_rows_read") != 0
                or not isinstance(perturbation, dict)
                or perturbation.get("plan_sha256") != file_sha256(plan_path)
                or perturbation.get("scenario_id") not in plan.get("scenario_ids", [])
                or perturbation_role not in {"background", "injection"}
                or perturbation.get("manifest_sha256")
                != plan.get("manifests", {})
                .get(str(perturbation_role), {})
                .get("sha256")
                or candidate_scoring.get("physical_time_domain_perturbation") is not True
                or candidate_scoring.get("fresh_time_frequency_transform") is not True
            ):
                raise ValueError("candidate scoring is not bound to the frozen calibration plan")
            calibration_timing_transfer = (
                validate_calibration_timing_transfer_compatibility(
                    calibration_timing_compatibility_report,
                    calibration_commit,
                    candidate_commit,
                )
            )
        else:
            raise ValueError(
                "cross-commit timing calibration requires generic compatibility or a "
                "frozen calibration plan plus timing-transfer compatibility"
            )
    provenance_matches = bool(
        calibration_scoring.get("available")
        and candidate_provenance.get("available")
        and all(
            str(calibration_scoring.get(field)) == str(candidate_scoring.get(field))
            for field in ("checkpoint_sha256", "config_sha256")
        )
        and (
            calibration_commit == candidate_commit
            or cross_commit_compatibility is not None
            or calibration_timing_transfer is not None
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
        "cross_commit_scoring_compatibility_report_sha256": (
            file_sha256(scoring_compatibility_report)
            if scoring_compatibility_report is not None
            else None
        ),
        "calibration_timing_transfer_compatibility_report_sha256": (
            file_sha256(calibration_timing_compatibility_report)
            if calibration_timing_compatibility_report is not None
            else None
        ),
        "calibration_timing_transfer_compatibility_report_path": (
            str(Path(calibration_timing_compatibility_report).resolve())
            if calibration_timing_compatibility_report is not None
            else None
        ),
        "calibration_perturbation_plan_sha256": (
            file_sha256(calibration_perturbation_plan)
            if calibration_perturbation_plan is not None
            else None
        ),
        "calibration_perturbation_plan_path": (
            str(Path(calibration_perturbation_plan).resolve())
            if calibration_perturbation_plan is not None
            else None
        ),
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


def build_detector_set_injection_candidate_rankings(
    injection_rows: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    split: str,
    detector_subsets: Iterable[Iterable[str]],
    pairwise_light_travel_time_seconds: dict[str, float],
    empirical_timing_uncertainty_seconds: float,
    truth_association_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce candidates over predeclared variable detector sets.

    Each selected network candidate must satisfy every pairwise light-travel
    limit plus the frozen empirical allowance. Missing detectors remain
    explicit: an injection is evaluated only over detector subsets contained
    in its ``valid_ifos`` and arrival-time metadata.
    """

    normalized_subsets: list[tuple[str, ...]] = []
    seen_subsets = set()
    for raw_subset in detector_subsets:
        subset = tuple(str(value) for value in raw_subset)
        if (
            len(subset) < 2
            or len(subset) != len(set(subset))
            or subset in seen_subsets
        ):
            raise ValueError("detector subsets must be unique sets of at least two IFOs")
        seen_subsets.add(subset)
        normalized_subsets.append(subset)
    if not normalized_subsets:
        raise ValueError("at least one detector subset is required")
    if (
        empirical_timing_uncertainty_seconds < 0
        or truth_association_window_seconds <= 0
    ):
        raise ValueError("detector-set injection timing settings are invalid")

    def pair_key(first: str, second: str) -> str:
        return "+".join(sorted((first, second)))

    required_pair_keys = {
        pair_key(first, second)
        for subset in normalized_subsets
        for first, second in combinations(subset, 2)
    }
    if set(pairwise_light_travel_time_seconds) != required_pair_keys or any(
        not np.isfinite(float(value)) or float(value) <= 0
        for value in pairwise_light_travel_time_seconds.values()
    ):
        raise ValueError(
            "pairwise light-travel limits must exactly cover the detector subsets"
        )

    parents = [row for row in injection_rows if str(row["split"]) == split]
    if not parents:
        raise ValueError(f"no scored injections for split {split}")
    by_id = {str(row["injection_id"]): row for row in parents}
    if len(by_id) != len(parents):
        raise ValueError("scored injection rows repeat injection IDs")
    candidate_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    calibration_hashes = set()
    checkpoint_hashes = set()
    config_hashes = set()
    code_commits = set()
    for row in candidates:
        if str(row["split"]) != split:
            continue
        injection_id = str(row["injection_id"])
        if injection_id not in by_id:
            raise ValueError(
                f"candidate references unknown {split} injection {injection_id}"
            )
        candidate_map.setdefault(injection_id, {}).setdefault(
            str(row["ifo"]), []
        ).append(row)
        if row.get("timing_calibration_report_sha256"):
            calibration_hashes.add(
                str(row["timing_calibration_report_sha256"])
            )
        if row.get("candidate_checkpoint_sha256"):
            checkpoint_hashes.add(str(row["candidate_checkpoint_sha256"]))
        if row.get("candidate_config_sha256"):
            config_hashes.add(str(row["candidate_config_sha256"]))
        if row.get("candidate_code_commit"):
            code_commits.add(str(row["candidate_code_commit"]))

    outputs = []
    eligibility_counts: Counter[str] = Counter()
    recovery_counts: Counter[str] = Counter()
    for injection_id, parent in sorted(by_id.items()):
        valid_ifos = {str(value) for value in parent["valid_ifos"]}
        arrivals = {
            str(ifo): float(value)
            for ifo, value in parent.get("detector_arrival_gps", {}).items()
        }
        eligible_subsets = [
            subset
            for subset in normalized_subsets
            if set(subset) <= valid_ifos and set(subset) <= set(arrivals)
        ]
        if not eligible_subsets:
            continue
        for subset in eligible_subsets:
            eligibility_counts["+".join(subset)] += 1

        networks = []
        by_ifo = candidate_map.get(injection_id, {})
        for subset in eligible_subsets:
            subset_name = "+".join(subset)
            for candidate_tuple in product(*(by_ifo.get(ifo, []) for ifo in subset)):
                if len(candidate_tuple) != len(subset):
                    continue
                if not all(
                    bool(row.get("timing_empirically_calibrated"))
                    for row in candidate_tuple
                ):
                    continue
                if not all(
                    np.isclose(
                        float(
                            row.get(
                                "empirical_timing_uncertainty_seconds", -1
                            )
                        ),
                        empirical_timing_uncertainty_seconds,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for row in candidate_tuple
                ):
                    continue
                if any(
                    abs(float(row["gps_peak"]) - arrivals[str(row["ifo"])])
                    > truth_association_window_seconds
                    for row in candidate_tuple
                ):
                    continue
                row_by_ifo = {
                    str(row["ifo"]): row for row in candidate_tuple
                }
                pairwise_separations = {}
                coherent = True
                for first, second in combinations(subset, 2):
                    key = pair_key(first, second)
                    separation = abs(
                        float(row_by_ifo[first]["gps_peak"])
                        - float(row_by_ifo[second]["gps_peak"])
                    )
                    limit = (
                        float(pairwise_light_travel_time_seconds[key])
                        + 2.0 * empirical_timing_uncertainty_seconds
                    )
                    pairwise_separations[key] = separation
                    if separation > limit:
                        coherent = False
                        break
                if not coherent:
                    continue
                provenance_fields = (
                    "timing_calibration_report_sha256",
                    "candidate_checkpoint_sha256",
                    "candidate_config_sha256",
                    "candidate_code_commit",
                )
                provenance = {
                    field: {str(row.get(field, "")) for row in candidate_tuple}
                    for field in provenance_fields
                }
                if any(
                    len(values) != 1 or not next(iter(values))
                    for values in provenance.values()
                ):
                    continue
                calibration_hashes.update(
                    provenance["timing_calibration_report_sha256"]
                )
                checkpoint_hashes.update(
                    provenance["candidate_checkpoint_sha256"]
                )
                config_hashes.update(provenance["candidate_config_sha256"])
                code_commits.update(provenance["candidate_code_commit"])
                chirp_scores = {
                    ifo: float(row_by_ifo[ifo]["chirp_score"]) for ifo in subset
                }
                glitch_scores = {
                    ifo: float(row_by_ifo[ifo]["glitch_score_at_peak"])
                    for ifo in subset
                }
                networks.append(
                    {
                        "detector_subset": subset_name,
                        "source_candidate_ids": {
                            ifo: row_by_ifo[ifo]["candidate_id"] for ifo in subset
                        },
                        "pairwise_peak_separation_seconds": pairwise_separations,
                        "chirp_scores": chirp_scores,
                        "glitch_scores": glitch_scores,
                        **network_ranking(
                            chirp_scores,
                            glitch_scores,
                            list(subset),
                        ),
                    }
                )
        selected = (
            max(
                networks,
                key=lambda row: (
                    float(row["ranking_score"]),
                    len(row["valid_ifos"]),
                    float(row["chirp_glitch_margin"]),
                ),
            )
            if networks
            else None
        )
        if selected is not None:
            recovery_counts[str(selected["detector_subset"])] += 1
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
                "eligible_detector_subsets": [
                    "+".join(subset) for subset in eligible_subsets
                ],
                "selected_detector_subset": (
                    selected["detector_subset"] if selected else None
                ),
                "candidate_network_found": selected is not None,
                "candidate_pair_found": selected is not None,
                "ranking_score": (
                    float(selected["ranking_score"]) if selected else 0.0
                ),
                "network_candidate": selected,
            }
        )

    report = {
        "status": "physical_variable_detector_set_injection_candidate_rankings",
        "split": split,
        "input_injections": len(parents),
        "eligible_detector_set_injections": len(outputs),
        "ranked_injections": len(outputs),
        "candidate_network_found": sum(
            row["candidate_network_found"] for row in outputs
        ),
        "excluded_missing_detector_or_arrival": len(parents) - len(outputs),
        "required_detector_subsets": [
            "+".join(subset) for subset in normalized_subsets
        ],
        "eligible_injections_by_detector_subset": dict(
            sorted(eligibility_counts.items())
        ),
        "selected_networks_by_detector_subset": dict(
            sorted(recovery_counts.items())
        ),
        "pairwise_light_travel_time_seconds": dict(
            sorted(
                (key, float(value))
                for key, value in pairwise_light_travel_time_seconds.items()
            )
        ),
        "empirical_timing_uncertainty_seconds": (
            empirical_timing_uncertainty_seconds
        ),
        "pairwise_allowed_peak_separation_seconds": {
            key: float(value) + 2.0 * empirical_timing_uncertainty_seconds
            for key, value in sorted(pairwise_light_travel_time_seconds.items())
        },
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
        "candidate_code_commit": (
            next(iter(code_commits)) if len(code_commits) == 1 else None
        ),
        "candidate_scoring_provenance_consistent": (
            len(checkpoint_hashes)
            == len(config_hashes)
            == len(code_commits)
            == 1
        ),
    }
    return outputs, report


def run_detector_set_injection_candidate_rankings(
    injection_trigger_manifest: str | Path,
    candidate_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    split: str,
    empirical_timing_uncertainty_seconds: float,
) -> dict[str, Any]:
    """Run the frozen H1/L1/V1 detector-set ranking policy."""

    config = load_yaml(config_path)
    settings = config.get("network_coherence")
    if (
        not isinstance(settings, dict)
        or settings.get("schema")
        != "h1_l1_v1_pairwise_light_travel_v1"
    ):
        raise ValueError("network-coherence configuration is invalid")
    maximum_uncertainty = float(
        settings["maximum_empirical_timing_uncertainty_seconds"]
    )
    if (
        not np.isfinite(empirical_timing_uncertainty_seconds)
        or empirical_timing_uncertainty_seconds < 0
        or empirical_timing_uncertainty_seconds > maximum_uncertainty
    ):
        raise ValueError(
            "empirical timing uncertainty exceeds the frozen network policy"
        )
    with Path(injection_trigger_manifest).open("r", encoding="utf-8") as handle:
        injections = [json.loads(line) for line in handle if line.strip()]
    with Path(candidate_manifest).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    rows, report = build_detector_set_injection_candidate_rankings(
        injections,
        candidates,
        split,
        settings["detector_subsets"],
        settings["pairwise_light_travel_time_seconds"],
        empirical_timing_uncertainty_seconds,
        float(settings["truth_association_window_seconds"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = (
        output
        / f"{split}_variable_detector_set_injection_candidate_rankings.jsonl"
    )
    atomic_write_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    result = {
        **report,
        "injection_trigger_manifest_path": str(
            Path(injection_trigger_manifest).resolve()
        ),
        "injection_trigger_manifest_sha256": file_sha256(
            injection_trigger_manifest
        ),
        "candidate_manifest_path": str(Path(candidate_manifest).resolve()),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_hash(config, 64),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        **execution_provenance(),
    }
    atomic_write_json(
        output
        / f"{split}_variable_detector_set_injection_candidate_ranking_report.json",
        result,
    )
    return result


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


def _normalize_detector_set_policy(
    detector_subsets: Iterable[Iterable[str]],
    pairwise_light_travel_time_seconds: dict[str, float],
) -> tuple[list[tuple[str, ...]], dict[str, float]]:
    normalized_subsets: list[tuple[str, ...]] = []
    seen_subsets: set[frozenset[str]] = set()
    for raw_subset in detector_subsets:
        subset = tuple(str(value) for value in raw_subset)
        identity = frozenset(subset)
        if (
            len(subset) < 2
            or len(subset) != len(identity)
            or identity in seen_subsets
        ):
            raise ValueError("detector subsets must be unique sets of at least two IFOs")
        seen_subsets.add(identity)
        normalized_subsets.append(subset)
    if not normalized_subsets:
        raise ValueError("at least one detector subset is required")

    def pair_key(first: str, second: str) -> str:
        return "+".join(sorted((first, second)))

    required_pairs = {
        pair_key(first, second)
        for subset in normalized_subsets
        for first, second in combinations(subset, 2)
    }
    limits = {
        str(key): float(value)
        for key, value in pairwise_light_travel_time_seconds.items()
    }
    if set(limits) != required_pairs or any(
        not np.isfinite(value) or value <= 0 for value in limits.values()
    ):
        raise ValueError(
            "pairwise light-travel limits must exactly cover the detector subsets"
        )
    return normalized_subsets, limits


def _cluster_detector_set_network_rows(
    rows: list[dict[str, Any]], cluster_window_seconds: float
) -> list[dict[str, Any]]:
    """Cluster one slide jointly across all detector-subset channels."""

    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (float(row["gps_peak"]), str(row["candidate_id"])),
    )
    groups = [[ordered[0]]]
    for row in ordered[1:]:
        if (
            float(row["gps_peak"]) - float(groups[-1][-1]["gps_peak"])
            <= cluster_window_seconds
        ):
            groups[-1].append(row)
        else:
            groups.append([row])
    return [
        max(
            group,
            key=lambda row: (
                float(row["ranking_score"]),
                len(row["valid_ifos"]),
                float(row["chirp_glitch_margin"]),
                str(row["candidate_id"]),
            ),
        )
        for group in groups
    ]


def build_detector_set_candidate_time_slides(
    candidates: Iterable[dict[str, Any]],
    background_windows: Iterable[dict[str, Any]],
    split: str,
    detector_subsets: Iterable[Iterable[str]],
    pairwise_light_travel_time_seconds: dict[str, float],
    empirical_timing_uncertainty_seconds: float,
    slide_offsets_seconds: Iterable[dict[str, Any]],
    cluster_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a duty-cycle-aware background over variable detector sets.

    Every slide assigns an independent source-window offset to each detector.
    Coincidences are formed for all eligible predeclared subsets, checked
    against every pairwise physical delay plus the frozen timing allowance,
    and then clustered jointly so one event cannot be counted once per subset.
    """

    subsets, physical_limits = _normalize_detector_set_policy(
        detector_subsets,
        pairwise_light_travel_time_seconds,
    )
    detectors = tuple(
        dict.fromkeys(ifo for subset in subsets for ifo in subset)
    )
    if (
        not np.isfinite(empirical_timing_uncertainty_seconds)
        or empirical_timing_uncertainty_seconds < 0
        or not np.isfinite(cluster_window_seconds)
        or cluster_window_seconds <= 0
    ):
        raise ValueError("detector-set time-slide timing settings are invalid")
    windows = [
        dict(row) for row in background_windows if str(row["split"]) == split
    ]
    if not windows:
        raise ValueError(f"No background windows for split {split}")
    durations = {
        float(row["gps_end"]) - float(row["gps_start"]) for row in windows
    }
    if (
        len(durations) != 1
        or not np.isfinite(next(iter(durations)))
        or next(iter(durations)) <= 0
    ):
        raise ValueError("Detector-set slides require one positive window duration")
    duration = next(iter(durations))
    by_start = {
        int(round(float(row["gps_start"]) * 1e9)): row for row in windows
    }
    if len(by_start) != len(windows):
        raise ValueError("Background windows have duplicate GPS starts")
    availability = {
        str(row["window_id"]): _available_ifos(row) for row in windows
    }

    schedules: list[dict[str, Any]] = []
    for ordinal, raw_schedule in enumerate(slide_offsets_seconds, start=1):
        raw_offsets = raw_schedule.get("offset_seconds", raw_schedule)
        offsets = {str(key): float(value) for key, value in raw_offsets.items()}
        if (
            set(offsets) != set(detectors)
            or any(not np.isfinite(value) for value in offsets.values())
            or 0.0 not in offsets.values()
            or len(set(offsets.values())) != len(offsets)
            or any(
                abs(first - second) < duration
                for first, second in combinations(offsets.values(), 2)
            )
        ):
            raise ValueError(
                "each slide must independently offset every detector by at least "
                "one window duration and retain one zero-offset reference"
            )
        slide_index = int(raw_schedule.get("slide_index", ordinal))
        slide_id = str(
            raw_schedule.get(
                "slide_id",
                "network-slide-"
                + canonical_hash(
                    {"split": split, "offset_seconds": offsets},
                    24,
                ),
            )
        )
        if slide_index < 1 or not slide_id:
            raise ValueError("detector-set time-slide identity is invalid")
        schedules.append(
            {
                "slide_number": ordinal,
                "slide_index": slide_index,
                "slide_id": slide_id,
                "offset_seconds": offsets,
            }
        )
    if not schedules:
        raise ValueError("at least one detector-set time slide is required")
    if (
        len({row["slide_id"] for row in schedules}) != len(schedules)
        or len({row["slide_index"] for row in schedules}) != len(schedules)
    ):
        raise ValueError("detector-set time-slide offsets repeat")

    candidates_by_window: dict[str, dict[str, list[dict[str, Any]]]] = {}
    relevant_candidates = []
    maximum_bin_width = 0.0
    window_ids = set(availability)
    for row in candidates:
        if str(row["split"]) != split:
            continue
        window_id = str(row["window_id"])
        if window_id not in window_ids:
            raise ValueError(
                f"Candidate references an unknown {split} window: {window_id}"
            )
        ifo = str(row["ifo"])
        if ifo not in detectors or ifo not in availability[window_id]:
            raise ValueError(
                f"Candidate {row.get('candidate_id')} uses unavailable detector {ifo}"
            )
        candidates_by_window.setdefault(window_id, {}).setdefault(ifo, []).append(
            row
        )
        relevant_candidates.append(row)
        maximum_bin_width = max(
            maximum_bin_width,
            float(row["bin_width_seconds"]),
        )

    def pair_key(first: str, second: str) -> str:
        return "+".join(sorted((first, second)))

    output = []
    slide_exposure = []
    eligible_subset_windows_total: Counter[str] = Counter()
    raw_subset_total: Counter[str] = Counter()
    selected_subset_total: Counter[str] = Counter()
    for schedule in schedules:
        offsets = schedule["offset_seconds"]
        offset_keys = {
            ifo: int(round(float(offset) * 1e9))
            for ifo, offset in offsets.items()
        }
        raw_networks = []
        exposure_intervals = []
        eligible_counts: Counter[str] = Counter()
        raw_counts: Counter[str] = Counter()
        skipped_no_subset = 0
        for base in sorted(windows, key=lambda row: float(row["gps_start"])):
            base_key = int(round(float(base["gps_start"]) * 1e9))
            source_windows = {
                ifo: by_start.get(base_key + offset_keys[ifo])
                for ifo in detectors
            }
            eligible_subsets = []
            for subset in subsets:
                if all(
                    source_windows[ifo] is not None
                    and ifo
                    in availability[str(source_windows[ifo]["window_id"])]
                    for ifo in subset
                ):
                    eligible_subsets.append(subset)
                    eligible_counts["+".join(subset)] += 1
            if not eligible_subsets:
                skipped_no_subset += 1
                continue
            exposure_intervals.append(
                (float(base["gps_start"]), float(base["gps_end"]))
            )
            for subset in eligible_subsets:
                subset_name = "+".join(subset)
                candidate_lists = [
                    candidates_by_window.get(
                        str(source_windows[ifo]["window_id"]),
                        {},
                    ).get(ifo, [])
                    for ifo in subset
                ]
                for candidate_tuple in product(*candidate_lists):
                    shifted_times = {
                        ifo: float(row["gps_peak"]) - float(offsets[ifo])
                        for ifo, row in zip(subset, candidate_tuple)
                    }
                    separations = {}
                    coherent = True
                    for first, second in combinations(subset, 2):
                        key = pair_key(first, second)
                        separation = abs(
                            shifted_times[first] - shifted_times[second]
                        )
                        separations[key] = separation
                        if separation > (
                            physical_limits[key]
                            + 2.0 * empirical_timing_uncertainty_seconds
                        ):
                            coherent = False
                            break
                    if not coherent:
                        continue
                    provenance_fields = (
                        "timing_calibration_report_sha256",
                        "candidate_checkpoint_sha256",
                        "candidate_config_sha256",
                        "candidate_code_commit",
                    )
                    provenance = {
                        field: {str(row.get(field, "")) for row in candidate_tuple}
                        for field in provenance_fields
                    }
                    if (
                        not all(
                            bool(row.get("timing_empirically_calibrated"))
                            and np.isclose(
                                float(
                                    row.get(
                                        "empirical_timing_uncertainty_seconds",
                                        -1,
                                    )
                                ),
                                empirical_timing_uncertainty_seconds,
                                rtol=0.0,
                                atol=1e-12,
                            )
                            for row in candidate_tuple
                        )
                        or any(
                            len(values) != 1 or not next(iter(values))
                            for values in provenance.values()
                        )
                    ):
                        continue
                    row_by_ifo = {
                        ifo: row
                        for ifo, row in zip(
                            subset,
                            candidate_tuple,
                        )
                    }
                    chirp_scores = {
                        ifo: float(row_by_ifo[ifo]["chirp_score"])
                        for ifo in subset
                    }
                    glitch_scores = {
                        ifo: float(row_by_ifo[ifo]["glitch_score_at_peak"])
                        for ifo in subset
                    }
                    source_ids = {
                        ifo: row_by_ifo[ifo]["candidate_id"] for ifo in subset
                    }
                    identity = {
                        "slide_id": schedule["slide_id"],
                        "detector_subset": subset_name,
                        "source_candidate_ids": source_ids,
                    }
                    raw_networks.append(
                        {
                            "candidate_id": (
                                "network-slide-candidate-"
                                + canonical_hash(identity, 24)
                            ),
                            "slide_id": schedule["slide_id"],
                            "slide_number": schedule["slide_number"],
                            "slide_index": schedule["slide_index"],
                            "split": split,
                            "detector_subset": subset_name,
                            "gps_peak": float(np.mean(list(shifted_times.values()))),
                            "pairwise_peak_separation_seconds": separations,
                            "offset_seconds": dict(offsets),
                            "source_candidate_ids": source_ids,
                            "source_window_ids": {
                                ifo: source_windows[ifo]["window_id"]
                                for ifo in subset
                            },
                            "source_gps_blocks": {
                                ifo: source_windows[ifo]["gps_block"]
                                for ifo in subset
                            },
                            "chirp_scores": chirp_scores,
                            "glitch_scores": glitch_scores,
                            **network_ranking(
                                chirp_scores,
                                glitch_scores,
                                list(subset),
                            ),
                        }
                    )
                    raw_counts[subset_name] += 1
        clustered = _cluster_detector_set_network_rows(
            raw_networks,
            cluster_window_seconds,
        )
        selected_counts = Counter(
            str(row["detector_subset"]) for row in clustered
        )
        output.extend(clustered)
        eligible_subset_windows_total.update(eligible_counts)
        raw_subset_total.update(raw_counts)
        selected_subset_total.update(selected_counts)
        slide_exposure.append(
            {
                **schedule,
                "eligible_windows_by_detector_subset": dict(
                    sorted(eligible_counts.items())
                ),
                "skipped_windows_without_eligible_subset": skipped_no_subset,
                "raw_coincidences": len(raw_networks),
                "raw_coincidences_by_detector_subset": dict(
                    sorted(raw_counts.items())
                ),
                "clustered_candidates": len(clustered),
                "clustered_candidates_by_detector_subset": dict(
                    sorted(selected_counts.items())
                ),
                "live_time_seconds": _union_duration(exposure_intervals),
            }
        )

    timing_resolutions = [
        float(row.get("timing_resolution_seconds", row["bin_width_seconds"]))
        for row in relevant_candidates
    ]
    calibrated = bool(relevant_candidates) and all(
        bool(row.get("timing_empirically_calibrated"))
        and np.isclose(
            float(row.get("empirical_timing_uncertainty_seconds", -1)),
            empirical_timing_uncertainty_seconds,
            rtol=0.0,
            atol=1e-12,
        )
        for row in relevant_candidates
    )
    provenance_values = {
        field: sorted(
            {
                str(row[field])
                for row in relevant_candidates
                if row.get(field)
            }
        )
        for field in (
            "timing_calibration_report_sha256",
            "candidate_checkpoint_sha256",
            "candidate_config_sha256",
            "candidate_code_commit",
        )
    }
    equivalent_live_time_seconds = sum(
        float(row["live_time_seconds"]) for row in slide_exposure
    )
    publication_timing_gate = (
        calibrated
        and bool(timing_resolutions)
        and max(timing_resolutions) <= 0.01
        and all(len(values) == 1 for values in provenance_values.values())
    )
    report = {
        "status": "variable_detector_set_time_slide_background",
        "scientific_claim_allowed": False,
        "split": split,
        "required_detector_subsets": [
            "+".join(subset) for subset in subsets
        ],
        "detectors": list(detectors),
        "slide_count": len(schedules),
        "slide_schedule": schedules,
        "slide_schedule_sha256": canonical_hash(schedules, 64),
        "window_duration_seconds": duration,
        "cluster_window_seconds": cluster_window_seconds,
        "pairwise_light_travel_time_seconds": dict(
            sorted(physical_limits.items())
        ),
        "empirical_timing_uncertainty_seconds": (
            empirical_timing_uncertainty_seconds
        ),
        "pairwise_allowed_peak_separation_seconds": {
            key: value + 2.0 * empirical_timing_uncertainty_seconds
            for key, value in sorted(physical_limits.items())
        },
        "input_windows": len(windows),
        "input_gps_blocks": sorted(
            {str(row["gps_block"]) for row in windows}
        ),
        "input_zero_lag_live_time_seconds": _union_duration(
            (float(row["gps_start"]), float(row["gps_end"]))
            for row in windows
        ),
        "input_candidates": len(relevant_candidates),
        "background_rows": len(output),
        "eligible_windows_by_detector_subset": dict(
            sorted(eligible_subset_windows_total.items())
        ),
        "raw_coincidences_by_detector_subset": dict(
            sorted(raw_subset_total.items())
        ),
        "clustered_candidates_by_detector_subset": dict(
            sorted(selected_subset_total.items())
        ),
        "slide_exposure": slide_exposure,
        "equivalent_live_time_seconds": equivalent_live_time_seconds,
        "equivalent_live_time_years": (
            equivalent_live_time_seconds / SECONDS_PER_YEAR
        ),
        "maximum_bin_width_seconds": maximum_bin_width or None,
        "maximum_timing_resolution_seconds": (
            max(timing_resolutions) if timing_resolutions else None
        ),
        "candidate_timing_empirically_calibrated": calibrated,
        "timing_calibration_report_sha256": (
            provenance_values["timing_calibration_report_sha256"][0]
            if len(provenance_values["timing_calibration_report_sha256"]) == 1
            else None
        ),
        "candidate_checkpoint_sha256": (
            provenance_values["candidate_checkpoint_sha256"][0]
            if len(provenance_values["candidate_checkpoint_sha256"]) == 1
            else None
        ),
        "candidate_config_sha256": (
            provenance_values["candidate_config_sha256"][0]
            if len(provenance_values["candidate_config_sha256"]) == 1
            else None
        ),
        "candidate_code_commit": (
            provenance_values["candidate_code_commit"][0]
            if len(provenance_values["candidate_code_commit"]) == 1
            else None
        ),
        "publication_timing_gate_passed": publication_timing_gate,
        "detector_duty_cycle_accounted": True,
        "detector_subset_channels_clustered_jointly": True,
        "live_time_counted_once_per_slide": True,
        "independent_pairwise_offsets": True,
        "scientific_blocker": (
            "timing gate passed, but publication use still requires a score-blind "
            "frozen schedule, adequate locked live time, and a validation-frozen "
            "threshold"
            if publication_timing_gate and equivalent_live_time_seconds > 0
            else "requires empirically calibrated <=10 ms candidates with common "
            "provenance and nonzero detector-duty-cycle-correct exposure"
        ),
    }
    return output, report


def build_detector_set_candidate_block_permutations(
    candidates: Iterable[dict[str, Any]],
    background_windows: Iterable[dict[str, Any]],
    schedule: dict[str, Any],
    empirical_timing_uncertainty_seconds: float,
    cluster_window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute frozen independent H1/L1/V1 circular block permutations."""

    if (
        schedule.get("status")
        != "frozen_detector_set_block_permutation_schedule"
        or schedule.get("method") != DETECTOR_SET_BLOCK_PERMUTATION_METHOD
        or schedule.get("selection_data") != DETECTOR_SET_BLOCK_SELECTION_DATA
        or schedule.get("candidate_scores_inspected") is not False
        or canonical_hash(
            detector_set_block_schedule_identity(schedule),
            32,
        )
        != schedule.get("schedule_id")
        or canonical_hash(schedule.get("permutations", []), 64)
        != schedule.get("permutations_sha256")
    ):
        raise ValueError("detector-set block schedule identity failed replay")
    split = str(schedule["split"])
    subsets, physical_limits = _normalize_detector_set_policy(
        schedule["detector_subsets"],
        schedule["pairwise_light_travel_time_seconds"],
    )
    detectors = tuple(str(value) for value in schedule["detectors"])
    if (
        set(detectors)
        != {ifo for subset in subsets for ifo in subset}
        or len(detectors) != len(set(detectors))
        or not np.isfinite(empirical_timing_uncertainty_seconds)
        or empirical_timing_uncertainty_seconds < 0
        or not np.isfinite(cluster_window_seconds)
        or cluster_window_seconds <= 0
    ):
        raise ValueError("detector-set block execution settings are invalid")
    windows = [
        dict(row)
        for row in background_windows
        if str(row.get("split")) == split
    ]
    if not windows:
        raise ValueError(f"No background windows for split {split}")
    duration = float(schedule["window_duration_seconds"])
    if any(
        not np.isclose(
            float(row["gps_end"]) - float(row["gps_start"]),
            duration,
            rtol=0.0,
            atol=1e-9,
        )
        for row in windows
    ):
        raise ValueError("background window duration differs from block schedule")
    ordered_blocks = [str(value) for value in schedule["ordered_gps_blocks"]]
    block_positions = {
        block: index for index, block in enumerate(ordered_blocks)
    }
    if len(block_positions) != len(ordered_blocks) or len(ordered_blocks) < 3:
        raise ValueError("detector-set block inventory is invalid")
    block_metadata: dict[str, dict[str, Any]] = {}
    windows_by_block_slot: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}
    availability: dict[str, set[str]] = {}
    all_slots: set[int] = set()
    for row in windows:
        block = str(row["gps_block"])
        if block not in block_positions:
            raise ValueError("background window uses an unscheduled GPS block")
        _, block_start, block_duration = parse_gps_block_identity(block)
        slot_value = (float(row["gps_start"]) - block_start) / duration
        slot = int(round(slot_value))
        if (
            not np.isclose(slot_value, slot, rtol=0.0, atol=1e-6)
            or slot < 0
            or (slot + 1) * duration > block_duration + 1e-9
            or (block, slot) in windows_by_block_slot
        ):
            raise ValueError("detector-set background block/slot is invalid")
        block_metadata.setdefault(
            block,
            {"gps_start": block_start, "duration": block_duration},
        )
        windows_by_block_slot[(block, slot)] = row
        availability[str(row["window_id"])] = _available_ifos(row)
        all_slots.add(slot)
    if set(block_metadata) != set(ordered_blocks):
        raise ValueError("detector-set block schedule contains an empty GPS block")

    candidates_by_window: dict[str, dict[str, list[dict[str, Any]]]] = {}
    relevant_candidates = []
    maximum_bin_width = 0.0
    for row in candidates:
        if str(row.get("split")) != split:
            continue
        window_id = str(row["window_id"])
        if window_id not in availability:
            raise ValueError(
                f"Candidate references an unknown {split} window: {window_id}"
            )
        ifo = str(row["ifo"])
        if ifo not in detectors or ifo not in availability[window_id]:
            raise ValueError(
                f"Candidate {row.get('candidate_id')} uses unavailable detector {ifo}"
            )
        candidates_by_window.setdefault(window_id, {}).setdefault(
            ifo,
            [],
        ).append(row)
        relevant_candidates.append(row)
        maximum_bin_width = max(
            maximum_bin_width,
            float(row["bin_width_seconds"]),
        )

    def pair_key(first: str, second: str) -> str:
        return "+".join(sorted((first, second)))

    output = []
    permutation_exposure = []
    eligible_subset_windows_total: Counter[str] = Counter()
    raw_subset_total: Counter[str] = Counter()
    selected_subset_total: Counter[str] = Counter()
    block_count = len(ordered_blocks)
    for permutation in schedule["permutations"]:
        shift_by_ifo = {
            str(key): int(value)
            for key, value in permutation["shift_by_ifo"].items()
        }
        if (
            set(shift_by_ifo) != set(detectors)
            or len(
                {
                    value % block_count
                    for value in shift_by_ifo.values()
                }
            )
            != len(detectors)
        ):
            raise ValueError("detector-set block shifts are not independent")
        raw_networks = []
        exposure_intervals = []
        eligible_counts: Counter[str] = Counter()
        raw_counts: Counter[str] = Counter()
        eligible_blocks = 0
        for base_index, base_block in enumerate(ordered_blocks):
            block_has_exposure = False
            for slot in sorted(all_slots):
                source_windows = {
                    ifo: windows_by_block_slot.get(
                        (
                            ordered_blocks[
                                (base_index + shift_by_ifo[ifo])
                                % block_count
                            ],
                            slot,
                        )
                    )
                    for ifo in detectors
                }
                eligible_subsets = [
                    subset
                    for subset in subsets
                    if all(
                        source_windows[ifo] is not None
                        and ifo
                        in availability[
                            str(source_windows[ifo]["window_id"])
                        ]
                        for ifo in subset
                    )
                ]
                if not eligible_subsets:
                    continue
                block_has_exposure = True
                base_time = (
                    float(block_metadata[base_block]["gps_start"])
                    + slot * duration
                )
                exposure_intervals.append(
                    (base_time, base_time + duration)
                )
                for subset in eligible_subsets:
                    subset_name = "+".join(subset)
                    eligible_counts[subset_name] += 1
                    candidate_lists = [
                        candidates_by_window.get(
                            str(source_windows[ifo]["window_id"]),
                            {},
                        ).get(ifo, [])
                        for ifo in subset
                    ]
                    for candidate_tuple in product(*candidate_lists):
                        shifted_times = {
                            ifo: (
                                float(row["gps_peak"])
                                - float(source_windows[ifo]["gps_start"])
                                + base_time
                            )
                            for ifo, row in zip(subset, candidate_tuple)
                        }
                        separations = {}
                        coherent = True
                        for first, second in combinations(subset, 2):
                            key = pair_key(first, second)
                            separation = abs(
                                shifted_times[first]
                                - shifted_times[second]
                            )
                            separations[key] = separation
                            if separation > (
                                physical_limits[key]
                                + 2.0
                                * empirical_timing_uncertainty_seconds
                            ):
                                coherent = False
                                break
                        if not coherent:
                            continue
                        provenance_fields = (
                            "timing_calibration_report_sha256",
                            "candidate_checkpoint_sha256",
                            "candidate_config_sha256",
                            "candidate_code_commit",
                        )
                        provenance = {
                            field: {
                                str(row.get(field, ""))
                                for row in candidate_tuple
                            }
                            for field in provenance_fields
                        }
                        if (
                            not all(
                                bool(
                                    row.get(
                                        "timing_empirically_calibrated"
                                    )
                                )
                                and np.isclose(
                                    float(
                                        row.get(
                                            "empirical_timing_uncertainty_seconds",
                                            -1,
                                        )
                                    ),
                                    empirical_timing_uncertainty_seconds,
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                                for row in candidate_tuple
                            )
                            or any(
                                len(values) != 1
                                or not next(iter(values))
                                for values in provenance.values()
                            )
                        ):
                            continue
                        row_by_ifo = {
                            ifo: row
                            for ifo, row in zip(subset, candidate_tuple)
                        }
                        chirp_scores = {
                            ifo: float(
                                row_by_ifo[ifo]["chirp_score"]
                            )
                            for ifo in subset
                        }
                        glitch_scores = {
                            ifo: float(
                                row_by_ifo[ifo][
                                    "glitch_score_at_peak"
                                ]
                            )
                            for ifo in subset
                        }
                        source_ids = {
                            ifo: row_by_ifo[ifo]["candidate_id"]
                            for ifo in subset
                        }
                        identity = {
                            "permutation_id": permutation[
                                "permutation_id"
                            ],
                            "base_gps_block": base_block,
                            "base_slot": slot,
                            "detector_subset": subset_name,
                            "source_candidate_ids": source_ids,
                        }
                        raw_networks.append(
                            {
                                "candidate_id": (
                                    "network-block-candidate-"
                                    + canonical_hash(identity, 24)
                                ),
                                "slide_id": permutation[
                                    "permutation_id"
                                ],
                                "slide_index": int(
                                    permutation["permutation_index"]
                                ),
                                "permutation_id": permutation[
                                    "permutation_id"
                                ],
                                "permutation_index": int(
                                    permutation["permutation_index"]
                                ),
                                "split": split,
                                "detector_subset": subset_name,
                                "base_gps_block": base_block,
                                "base_slot": slot,
                                "base_gps_start": base_time,
                                "gps_peak": float(
                                    np.mean(
                                        list(shifted_times.values())
                                    )
                                ),
                                "pairwise_peak_separation_seconds": (
                                    separations
                                ),
                                "shift_by_ifo": shift_by_ifo,
                                "source_candidate_ids": source_ids,
                                "source_window_ids": {
                                    ifo: source_windows[ifo][
                                        "window_id"
                                    ]
                                    for ifo in subset
                                },
                                "source_gps_blocks": {
                                    ifo: source_windows[ifo][
                                        "gps_block"
                                    ]
                                    for ifo in subset
                                },
                                "chirp_scores": chirp_scores,
                                "glitch_scores": glitch_scores,
                                **network_ranking(
                                    chirp_scores,
                                    glitch_scores,
                                    list(subset),
                                ),
                            }
                        )
                        raw_counts[subset_name] += 1
            if block_has_exposure:
                eligible_blocks += 1
        clustered = _cluster_detector_set_network_rows(
            raw_networks,
            cluster_window_seconds,
        )
        selected_counts = Counter(
            str(row["detector_subset"]) for row in clustered
        )
        live_time_seconds = _union_duration(exposure_intervals)
        if (
            int(permutation["eligible_blocks"]) != eligible_blocks
            or int(permutation["eligible_windows"])
            != int(round(live_time_seconds / duration))
            or {
                str(key): int(value)
                for key, value in permutation[
                    "eligible_windows_by_detector_subset"
                ].items()
            }
            != dict(eligible_counts)
            or not np.isclose(
                float(permutation["live_time_seconds"]),
                live_time_seconds,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "executed detector-set block exposure differs from schedule"
            )
        output.extend(clustered)
        eligible_subset_windows_total.update(eligible_counts)
        raw_subset_total.update(raw_counts)
        selected_subset_total.update(selected_counts)
        permutation_exposure.append(
            {
                **permutation,
                "slide_index": int(permutation["permutation_index"]),
                "raw_coincidences": len(raw_networks),
                "raw_coincidences_by_detector_subset": dict(
                    sorted(raw_counts.items())
                ),
                "clustered_candidates": len(clustered),
                "clustered_candidates_by_detector_subset": dict(
                    sorted(selected_counts.items())
                ),
            }
        )

    timing_resolutions = [
        float(
            row.get(
                "timing_resolution_seconds",
                row["bin_width_seconds"],
            )
        )
        for row in relevant_candidates
    ]
    calibrated = bool(relevant_candidates) and all(
        bool(row.get("timing_empirically_calibrated"))
        and np.isclose(
            float(
                row.get(
                    "empirical_timing_uncertainty_seconds",
                    -1,
                )
            ),
            empirical_timing_uncertainty_seconds,
            rtol=0.0,
            atol=1e-12,
        )
        for row in relevant_candidates
    )
    provenance_values = {
        field: sorted(
            {
                str(row[field])
                for row in relevant_candidates
                if row.get(field)
            }
        )
        for field in (
            "timing_calibration_report_sha256",
            "candidate_checkpoint_sha256",
            "candidate_config_sha256",
            "candidate_code_commit",
        )
    }
    equivalent_live_time_seconds = sum(
        float(row["live_time_seconds"])
        for row in permutation_exposure
    )
    publication_timing_gate = (
        calibrated
        and bool(timing_resolutions)
        and max(timing_resolutions) <= 0.01
        and all(
            len(values) == 1 for values in provenance_values.values()
        )
    )
    report = {
        "status": "variable_detector_set_block_permutation_background",
        "scientific_claim_allowed": False,
        "split": split,
        "background_pairing_method": (
            DETECTOR_SET_BLOCK_PERMUTATION_METHOD
        ),
        "required_detector_subsets": [
            "+".join(subset) for subset in subsets
        ],
        "detectors": list(detectors),
        "pairwise_light_travel_time_seconds": dict(
            sorted(physical_limits.items())
        ),
        "empirical_timing_uncertainty_seconds": (
            empirical_timing_uncertainty_seconds
        ),
        "pairwise_allowed_peak_separation_seconds": {
            key: value
            + 2.0 * empirical_timing_uncertainty_seconds
            for key, value in sorted(physical_limits.items())
        },
        "cluster_window_seconds": cluster_window_seconds,
        "window_duration_seconds": duration,
        "input_windows": len(windows),
        "input_gps_blocks": ordered_blocks,
        "input_zero_lag_live_time_seconds": _union_duration(
            (
                float(row["gps_start"]),
                float(row["gps_end"]),
            )
            for row in windows
        ),
        "input_candidates": len(relevant_candidates),
        "background_rows": len(output),
        "slide_count": len(permutation_exposure),
        "slide_indices": [
            int(row["permutation_index"])
            for row in permutation_exposure
        ],
        "slide_indices_sha256": canonical_hash(
            [
                int(row["permutation_index"])
                for row in permutation_exposure
            ],
            64,
        ),
        "slide_exposure": permutation_exposure,
        "eligible_windows_by_detector_subset": dict(
            sorted(eligible_subset_windows_total.items())
        ),
        "raw_coincidences_by_detector_subset": dict(
            sorted(raw_subset_total.items())
        ),
        "clustered_candidates_by_detector_subset": dict(
            sorted(selected_subset_total.items())
        ),
        "equivalent_live_time_seconds": equivalent_live_time_seconds,
        "equivalent_live_time_years": (
            equivalent_live_time_seconds / SECONDS_PER_YEAR
        ),
        "maximum_bin_width_seconds": maximum_bin_width or None,
        "maximum_timing_resolution_seconds": (
            max(timing_resolutions) if timing_resolutions else None
        ),
        "candidate_timing_empirically_calibrated": calibrated,
        "timing_calibration_report_sha256": (
            provenance_values[
                "timing_calibration_report_sha256"
            ][0]
            if len(
                provenance_values[
                    "timing_calibration_report_sha256"
                ]
            )
            == 1
            else None
        ),
        "candidate_checkpoint_sha256": (
            provenance_values["candidate_checkpoint_sha256"][0]
            if len(
                provenance_values["candidate_checkpoint_sha256"]
            )
            == 1
            else None
        ),
        "candidate_config_sha256": (
            provenance_values["candidate_config_sha256"][0]
            if len(provenance_values["candidate_config_sha256"])
            == 1
            else None
        ),
        "candidate_code_commit": (
            provenance_values["candidate_code_commit"][0]
            if len(provenance_values["candidate_code_commit"])
            == 1
            else None
        ),
        "publication_timing_gate_passed": publication_timing_gate,
        "detector_duty_cycle_accounted": True,
        "detector_subset_channels_clustered_jointly": True,
        "live_time_counted_once_per_permutation": True,
        "independent_pairwise_block_shifts": True,
        "execution_schedule_complete": True,
    }
    return output, report


def run_detector_set_candidate_block_permutations(
    candidates_path: str | Path,
    background_manifest: str | Path,
    schedule_path: str | Path,
    output_dir: str | Path,
    empirical_timing_uncertainty_seconds: float,
    cluster_window_seconds: float,
) -> dict[str, Any]:
    """Execute and persist a frozen variable-detector block background."""

    with Path(schedule_path).open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    if schedule.get("background_manifest_sha256") != file_sha256(
        background_manifest
    ):
        raise ValueError(
            "detector-set block schedule background hash differs"
        )
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        windows = [
            json.loads(line) for line in handle if line.strip()
        ]
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        candidates = [
            json.loads(line) for line in handle if line.strip()
        ]
    rows, report = build_detector_set_candidate_block_permutations(
        candidates,
        windows,
        schedule,
        empirical_timing_uncertainty_seconds,
        cluster_window_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split = str(schedule["split"])
    manifest = (
        output
        / f"{split}_detector_set_block_permutation_background.jsonl"
    )
    atomic_write_text(
        manifest,
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
    )
    result = {
        **report,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "candidate_manifest_path": str(Path(candidates_path).resolve()),
        "candidate_manifest_sha256": file_sha256(candidates_path),
        "background_manifest_path": str(
            Path(background_manifest).resolve()
        ),
        "background_manifest_sha256": file_sha256(
            background_manifest
        ),
        "slide_schedule_path": str(Path(schedule_path).resolve()),
        "slide_schedule_sha256": file_sha256(schedule_path),
        "slide_schedule_id": schedule["schedule_id"],
        "slide_schedule_count": schedule["selected_shift_count"],
        "network_config_sha256": schedule["network_config_sha256"],
        **execution_provenance(),
    }
    report_path = (
        output
        / f"{split}_detector_set_block_permutation_report.json"
    )
    atomic_write_json(report_path, result)
    return result


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
    slide_start_index: int = 1,
    slide_indices: Iterable[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reference_ifo == shifted_ifo:
        raise ValueError("reference and shifted IFOs must differ")
    indices = normalize_candidate_slide_indices(
        slide_count, slide_start_index, slide_indices
    )
    if min(step_seconds, coincidence_window_seconds, cluster_window_seconds) <= 0:
        raise ValueError("slide step, coincidence, and cluster values must be positive")
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
    for slide_index in indices:
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
        "slide_start_index": min(indices),
        "slide_stop_index_exclusive": max(indices) + 1,
        "slide_indices": indices,
        "slide_indices_sha256": canonical_hash(indices, 64),
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
    slide_start_index: int = 1,
    slide_schedule_path: str | Path | None = None,
    schedule_offset: int = 0,
) -> dict[str, Any]:
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        windows = [json.loads(line) for line in handle if line.strip()]
    schedule: dict[str, Any] | None = None
    selected_indices: list[int] | None = None
    if slide_schedule_path is not None:
        schedule_path = Path(slide_schedule_path).resolve()
        with schedule_path.open("r", encoding="utf-8") as handle:
            schedule = json.load(handle)
        if schedule.get("status") != "frozen_candidate_time_slide_schedule":
            raise ValueError("candidate time-slide schedule has the wrong status")
        expected_contract = {
            "split": split,
            "reference_ifo": reference_ifo,
            "shifted_ifo": shifted_ifo,
        }
        if any(schedule.get(field) != value for field, value in expected_contract.items()):
            raise ValueError("candidate time-slide schedule contract differs from runner")
        if not np.isclose(
            float(schedule.get("step_seconds", -1)),
            step_seconds,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("candidate time-slide schedule step differs from runner")
        if schedule.get("background_manifest_sha256") != file_sha256(
            background_manifest
        ):
            raise ValueError("candidate time-slide schedule background hash differs")
        all_indices = [int(value) for value in schedule.get("slide_indices", [])]
        schedule_identity = candidate_slide_schedule_identity(schedule)
        if (
            schedule.get("candidate_scores_inspected") is not False
            or schedule.get("selection_data")
            != "background_gps_and_detector_availability_only"
            or int(schedule.get("slide_count", -1)) != len(all_indices)
            or canonical_hash(all_indices, 64) != schedule.get("slide_indices_sha256")
            or canonical_hash(schedule_identity, 32) != schedule.get("schedule_id")
        ):
            raise ValueError("candidate time-slide schedule index hash differs")
        if schedule_offset < 0:
            raise ValueError("candidate time-slide schedule offset must be non-negative")
        selected_indices = all_indices[schedule_offset : schedule_offset + slide_count]
        if len(selected_indices) != slide_count:
            raise ValueError("candidate time-slide schedule shard exceeds frozen schedule")
    elif schedule_offset != 0:
        raise ValueError("schedule offset requires a frozen candidate time-slide schedule")
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
        slide_start_index,
        selected_indices,
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
        "slide_schedule_path": str(Path(slide_schedule_path).resolve())
        if slide_schedule_path is not None
        else None,
        "slide_schedule_sha256": file_sha256(slide_schedule_path)
        if slide_schedule_path is not None
        else None,
        "slide_schedule_id": schedule.get("schedule_id") if schedule else None,
        "slide_schedule_count": schedule.get("slide_count") if schedule else None,
        "schedule_offset": schedule_offset if schedule else None,
        "schedule_stop_offset_exclusive": schedule_offset + slide_count
        if schedule
        else None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / f"{split}_candidate_time_slide_report.json", result)
    return result


def run_candidate_block_permutations(
    candidates_path: str | Path,
    background_manifest: str | Path,
    schedule_path: str | Path,
    output_dir: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    coincidence_window_seconds: float,
    cluster_window_seconds: float,
    physical_delay_limit_seconds: float,
    empirical_timing_uncertainty_seconds: float,
) -> dict[str, Any]:
    """Execute a frozen circular GPS-block permutation background schedule."""

    if reference_ifo == shifted_ifo:
        raise ValueError("block permutations require different detectors")
    if (
        min(
            coincidence_window_seconds,
            cluster_window_seconds,
            physical_delay_limit_seconds,
        )
        <= 0
        or empirical_timing_uncertainty_seconds < 0
    ):
        raise ValueError("block permutation timing parameters are invalid")
    expected_window = (
        physical_delay_limit_seconds + 2 * empirical_timing_uncertainty_seconds
    )
    if not np.isclose(
        coincidence_window_seconds, expected_window, rtol=0.0, atol=1e-12
    ):
        raise ValueError("block permutation coincidence window differs from physics")
    with Path(schedule_path).open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    if schedule.get("status") != "frozen_candidate_block_permutation_schedule":
        raise ValueError("candidate block-permutation schedule has the wrong status")
    if (
        schedule.get("candidate_scores_inspected") is not False
        or schedule.get("method") != CANDIDATE_BLOCK_PERMUTATION_METHOD
        or schedule.get("selection_data") != CANDIDATE_BLOCK_SELECTION_DATA
        or canonical_hash(candidate_block_schedule_identity(schedule), 32)
        != schedule.get("schedule_id")
    ):
        raise ValueError("candidate block-permutation schedule identity differs")
    if any(
        schedule.get(field) != value
        for field, value in {
            "split": split,
            "reference_ifo": reference_ifo,
            "shifted_ifo": shifted_ifo,
        }.items()
    ):
        raise ValueError("candidate block-permutation schedule contract differs")
    if schedule.get("background_manifest_sha256") != file_sha256(background_manifest):
        raise ValueError("candidate block schedule background hash differs")
    shifts = [int(value) for value in schedule.get("shift_indices", [])]
    if (
        not shifts
        or shifts != sorted(set(shifts))
        or any(value <= 0 for value in shifts)
        or len(shifts) != int(schedule.get("selected_shift_count", -1))
        or canonical_hash(shifts, 64) != schedule.get("shift_indices_sha256")
    ):
        raise ValueError("candidate block schedule shift hash differs")
    selected_shift_rows = schedule.get("selected_shifts", [])
    expected_exposure = {int(row["shift_index"]): row for row in selected_shift_rows}
    if (
        len(selected_shift_rows) != len(shifts)
        or len(expected_exposure) != len(shifts)
        or set(expected_exposure) != set(shifts)
    ):
        raise ValueError("candidate block schedule exposure rows differ from shifts")

    output = Path(output_dir)
    report_path = output / f"{split}_candidate_time_slide_report.json"
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        manifest_value = prior.get("manifest_path")
        immutable_fields = {
            "status": "subwindow_clustered_time_slide_integration_only",
            "background_pairing_method": schedule["method"],
            "split": split,
            "reference_ifo": reference_ifo,
            "shifted_ifo": shifted_ifo,
            "candidate_manifest_sha256": file_sha256(candidates_path),
            "background_manifest_sha256": file_sha256(background_manifest),
            "slide_schedule_sha256": file_sha256(schedule_path),
            "slide_schedule_id": schedule["schedule_id"],
            "slide_schedule_count": len(shifts),
            "execution_schedule_complete": True,
        }
        if any(
            prior.get(field) != value for field, value in immutable_fields.items()
        ):
            raise ValueError("completed candidate block background has another identity")
        float_fields = {
            "coincidence_window_seconds": coincidence_window_seconds,
            "cluster_window_seconds": cluster_window_seconds,
            "physical_delay_limit_seconds": physical_delay_limit_seconds,
            "empirical_timing_uncertainty_seconds": (
                empirical_timing_uncertainty_seconds
            ),
        }
        if any(
            not np.isclose(
                float(prior.get(field, float("nan"))),
                value,
                rtol=0.0,
                atol=1e-12,
            )
            for field, value in float_fields.items()
        ):
            raise ValueError("completed candidate block background timing differs")
        if (
            not manifest_value
            or not Path(str(manifest_value)).is_file()
            or file_sha256(manifest_value) != str(prior.get("manifest_sha256"))
        ):
            raise ValueError("completed candidate block background manifest is invalid")
        return prior

    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        windows = [json.loads(line) for line in handle if line.strip()]
    windows = [row for row in windows if str(row.get("split")) == split]
    if not windows:
        raise ValueError(f"no background windows for split {split}")
    window_duration = float(schedule["window_duration_seconds"])
    by_block: dict[str, dict[int, dict[str, Any]]] = {}
    block_starts: dict[str, float] = {}
    by_window: dict[str, dict[str, Any]] = {}
    for row in windows:
        block = str(row["gps_block"])
        _, block_start, _ = parse_gps_block_identity(block)
        offset = (float(row["gps_start"]) - block_start) / window_duration
        slot = int(round(offset))
        if not np.isclose(offset, slot, rtol=0.0, atol=1e-6):
            raise ValueError("background window is not block-slot aligned")
        if slot in by_block.setdefault(block, {}):
            raise ValueError(f"GPS block {block} repeats slot {slot}")
        by_block[block][slot] = row
        block_starts[block] = block_start
        window_id = str(row["window_id"])
        if window_id in by_window:
            raise ValueError("background window ID repeats")
        by_window[window_id] = row
    ordered = [str(value) for value in schedule["ordered_gps_blocks"]]
    if set(ordered) != set(by_block) or len(ordered) != len(by_block):
        raise ValueError("candidate block schedule GPS inventory differs")
    if any(value >= len(ordered) for value in shifts):
        raise ValueError("candidate block schedule shift exceeds the circular range")

    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]
    candidates_by_window: dict[str, dict[str, list[dict[str, Any]]]] = {}
    relevant = []
    for row in candidates:
        if str(row.get("split")) != split:
            raise ValueError("candidate block input mixes data splits")
        window_id = str(row["window_id"])
        if window_id not in by_window:
            raise ValueError("candidate block input references an unknown window")
        ifo = str(row["ifo"])
        if ifo not in _available_ifos(by_window[window_id]):
            raise ValueError("candidate block input uses an unavailable detector")
        candidates_by_window.setdefault(window_id, {}).setdefault(ifo, []).append(row)
        if ifo in {reference_ifo, shifted_ifo}:
            relevant.append(row)
    if not relevant:
        raise ValueError("candidate block input has no relevant detector candidates")
    calibrated = all(bool(row.get("timing_empirically_calibrated")) for row in relevant)
    uncertainties_match = calibrated and all(
        np.isclose(
            float(row["empirical_timing_uncertainty_seconds"]),
            empirical_timing_uncertainty_seconds,
            rtol=0.0,
            atol=1e-12,
        )
        for row in relevant
    )
    calibration_hashes = {
        str(row.get("timing_calibration_report_sha256")) for row in relevant
    }
    checkpoint_hashes = {
        str(row.get("candidate_checkpoint_sha256")) for row in relevant
    }
    config_hashes = {str(row.get("candidate_config_sha256")) for row in relevant}
    code_commits = {str(row.get("candidate_code_commit")) for row in relevant}
    timing_resolutions = [
        float(row.get("timing_resolution_seconds", row["bin_width_seconds"]))
        for row in relevant
    ]
    provenance_complete = all(
        len(values) == 1 and None not in values and "None" not in values
        for values in (
            calibration_hashes,
            checkpoint_hashes,
            config_hashes,
            code_commits,
        )
    )
    timing_gate = bool(
        calibrated
        and uncertainties_match
        and provenance_complete
        and max(timing_resolutions) <= 0.01
    )

    output_rows = []
    exposure_rows = []
    for shift in shifts:
        raw = []
        paired_windows = 0
        paired_blocks = 0
        for index, reference_block in enumerate(ordered):
            shifted_block = ordered[(index + shift) % len(ordered)]
            common_slots = set(by_block[reference_block]) & set(by_block[shifted_block])
            block_contributed = False
            for slot in sorted(common_slots):
                reference_window = by_block[reference_block][slot]
                shifted_window = by_block[shifted_block][slot]
                if reference_ifo not in _available_ifos(
                    reference_window
                ) or shifted_ifo not in _available_ifos(shifted_window):
                    continue
                paired_windows += 1
                block_contributed = True
                reference_candidates = candidates_by_window.get(
                    str(reference_window["window_id"]), {}
                ).get(reference_ifo, [])
                shifted_candidates = candidates_by_window.get(
                    str(shifted_window["window_id"]), {}
                ).get(shifted_ifo, [])
                for first in reference_candidates:
                    first_relative = float(first["gps_peak"]) - float(
                        reference_window["gps_start"]
                    )
                    for second in shifted_candidates:
                        second_relative = float(second["gps_peak"]) - float(
                            shifted_window["gps_start"]
                        )
                        separation = abs(first_relative - second_relative)
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
                                "candidate_id": "block-permutation-"
                                + canonical_hash(
                                    {
                                        "shift": shift,
                                        "first": first["candidate_id"],
                                        "second": second["candidate_id"],
                                    },
                                    24,
                                ),
                                "slide_id": f"block-shift-{shift:06d}",
                                "slide_index": shift,
                                "split": split,
                                "gps_peak": float(reference_window["gps_start"])
                                + (first_relative + second_relative) / 2,
                                "peak_separation_seconds": separation,
                                "background_pairing_method": schedule["method"],
                                "source_candidate_ids": {
                                    reference_ifo: first["candidate_id"],
                                    shifted_ifo: second["candidate_id"],
                                },
                                "source_window_ids": {
                                    reference_ifo: reference_window["window_id"],
                                    shifted_ifo: shifted_window["window_id"],
                                },
                                "source_gps_blocks": {
                                    reference_ifo: reference_block,
                                    shifted_ifo: shifted_block,
                                },
                                "relative_peak_seconds": {
                                    reference_ifo: first_relative,
                                    shifted_ifo: second_relative,
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
            paired_blocks += int(block_contributed)
        clustered = _cluster_network_rows(raw, cluster_window_seconds)
        output_rows.extend(clustered)
        live_time = paired_windows * window_duration
        expected = expected_exposure[shift]
        if (
            paired_windows != int(expected["paired_windows"])
            or paired_blocks != int(expected["paired_blocks"])
            or not np.isclose(
                live_time, float(expected["live_time_seconds"]), rtol=0.0, atol=1e-9
            )
        ):
            raise ValueError(
                "executed block-permutation exposure differs from schedule"
            )
        exposure_rows.append(
            {
                "slide_index": shift,
                "paired_blocks": paired_blocks,
                "paired_windows": paired_windows,
                "raw_coincidences": len(raw),
                "clustered_candidates": len(clustered),
                "live_time_seconds": live_time,
            }
        )
    exposure_seconds = sum(row["live_time_seconds"] for row in exposure_rows)
    if not np.isclose(
        exposure_seconds,
        float(schedule["selected_equivalent_live_time_seconds"]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("executed block exposure differs from frozen total")

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{split}_candidate_block_permutation_background.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    )
    result = {
        "status": "subwindow_clustered_time_slide_integration_only",
        "background_pairing_method": schedule["method"],
        "scientific_claim_allowed": False,
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "slide_count": len(shifts),
        "slide_indices": shifts,
        "slide_indices_sha256": canonical_hash(shifts, 64),
        "coincidence_window_seconds": coincidence_window_seconds,
        "cluster_window_seconds": cluster_window_seconds,
        "physical_delay_limit_seconds": physical_delay_limit_seconds,
        "empirical_timing_uncertainty_seconds": empirical_timing_uncertainty_seconds,
        "expected_coincidence_window_seconds": expected_window,
        "coincidence_window_matches_physics": True,
        "input_windows": len(windows),
        "input_gps_blocks": ordered,
        "input_zero_lag_live_time_seconds": _union_duration(
            (float(row["gps_start"]), float(row["gps_end"])) for row in windows
        ),
        "input_candidates": len(candidates),
        "background_rows": len(output_rows),
        "slide_exposure": exposure_rows,
        "equivalent_live_time_seconds": exposure_seconds,
        "equivalent_live_time_years": exposure_seconds / SECONDS_PER_YEAR,
        "maximum_bin_width_seconds": max(
            float(row["bin_width_seconds"]) for row in relevant
        ),
        "maximum_timing_resolution_seconds": max(timing_resolutions),
        "candidate_timing_empirically_calibrated": calibrated,
        "timing_uncertainty_matches_candidates": uncertainties_match,
        "timing_calibration_report_sha256": next(iter(calibration_hashes))
        if len(calibration_hashes) == 1
        else None,
        "candidate_checkpoint_sha256": next(iter(checkpoint_hashes))
        if len(checkpoint_hashes) == 1
        else None,
        "candidate_config_sha256": next(iter(config_hashes))
        if len(config_hashes) == 1
        else None,
        "candidate_code_commit": next(iter(code_commits))
        if len(code_commits) == 1
        else None,
        "publication_timing_gate_passed": timing_gate,
        "candidate_manifest_sha256": file_sha256(candidates_path),
        "background_manifest_sha256": file_sha256(background_manifest),
        "slide_schedule_path": str(Path(schedule_path).resolve()),
        "slide_schedule_sha256": file_sha256(schedule_path),
        "slide_schedule_id": schedule["schedule_id"],
        "slide_schedule_count": len(shifts),
        "execution_schedule_complete": True,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "scientific_blocker": (
            "validation schedule executed; frozen threshold and locked test remain required"
            if timing_gate and schedule.get("schedule_exposure_target_reached")
            else "timing or exposure gate did not pass"
        ),
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def merge_candidate_time_slide_shards(
    report_paths: Iterable[str | Path],
    output_dir: str | Path,
    split: str,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in report_paths]
    if not paths:
        raise ValueError("candidate time-slide merge requires at least one report")
    reports = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") != "subwindow_clustered_time_slide_integration_only":
            raise ValueError(f"candidate time-slide shard has wrong status: {path}")
        if str(report.get("split")) != split:
            raise ValueError(f"candidate time-slide shard split differs from {split}: {path}")
        reports.append(report)

    common_fields = (
        "reference_ifo",
        "shifted_ifo",
        "step_seconds",
        "coincidence_window_seconds",
        "cluster_window_seconds",
        "physical_delay_limit_seconds",
        "empirical_timing_uncertainty_seconds",
        "expected_coincidence_window_seconds",
        "coincidence_window_matches_physics",
        "input_windows",
        "input_gps_blocks",
        "input_zero_lag_live_time_seconds",
        "input_candidates",
        "candidate_timing_empirically_calibrated",
        "timing_uncertainty_matches_candidates",
        "timing_calibration_report_sha256",
        "candidate_checkpoint_sha256",
        "candidate_config_sha256",
        "candidate_code_commit",
        "publication_timing_gate_passed",
        "candidate_manifest_sha256",
        "background_manifest_sha256",
        "slide_schedule_path",
        "slide_schedule_sha256",
        "slide_schedule_id",
        "slide_schedule_count",
    )
    first = reports[0]
    for field in common_fields:
        if any(report.get(field) != first.get(field) for report in reports[1:]):
            raise ValueError(f"candidate time-slide shard field differs: {field}")

    exposures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_slide_indices: set[int] = set()
    seen_candidate_ids: set[str] = set()
    source_ranges = []
    for path, report in zip(paths, reports):
        shard_exposures = report.get("slide_exposure")
        if not isinstance(shard_exposures, list) or len(shard_exposures) != int(
            report.get("slide_count", -1)
        ):
            raise ValueError(f"candidate time-slide shard exposure is incomplete: {path}")
        shard_indices = {int(row["slide_index"]) for row in shard_exposures}
        if len(shard_indices) != len(shard_exposures):
            raise ValueError(f"candidate time-slide shard repeats an offset: {path}")
        reported_indices = [int(value) for value in report.get("slide_indices", [])]
        if (
            reported_indices != sorted(shard_indices)
            or canonical_hash(reported_indices, 64)
            != report.get("slide_indices_sha256")
        ):
            raise ValueError(f"candidate time-slide shard index hash differs: {path}")
        if (
            int(report.get("slide_start_index", -1)) != min(shard_indices)
            or int(report.get("slide_stop_index_exclusive", -1))
            != max(shard_indices) + 1
        ):
            raise ValueError(f"candidate time-slide shard range metadata differs: {path}")
        if any(
            not np.isclose(
                float(row["offset_seconds"]),
                int(row["slide_index"]) * float(first["step_seconds"]),
                rtol=0.0,
                atol=1e-12,
            )
            for row in shard_exposures
        ):
            raise ValueError(f"candidate time-slide shard offset differs from index: {path}")
        overlap = seen_slide_indices & shard_indices
        if overlap:
            raise ValueError(f"candidate time-slide shards repeat offsets: {sorted(overlap)}")
        seen_slide_indices.update(shard_indices)
        exposures.extend(shard_exposures)
        manifest = Path(str(report["manifest_path"])).resolve()
        if not manifest.is_file() or file_sha256(manifest) != str(report["manifest_sha256"]):
            raise ValueError(f"candidate time-slide shard manifest hash mismatch: {path}")
        with manifest.open("r", encoding="utf-8") as handle:
            shard_rows = [json.loads(line) for line in handle if line.strip()]
        if len(shard_rows) != int(report.get("background_rows", -1)):
            raise ValueError(f"candidate time-slide shard row count mismatch: {path}")
        if any(int(row["slide_index"]) not in shard_indices for row in shard_rows):
            raise ValueError(f"candidate time-slide row lies outside its shard: {path}")
        for row in shard_rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen_candidate_ids:
                raise ValueError(f"candidate time-slide shards repeat candidate {candidate_id}")
            seen_candidate_ids.add(candidate_id)
        rows.extend(shard_rows)
        source_ranges.append(
            {
                "report_path": str(path),
                "report_sha256": file_sha256(path),
                "slide_start_index": min(shard_indices),
                "slide_stop_index_exclusive": max(shard_indices) + 1,
                "slide_count": len(shard_indices),
            }
        )

    ordered_indices = sorted(seen_slide_indices)
    expected_indices = list(range(ordered_indices[0], ordered_indices[-1] + 1))
    contiguous = ordered_indices == expected_indices
    schedule_complete: bool | None = None
    if first.get("slide_schedule_sha256") is not None:
        schedule_path = Path(str(first["slide_schedule_path"])).resolve()
        if (
            not schedule_path.is_file()
            or file_sha256(schedule_path) != str(first["slide_schedule_sha256"])
        ):
            raise ValueError("candidate time-slide frozen schedule hash mismatch")
        with schedule_path.open("r", encoding="utf-8") as handle:
            schedule = json.load(handle)
        schedule_indices = [int(value) for value in schedule.get("slide_indices", [])]
        schedule_identity = candidate_slide_schedule_identity(schedule)
        if (
            schedule.get("status") != "frozen_candidate_time_slide_schedule"
            or schedule.get("schedule_id") != first.get("slide_schedule_id")
            or schedule.get("candidate_scores_inspected") is not False
            or schedule.get("selection_data")
            != "background_gps_and_detector_availability_only"
            or schedule.get("background_manifest_sha256")
            != first.get("background_manifest_sha256")
            or int(schedule.get("slide_count", -1)) != len(schedule_indices)
            or canonical_hash(schedule_indices, 64)
            != schedule.get("slide_indices_sha256")
            or canonical_hash(schedule_identity, 32) != schedule.get("schedule_id")
        ):
            raise ValueError("candidate time-slide frozen schedule identity mismatch")
        expected_schedule_indices = schedule_indices
        if not set(ordered_indices).issubset(expected_schedule_indices):
            raise ValueError("candidate time-slide shard contains an unscheduled offset")
        schedule_complete = ordered_indices == expected_schedule_indices
    execution_complete = schedule_complete if schedule_complete is not None else contiguous
    exposures.sort(key=lambda row: int(row["slide_index"]))
    rows.sort(key=lambda row: (int(row["slide_index"]), float(row["gps_peak"])))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{split}_candidate_time_slide_background.jsonl"
    report_path = output / f"{split}_candidate_time_slide_report.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("merged candidate time-slide outputs are immutable")
    atomic_write_text(
        manifest_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    equivalent_seconds = sum(float(row["live_time_seconds"]) for row in exposures)
    result = {
        **{field: first.get(field) for field in common_fields},
        "status": "subwindow_clustered_time_slide_integration_only",
        "scientific_claim_allowed": False,
        "split": split,
        "slide_count": len(ordered_indices),
        "slide_start_index": ordered_indices[0],
        "slide_stop_index_exclusive": ordered_indices[-1] + 1,
        "slide_indices_contiguous": contiguous,
        "slide_schedule_complete": schedule_complete,
        "execution_schedule_complete": execution_complete,
        "source_publication_timing_gate_passed": bool(
            first.get("publication_timing_gate_passed")
        ),
        "publication_timing_gate_passed": bool(
            first.get("publication_timing_gate_passed")
        )
        and execution_complete,
        "source_shards": sorted(source_ranges, key=lambda row: row["slide_start_index"]),
        "background_rows": len(rows),
        "slide_exposure": exposures,
        "equivalent_live_time_seconds": equivalent_seconds,
        "equivalent_live_time_years": equivalent_seconds / SECONDS_PER_YEAR,
        "maximum_bin_width_seconds": max(
            (
                float(report["maximum_bin_width_seconds"])
                for report in reports
                if report.get("maximum_bin_width_seconds") is not None
            ),
            default=None,
        ),
        "maximum_timing_resolution_seconds": max(
            (
                float(report["maximum_timing_resolution_seconds"])
                for report in reports
                if report.get("maximum_timing_resolution_seconds") is not None
            ),
            default=None,
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "scientific_blocker": (
            "merged slide shards do not complete the frozen offset schedule"
            if not execution_complete
            else "adequate exposure, validation-only threshold freeze and locked-test evaluation "
            "remain required"
        ),
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result
