from __future__ import annotations

import numpy as np
import pytest

from gwyolo.candidates import (
    _local_envelope_timing_refinement,
    build_injection_candidate_rankings,
    build_candidate_time_slides,
    calibrate_candidate_timing_rows,
    extract_temporal_clusters,
)


def test_temporal_clusters_preserve_multiple_candidates_and_refine_peak() -> None:
    chirp = np.zeros((2, 1, 1, 8), dtype=np.float32)
    glitch = np.zeros_like(chirp)
    chirp[0, 0, 0] = [0.0, 0.1, 0.6, 0.9, 0.2, 0.0, 0.7, 0.0]
    chirp[1, 0, 0] = [0.0, 0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0]
    glitch[0, 0, 0, 3] = 0.25
    rows = extract_temporal_clusters(chirp, glitch, ["H1", "L1"], 1000, 8, 0.5)
    assert len(rows) == 3
    h1 = [row for row in rows if row["ifo"] == "H1"]
    assert len(h1) == 2
    assert h1[0]["start_bin"] == 2
    assert h1[0]["stop_bin_exclusive"] == 4
    assert h1[0]["peak_bin"] == 3
    assert h1[0]["gps_start"] == 1002
    assert h1[0]["gps_end"] == 1004
    assert h1[0]["chirp_glitch_margin"] == np.float32(0.9) - np.float32(0.25)
    assert 1003.0 < h1[0]["gps_peak"] < 1004.0
    assert h1[0]["timing_uncertainty_floor_seconds"] == 0.5


def test_local_strain_timing_refines_every_mask_cluster_at_sample_resolution() -> None:
    strain = np.zeros((1, 128), dtype=np.float32)
    strain[0, 52] = 10.0
    clusters = [
        {
            "ifo": "H1",
            "gps_start": 102.0,
            "gps_end": 104.0,
            "gps_peak": 103.0,
            "bin_width_seconds": 1.0,
        }
    ]
    rows = _local_envelope_timing_refinement(
        clusters, strain, ["H1"], 100.0, sample_rate=16
    )
    assert rows[0]["gps_peak"] == 103.25
    assert rows[0]["mask_profile_gps_peak"] == 103.0
    assert rows[0]["timing_resolution_seconds"] == 1 / 16
    assert rows[0]["timing_empirically_calibrated"] is False


def test_candidate_time_slides_use_all_candidates_but_cluster_network_events() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": f"b{index}",
            "ifos": ["H1", "L1"],
        }
        for index in range(3)
    ]
    candidates = [
        {
            "candidate_id": "h1-a",
            "window_id": "w0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 1.0,
            "chirp_score": 0.8,
            "glitch_score_at_peak": 0.1,
            "bin_width_seconds": 0.01,
        },
        {
            "candidate_id": "h1-b",
            "window_id": "w0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 1.02,
            "chirp_score": 0.6,
            "glitch_score_at_peak": 0.1,
            "bin_width_seconds": 0.01,
        },
        {
            "candidate_id": "l1",
            "window_id": "w1",
            "split": "val",
            "ifo": "L1",
            "gps_peak": 9.005,
            "chirp_score": 0.7,
            "glitch_score_at_peak": 0.2,
            "bin_width_seconds": 0.01,
        },
    ]
    rows, report = build_candidate_time_slides(
        candidates,
        windows,
        "val",
        "H1",
        "L1",
        slide_count=1,
        step_seconds=8,
        coincidence_window_seconds=0.03,
        cluster_window_seconds=0.1,
    )
    assert report["equivalent_live_time_seconds"] == 16
    assert report["slide_exposure"][0]["raw_coincidences"] == 2
    assert report["slide_exposure"][0]["clustered_candidates"] == 1
    assert len(rows) == 1
    assert rows[0]["ranking_score"] == 0.7
    assert rows[0]["peak_separation_seconds"] < 0.01


def test_candidate_slide_exposure_requires_the_contributing_ifos() -> None:
    windows = [
        {
            "window_id": "w0",
            "split": "val",
            "gps_start": 0,
            "gps_end": 8,
            "gps_block": "b0",
            "ifos": ["H1", "L1"],
        },
        {
            "window_id": "w1",
            "split": "val",
            "gps_start": 8,
            "gps_end": 16,
            "gps_block": "b1",
            "ifos": ["H1"],
        },
        {
            "window_id": "w2",
            "split": "val",
            "gps_start": 16,
            "gps_end": 24,
            "gps_block": "b2",
            "ifos": ["H1", "L1"],
        },
    ]
    _, report = build_candidate_time_slides(
        [], windows, "val", "H1", "L1", 1, 8, 0.03, 0.1
    )
    assert report["equivalent_live_time_seconds"] == 8
    assert report["slide_exposure"][0]["paired_windows"] == 1
    assert report["slide_exposure"][0]["skipped_unavailable_pairs"] == 1


def test_candidate_slide_publication_timing_gate_needs_calibration_and_physics() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": f"b{index}",
            "ifos": ["H1", "L1"],
        }
        for index in range(2)
    ]
    candidates = [
        {
            "candidate_id": "h1",
            "window_id": "w0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 1.0,
            "chirp_score": 0.8,
            "glitch_score_at_peak": 0.1,
            "bin_width_seconds": 0.08,
            "timing_resolution_seconds": 1 / 1024,
            "timing_empirically_calibrated": True,
            "empirical_timing_uncertainty_seconds": 0.001,
            "timing_calibration_report_sha256": "a" * 64,
        },
        {
            "candidate_id": "l1",
            "window_id": "w1",
            "split": "val",
            "ifo": "L1",
            "gps_peak": 9.005,
            "chirp_score": 0.7,
            "glitch_score_at_peak": 0.2,
            "bin_width_seconds": 0.08,
            "timing_resolution_seconds": 1 / 1024,
            "timing_empirically_calibrated": True,
            "empirical_timing_uncertainty_seconds": 0.001,
            "timing_calibration_report_sha256": "a" * 64,
        },
    ]
    _, report = build_candidate_time_slides(
        candidates,
        windows,
        "val",
        "H1",
        "L1",
        1,
        8,
        0.012,
        0.1,
        physical_delay_limit_seconds=0.010,
        empirical_timing_uncertainty_seconds=0.001,
    )
    assert report["publication_timing_gate_passed"] is True
    _, uncalibrated = build_candidate_time_slides(
        [{**row, "timing_empirically_calibrated": False} for row in candidates],
        windows,
        "val",
        "H1",
        "L1",
        1,
        8,
        0.012,
        0.1,
        physical_delay_limit_seconds=0.010,
        empirical_timing_uncertainty_seconds=0.001,
    )
    assert uncalibrated["publication_timing_gate_passed"] is False


def test_candidate_timing_calibration_uses_nearest_candidate_once_per_target() -> None:
    candidates = [
        {
            "injection_id": "i1",
            "ifo": "H1",
            "gps_peak": 100.004,
            "timing_method": "strain",
            "timing_resolution_seconds": 1 / 1024,
        },
        {
            "injection_id": "i1",
            "ifo": "H1",
            "gps_peak": 100.001,
            "timing_method": "strain",
            "timing_resolution_seconds": 1 / 1024,
        },
        {
            "injection_id": "i2",
            "ifo": "L1",
            "gps_peak": 200.003,
            "timing_method": "strain",
            "timing_resolution_seconds": 1 / 1024,
        },
    ]
    report = calibrate_candidate_timing_rows(
        candidates,
        {"i1": {"H1": 100.0}, "i2": {"L1": 200.0}},
        association_window_seconds=0.01,
        uncertainty_quantile=0.9,
        minimum_matches_per_method=2,
    )
    method = report["methods"]["strain"]
    assert method["matches"] == 2
    assert method["conditional_match_fraction"] == 1.0
    assert method["empirical_timing_uncertainty_seconds"] == pytest.approx(0.0028)
    assert method["calibration_gate_passed"] is True


def test_injection_candidate_ranking_keeps_missed_injections_in_vt_denominator() -> None:
    parents = [
        {
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "split": "val",
            "source_family": "bbh",
            "gps_block": f"g{index}",
            "gps_time": 100.0 + index,
            "vt_weight": 2.0 + index,
            "vt_weight_unit": "Mpc^3 yr",
            "valid_ifos": ["H1", "L1"],
            "detector_arrival_gps": {
                "H1": 100.0 + index,
                "L1": 100.005 + index,
            },
        }
        for index in range(2)
    ]
    candidates = [
        {
            "candidate_id": "h",
            "injection_id": "i0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 100.001,
            "chirp_score": 0.8,
            "glitch_score_at_peak": 0.1,
            "timing_empirically_calibrated": True,
            "empirical_timing_uncertainty_seconds": 0.001,
            "timing_calibration_report_sha256": "a" * 64,
        },
        {
            "candidate_id": "l",
            "injection_id": "i0",
            "split": "val",
            "ifo": "L1",
            "gps_peak": 100.006,
            "chirp_score": 0.7,
            "glitch_score_at_peak": 0.2,
            "timing_empirically_calibrated": True,
            "empirical_timing_uncertainty_seconds": 0.001,
            "timing_calibration_report_sha256": "a" * 64,
        },
    ]
    rows, report = build_injection_candidate_rankings(
        parents, candidates, "val", "H1", "L1", 0.010, 0.001, 0.02
    )
    assert len(rows) == 2
    assert rows[0]["ranking_score"] == 0.7
    assert rows[0]["candidate_pair_found"] is True
    assert rows[1]["ranking_score"] == 0.0
    assert rows[1]["candidate_pair_found"] is False
    assert report["candidate_pair_found"] == 1
