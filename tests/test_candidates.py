from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.candidates import (
    _local_envelope_timing_refinement,
    build_candidate_time_slides,
    build_detector_set_candidate_time_slides,
    build_detector_set_injection_candidate_rankings,
    build_injection_candidate_rankings,
    calibrate_candidate_timing_rows,
    candidate_proposal_coverage,
    extract_temporal_clusters,
    merge_candidate_time_slide_shards,
    run_apply_candidate_timing_calibration,
    run_candidate_block_permutations,
    run_candidate_time_slides,
    select_candidate_proposal_threshold,
)
from gwyolo.exposure import (
    freeze_candidate_block_permutation_schedule,
    freeze_candidate_time_slide_schedule,
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


def test_candidate_proposal_coverage_preserves_misses_by_hand() -> None:
    injections = [
        {
            "injection_id": "i1",
            "waveform_id": "w1",
            "source_family": "BBH",
            "optimal_snr_stratum": "snr_8_15",
            "optimal_snr_by_ifo": {"H1": 10.0, "L1": 9.0},
            "detector_arrival_gps": {"H1": 10.0, "L1": 10.005},
            "analysis_start_index": 0,
            "analysis_stop_index": 8,
            "sample_rate": 1,
        },
        {
            "injection_id": "i2",
            "waveform_id": "w2",
            "source_family": "BNS",
            "optimal_snr_stratum": "snr_4_8",
            "optimal_snr_by_ifo": {"H1": 5.0, "L1": 6.0},
            "detector_arrival_gps": {"H1": 20.0, "L1": 20.005},
            "analysis_start_index": 0,
            "analysis_stop_index": 8,
            "sample_rate": 1,
        },
    ]
    candidates = [
        {
            "candidate_id": "c1",
            "injection_id": "i1",
            "ifo": "H1",
            "gps_start": 9.9,
            "gps_end": 10.1,
            "gps_peak": 10.02,
        },
        {
            "candidate_id": "c2",
            "injection_id": "i1",
            "ifo": "L1",
            "gps_start": 9.0,
            "gps_end": 9.5,
            "gps_peak": 9.4,
        },
        {
            "candidate_id": "c3",
            "injection_id": "i2",
            "ifo": "L1",
            "gps_start": 19.8,
            "gps_end": 20.0,
            "gps_peak": 19.9,
        },
    ]

    report = candidate_proposal_coverage(injections, candidates, padding_seconds=0.6)
    all_rows = report["groups"]["all"]
    assert all_rows["expected_detector_arrivals"] == 4
    assert all_rows["any_proposal_fraction"] == 0.75
    assert all_rows["interval_coverage_fraction"] == 0.25
    assert all_rows["padded_coverage_fraction"] == 0.75
    assert all_rows["proposal_count_quantiles"]["0.0"] == 0.0
    assert all_rows["proposal_union_fraction_of_analysis_quantiles"]["1.0"] == pytest.approx(
        0.5 / 8
    )
    assert all_rows["minimum_containing_proposal_width_seconds_quantiles"][
        "0.5"
    ] == pytest.approx(0.2)

    with pytest.raises(ValueError, match="duplicate proposal candidate"):
        candidate_proposal_coverage(injections, candidates + candidates[:1], 0.6)


def _proposal_threshold_audit(
    threshold: float, coverage: float, median_union: float, median_width: float
) -> dict[str, object]:
    all_group = {
        "padded_coverage_fraction": coverage,
        "proposal_union_fraction_of_analysis_quantiles": {
            "0.5": median_union,
            "0.9": median_union + 0.1,
        },
        "minimum_containing_proposal_width_seconds_quantiles": {
            "0.5": median_width
        },
    }
    return {
        "status": "validation_only_all_instance_candidate_proposal_coverage",
        "injection_manifest_sha256": "a" * 64,
        "padding_seconds": 0.5,
        "candidates": 10,
        "audit_report_sha256": f"{int(threshold * 10)}" * 64,
        "candidate_extraction_provenance": {
            "available": True,
            "chirp_threshold": threshold,
            "scoring": {
                "checkpoint_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "trigger_manifest_sha256": "d" * 64,
            },
        },
        "groups": {
            "all": all_group,
            "family:BBH": {"padded_coverage_fraction": coverage},
        },
    }


def test_candidate_proposal_threshold_requires_coverage_and_compactness() -> None:
    settings = {
        "required_groups": ["family:BBH"],
        "minimum_all_padded_coverage": 0.95,
        "minimum_required_group_padded_coverage": 0.9,
        "maximum_median_union_fraction": 0.5,
        "maximum_p90_union_fraction": 0.8,
        "maximum_median_containing_width_seconds": 2.0,
    }
    wide = _proposal_threshold_audit(0.3, 0.99, 0.9, 7.0)
    compact = _proposal_threshold_audit(0.5, 0.96, 0.4, 1.5)

    result = select_candidate_proposal_threshold([wide, compact], settings)
    assert result["promotion_allowed"] is True
    assert result["selected"]["chirp_threshold"] == 0.5
    assert result["records"][0]["qualified"] is False

    failed = select_candidate_proposal_threshold([wide], settings)
    assert failed["promotion_allowed"] is False
    assert failed["selected"] is None


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


def test_candidate_time_slide_runner_preserves_declared_parameters(
    tmp_path: Path,
) -> None:
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
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    background_path = tmp_path / "background.jsonl"
    background_path.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    report = run_candidate_time_slides(
        candidate_path,
        background_path,
        tmp_path / "slides",
        "val",
        "H1",
        "L1",
        1,
        8.0,
        0.03,
        0.1,
    )
    assert report["slide_count"] == 1
    assert report["slide_start_index"] == 1
    assert report["slide_stop_index_exclusive"] == 2
    assert report["step_seconds"] == 8.0
    assert report["coincidence_window_seconds"] == 0.03
    assert report["cluster_window_seconds"] == 0.1
    assert report["background_rows"] == 1
    assert Path(report["manifest_path"]).is_file()


def test_candidate_block_permutations_execute_frozen_relative_slots(
    tmp_path: Path,
) -> None:
    windows = []
    candidates = []
    for block_index in range(3):
        block_start = 1000 + block_index * 256
        for slot in range(2):
            windows.append(
                {
                    "window_id": f"w-{block_index}-{slot}",
                    "split": "val",
                    "gps_start": block_start + slot * 8,
                    "gps_end": block_start + (slot + 1) * 8,
                    "gps_block": f"gps:{block_start}:256",
                    "ifos": ["H1", "L1"],
                }
            )
        for ifo, offset in (("H1", 1.0), ("L1", 1.005)):
            candidates.append(
                {
                    "candidate_id": f"{ifo}-{block_index}",
                    "window_id": f"w-{block_index}-0",
                    "split": "val",
                    "ifo": ifo,
                    "gps_peak": block_start + offset,
                    "chirp_score": 0.8 if ifo == "H1" else 0.7,
                    "glitch_score_at_peak": 0.1,
                    "bin_width_seconds": 0.08,
                    "timing_resolution_seconds": 1 / 1024,
                    "timing_empirically_calibrated": True,
                    "empirical_timing_uncertainty_seconds": 0.001,
                    "timing_calibration_report_sha256": "a" * 64,
                    "candidate_checkpoint_sha256": "b" * 64,
                    "candidate_config_sha256": "c" * 64,
                    "candidate_code_commit": "deadbee",
                }
            )
    background_path = tmp_path / "background.jsonl"
    background_path.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    schedule_path = tmp_path / "block-schedule.json"
    schedule = freeze_candidate_block_permutation_schedule(
        background_path,
        schedule_path,
        "val",
        "H1",
        "L1",
        target_far_per_year=1_000_000,
        maximum_shifts=2,
    )
    report = run_candidate_block_permutations(
        candidate_path,
        background_path,
        schedule_path,
        tmp_path / "block-background",
        "val",
        "H1",
        "L1",
        coincidence_window_seconds=0.012,
        cluster_window_seconds=0.1,
        physical_delay_limit_seconds=0.010,
        empirical_timing_uncertainty_seconds=0.001,
    )
    assert report["publication_timing_gate_passed"] is True
    assert report["equivalent_live_time_seconds"] == 96
    assert report["background_rows"] == 6
    assert [row["paired_windows"] for row in report["slide_exposure"]] == [6, 6]
    assert report["slide_schedule_id"] == schedule["schedule_id"]
    rows = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    assert {row["ranking_score"] for row in rows} == {0.7}
    assert {row["background_pairing_method"] for row in rows} == {
        "circular_gps_block_relative_window_permutation_v1"
    }
    resumed = run_candidate_block_permutations(
        candidate_path,
        background_path,
        schedule_path,
        tmp_path / "block-background",
        "val",
        "H1",
        "L1",
        coincidence_window_seconds=0.012,
        cluster_window_seconds=0.1,
        physical_delay_limit_seconds=0.010,
        empirical_timing_uncertainty_seconds=0.001,
    )
    assert resumed == report
    with pytest.raises(ValueError, match="timing differs"):
        run_candidate_block_permutations(
            candidate_path,
            background_path,
            schedule_path,
            tmp_path / "block-background",
            "val",
            "H1",
            "L1",
            coincidence_window_seconds=0.012,
            cluster_window_seconds=0.2,
            physical_delay_limit_seconds=0.010,
            empirical_timing_uncertainty_seconds=0.001,
        )


def test_candidate_time_slide_shards_merge_absolute_nonoverlapping_offsets(
    tmp_path: Path,
) -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": f"b{index}",
            "ifos": ["H1", "L1"],
        }
        for index in range(4)
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
            "bin_width_seconds": 0.01,
        },
        *[
            {
                "candidate_id": f"l1-{index}",
                "window_id": f"w{index}",
                "split": "val",
                "ifo": "L1",
                "gps_peak": index * 8 + 1.005,
                "chirp_score": 0.7,
                "glitch_score_at_peak": 0.2,
                "bin_width_seconds": 0.01,
            }
            for index in (1, 3)
        ],
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    background_path = tmp_path / "background.jsonl"
    background_path.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    schedule_path = tmp_path / "schedule.json"
    freeze_candidate_time_slide_schedule(
        background_path,
        schedule_path,
        "val",
        "H1",
        "L1",
        8.0,
        [1, 3],
        1.0,
    )
    reports = []
    for schedule_offset, slide_index in enumerate((1, 3)):
        output = tmp_path / f"shard-{slide_index}"
        run_candidate_time_slides(
            candidate_path,
            background_path,
            output,
            "val",
            "H1",
            "L1",
            1,
            8.0,
            0.03,
            0.1,
            slide_schedule_path=schedule_path,
            schedule_offset=schedule_offset,
        )
        reports.append(output / "val_candidate_time_slide_report.json")
    merged = merge_candidate_time_slide_shards(
        reports, tmp_path / "merged", "val"
    )
    assert merged["slide_count"] == 2
    assert merged["slide_start_index"] == 1
    assert merged["slide_stop_index_exclusive"] == 4
    assert merged["slide_indices_contiguous"] is False
    assert merged["slide_schedule_complete"] is True
    assert merged["execution_schedule_complete"] is True
    assert merged["equivalent_live_time_seconds"] == 32
    assert merged["background_rows"] == 2
    assert [row["slide_index"] for row in merged["slide_exposure"]] == [1, 3]

    with pytest.raises(ValueError, match="repeat offsets"):
        merge_candidate_time_slide_shards(
            [reports[0], reports[0]], tmp_path / "duplicate", "val"
        )

    tampered_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    tampered_schedule["candidate_scores_inspected"] = True
    tampered_path = tmp_path / "tampered-schedule.json"
    tampered_path.write_text(json.dumps(tampered_schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule index hash differs"):
        run_candidate_time_slides(
            candidate_path,
            background_path,
            tmp_path / "tampered-output",
            "val",
            "H1",
            "L1",
            1,
            8.0,
            0.03,
            0.1,
            slide_schedule_path=tampered_path,
            schedule_offset=0,
        )


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
            "candidate_checkpoint_sha256": "b" * 64,
            "candidate_config_sha256": "c" * 64,
            "candidate_code_commit": "deadbee",
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
            "candidate_checkpoint_sha256": "b" * 64,
            "candidate_config_sha256": "c" * 64,
            "candidate_code_commit": "deadbee",
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
    assert method["empirical_uncertainty_gate"] is True
    assert method["calibration_gate_passed"] is True

    rejected = calibrate_candidate_timing_rows(
        [
            {**row, "gps_peak": row["gps_peak"] + 0.02}
            for row in candidates
        ],
        {"i1": {"H1": 100.0}, "i2": {"L1": 200.0}},
        association_window_seconds=0.05,
        uncertainty_quantile=0.9,
        minimum_matches_per_method=2,
        maximum_empirical_timing_uncertainty_seconds=0.01,
    )["methods"]["strain"]
    assert rejected["maximum_resolution_seconds"] < 0.01
    assert rejected["minimum_matches_gate"] is True
    assert rejected["empirical_uncertainty_gate"] is False
    assert rejected["calibration_gate_passed"] is False


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
            "candidate_checkpoint_sha256": "b" * 64,
            "candidate_config_sha256": "c" * 64,
            "candidate_code_commit": "deadbee",
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
            "candidate_checkpoint_sha256": "b" * 64,
            "candidate_config_sha256": "c" * 64,
            "candidate_code_commit": "deadbee",
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


def test_detector_set_injection_ranking_supports_hlv_and_missing_ifos() -> None:
    parents = [
        {
            "injection_id": "i0",
            "waveform_id": "w0",
            "split": "val",
            "source_family": "BBH",
            "gps_block": "g0",
            "gps_time": 100.0,
            "vt_weight": 2.0,
            "vt_weight_unit": "Mpc^3 yr",
            "valid_ifos": ["H1", "L1", "V1"],
            "detector_arrival_gps": {
                "H1": 100.000,
                "L1": 100.005,
                "V1": 100.020,
            },
        },
        {
            "injection_id": "i1",
            "waveform_id": "w1",
            "split": "val",
            "source_family": "NSBH",
            "gps_block": "g1",
            "gps_time": 101.0,
            "vt_weight": 3.0,
            "vt_weight_unit": "Mpc^3 yr",
            "valid_ifos": ["H1", "V1"],
            "detector_arrival_gps": {
                "H1": 101.000,
                "V1": 101.020,
            },
        },
        {
            "injection_id": "i2",
            "waveform_id": "w2",
            "split": "val",
            "source_family": "BNS",
            "gps_block": "g2",
            "gps_time": 102.0,
            "vt_weight": 4.0,
            "vt_weight_unit": "Mpc^3 yr",
            "valid_ifos": ["H1"],
            "detector_arrival_gps": {"H1": 102.000},
        },
    ]
    scores = {
        "i0": {"H1": 0.8, "L1": 0.7, "V1": 0.9},
        "i1": {"H1": 0.6, "V1": 0.75},
    }
    peaks = {
        "i0": {"H1": 100.001, "L1": 100.006, "V1": 100.021},
        "i1": {"H1": 101.001, "V1": 101.021},
    }
    candidates = []
    for injection_id, by_ifo in scores.items():
        for ifo, chirp_score in by_ifo.items():
            candidates.append(
                {
                    "candidate_id": f"{injection_id}-{ifo}",
                    "injection_id": injection_id,
                    "split": "val",
                    "ifo": ifo,
                    "gps_peak": peaks[injection_id][ifo],
                    "chirp_score": chirp_score,
                    "glitch_score_at_peak": 0.1,
                    "timing_empirically_calibrated": True,
                    "empirical_timing_uncertainty_seconds": 0.001,
                    "timing_calibration_report_sha256": "a" * 64,
                    "candidate_checkpoint_sha256": "b" * 64,
                    "candidate_config_sha256": "c" * 64,
                    "candidate_code_commit": "deadbee",
                }
            )
    subsets = (
        ("H1", "L1"),
        ("H1", "V1"),
        ("L1", "V1"),
        ("H1", "L1", "V1"),
    )
    limits = {
        "H1+L1": 0.010012846152267725,
        "H1+V1": 0.027287979933397113,
        "L1+V1": 0.02644834101635671,
    }
    rows, report = build_detector_set_injection_candidate_rankings(
        parents,
        candidates,
        "val",
        subsets,
        limits,
        empirical_timing_uncertainty_seconds=0.001,
        truth_association_window_seconds=0.02,
    )

    assert [row["injection_id"] for row in rows] == ["i0", "i1"]
    assert rows[0]["selected_detector_subset"] == "H1+L1+V1"
    assert rows[0]["ranking_score"] == pytest.approx(0.8)
    assert rows[1]["selected_detector_subset"] == "H1+V1"
    assert rows[1]["ranking_score"] == pytest.approx(0.6)
    assert report["excluded_missing_detector_or_arrival"] == 1
    assert report["eligible_injections_by_detector_subset"] == {
        "H1+L1": 1,
        "H1+L1+V1": 1,
        "H1+V1": 2,
        "L1+V1": 1,
    }
    assert report["selected_networks_by_detector_subset"] == {
        "H1+L1+V1": 1,
        "H1+V1": 1,
    }
    with pytest.raises(ValueError, match="exactly cover"):
        build_detector_set_injection_candidate_rankings(
            parents,
            candidates,
            "val",
            subsets,
            {"H1+L1": limits["H1+L1"]},
            0.001,
            0.02,
        )


def test_detector_set_time_slides_use_independent_offsets_and_union_exposure() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "test",
            "gps_start": index * 10.0,
            "gps_end": (index + 1) * 10.0,
            "gps_block": f"g{index}",
            "ifos": ifos,
        }
        for index, ifos in enumerate(
            (
                ["V1"],
                ["H1"],
                ["L1"],
                ["H1", "L1", "V1"],
            )
        )
    ]
    candidates = []
    for window_id, ifo, peak, score in (
        ("w1", "H1", 10.001, 0.8),
        ("w2", "L1", 20.006, 0.7),
        ("w0", "V1", 0.021, 0.9),
    ):
        candidates.append(
            {
                "candidate_id": f"{window_id}-{ifo}",
                "window_id": window_id,
                "split": "test",
                "ifo": ifo,
                "gps_peak": peak,
                "chirp_score": score,
                "glitch_score_at_peak": 0.1,
                "bin_width_seconds": 0.005,
                "timing_resolution_seconds": 0.005,
                "timing_empirically_calibrated": True,
                "empirical_timing_uncertainty_seconds": 0.001,
                "timing_calibration_report_sha256": "a" * 64,
                "candidate_checkpoint_sha256": "b" * 64,
                "candidate_config_sha256": "c" * 64,
                "candidate_code_commit": "deadbee",
            }
        )
    subsets = (
        ("H1", "L1"),
        ("H1", "V1"),
        ("L1", "V1"),
        ("H1", "L1", "V1"),
    )
    limits = {
        "H1+L1": 0.010012846152267725,
        "H1+V1": 0.027287979933397113,
        "L1+V1": 0.02644834101635671,
    }
    rows, report = build_detector_set_candidate_time_slides(
        candidates,
        windows,
        "test",
        subsets,
        limits,
        0.001,
        [{"H1": 0.0, "L1": 10.0, "V1": -10.0}],
        0.1,
    )

    assert len(rows) == 1
    assert rows[0]["detector_subset"] == "H1+L1+V1"
    assert rows[0]["source_window_ids"] == {
        "H1": "w1",
        "L1": "w2",
        "V1": "w0",
    }
    assert rows[0]["ranking_score"] == pytest.approx(0.8)
    assert report["eligible_windows_by_detector_subset"] == {
        "H1+L1": 1,
        "H1+L1+V1": 1,
        "H1+V1": 1,
        "L1+V1": 1,
    }
    assert report["slide_exposure"][0]["raw_coincidences"] == 4
    assert report["slide_exposure"][0]["clustered_candidates"] == 1
    assert report["equivalent_live_time_seconds"] == 10.0
    assert report["live_time_counted_once_per_slide"] is True
    assert report["publication_timing_gate_passed"] is True

    with pytest.raises(ValueError, match="independently offset"):
        build_detector_set_candidate_time_slides(
            candidates,
            windows,
            "test",
            subsets,
            limits,
            0.001,
            [{"H1": 0.0, "L1": 10.0, "V1": 10.0}],
            0.1,
        )


def test_calibration_stress_can_reuse_frozen_timing_only_with_narrow_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwyolo.io import file_sha256

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "status": "frozen_validation_calibration_perturbation_plan",
                "passed": True,
                "test_rows_read": 0,
                "scenario_ids": ["stress"],
                "manifests": {"background": {"sha256": "background-manifest"}},
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "c1",
                "timing_method": "strain",
                "timing_resolution_seconds": 0.001,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current_scoring = {
        "available": True,
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "code_commit": "candidate-commit",
        "calibration_perturbation": {
            "plan_sha256": file_sha256(plan_path),
            "scenario_id": "stress",
            "role": "background",
            "manifest_sha256": "background-manifest",
        },
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform": True,
    }
    (tmp_path / "candidate_extraction_report.json").write_text(
        json.dumps(
            {
                "manifest_sha256": file_sha256(candidates),
                "chirp_threshold": 0.3,
                "minimum_bins": 1,
                "source_scoring_provenance": current_scoring,
            }
        ),
        encoding="utf-8",
    )
    timing = tmp_path / "timing.json"
    timing.write_text(
        json.dumps(
            {
                "status": "validation_only_candidate_timing_calibration",
                "source_scoring_provenance": {
                    "available": True,
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "code_commit": "reference-commit",
                },
                "methods": {
                    "strain": {
                        "calibration_gate_passed": True,
                        "maximum_resolution_seconds": 0.002,
                        "empirical_timing_uncertainty_seconds": 0.003,
                        "uncertainty_quantile": 0.99,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    transfer = tmp_path / "transfer.json"
    transfer.write_text(json.dumps({"passed": True}), encoding="utf-8")
    monkeypatch.setattr(
        "gwyolo.code_compatibility.validate_calibration_timing_transfer_compatibility",
        lambda *args: {"passed": True},
    )

    result = run_apply_candidate_timing_calibration(
        candidates,
        timing,
        tmp_path / "calibrated.jsonl",
        calibration_perturbation_plan=plan_path,
        calibration_timing_compatibility_report=transfer,
    )

    assert result["scoring_provenance_matches"] is True
    assert result["uncalibrated_candidates"] == 0
    assert result["calibration_perturbation_plan_sha256"] == file_sha256(plan_path)
    assert result["calibration_timing_transfer_compatibility_report_sha256"] == file_sha256(
        transfer
    )
