from __future__ import annotations

import json
from pathlib import Path

from gwyolo.io import canonical_hash, file_sha256
from gwyolo.locked_streaming import reduce_locked_o4b_search_inputs


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_locked_search_reducer_builds_shared_hlv_background_and_rankings(
    tmp_path: Path,
) -> None:
    work = tmp_path / "execution"
    inputs = tmp_path / "inputs"
    work.mkdir()
    inputs.mkdir()
    suite_path = tmp_path / "suite.json"
    execution_path = tmp_path / "execution.json"
    access_path = tmp_path / "access.json"
    merge_path = work / "merge.json"
    schedule_path = tmp_path / "network-schedule.json"
    report_path = work / "search-reduction.json"

    background_rows = [
        {
            "window_id": f"w{index}",
            "split": "test",
            "gps_start": index * 10.0,
            "gps_end": (index + 1) * 10.0,
            "gps_block": f"gps:{index * 10}:10",
            "ifos": ["H1", "L1", "V1"],
        }
        for index in range(3)
    ]
    raw_background = inputs / "raw-background.jsonl"
    mask_background = inputs / "mask-background.jsonl"
    _write_jsonl(raw_background, background_rows)
    _write_jsonl(mask_background, background_rows)

    provenance = {
        "timing_empirically_calibrated": True,
        "empirical_timing_uncertainty_seconds": 0.001,
        "timing_calibration_report_sha256": "a" * 64,
        "candidate_config_sha256": "c" * 64,
        "candidate_code_commit": "abc123",
        "bin_width_seconds": 0.005,
        "timing_resolution_seconds": 0.005,
        "glitch_score_at_peak": 0.1,
    }

    def background_candidates(arm: str) -> list[dict]:
        checkpoint = "b" * 64 if arm == "raw" else "d" * 64
        rows = []
        detector_delays = {"H1": 0.001, "L1": 0.006, "V1": 0.021}
        for index in range(3):
            for ifo, delay in detector_delays.items():
                rows.append(
                    {
                        **provenance,
                        "candidate_checkpoint_sha256": checkpoint,
                        "candidate_id": f"{arm}-bg-{index}-{ifo}",
                        "window_id": f"w{index}",
                        "split": "test",
                        "ifo": ifo,
                        "gps_peak": index * 10.0 + delay,
                        "chirp_score": 0.7,
                    }
                )
        return rows

    trigger_rows = [
        {
            "injection_id": "i0",
            "waveform_id": "wave0",
            "split": "test",
            "source_family": "BBH",
            "stratum": "BBH",
            "gps_block": "inj-block",
            "gps_time": 100.0,
            "vt_weight": 2.0,
            "vt_weight_unit": "Mpc^3 yr",
            "valid_ifos": ["H1", "L1", "V1"],
            "detector_arrival_gps": {
                "H1": 100.0,
                "L1": 100.005,
                "V1": 100.020,
            },
        }
    ]

    def injection_candidates(arm: str) -> list[dict]:
        checkpoint = "b" * 64 if arm == "raw" else "d" * 64
        peaks = {"H1": 100.001, "L1": 100.006, "V1": 100.021}
        return [
            {
                **provenance,
                "candidate_checkpoint_sha256": checkpoint,
                "candidate_id": f"{arm}-inj-{ifo}",
                "injection_id": "i0",
                "split": "test",
                "ifo": ifo,
                "gps_peak": peak,
                "chirp_score": 0.8,
            }
            for ifo, peak in peaks.items()
        ]

    source_paths = {
        "raw_background_candidates": work / "raw-bg.jsonl",
        "raw_injection_candidates": work / "raw-inj.jsonl",
        "mask_background_candidates": work / "mask-bg.jsonl",
        "mask_injection_candidates": work / "mask-inj.jsonl",
        "injection_trigger_manifest": work / "triggers.jsonl",
        "raw_background_manifest": raw_background,
        "mask_background_manifest": mask_background,
    }
    _write_jsonl(
        source_paths["raw_background_candidates"],
        background_candidates("raw"),
    )
    _write_jsonl(
        source_paths["mask_background_candidates"],
        background_candidates("mask"),
    )
    _write_jsonl(
        source_paths["raw_injection_candidates"],
        injection_candidates("raw"),
    )
    _write_jsonl(
        source_paths["mask_injection_candidates"],
        injection_candidates("mask"),
    )
    _write_jsonl(source_paths["injection_trigger_manifest"], trigger_rows)

    subsets = [
        ["H1", "L1"],
        ["H1", "V1"],
        ["L1", "V1"],
        ["H1", "L1", "V1"],
    ]
    limits = {
        "H1+L1": 0.010012846152267725,
        "H1+V1": 0.027287979933397113,
        "L1+V1": 0.02644834101635671,
    }
    slides = [
        {
            "slide_index": 1,
            "slide_id": "locked-network-slide-1",
            "offset_seconds": {"H1": 0.0, "L1": 10.0, "V1": -10.0},
            "eligible_windows_by_detector_subset": {
                "H1+L1": 2,
                "H1+L1+V1": 1,
                "H1+V1": 2,
                "L1+V1": 1,
            },
            "predicted_live_time_seconds": 30.0,
        }
    ]
    schedule_identity = {
        "schema": "independent_symmetric_detector_offsets_v1",
        "split": "test",
        "availability_manifest_sha256": "e" * 64,
        "selection_data": "background_gps_and_detector_availability_only",
        "candidate_scores_inspected": False,
        "detectors": ["H1", "L1", "V1"],
        "detector_subsets": subsets,
        "pairwise_light_travel_time_seconds": limits,
        "cluster_window_seconds": 0.1,
        "maximum_empirical_timing_uncertainty_seconds": 0.01,
        "window_duration_seconds": 10.0,
        "minimum_background_shifts": 1,
        "minimum_test_live_time_years": 1e-8,
        "target_far_per_year": 0.1,
        "slides": slides,
    }
    schedule = {
        "status": "frozen_score_blind_network_time_slide_schedule",
        "passed": True,
        **schedule_identity,
        "slide_count": 1,
        "equivalent_live_time_seconds_predicted": 30.0,
        "equivalent_live_time_years_predicted": 30.0 / 31_557_600.0,
        "eligible_windows_by_detector_subset": slides[0][
            "eligible_windows_by_detector_subset"
        ],
        "schedule_id": canonical_hash(schedule_identity, 32),
        "schedule_sha256": canonical_hash(slides, 64),
    }
    _write_json(schedule_path, schedule)

    suite_inputs = {
        "raw_test_time_slide_report": str(
            (inputs / "raw-time-slides.json").resolve()
        ),
        "mask_test_time_slide_report": str(
            (inputs / "mask-time-slides.json").resolve()
        ),
        "raw_test_background_manifest": str(raw_background.resolve()),
        "mask_test_background_manifest": str(mask_background.resolve()),
        "raw_test_injection_ranking_report": str(
            (inputs / "raw-rankings.json").resolve()
        ),
        "mask_test_injection_ranking_report": str(
            (inputs / "mask-rankings.json").resolve()
        ),
    }
    suite = {
        "status": "frozen_locked_evaluation_suite_plan",
        "passed": True,
        "code_commit": "abc123",
        "inputs": suite_inputs,
        "endpoints": {
            "minimum_test_live_time_years": 1e-8,
            "minimum_background_shifts": 1,
            "minimum_test_injections": 1,
            "required_detector_subsets": [
                "H1+L1",
                "H1+V1",
                "L1+V1",
                "H1+L1+V1",
            ],
        },
    }
    _write_json(suite_path, suite)
    execution = {
        "status": "frozen_locked_o4b_streaming_execution_plan",
        "passed": True,
        "code_commit": "abc123",
        "freeze_identity": {"work_root": str(work.resolve())},
        "network_time_slide_schedule": schedule,
        "network_time_slide_schedule_path": str(schedule_path.resolve()),
        "network_time_slide_schedule_sha256": file_sha256(schedule_path),
        "search_input_reduction_report_path": str(report_path.resolve()),
    }
    _write_json(execution_path, execution)
    access = {
        "status": "locked_evaluation_corpus_opened_once",
        "code_commit": "abc123",
        "frozen_artifacts": {
            "locked_suite_plan": {
                "path": str(suite_path.resolve()),
                "sha256": file_sha256(suite_path),
            },
            "locked_execution_plan": {
                "path": str(execution_path.resolve()),
                "sha256": file_sha256(execution_path),
            },
        },
    }
    _write_json(access_path, access)
    readiness = {
        "minimum_background_gps_blocks": True,
        "minimum_test_injections": True,
        "minimum_injection_gps_blocks": True,
        "minimum_locked_ood_rows": True,
        "minimum_paired_pe_injections": True,
        "required_detector_subsets": True,
        "required_source_families": True,
    }
    merged = {
        "status": "merged_locked_o4b_streaming_suite_input_sources",
        "passed": True,
        "code_commit": "abc123",
        "endpoint_source_readiness": readiness,
        "artifacts": {
            label: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for label, path in source_paths.items()
        },
    }
    _write_json(merge_path, merged)

    result = reduce_locked_o4b_search_inputs(
        suite_path,
        execution_path,
        access_path,
        merge_path,
        "abc123",
    )
    replay = reduce_locked_o4b_search_inputs(
        suite_path,
        execution_path,
        access_path,
        merge_path,
        "abc123",
    )

    assert replay == result
    assert result["passed"] is True
    assert result["raw_mask_shared_schedule"] is True
    assert result["detector_subset_channels_clustered_jointly"] is True
    raw_slide = json.loads(
        Path(suite_inputs["raw_test_time_slide_report"]).read_text(
            encoding="utf-8"
        )
    )
    raw_ranking = json.loads(
        Path(suite_inputs["raw_test_injection_ranking_report"]).read_text(
            encoding="utf-8"
        )
    )
    assert raw_slide["equivalent_live_time_seconds"] == 30.0
    assert raw_slide["slide_schedule_id"] == schedule["schedule_id"]
    assert raw_ranking["ranked_injections"] == 1
    assert raw_ranking["selected_networks_by_detector_subset"] == {
        "H1+L1+V1": 1
    }
