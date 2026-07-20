from gwyolo.candidate_refiner import label_candidate_refiner_rows


def _parent(injection_id: str, split: str, gps: float):
    return {
        "injection_id": injection_id,
        "waveform_id": f"wave-{injection_id}",
        "split": split,
        "gps_block": f"block-{injection_id}",
        "detector_arrival_gps": {"H1": gps, "L1": gps + 0.01},
    }


def _candidate(candidate_id: str, injection_id: str, split: str, ifo: str, start: float):
    return {
        "candidate_id": candidate_id,
        "injection_id": injection_id,
        "waveform_id": f"wave-{injection_id}",
        "split": split,
        "source_family": "BBH",
        "gps_block": f"block-{injection_id}",
        "ifo": ifo,
        "gps_start": start,
        "gps_end": start + 0.1,
        "gps_peak": start + 0.05,
        "proposal_score": 0.7,
    }


def test_candidate_refiner_plan_retains_all_and_counts_coverage_by_hand() -> None:
    parents = [_parent("i1", "val", 100.0)]
    candidates = [
        _candidate("c1", "i1", "val", "H1", 99.8),
        _candidate("c2", "i1", "val", "H1", 103.0),
        _candidate("c3", "i1", "val", "L1", 99.9),
    ]
    rows, report = label_candidate_refiner_rows(
        parents,
        candidates,
        "val",
        positive_padding_seconds=0.5,
        validation_selection_fraction=0.2,
        seed=7,
    )
    assert len(rows) == 3
    assert [row["refiner_positive"] for row in rows] == [True, False, True]
    assert len({row["refiner_role"] for row in rows}) == 1
    assert report["positive_candidates"] == 2
    assert report["negative_candidates"] == 1
    assert report["expected_detector_arrivals"] == 2
    assert report["arrivals_with_positive_candidate"] == 2
    assert report["positive_candidate_coverage_fraction"] == 1.0
    assert report["all_connected_candidates_retained"] is True
