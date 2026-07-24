from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.overlaps import audit_physical_overlap_expansion_capacity

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _hard_endpoint(path: Path, next_scale: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "completed_group_safe_physical_overlap_data_scaling_curve",
                "passed": True,
                "test_rows_read": 0,
                "test_evaluation": None,
                "hard_endpoint_binding": {"passed": True},
                "scale_promotion_authorized": True,
                "authorized_next_physical_scale": next_scale,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_expansion_capacity_separates_new_data_from_detector_set_change(
    tmp_path: Path,
) -> None:
    current = _jsonl(
        tmp_path / "current.jsonl",
        [
            {
                "mixture_id": f"m-{index}",
                "injection_id": f"i-{index}",
                "waveform_id": f"w-{index}",
                "glitch_id": f"g-{index}",
                "split": "train",
                "available_ifos": ["H1", "L1"],
            }
            for index in range(2)
        ],
    )
    glitches = _jsonl(
        tmp_path / "glitches.jsonl",
        [
            {
                "glitch_id": f"g-{index}",
                "split": "train",
                "ifo": "H1",
                "available_ifos": (
                    ["H1", "L1"] if index < 3 else ["H1", "L1", "V1"]
                ),
            }
            for index in range(5)
        ],
    )
    injections = _jsonl(
        tmp_path / "injections.jsonl",
        [
            {
                "injection_id": f"i-{index}",
                "waveform_id": f"w-{index}",
                "split": "train",
                "ifos": ["H1", "L1", "V1"],
            }
            for index in range(5)
        ],
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
                "train_manifest_sha256": file_sha256(glitches),
                "split_audit": {
                    "cross_split_overlaps": {
                        "glitch_id": [],
                        "network_gps_block": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    injection_audit = tmp_path / "injection-audit.json"
    injection_audit.write_text(
        json.dumps(
            {
                "status": "verified_physical_detector_set_expansion",
                "passed": True,
                "test_rows_read": 0,
                "test_evaluation": None,
                "selected_split": "train",
                "same_distribution_data_scaling_claim_allowed": False,
                "manifest_sha256": file_sha256(injections),
            }
        ),
        encoding="utf-8",
    )
    report = audit_physical_overlap_expansion_capacity(
        _hard_endpoint(tmp_path / "hard.json", 4),
        current,
        glitches,
        injections,
        audit,
        tmp_path / "capacity.json",
        seed=7,
        candidate_injection_audit_path=injection_audit,
    )
    assert report["current_physical_groups"] == 2
    assert report["maximum_same_distribution_physical_groups"] == 3
    assert report["maximum_all_detector_set_physical_groups"] == 5
    assert report["same_distribution_capacity_ready"] is False
    assert report["all_detector_set_capacity_ready"] is True
    assert report["next_scale_training_authorized"] is False
    assert (
        report["expansion_mode"]
        == "detector_set_expansion_requires_separate_ablation"
    )
    assert report["inputs"]["candidate_injection_audit"]["sha256"] == file_sha256(
        injection_audit
    )

    injection_audit.write_text(
        injection_audit.read_text(encoding="utf-8").replace(
            file_sha256(injections), "0" * 64
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="detector-expanded injection corpus failed"
    ):
        audit_physical_overlap_expansion_capacity(
            _hard_endpoint(tmp_path / "hard-tamper.json", 4),
            current,
            glitches,
            injections,
            audit,
            tmp_path / "capacity-tamper.json",
            seed=7,
            candidate_injection_audit_path=injection_audit,
        )


def test_expansion_capacity_reports_hand_calculated_new_source_gap(
    tmp_path: Path,
) -> None:
    current_rows = [
        {
            "mixture_id": f"m-{index}",
            "injection_id": f"i-{index}",
            "waveform_id": f"w-{index}",
            "glitch_id": f"g-{index}",
            "split": "train",
            "available_ifos": ["H1", "L1"],
        }
        for index in range(2)
    ]
    current = _jsonl(tmp_path / "current.jsonl", current_rows)
    glitches = _jsonl(
        tmp_path / "glitches.jsonl",
        [
            {
                "glitch_id": f"g-{index}",
                "split": "train",
                "ifo": "H1",
                "available_ifos": ["H1", "L1"],
            }
            for index in range(3)
        ],
    )
    injections = _jsonl(
        tmp_path / "injections.jsonl",
        [
            {
                "injection_id": f"i-{index}",
                "waveform_id": f"w-{index}",
                "split": "train",
                "ifos": ["H1", "L1"],
            }
            for index in range(6)
        ],
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
                "train_manifest_sha256": file_sha256(glitches),
                "split_audit": {"cross_split_overlaps": {"glitch_id": []}},
            }
        ),
        encoding="utf-8",
    )
    report = audit_physical_overlap_expansion_capacity(
        _hard_endpoint(tmp_path / "hard.json", 5),
        current,
        glitches,
        injections,
        audit,
        tmp_path / "capacity.json",
        seed=11,
    )
    assert report["maximum_same_distribution_physical_groups"] == 3
    assert report["maximum_all_detector_set_physical_groups"] == 3
    assert report["minimum_new_detector_compatible_physical_groups"] == 2
    assert report["expansion_mode"] == "new_physical_sources_required"
    assert report["next_scale_training_authorized"] is False


def test_capacity_queue_waits_for_validation_and_never_opens_test_data() -> None:
    script = (
        ROOT / "scripts/queue_physical_overlap_expansion_capacity.sh"
    ).read_text(encoding="utf-8")
    assert 'while [[ ! -s "$HARD_ENDPOINT_REPORT" ]]' in script
    assert "physical-overlap-expansion-capacity" in script
    assert "physical_overlap_expansion_capacity_queue_upstream_incomplete" in script
    assert '"test_rows_read": 0' in script
    assert '"test_evaluation": None' in script
    assert "next-scale training remains unauthorized" in script
    assert "--candidate-injection-audit" in script
