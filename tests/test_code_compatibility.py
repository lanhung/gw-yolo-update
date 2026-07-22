from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gwyolo.code_compatibility import (
    audit_calibration_timing_transfer_compatibility,
    audit_candidate_scoring_implementation_compatibility,
    validate_calibration_timing_transfer_compatibility,
    validate_candidate_scoring_compatibility,
)


def _repository(
    root: Path,
    source: str,
    extra: str = "",
    candidates: str = "",
) -> str:
    package = root / "src" / "gwyolo"
    package.mkdir(parents=True)
    (package / "trigger.py").write_text(source, encoding="utf-8")
    if extra:
        (package / "streaming.py").write_text(extra, encoding="utf-8")
    if candidates:
        (package / "candidates.py").write_text(candidates, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=GW-YOLO test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_candidate_scoring_compatibility_allows_orchestration_only_change(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference_commit = _repository(reference, "SCORER = 1\n", "OLD = True\n")
    candidate_commit = _repository(candidate, "SCORER = 1\n", "NEW = True\n")
    report_path = tmp_path / "compatibility.json"

    report = audit_candidate_scoring_implementation_compatibility(
        reference,
        candidate,
        reference_commit,
        candidate_commit,
        report_path,
    )

    assert report["passed"] is True
    assert report["differences"] == []
    assert report["compared_files"] == 1
    assert (
        validate_candidate_scoring_compatibility(report_path, reference_commit, candidate_commit)
        == report
    )


def test_candidate_scoring_compatibility_retains_mismatch_report(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference_commit = _repository(reference, "SCORER = 1\n")
    candidate_commit = _repository(candidate, "SCORER = 2\n")
    report_path = tmp_path / "compatibility.json"

    with pytest.raises(ValueError, match="differs"):
        audit_candidate_scoring_implementation_compatibility(
            reference,
            candidate,
            reference_commit,
            candidate_commit,
            report_path,
        )

    assert report_path.is_file()
    with pytest.raises(ValueError, match="failed replay"):
        validate_candidate_scoring_compatibility(report_path, reference_commit, candidate_commit)


def test_candidate_scoring_compatibility_normalizes_only_timing_apply(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference_source = """\
def extract_temporal_clusters(value):
    return value + 1

def run_apply_candidate_timing_calibration(value):
    return value
"""
    candidate_source = reference_source.replace(
        "def run_apply_candidate_timing_calibration(value):\n    return value\n",
        'def run_apply_candidate_timing_calibration(value):\n    return {"value": value}\n',
    )
    reference_commit = _repository(reference, "SCORER = 1\n", candidates=reference_source)
    candidate_commit = _repository(candidate, "SCORER = 1\n", candidates=candidate_source)
    report_path = tmp_path / "normalized.json"

    report = audit_candidate_scoring_implementation_compatibility(
        reference,
        candidate,
        reference_commit,
        candidate_commit,
        report_path,
    )

    assert report["passed"] is True
    assert report["normalized_orchestration_functions"] == {
        "src/gwyolo/candidates.py": ["run_apply_candidate_timing_calibration"]
    }

    changed_extraction = tmp_path / "changed-extraction"
    changed_commit = _repository(
        changed_extraction,
        "SCORER = 1\n",
        candidates=candidate_source.replace("value + 1", "value + 2"),
    )
    mismatch = tmp_path / "extraction-mismatch.json"
    with pytest.raises(ValueError, match="differs"):
        audit_candidate_scoring_implementation_compatibility(
            reference,
            changed_extraction,
            reference_commit,
            changed_commit,
            mismatch,
        )


def _timing_candidates_source(offset: int = 1) -> str:
    return f"""\
def _active_runs(value):
    return value

def _parabolic_offset(value):
    return value + {offset}

def extract_temporal_clusters(value):
    return value

def _clusters_from_scored_row(value):
    return value

def build_injection_candidate_rankings(value):
    return value

def _cluster_network_rows(value):
    return value

def run_candidate_block_permutations(value):
    return value

def unrelated_calibration_orchestration(value):
    return value
"""


def test_calibration_timing_transfer_compares_only_predeclared_core_functions(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference_commit = _repository(
        reference,
        "def network_ranking(value):\n    return value\n",
        candidates=_timing_candidates_source(),
    )
    candidate_commit = _repository(
        candidate,
        "def network_ranking(value):\n    return value\n\nCALIBRATION = True\n",
        candidates=_timing_candidates_source().replace(
            "return value\n\ndef unrelated", "return value\n\ndef new_wrapper(value):\n    return value\n\ndef unrelated", 1
        ),
    )
    report_path = tmp_path / "timing-transfer.json"

    report = audit_calibration_timing_transfer_compatibility(
        reference,
        candidate,
        reference_commit,
        candidate_commit,
        report_path,
    )

    assert report["passed"] is True
    assert report["compared_functions"] == 8
    assert (
        validate_calibration_timing_transfer_compatibility(
            report_path, reference_commit, candidate_commit
        )
        == report
    )

    changed = tmp_path / "changed"
    changed_commit = _repository(
        changed,
        "def network_ranking(value):\n    return value\n",
        candidates=_timing_candidates_source(offset=2),
    )
    mismatch = tmp_path / "timing-mismatch.json"
    with pytest.raises(ValueError, match="timing/ranking semantics"):
        audit_calibration_timing_transfer_compatibility(
            reference,
            changed,
            reference_commit,
            changed_commit,
            mismatch,
        )
