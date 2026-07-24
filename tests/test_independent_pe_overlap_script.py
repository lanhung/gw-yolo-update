from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_independent_pe_overlap.sh"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_pe_overlap_binds_endpoint_detector_sets_and_joint_audit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "VALIDATION_GLITCH_MANIFEST",
        "GRAVITYSPY_CORPUS_AUDIT",
        "TRAIN_OVERLAP_MANIFEST",
        "MINIMUM_OVERLAP_ROWS",
        "frozen_gps_and_purpose_disjoint_validation_endpoint",
        "endpoint_component_reports",
        "excluded_detector_incompatible_glitch_rows",
        "physical-overlap-materialize",
        "physical-overlap-audit",
        "passed_physical_overlap_group_audit",
        "test_rows_read",
    ):
        assert token in source
    assert '--split val' in source
    assert '--manifest "$TRAIN_OVERLAP_MANIFEST"' in source
    assert '--manifest "$overlap_manifest"' in source
    assert "required <= supported" in source


def test_independent_pe_overlap_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_independent_pe_overlap_preflight_excludes_missing_detector_sets(
    tmp_path: Path,
) -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    component_reports = {}
    for label in (
        "purpose_partition",
        "injection_plan",
        "waveform_validation",
        "materialization",
        "snr_annotation",
        "arrival_annotation",
    ):
        path = tmp_path / f"{label}.json"
        path.write_text("{}\n", encoding="utf-8")
        component_reports[label] = {"path": str(path), "sha256": _digest(path)}

    injections = tmp_path / "injections.jsonl"
    _write_jsonl(
        injections,
        [
            {
                "split": "val",
                "injection_id": "independent-inj-1",
                "waveform_id": "independent-wave-1",
                "gps_block": "independent-injection-gps",
                "ifos": ["H1", "L1"],
                "materialized_sha256": "signal-hash-1",
            },
            {
                "split": "val",
                "injection_id": "independent-inj-2",
                "waveform_id": "independent-wave-2",
                "gps_block": "independent-injection-gps-2",
                "ifos": ["H1", "L1"],
                "materialized_sha256": "signal-hash-2",
            },
        ],
    )
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "status": "frozen_gps_and_purpose_disjoint_validation_endpoint",
                "passed": True,
                "test_rows_read": 0,
                "test_evaluation": None,
                "purpose_gps_block_overlap": 0,
                "rows": 2,
                "injection_arrival_manifest_path": str(injections),
                "injection_arrival_manifest_sha256": _digest(injections),
                "component_reports": component_reports,
            }
        ),
        encoding="utf-8",
    )
    glitches = tmp_path / "glitches.jsonl"
    _write_jsonl(
        glitches,
        [
            {
                "split": "val",
                "glitch_id": "val-glitch-hl",
                "ifo": "H1",
                "available_ifos": ["H1", "L1"],
                "network_gps_block": "val-glitch-gps-hl",
            },
            {
                "split": "val",
                "glitch_id": "val-glitch-hlv",
                "ifo": "V1",
                "available_ifos": ["H1", "L1", "V1"],
                "network_gps_block": "val-glitch-gps-hlv",
            },
        ],
    )
    audit = tmp_path / "corpus-audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
                "validation_manifest_sha256": _digest(glitches),
                "split_audit": {"cross_split_overlaps": {"train__val": []}},
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train-overlap.jsonl"
    _write_jsonl(
        train,
        [
            {
                "split": "train",
                "injection_id": "train-inj",
                "waveform_id": "train-wave",
                "glitch_id": "train-glitch",
                "injection_gps_block": "train-injection-gps",
                "network_gps_block": "train-glitch-gps",
                "gravityspy_corpus_audit_sha256": _digest(audit),
            }
        ],
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            snippets[0],
            str(endpoint),
            str(glitches),
            str(audit),
            str(train),
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [str(injections.resolve()), "1", "1"]

    config = tmp_path / "overlap.yaml"
    config.write_text("overlap_factory: {}\n", encoding="utf-8")
    overlap_manifest = tmp_path / "physical_overlap_val_manifest.jsonl"
    _write_jsonl(
        overlap_manifest,
        [
            {
                "split": "val",
                "mixture_id": "mixture-1",
                "injection_id": "independent-inj-1",
                "waveform_id": "independent-wave-1",
                "glitch_id": "val-glitch-hl",
                "injection_materialized_sha256": "signal-hash-1",
                "injection_gps_block": "independent-injection-gps",
                "available_ifos": ["H1", "L1"],
                "gravityspy_corpus_audit_sha256": _digest(audit),
            }
        ],
    )
    overlap_report = tmp_path / "physical_overlap_report.json"
    overlap_report.write_text(
        json.dumps(
            {
                "status": "verified_real_glitch_physical_overlap_training_data",
                "scientific_claim_allowed": False,
                "search_claim_allowed": False,
                "split": "val",
                "rows": 1,
                "manifest_path": str(overlap_manifest),
                "manifest_sha256": _digest(overlap_manifest),
                "gravityspy_manifest_sha256": _digest(glitches),
                "injection_manifest_sha256": _digest(injections),
                "config_sha256": _digest(config),
                "gravityspy_corpus_audit_sha256": _digest(audit),
                "rendered_image_count": 0,
                "unique_physical_counts": {
                    "mixtures": 1,
                    "injections": 1,
                    "waveforms": 1,
                    "glitches": 1,
                },
                "aligned_network_rows": 1,
                "single_ifo_rows": 0,
                "weak_masks": 0,
                "automatic_pseudo_masks": 1,
                "human_pixel_masks": 0,
                "manual_annotation_required": False,
                "automatic_mask_policy": {
                    "human_ground_truth_claimed": False,
                },
                "code_commit": "test-commit",
            }
        ),
        encoding="utf-8",
    )
    joint_audit = tmp_path / "joint-audit.json"
    joint_audit.write_text(
        json.dumps(
            {
                "status": "passed_physical_overlap_group_audit",
                "passed": True,
                "manifest_sha256_by_split": {
                    "train": _digest(train),
                    "val": _digest(overlap_manifest),
                },
                "rows_by_split": {"train": 1, "val": 1},
                "cross_split_overlaps": {
                    "train__val": {"injection_id": [], "glitch_id": []}
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "independent_pe_overlap_report.json"
    finalized = subprocess.run(
        [
            sys.executable,
            "-c",
            snippets[1],
            str(endpoint),
            str(glitches),
            str(audit),
            str(train),
            str(config),
            str(overlap_report),
            str(overlap_manifest),
            str(joint_audit),
            "1",
            "1",
            "1",
            "20260726",
            "test-commit",
            str(receipt),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finalized.returncode == 0, finalized.stderr
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    assert frozen["status"] == "verified_independent_validation_pe_overlap"
    assert frozen["rows"] == 1
    assert frozen["excluded_detector_incompatible_glitch_rows"] == 1
    assert frozen["test_rows_read"] == 0


def test_independent_pe_overlap_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_independent_pe_overlap_rejects_missing_endpoint(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "src" / "gwyolo").mkdir(parents=True)
    task_python = tmp_path / "python"
    task_python.symlink_to(sys.executable)
    existing = {}
    for name in (
        "validation_glitch_manifest",
        "gravityspy_corpus_audit",
        "train_overlap_manifest",
        "materialization_config",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        existing[name.upper()] = str(path)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": str(task_python),
            "TASK_CODE_DIR": str(code),
            "GWYOLO_CODE_COMMIT": "commit",
            "INDEPENDENT_VALIDATION_ENDPOINT_REPORT": str(tmp_path / "missing.json"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            **existing,
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "required independent PE overlap input is absent" in completed.stderr
