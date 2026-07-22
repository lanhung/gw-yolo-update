from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.ood import bind_source_safe_detector_set_ood_validation
from gwyolo.publication import run_publication_evidence_audit


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
                "split_audit": {
                    "cross_split_overlaps": {
                        "source_file": [],
                        "gps_block": [],
                        "glitch_id": [],
                    }
                },
                "train_manifest_sha256": "train-hash",
                "validation_manifest_sha256": "validation-hash",
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "frozen_score_blind_held_glitch_family_protocol",
                "model_scores_used_for_selection": False,
                "unknown_scores_opened_before_selection": False,
                "identity": {
                    "train_manifest_sha256": "train-hash",
                    "validation_manifest_sha256": "validation-hash",
                },
                "base_split_audit": {
                    "passed": True,
                    "cross_split_overlaps": {"source_file": [], "gps_block": []},
                },
                "selected": {"glitch_family": "Blip"},
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split_manifests = {}
    for role in ("known_train", "known_calibration", "heldout_evaluation"):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(json.dumps({"split": "val", "role": role}) + "\n", encoding="utf-8")
        split_manifests[role] = path
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calibration = tmp_path / "known.jsonl"
    calibration.write_text('{"split":"val"}\n', encoding="utf-8")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"split":"val"}\n', encoding="utf-8")
    embedding = tmp_path / "embedding.json"
    embedding.write_text(
        json.dumps(
            {
                "status": "known_family_embedding_heldout_ood_validation",
                "scientific_claim_allowed": False,
                "architecture": "detector_set",
                "ood_score_method": "logit_energy",
                "device": "cuda",
                "test_evaluation": None,
                "auxiliary_policy": (
                    "attribution_or_review_only; cannot veto a strain-coherent candidate"
                ),
                "ood_score_fit": {"heldout_scores_used_for_method_or_fit_selection": False},
                "run_identity": {
                    f"{role}_manifest_sha256": file_sha256(path)
                    for role, path in split_manifests.items()
                },
                "ood_evaluation": {
                    "status": "frozen_known_only_ood_abstention_evaluation",
                    "calibration": {
                        "selection_data": "known_validation_only",
                        "unknown_scores_used_for_selection": False,
                    },
                    "observing_run_strata": {"O2": {"rows": 2}, "O3": {"rows": 3}},
                    "unknown_false_acceptance": {
                        "count": 1,
                        "total": 3,
                        "rate": 1 / 3,
                    },
                },
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "known_calibration_scores_path": str(calibration.resolve()),
                "known_calibration_scores_sha256": file_sha256(calibration),
                "heldout_evaluation_scores_path": str(heldout.resolve()),
                "heldout_evaluation_scores_sha256": file_sha256(heldout),
            }
        ),
        encoding="utf-8",
    )
    split.write_text(
        json.dumps(
            {
                "status": "frozen_leave_one_glitch_family_out_split",
                "held_out_family": "Blip",
                "train_manifest_sha256": "train-hash",
                "validation_manifest_sha256": "validation-hash",
                "split_audit": {
                    "passed": True,
                    "gps_block_overlaps": {
                        "train_calibration": [],
                        "train_evaluation": [],
                        "calibration_evaluation": [],
                    },
                },
                "base_split_audit": {
                    "passed": True,
                    "cross_split_overlaps": {"source_file": [], "gps_block": []},
                },
                "artifacts": {
                    role: _identity(path) for role, path in split_manifests.items()
                },
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "gravityspy_corpus_audit": corpus,
        "held_family_protocol": protocol,
        "split_report": split,
        "embedding_report": embedding,
        "checkpoint": checkpoint,
        "known_calibration_scores": calibration,
        "heldout_evaluation_scores": heldout,
    }
    source = tmp_path / "source-receipt.json"
    source.write_text(
        json.dumps(
            {
                "status": "completed_source_safe_detector_set_ood_validation",
                "passed": True,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "test_evaluation": None,
                "artifacts": {label: _identity(path) for label, path in paths.items()},
            }
        ),
        encoding="utf-8",
    )
    return source, corpus, paths


def test_ood_endpoint_binds_source_safe_chain_and_passes_official_ledger(
    tmp_path: Path,
) -> None:
    source, corpus, _ = _write_inputs(tmp_path)
    endpoint = tmp_path / "endpoint.json"
    result = bind_source_safe_detector_set_ood_validation(source, corpus, endpoint)

    assert result["status"] == "bound_source_safe_detector_set_ood_validation"
    assert result["source_safe_corpus_gate"] is True
    assert result["test_rows_read"] == 0
    assert len(result["artifacts"]) == 11

    protocol = Path(__file__).resolve().parents[1] / "configs/publication_validation_evidence.yaml"
    audit = run_publication_evidence_audit(
        protocol,
        [f"detector_set_ood_transfer={endpoint}"],
        tmp_path / "audit.json",
    )
    gate = next(row for row in audit["requirements"] if row["id"] == "detector_set_ood_transfer")
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 11


def test_ood_endpoint_rejects_changed_score_artifact(tmp_path: Path) -> None:
    source, corpus, paths = _write_inputs(tmp_path)
    paths["heldout_evaluation_scores"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed hash replay"):
        bind_source_safe_detector_set_ood_validation(
            source,
            corpus,
            tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()
