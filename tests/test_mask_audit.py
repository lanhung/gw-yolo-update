from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import canonical_hash, file_sha256
from gwyolo.mask_audit import (
    _resolve_mask_checkpoint_selection,
    bind_raw_mask_human_consensus_publication_evidence,
    binary_mask_iou,
    binary_mask_metrics,
    evaluate_gravityspy_mask_audit,
    evaluate_gravityspy_mask_segmentation,
    materialize_gravityspy_mask_consensus,
    plan_gravityspy_mask_audit,
    predict_gravityspy_mask_segmentation,
)


def test_binary_mask_iou_is_hand_calculated() -> None:
    left = np.asarray([1, 1, 0, 0])
    right = np.asarray([1, 0, 1, 0])
    assert binary_mask_iou(left, right) == pytest.approx(1 / 3)


def test_binary_mask_metrics_are_hand_calculated() -> None:
    expected = np.asarray([1, 1, 0, 0])
    predicted = np.asarray([1, 0, 1, 0])
    metrics = binary_mask_metrics(expected, predicted)
    assert metrics == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "iou": pytest.approx(1 / 3),
        "dice": 0.5,
    }


def test_mask_audit_plan_requires_three_blinded_annotators(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    np.savez(
        sample,
        features=np.asarray([[0.1, 0.8], [0.2, 0.0]], dtype=np.float32),
        glitch_mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
    )
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "glitch_id": "g1",
                "ml_label": "Blip",
                "ifo": "H1",
                "observing_run": "O3a",
                "network_gps_block": "O3a:1",
                "path": str(sample),
                "sha256": file_sha256(sample),
            }
        )
        + "\n"
    )
    report = plan_gravityspy_mask_audit(manifest, tmp_path / "audit", per_label=1)
    task = json.loads(Path(report["task_manifest_path"]).read_text())
    annotation_task = json.loads(
        Path(report["annotation_task_manifest_path"]).read_text()
    )
    assert task["required_independent_annotators"] == 3
    assert "three independent" in report["scientific_blocker"]
    assert "exact_command" in report and "environment" in report
    assert "numeric_sample_path" not in annotation_task
    assert "weak_mask_key" not in annotation_task
    assert task["annotation_task_hash"] == annotation_task["annotation_task_hash"]
    assert report["mask_targets_exposed_to_annotators"] is False
    with np.load(annotation_task["blinded_input_path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"features"}
    assert binary_mask_iou(np.zeros(3), np.zeros(3)) == 1.0


def test_mask_audit_requires_blinded_independent_annotations(tmp_path) -> None:
    weak = tmp_path / "weak.npz"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    third = tmp_path / "third.npz"
    np.savez(weak, glitch_mask=np.asarray([1, 1, 0, 0]))
    blind = tmp_path / "blind.npz"
    np.savez(blind, features=np.asarray([0.4, 0.2, 0.8, 0.1]))
    np.savez(first, mask=np.asarray([1, 0, 1, 0]))
    np.savez(second, mask=np.asarray([1, 0, 1, 0]))
    np.savez(third, mask=np.asarray([1, 0, 1, 0]))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "audit_id": "a1",
                "glitch_id": "g1",
                "ml_label": "Blip",
                "numeric_sample_path": str(weak),
                "numeric_sample_sha256": file_sha256(weak),
                "weak_mask_key": "glitch_mask",
                "blinded_input_path": str(blind),
                "blinded_input_sha256": file_sha256(blind),
                "blinded_input_keys": ["features"],
                "mask_shape": [4],
                "required_independent_annotators": 3,
                "required_annotation_key": "mask",
            }
        )
        + "\n"
    )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        "".join(
            json.dumps(
                {
                    "audit_id": "a1",
                    "annotator_id": annotator,
                    "mask_path": str(path),
                    "mask_sha256": file_sha256(path),
                    "blinded_to_weak_mask": True,
                    "protocol_version": "v1",
                    "annotation_task_hash": "task-hash-v1",
                }
            )
            + "\n"
            for annotator, path in (("one", first), ("two", second), ("three", third))
        )
    )
    task_row = json.loads(tasks.read_text())
    task_row["annotation_task_hash"] = "task-hash-v1"
    tasks.write_text(json.dumps(task_row) + "\n")
    report = evaluate_gravityspy_mask_audit(
        tasks, annotations, tmp_path / "report.json"
    )
    assert report["overall"]["mean_interannotator_iou"] == 1.0
    assert report["overall"]["weak_consensus_iou_mean"] == pytest.approx(1 / 3)
    consensus = materialize_gravityspy_mask_consensus(
        tasks, annotations, tmp_path / "report.json", tmp_path / "consensus"
    )
    row = json.loads(Path(consensus["manifest_path"]).read_text())
    with np.load(row["path"], allow_pickle=False) as arrays:
        assert arrays["mask"].tolist() == [1, 0, 1, 0]
    assert consensus["training_allowed"] is False
    assert row["human_pixel_mask"] is True
    assert row["training_allowed"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        materialize_gravityspy_mask_consensus(
            tasks, annotations, tmp_path / "report.json", tmp_path / "consensus"
        )

    tampered_report = json.loads((tmp_path / "report.json").read_text())
    tampered_report["evaluated_tasks"][0]["weak_consensus_iou"] = 1.0
    tampered_path = tmp_path / "tampered-report.json"
    tampered_path.write_text(json.dumps(tampered_report))
    with pytest.raises(ValueError, match="metrics differ"):
        materialize_gravityspy_mask_consensus(
            tasks, annotations, tampered_path, tmp_path / "tampered-consensus"
        )

    annotation_rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    annotation_rows[-1]["protocol_version"] = "v2"
    mixed = tmp_path / "mixed-protocol.jsonl"
    mixed.write_text("".join(json.dumps(row) + "\n" for row in annotation_rows))
    with pytest.raises(ValueError, match="mix protocol versions"):
        evaluate_gravityspy_mask_audit(
            tasks, mixed, tmp_path / "mixed-protocol-report.json"
        )

    annotation_rows[-1]["protocol_version"] = "v1"
    annotation_rows[-1]["annotator_id"] = "two"
    repeated = tmp_path / "repeated-annotator.jsonl"
    repeated.write_text("".join(json.dumps(row) + "\n" for row in annotation_rows))
    with pytest.raises(ValueError, match="lacks independent annotators"):
        evaluate_gravityspy_mask_audit(
            tasks, repeated, tmp_path / "repeated-annotator-report.json"
        )


def test_human_consensus_segmentation_is_validation_only_and_hash_bound(
    tmp_path: Path,
) -> None:
    gold_masks = []
    prediction_masks = []
    for index, (gold, probability) in enumerate(
        (
            ([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2]),
            ([0, 0, 0, 0], [0.1, 0.2, 0.1, 0.0]),
        )
    ):
        gold_path = tmp_path / f"gold-{index}.npz"
        prediction_path = tmp_path / f"prediction-{index}.npz"
        np.savez(gold_path, mask=np.asarray(gold, dtype=np.uint8))
        np.savez(
            prediction_path,
            mask_probability=np.asarray(probability, dtype=np.float32),
        )
        gold_masks.append(gold_path)
        prediction_masks.append(prediction_path)

    gold_manifest = tmp_path / "gold.jsonl"
    gold_rows = [
        {
            "audit_id": f"a{index}",
            "glitch_id": f"g{index}",
            "ml_label": label,
            "split": "val",
            "training_allowed": False,
            "human_pixel_mask": True,
            "mask_key": "mask",
            "mask_shape": [4],
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for index, (label, path) in enumerate(
            zip(("Blip", "Tomte"), gold_masks)
        )
    ]
    gold_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8"
    )
    gold_report = tmp_path / "gold-report.json"
    gold_report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_human_consensus_mask_bank",
                "training_allowed": False,
                "tasks": 2,
                "manifest_path": str(gold_manifest),
                "manifest_sha256": file_sha256(gold_manifest),
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    config = tmp_path / "config.yaml"
    selection = tmp_path / "selection.json"
    model.write_bytes(b"model")
    config.write_text("model: dual-mask\n", encoding="utf-8")
    selection.write_text('{"split":"val"}\n', encoding="utf-8")
    prediction_rows = [
        {
            "audit_id": gold["audit_id"],
            "glitch_id": gold["glitch_id"],
            "split": "val",
            "path": str(path),
            "sha256": file_sha256(path),
            "mask_key": "mask_probability",
            "threshold": 0.5,
            "checkpoint_selection_split": "val",
            "threshold_selection_split": "val",
            "model_path": str(model),
            "model_sha256": file_sha256(model),
            "config_path": str(config),
            "config_sha256": file_sha256(config),
            "checkpoint_selection_report_path": str(selection),
            "checkpoint_selection_report_sha256": file_sha256(selection),
            "threshold_selection_report_path": str(selection),
            "threshold_selection_report_sha256": file_sha256(selection),
        }
        for gold, path in zip(gold_rows, prediction_masks)
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    output = tmp_path / "segmentation.json"
    report = evaluate_gravityspy_mask_segmentation(
        gold_report, predictions, output, bootstrap_replicates=500, bootstrap_seed=7
    )
    assert report["test_evaluation"] is False
    assert report["promotion_evidence_allowed"] is True
    assert report["overall"]["macro"]["iou"] == pytest.approx(2 / 3)
    assert report["overall"]["macro"]["dice"] == pytest.approx(0.75)
    assert report["overall"]["pooled_pixels"]["iou"] == pytest.approx(1 / 3)
    assert report["overall"]["pooled_pixels"]["true_negative"] == 5
    assert report["overall"]["iou_ge_0_5"] == 1
    assert report["overall"]["macro_paired_task_bootstrap_95"]["iou"] == [
        pytest.approx(1 / 3),
        1.0,
    ]
    assert report["by_label"]["Blip"]["macro_paired_task_bootstrap_95"][
        "iou"
    ] == [None, None]
    with pytest.raises(FileExistsError, match="immutable"):
        evaluate_gravityspy_mask_segmentation(gold_report, predictions, output)

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(json.dumps(prediction_rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_gravityspy_mask_segmentation(
            gold_report, incomplete, tmp_path / "incomplete-report.json"
        )

    prediction_rows[0]["threshold_selection_split"] = "test"
    test_selected = tmp_path / "test-selected.jsonl"
    test_selected.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation-only"):
        evaluate_gravityspy_mask_segmentation(
            gold_report, test_selected, tmp_path / "test-selected-report.json"
        )

    prediction_rows[0]["threshold_selection_split"] = "val"
    selection.write_text('{"test_metrics":{"iou":1.0}}\n', encoding="utf-8")
    for row in prediction_rows:
        row["checkpoint_selection_report_sha256"] = file_sha256(selection)
        row["threshold_selection_report_sha256"] = file_sha256(selection)
    test_report_selected = tmp_path / "test-report-selected.jsonl"
    test_report_selected.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains test selection"):
        evaluate_gravityspy_mask_segmentation(
            gold_report,
            test_report_selected,
            tmp_path / "test-report-selected-output.json",
        )


def test_raw_mask_publication_binding_requires_frozen_human_consensus_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", commit)
    task_rows = []
    annotation_rows = []
    for index in range(2):
        weak = tmp_path / f"weak-{index}.npz"
        blind = tmp_path / f"blind-{index}.npz"
        np.savez(
            weak,
            features=np.asarray([0.8, 0.7, 0.1, 0.0]),
            glitch_mask=np.asarray([1, 1, 0, 0]),
        )
        np.savez(blind, features=np.asarray([0.8, 0.7, 0.1, 0.0]))
        task_hash = f"task-hash-{index}"
        task_rows.append(
            {
                "audit_id": f"a{index}",
                "glitch_id": f"g{index}",
                "ml_label": "Blip",
                "numeric_sample_path": str(weak),
                "numeric_sample_sha256": file_sha256(weak),
                "weak_mask_key": "glitch_mask",
                "blinded_input_path": str(blind),
                "blinded_input_sha256": file_sha256(blind),
                "blinded_input_keys": ["features"],
                "mask_shape": [4],
                "required_independent_annotators": 3,
                "required_annotation_key": "mask",
                "annotation_task_hash": task_hash,
            }
        )
        for annotator in ("one", "two", "three"):
            mask = tmp_path / f"human-{index}-{annotator}.npz"
            np.savez(mask, mask=np.asarray([1, 1, 0, 0]))
            annotation_rows.append(
                {
                    "audit_id": f"a{index}",
                    "annotator_id": annotator,
                    "mask_path": str(mask),
                    "mask_sha256": file_sha256(mask),
                    "blinded_to_weak_mask": True,
                    "protocol_version": "v1",
                    "annotation_task_hash": task_hash,
                }
            )
    tasks = tmp_path / "tasks.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    tasks.write_text("".join(json.dumps(row) + "\n" for row in task_rows))
    annotations.write_text(
        "".join(json.dumps(row) + "\n" for row in annotation_rows)
    )
    audit_path = tmp_path / "audit.json"
    evaluate_gravityspy_mask_audit(tasks, annotations, audit_path)
    gold = materialize_gravityspy_mask_consensus(
        tasks, annotations, audit_path, tmp_path / "gold"
    )
    gold_rows = [
        json.loads(line)
        for line in Path(gold["manifest_path"]).read_text().splitlines()
    ]

    model = tmp_path / "model.pt"
    config = tmp_path / "model.yaml"
    selection = tmp_path / "selection.json"
    model.write_bytes(b"model")
    config.write_text("model: detector-set\n")
    selection.write_text('{"selection_split":"val"}\n')
    prediction_rows = []
    for row in gold_rows:
        prediction = tmp_path / f"prediction-{row['audit_id']}.npz"
        np.savez(
            prediction,
            mask_probability=np.asarray([0.9, 0.9, 0.1, 0.1], dtype=np.float32),
        )
        prediction_rows.append(
            {
                "audit_id": row["audit_id"],
                "glitch_id": row["glitch_id"],
                "split": "val",
                "path": str(prediction),
                "sha256": file_sha256(prediction),
                "mask_key": "mask_probability",
                "threshold": 0.5,
                "checkpoint_selection_split": "val",
                "threshold_selection_split": "val",
                "model_path": str(model),
                "model_sha256": file_sha256(model),
                "config_path": str(config),
                "config_sha256": file_sha256(config),
                "checkpoint_selection_report_path": str(selection),
                "checkpoint_selection_report_sha256": file_sha256(selection),
                "threshold_selection_report_path": str(selection),
                "threshold_selection_report_sha256": file_sha256(selection),
            }
        )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows)
    )
    segmentation_path = tmp_path / "segmentation.json"
    evaluate_gravityspy_mask_segmentation(
        tmp_path / "gold" / "gravityspy_human_consensus_mask_report.json",
        predictions,
        segmentation_path,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    artifact_names = (
        "source_background_receipt",
        "background_plan_authorization",
        "parent_plan",
        "merge_report",
        "paired_validation_comparison",
        "mask_validation_receipt",
        "mask_timing_receipt",
        "raw_arm_merge",
        "mask_arm_merge",
        "raw_calibration",
        "mask_calibration",
    )
    artifacts = {}
    for name in artifact_names:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}))
        artifacts[name] = {"path": str(path), "sha256": file_sha256(path)}
    raw_endpoint = tmp_path / "raw-mask-endpoint.json"
    raw_endpoint.write_text(
        json.dumps(
            {
                "status": "bound_validation_raw_mask_continuous_background_evidence",
                "passed": True,
                "mask_locked_test_arm_eligible": True,
                "scientific_claim_allowed": False,
                "locked_test_prerequisites_satisfied": False,
                "test_rows_read": 0,
                "code_commit": commit,
                **{
                    field: artifacts[field]
                    for field in (
                        "source_background_receipt",
                        "background_plan_authorization",
                        "parent_plan",
                        "merge_report",
                        "paired_validation_comparison",
                        "mask_validation_receipt",
                        "mask_timing_receipt",
                    )
                },
                "arm_merges": {
                    "raw": artifacts["raw_arm_merge"],
                    "mask": artifacts["mask_arm_merge"],
                },
                "calibrations": {
                    "raw": artifacts["raw_calibration"],
                    "mask": artifacts["mask_calibration"],
                },
            }
        )
    )
    passing_config = tmp_path / "passing-gate.yaml"
    passing_config.write_text(
        json.dumps(
            {
                "human_mask_publication_gate": {
                    "schema": "human_mask_publication_gate_v1",
                    "protocol": "frozen-before-results",
                    "minimum_tasks": 2,
                    "minimum_unique_glitches": 2,
                    "minimum_labels": 1,
                    "minimum_well_supported_labels": 1,
                    "minimum_support_for_well_supported_label": 2,
                    "minimum_bootstrap_replicates": 100,
                    "minimum_macro_iou_lower_95": 0.9,
                    "minimum_iou_ge_0_5_wilson_lower_95": 0.3,
                    "require_validation_only": True,
                    "require_three_blinded_independent_annotators": True,
                    "require_complete_prediction_coverage": True,
                }
            }
        )
    )
    bound = bind_raw_mask_human_consensus_publication_evidence(
        raw_endpoint,
        segmentation_path,
        passing_config,
        tmp_path / "bound.json",
    )
    assert bound["passed"] is True
    assert bound["observed"] == {
        "tasks": 2,
        "unique_glitches": 2,
        "labels": 1,
        "well_supported_labels": 1,
        "well_supported_label_names": ["Blip"],
        "under_supported_labels": {},
        "bootstrap_replicates": 100,
        "macro_iou_lower_95": 1.0,
        "iou_ge_0_5_wilson_lower_95": pytest.approx(0.3423802275),
    }

    failing_config = tmp_path / "failing-gate.yaml"
    failed_gate = json.loads(passing_config.read_text())
    failed_gate["human_mask_publication_gate"][
        "minimum_iou_ge_0_5_wilson_lower_95"
    ] = 0.5
    failing_config.write_text(json.dumps(failed_gate))
    failed = bind_raw_mask_human_consensus_publication_evidence(
        raw_endpoint,
        segmentation_path,
        failing_config,
        tmp_path / "failed.json",
    )
    assert failed["passed"] is False
    assert failed["checks"]["minimum_iou_ge_0_5_wilson_lower_95"] is False


def test_detector_set_champion_exports_gold_task_probabilities(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from gwyolo.numeric import DetectorSetQNet

    config = {
        "overlap_training": {
            "model_ifos": ["H1", "L1"],
            "q_values": [4.0],
            "tensor": {"frequency_bins": 4, "time_bins": 4},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model = DetectorSetQNet(2, 1, base_channels=2)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "detector_set",
            "model_ifos": ["H1", "L1"],
            "q_values": [4.0],
            "input_channels": 2,
            "base_channels": 2,
            "config_hash": canonical_hash(config),
        },
        checkpoint,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "validation_selected_real_glitch_overlap_finetune",
                "seed": 7,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "config_hash": canonical_hash(config),
                "config_file_sha256": file_sha256(config_path),
                "validation_selected_thresholds": {"chirp": 0.4, "glitch": 0.5},
            }
        ),
        encoding="utf-8",
    )
    numeric = tmp_path / "numeric.npz"
    features = np.zeros((2, 1, 4, 4), dtype=np.float32)
    features[0] = 0.25
    np.savez(
        numeric,
        features=features,
        detector_availability=np.asarray([1, 0], dtype=np.uint8),
        ifos=np.asarray(["H1", "L1"]),
        q_values=np.asarray([4.0], dtype=np.float32),
    )
    gold_mask = tmp_path / "gold-mask.npz"
    np.savez(gold_mask, mask=np.zeros((2, 1, 4, 4), dtype=np.uint8))
    gold_manifest = tmp_path / "gold-manifest.jsonl"
    gold_manifest.write_text(
        json.dumps(
            {
                "audit_id": "a1",
                "glitch_id": "g1",
                "ml_label": "Blip",
                "split": "val",
                "training_allowed": False,
                "human_pixel_mask": True,
                "mask_key": "mask",
                "mask_shape": [2, 1, 4, 4],
                "path": str(gold_mask),
                "sha256": file_sha256(gold_mask),
                "numeric_sample_path": str(numeric),
                "numeric_sample_sha256": file_sha256(numeric),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold_report = tmp_path / "gold-report.json"
    gold_report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_human_consensus_mask_bank",
                "training_allowed": False,
                "tasks": 1,
                "manifest_path": str(gold_manifest),
                "manifest_sha256": file_sha256(gold_manifest),
            }
        ),
        encoding="utf-8",
    )
    prediction = predict_gravityspy_mask_segmentation(
        gold_report, selection, config_path, tmp_path / "prediction"
    )
    assert prediction["tasks"] == 1
    assert prediction["test_evaluation"] is False
    row = json.loads(Path(prediction["prediction_manifest_path"]).read_text())
    with np.load(row["path"], allow_pickle=False) as arrays:
        probability = arrays["mask_probability"]
        assert probability.shape == (2, 1, 4, 4)
        assert np.all(probability[1] < 1e-6)
    evaluated = evaluate_gravityspy_mask_segmentation(
        gold_report,
        prediction["prediction_manifest_path"],
        tmp_path / "evaluation.json",
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert evaluated["tasks"] == 1
    assert evaluated["model_sha256"] == file_sha256(checkpoint)


def test_five_seed_mask_selection_resolves_its_exact_threshold_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"selected-model")
    selected_report = tmp_path / "seed-7.json"
    selected_report.write_text(
        json.dumps(
            {
                "status": "validation_selected_real_glitch_overlap_finetune",
                "seed": 7,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "config_hash": "config-hash",
                "config_file_sha256": "config-file-sha",
                "validation_selected_thresholds": {"glitch": 0.4},
            }
        ),
        encoding="utf-8",
    )
    other_report = tmp_path / "seed-8.json"
    other_report.write_text(
        json.dumps(
            {
                "status": "validation_selected_real_glitch_overlap_finetune",
                "seed": 8,
                "checkpoint_sha256": "another-checkpoint",
                "validation_selected_thresholds": {"glitch": 0.7},
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "five-seed.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed_five_seed_source_safe_overlap_validation",
                "passed": True,
                "test_data_opened": False,
                "selected_seed": 7,
                "selected_checkpoint_path": str(checkpoint),
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "finetune_reports": [
                    {
                        "path": str(selected_report),
                        "sha256": file_sha256(selected_report),
                    },
                    {"path": str(other_report), "sha256": file_sha256(other_report)},
                ],
            }
        ),
        encoding="utf-8",
    )
    resolved = _resolve_mask_checkpoint_selection(summary)
    assert resolved["checkpoint_sha256"] == file_sha256(checkpoint)
    assert resolved["threshold"] == 0.4
    assert resolved["threshold_report_sha256"] == file_sha256(selected_report)
