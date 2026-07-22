from __future__ import annotations

import pytest

from gwyolo.ood import (
    DetectorSetGlitchOODDataset,
    build_leave_one_family_out_split,
    calibrate_known_only_abstention,
    class_conditional_mahalanobis_scores,
    evaluate_frozen_ood_threshold,
    fit_class_conditional_mahalanobis,
    freeze_ood_held_family_protocol,
    ood_auc,
    run_locked_ood_transfer_evaluation,
    supervised_contrastive_loss,
)


def _row(
    glitch_id: str,
    gps_block: str,
    family: str,
    score: float,
    unknown: bool,
    split: str,
) -> dict:
    return {
        "glitch_id": glitch_id,
        "gps_block": gps_block,
        "glitch_family": family,
        "observing_run": "O4a" if unknown else "O3b",
        "ood_score": score,
        "is_unknown": unknown,
        "split": split,
    }


def test_known_only_ood_calibration_is_tie_safe() -> None:
    result = calibrate_known_only_abstention([0.1, 0.2, 0.2, 0.3], 0.5)
    assert result["maximum_known_abstentions"] == 2
    assert result["threshold"] == 0.3
    assert result["observed_known_abstentions"] == 1
    assert result["unknown_scores_used_for_selection"] is False


def test_ood_auc_pair_count_handles_ties_by_hand() -> None:
    rows = [
        {"ood_score": 0.1, "is_unknown": False},
        {"ood_score": 0.4, "is_unknown": False},
        {"ood_score": 0.4, "is_unknown": True},
        {"ood_score": 0.8, "is_unknown": True},
    ]
    # Pair wins: 1 + 0.5 + 1 + 1 = 3.5 of four.
    assert ood_auc(rows) == 0.875


def test_class_conditional_mahalanobis_scores_by_hand() -> None:
    import numpy as np

    train = np.asarray([[0.0], [2.0], [8.0], [10.0]])
    targets = np.asarray([0, 0, 1, 1])
    fit = fit_class_conditional_mahalanobis(
        train, targets, class_count=2, shrinkage=0.0, epsilon=1.0
    )
    # Centers are 1 and 9; pooled within-class variance is 2 and epsilon makes it 3.
    scores = class_conditional_mahalanobis_scores(
        np.asarray([[1.0], [4.0], [9.0]]), fit
    )
    assert scores.tolist() == pytest.approx([0.0, 3.0, 0.0])


def test_class_conditional_mahalanobis_rejects_missing_class() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="every contiguous known class"):
        fit_class_conditional_mahalanobis(
            np.asarray([[0.0], [1.0], [2.0]]),
            np.asarray([0, 0, 0]),
            class_count=2,
        )


def test_supervised_contrastive_loss_by_hand() -> None:
    torch = pytest.importorskip("torch")

    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    measured = supervised_contrastive_loss(embeddings, targets, temperature=1.0)
    expected = torch.log(torch.exp(torch.tensor(1.0)) + 2.0) - 1.0
    assert float(measured) == pytest.approx(float(expected), abs=1e-7)


def test_supervised_contrastive_loss_ignores_anchors_without_positive_pairs() -> None:
    torch = pytest.importorskip("torch")

    embeddings = torch.eye(3)
    targets = torch.tensor([0, 1, 2])
    measured = supervised_contrastive_loss(embeddings, targets)
    assert float(measured) == 0.0


def test_frozen_ood_evaluation_reports_false_acceptance_and_leakage() -> None:
    calibration = [
        _row("c0", "cb0", "Blip", 0.1, False, "val"),
        _row("c1", "cb1", "Blip", 0.2, False, "val"),
        _row("c2", "cb2", "Koi_Fish", 0.3, False, "val"),
        _row("c3", "cb3", "Koi_Fish", 0.4, False, "val"),
    ]
    evaluation = [
        _row("e0", "eb0", "Blip", 0.1, False, "test"),
        _row("e1", "eb1", "Koi_Fish", 0.5, False, "test"),
        _row("u0", "ub0", "Held_Out", 0.8, True, "test"),
        _row("u1", "ub1", "Held_Out", 0.2, True, "test"),
    ]
    result = evaluate_frozen_ood_threshold(calibration, evaluation, 0.25)
    assert result["calibration"]["threshold"] == 0.4
    assert result["known_false_abstention"]["rate"] == 0.5
    assert result["unknown_true_abstention"]["rate"] == 0.5
    assert result["unknown_false_acceptance"]["rate"] == 0.5
    assert result["calibration"]["unknown_scores_used_for_selection"] is False
    leaked = [*evaluation]
    leaked[0] = {**leaked[0], "glitch_id": "c0"}
    with pytest.raises(ValueError, match="group leakage"):
        evaluate_frozen_ood_threshold(calibration, leaked, 0.25)


def test_locked_ood_transfer_reuses_validation_threshold_by_hand(
    tmp_path, monkeypatch
) -> None:
    import json

    from gwyolo.io import file_sha256

    endpoints = {"minimum_locked_ood_rows": 4}

    def binding(_plan, _access, output_key, output_path):
        return {
            "output_key": output_key,
            "output_path": str(output_path.resolve()),
            "endpoints": endpoints,
        }

    monkeypatch.setattr(
        "gwyolo.evaluation_lock.validate_locked_evaluation_suite_access", binding
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "".join(
            json.dumps(_row(f"c{i}", f"cb{i}", "Known", score, False, "val"))
            + "\n"
            for i, score in enumerate((0.1, 0.2, 0.3, 0.4))
        ),
        encoding="utf-8",
    )
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps(_row("v0", "vb0", "Held", 0.8, True, "val")) + "\n",
        encoding="utf-8",
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "known_family_embedding_heldout_ood_validation",
                "architecture": "detector_set",
                "ood_score_method": "logit_energy",
                "test_evaluation": None,
                "ood_score_fit": {
                    "heldout_scores_used_for_method_or_fit_selection": False
                },
                "ood_evaluation": {
                    "status": "frozen_known_only_ood_abstention_evaluation",
                    "calibration": {
                        "threshold": 0.4,
                        "selection_data": "known_validation_only",
                        "unknown_scores_used_for_selection": False,
                    },
                },
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "known_calibration_scores_path": str(calibration),
                "known_calibration_scores_sha256": file_sha256(calibration),
                "heldout_evaluation_scores_path": str(heldout),
                "heldout_evaluation_scores_sha256": file_sha256(heldout),
            }
        ),
        encoding="utf-8",
    )
    locked_rows = [
        _row("t0", "tb0", "Known", 0.1, False, "test"),
        _row("t1", "tb1", "Known", 0.5, False, "test"),
        _row("u0", "ub0", "New", 0.8, True, "test"),
        _row("u1", "ub1", "New", 0.2, True, "test"),
    ]
    for row, ifos in zip(
        locked_rows,
        (("H1", "L1"), ("H1", "V1"), ("H1", "L1", "V1"), ("L1", "V1")),
    ):
        row.update(
            {
                "observing_run": "O4b",
                "available_ifos": list(ifos),
                "embedding_checkpoint_sha256": file_sha256(checkpoint),
                "ood_score_method": "logit_energy",
            }
        )
    locked = tmp_path / "locked.jsonl"
    locked.write_text(
        "".join(json.dumps(row) + "\n" for row in locked_rows), encoding="utf-8"
    )
    result = run_locked_ood_transfer_evaluation(
        validation,
        locked,
        tmp_path / "plan.json",
        tmp_path / "access.json",
        tmp_path / "result.json",
    )
    assert result["threshold"] == 0.4
    assert result["threshold_refits_on_test"] == 0
    assert result["known_false_abstention"]["rate"] == 0.5
    assert result["unknown_false_acceptance"]["rate"] == 0.5
    assert len(result["detector_subset_strata"]) == 4


def test_leave_one_family_out_split_excludes_whole_gps_blocks(tmp_path) -> None:
    import json

    train = [
        {"glitch_id": "t0", "network_gps_block": "tb0", "ml_label": "Held", "observing_run": "O3b", "split": "train"},
        {"glitch_id": "t1", "network_gps_block": "tb0", "ml_label": "Known", "observing_run": "O3b", "split": "train"},
        {"glitch_id": "t2", "network_gps_block": "tb1", "ml_label": "Known", "observing_run": "O3b", "split": "train"},
    ]
    validation = [
        {"glitch_id": "v0", "network_gps_block": "vb0", "ml_label": "Held", "observing_run": "O3b", "split": "val"},
        {"glitch_id": "v1", "network_gps_block": "vb1", "ml_label": "Known", "observing_run": "O3b", "split": "val"},
        {"glitch_id": "v2", "network_gps_block": "vb2", "ml_label": "Known", "observing_run": "O3b", "split": "val"},
    ]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "val.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train), encoding="utf-8")
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8"
    )
    result = build_leave_one_family_out_split(
        train_path, validation_path, "Held", tmp_path / "out", seed=1
    )
    known_train = [
        json.loads(line)
        for line in (tmp_path / "out" / "known_train.jsonl").read_text().splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in (tmp_path / "out" / "heldout_evaluation.jsonl").read_text().splitlines()
    ]
    assert {row["glitch_id"] for row in known_train} == {"t2"}
    assert any(row["is_unknown"] for row in evaluation)
    assert any(not row["is_unknown"] for row in evaluation)
    assert result["split_audit"]["passed"] is True


def test_detector_set_ood_dataset_requires_matching_explicit_availability(
    tmp_path,
) -> None:
    import numpy as np

    from gwyolo.io import file_sha256

    sample = tmp_path / "network.npz"
    np.savez_compressed(
        sample,
        features=np.zeros((3, 2, 5, 4), dtype=np.float32),
        detector_availability=np.asarray([1, 0, 1], dtype=np.uint8),
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4.0, 8.0], dtype=np.float32),
    )
    row = {
        "glitch_id": "g0",
        "path": str(sample),
        "sha256": file_sha256(sample),
        "aligned_network_context": True,
        "detector_availability": [1, 0, 1],
        "available_ifos": ["H1", "V1"],
        "ifo": "H1",
        "glitch_family": "Known",
    }
    dataset = DetectorSetGlitchOODDataset(
        [row], ("H1", "L1", "V1"), (4.0, 8.0), {"Known": 0}
    )
    features, availability, target = dataset[0]
    assert features.shape == (3, 2, 5, 4)
    assert availability.tolist() == [1.0, 0.0, 1.0]
    assert target == 0
    dataset.rows[0] = {**row, "detector_availability": [1, 1, 0]}
    dataset.cache = [None]
    with pytest.raises(ValueError, match="row/array detector availability differs"):
        dataset[0]


def test_held_family_freeze_is_score_blind_and_excludes_opened_families(
    tmp_path,
) -> None:
    import json

    def rows(split: str, family: str, count: int, block_prefix: str) -> list[dict]:
        return [
            {
                "glitch_id": f"{split}-{family}-{index}",
                "network_gps_block": f"{block_prefix}-{index // 2}",
                "ml_label": family,
                "observing_run": "O3b",
                "split": split,
                # This deliberately adversarial field must not enter selection.
                "ood_score": 1_000.0 if family == "Smaller" else -1_000.0,
            }
            for index in range(count)
        ]

    train = rows("train", "Opened", 8, "to")
    train += rows("train", "Larger", 9, "tl")
    train += rows("train", "Smaller", 10, "ts")
    validation = rows("val", "Opened", 8, "vo")
    validation += rows("val", "Larger", 7, "vl")
    validation += rows("val", "Smaller", 6, "vs")
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "val.jsonl"
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train), encoding="utf-8"
    )
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8"
    )
    result = freeze_ood_held_family_protocol(
        train_path,
        validation_path,
        tmp_path / "protocol.json",
        excluded_families=["Opened"],
        minimum_train_rows=5,
        minimum_validation_rows=5,
        minimum_validation_gps_blocks=3,
    )
    assert result["selected"]["glitch_family"] == "Larger"
    assert result["model_scores_used_for_selection"] is False
    assert result["identity"]["excluded_families"] == ["Opened"]
