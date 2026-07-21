from __future__ import annotations

import pytest

from gwyolo.ood import (
    build_leave_one_family_out_split,
    calibrate_known_only_abstention,
    class_conditional_mahalanobis_scores,
    evaluate_frozen_ood_threshold,
    fit_class_conditional_mahalanobis,
    ood_auc,
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
