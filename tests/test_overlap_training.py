from __future__ import annotations

import json

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.overlap_training import (
    PhysicalOverlapDataset,
    _train_epoch,
    glitch_family_sampling_weights,
    overlap_training_split_audit,
    promote_overlap_sampling_arm,
    resolve_overlap_training_control,
    summarize_overlap_five_seed_promotion,
    summarize_glitch_family_counts,
    summarize_physical_overlap_data_scaling,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _row(split: str, suffix: str) -> dict:
    return {
        "split": split,
        "mixture_id": f"m-{suffix}",
        "injection_id": f"i-{suffix}",
        "waveform_id": f"w-{suffix}",
        "glitch_id": f"g-{suffix}",
        "injection_gps_block": f"ib-{suffix}",
        "network_gps_block": f"gb-{suffix}",
    }


def test_overlap_training_split_audit_covers_both_physical_group_families() -> None:
    report = overlap_training_split_audit([_row("train", "a")], [_row("val", "b")])
    assert report["passed"]
    assert all(not values for values in report["cross_split_overlaps"].values())
    leaked = _row("val", "b")
    leaked["glitch_id"] = "g-a"
    with pytest.raises(ValueError, match="split leakage"):
        overlap_training_split_audit([_row("train", "a")], [leaked])


def test_glitch_family_sampling_is_bounded_and_adds_no_physical_rows() -> None:
    rows = [
        {"ml_label": label}
        for label in ["common"] * 16 + ["medium"] * 4 + ["rare"]
    ]
    weights, report = glitch_family_sampling_weights(
        rows, exponent=0.5, maximum_weight_ratio=3.0, minimum_family_count=2
    )
    by_label = report["family_relative_weights"]
    assert by_label == {"common": 1.0, "medium": 2.0, "rare": 1.0}
    assert weights.shape == (21,)
    assert weights.mean() == pytest.approx(1.0)
    assert report["physical_rows"] == report["sample_draws_per_epoch"] == 21
    assert report["adds_independent_physical_examples"] is False
    assert report["families_below_minimum_count_not_boosted"] == ["rare"]


def test_overlap_training_controls_separate_epochs_from_optimizer_updates() -> None:
    assert resolve_overlap_training_control(
        {"training_control": "fixed_epochs", "epochs": 20}, 13
    ) == {
        "control": "fixed_epochs",
        "maximum_epochs": 20,
        "target_optimizer_updates": None,
    }
    assert resolve_overlap_training_control(
        {
            "training_control": "fixed_optimizer_updates",
            "epochs": 40,
            "max_optimizer_updates": 401,
        },
        13,
    ) == {
        "control": "fixed_optimizer_updates",
        "maximum_epochs": 40,
        "target_optimizer_updates": 401,
        "minimum_epochs_required": 31,
    }
    with pytest.raises(ValueError, match="safety cap"):
        resolve_overlap_training_control(
            {
                "training_control": "fixed_optimizer_updates",
                "epochs": 30,
                "max_optimizer_updates": 401,
            },
            13,
        )


@pytest.mark.skipif(torch is None, reason="torch is optional")
def test_overlap_train_epoch_stops_at_exact_update_budget() -> None:
    class TinyDetectorSet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, features, availability):
            return self.bias.reshape(1, 2, 1, 1, 1).expand(
                features.shape[0], 2, 1, features.shape[-2], features.shape[-1]
            )

    student = TinyDetectorSet()
    teacher = TinyDetectorSet()
    overlap_batch = (
        torch.zeros((1, 1, 2, 2)),
        torch.zeros((1, 2, 1, 2, 2)),
        torch.ones((1, 1)),
    )
    clean_batch = (
        torch.zeros((1, 1, 2, 2)),
        torch.zeros((1, 1, 2, 2)),
        torch.ones((1, 1)),
    )
    result = _train_epoch(
        student,
        teacher,
        "detector_set",
        [overlap_batch, overlap_batch, overlap_batch],
        [clean_batch],
        torch.device("cpu"),
        torch.optim.SGD(student.parameters(), lr=0.01),
        1,
        {
            "positive_weights": [1.0, 1.0],
            "class_weights": [1.0, 1.0],
            "clean_chirp_positive_weight": 1.0,
        },
        maximum_optimizer_updates=1,
    )
    assert result["optimizer_updates"] == 1


def test_overlap_data_scaling_requires_paired_gain_under_both_controls(
    tmp_path,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {"status": "passed_physical_overlap_group_audit", "passed": True}
        )
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text('{"split":"val"}\n')
    corpus_audit = tmp_path / "corpus-audit.json"
    corpus_audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
            }
        )
    )
    subsets = []
    for scale in (10, 20):
        manifest = tmp_path / f"train-{scale}.jsonl"
        manifest.write_text(
            "".join(json.dumps({"split": "train", "row": index}) + "\n" for index in range(scale))
        )
        subsets.append(
            {
                "scale": scale,
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
            }
        )
    subset_report = tmp_path / "subsets.json"
    subset_report.write_text(
        json.dumps(
            {
                "status": "frozen_group_safe_physical_overlap_scaling_subsets",
                "passed": True,
                "test_rows_read": 0,
                "required_training_controls": [
                    "fixed_epochs",
                    "fixed_optimizer_updates",
                ],
                "scales": [10, 20],
                "subsets": subsets,
                "validation_manifest_sha256": file_sha256(validation),
                "gravityspy_corpus_audit": {
                    "path": str(corpus_audit),
                    "sha256": file_sha256(corpus_audit),
                },
                "train_validation_group_audit": {
                    "path": str(audit),
                    "sha256": file_sha256(audit),
                },
            }
        )
    )
    reports = []
    for control in ("fixed_epochs", "fixed_optimizer_updates"):
        for scale, glitch_iou in ((10, 0.20), (20, 0.23)):
            manifest_hash = next(
                row["manifest_sha256"] for row in subsets if row["scale"] == scale
            )
            for seed in range(5):
                checkpoint = tmp_path / f"{control}-{scale}-{seed}.pt"
                checkpoint.write_bytes(f"{control}-{scale}-{seed}".encode())
                report = tmp_path / f"{control}-{scale}-{seed}.json"
                report.write_text(
                    json.dumps(
                        {
                            "status": (
                                "validation_selected_real_glitch_overlap_finetune"
                            ),
                            "seed": seed,
                            "training_control": {
                                "control": control,
                                "target_optimizer_updates": (
                                    400 if control == "fixed_optimizer_updates" else None
                                ),
                            },
                            "completed_optimizer_updates": (
                                400 if control == "fixed_optimizer_updates" else 50
                            ),
                            "overlap_train_manifest_sha256": manifest_hash,
                            "overlap_validation_manifest_sha256": file_sha256(
                                validation
                            ),
                            "clean_train_manifest_sha256": "clean-train",
                            "clean_validation_manifest_sha256": "clean-val",
                            "pretrained_checkpoint_sha256": "pretrained",
                            "config_file_sha256": f"config-{control}",
                            "split_audit": {
                                "passed": True,
                                "cross_split_overlaps": {
                                    "mixture_id": [],
                                    "waveform_id": [],
                                    "glitch_id": [],
                                },
                            },
                            "best_epoch": 2,
                            "history": [
                                {
                                    "epoch": 2,
                                    "checkpoint_eligible": True,
                                    "clean_chirp_iou_retention": 0.97,
                                }
                            ],
                            "calibrated_overlap_validation": {
                                "mean_iou": (0.80 + glitch_iou + seed * 0.001) / 2,
                                "chirp": {"iou": 0.80},
                                "glitch": {"iou": glitch_iou + seed * 0.001},
                            },
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": file_sha256(checkpoint),
                        }
                    )
                )
                reports.append(report)

    result = summarize_physical_overlap_data_scaling(
        subset_report,
        reports,
        tmp_path / "summary.json",
        bootstrap_replicates=200,
        bootstrap_seed=9,
    )
    assert result["promote_more_same_distribution_data"] is True
    assert result["diagnosis"] == "data_limited_at_frozen_overlap_endpoint"
    for control in ("fixed_epochs", "fixed_optimizer_updates"):
        comparison = result["adjacent_scale_comparisons"][control][0]
        assert comparison["mean_glitch_iou_delta"] == pytest.approx(0.03)
        assert comparison["paired_bootstrap_95_interval"] == pytest.approx(
            [0.03, 0.03]
        )


def test_glitch_family_metrics_use_hand_calculated_pixel_counts() -> None:
    summary = summarize_glitch_family_counts(
        {
            "Blip": np.asarray([[9, 0, 0], [2, 1, 1]]),
            "Tomte": np.asarray([[8, 0, 0], [3, 0, 1]]),
        },
        {"Blip": 2, "Tomte": 1},
    )
    assert summary["Blip"]["physical_rows"] == 2
    assert summary["Blip"]["iou"] == pytest.approx(0.5)
    assert summary["Blip"]["dice"] == pytest.approx(2 / 3)
    assert summary["Tomte"]["recall"] == pytest.approx(0.75)


def test_overlap_sampling_promotion_uses_only_paired_audited_validation(tmp_path) -> None:
    audit = tmp_path / "corpus-audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
            }
        )
    )
    audit_hash = file_sha256(audit)
    manifests = {}
    for split in ("train", "val"):
        path = tmp_path / f"overlap-{split}.jsonl"
        rows = [
            {
                "split": split,
                "mixture_id": f"{split}-{index}",
                "gravityspy_corpus_audit_sha256": audit_hash,
            }
            for index in range(10)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifests[split] = path

    common = {
        "status": "validation_selected_real_glitch_overlap_finetune",
        "overlap_train_manifest_sha256": file_sha256(manifests["train"]),
        "overlap_validation_manifest_sha256": file_sha256(manifests["val"]),
        "clean_train_manifest_sha256": "clean-train",
        "clean_validation_manifest_sha256": "clean-val",
        "pretrained_checkpoint_sha256": "pretrained",
        "config_file_sha256": "config-file",
        "seed": 7,
        "best_epoch": 2,
        "history": [
            {
                "epoch": 2,
                "clean_chirp_iou_retention": 0.96,
                "checkpoint_eligible": True,
            }
        ],
    }
    reports = {}
    for name, chirp, glitch, family_ious in (
        ("uniform", 0.80, 0.18, {"Blip": 0.20, "Tomte": 0.10}),
        ("family", 0.80, 0.19, {"Blip": 0.22, "Tomte": 0.12}),
    ):
        path = tmp_path / f"{name}.json"
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        path.write_text(
            json.dumps(
                {
                    **common,
                    "calibrated_overlap_validation": {
                        "chirp": {"iou": chirp},
                        "glitch": {"iou": glitch},
                        "mean_iou": (chirp + glitch) / 2,
                        "by_glitch_family": {
                            label: {"physical_rows": 5, "iou": iou}
                            for label, iou in family_ious.items()
                        },
                    },
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                }
            )
        )
        reports[name] = path
    config = tmp_path / "promotion.yaml"
    config.write_text(
        """overlap_sampling_promotion:
  minimum_clean_chirp_iou_retention: 0.95
  minimum_glitch_iou: 0.10
  minimum_family_median_iou: 0.05
  maximum_zero_iou_families: 0
  minimum_validation_rows_per_family: 5
  balanced_minimum_overall_glitch_delta: -0.005
  balanced_minimum_chirp_delta: -0.005
  balanced_minimum_worst_family_delta: 0.0
  balanced_minimum_median_family_delta: 0.005
  maximum_family_regression: 0.02
  maximum_regressed_families: 0
"""
    )
    result = promote_overlap_sampling_arm(
        reports["uniform"],
        reports["family"],
        manifests["train"],
        manifests["val"],
        audit,
        config,
        tmp_path / "promotion.json",
    )
    assert result["passed"]
    assert result["promoted_arm"] == "family_balanced"
    assert result["scale_to_five_seeds"]
    assert result["test_data_opened"] is False
    five_reports = [reports["family"]]
    family_payload = json.loads(reports["family"].read_text())
    for seed in (8, 9, 10, 11):
        path = tmp_path / f"family-seed-{seed}.json"
        payload = dict(family_payload)
        payload["seed"] = seed
        checkpoint = tmp_path / f"family-seed-{seed}.pt"
        checkpoint.write_bytes(str(seed).encode())
        payload["checkpoint_path"] = str(checkpoint)
        payload["checkpoint_sha256"] = file_sha256(checkpoint)
        payload["calibrated_overlap_validation"] = dict(
            family_payload["calibrated_overlap_validation"]
        )
        payload["calibrated_overlap_validation"]["mean_iou"] -= (
            seed - 7
        ) * 0.001
        path.write_text(json.dumps(payload))
        five_reports.append(path)
    summary = summarize_overlap_five_seed_promotion(
        tmp_path / "promotion.json",
        five_reports,
        tmp_path / "five-seed-summary.json",
    )
    assert summary["passed"]
    assert summary["seeds"] == [7, 8, 9, 10, 11]
    assert summary["metrics"]["overlap_glitch_iou"]["mean"] == pytest.approx(0.19)
    assert summary["metrics"]["overlap_glitch_iou"][
        "sample_standard_deviation"
    ] == pytest.approx(0.0)
    assert summary["selected_seed"] == 7
    assert summary["checkpoint_selection"] == (
        "maximum_validation_overlap_mean_iou_then_seed"
    )


def test_overlap_dataset_preserves_both_masks_and_availability(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    features = np.zeros((3, 2, 4, 5), dtype=np.float16)
    chirp = np.zeros_like(features, dtype=np.uint8)
    glitch = np.zeros_like(features, dtype=np.uint8)
    features[1] = 2
    chirp[1, :, 1, 2] = 1
    glitch[1, :, 2, 3] = 1
    np.savez(
        sample,
        features=features,
        chirp_mask=chirp,
        glitch_mask=glitch,
        detector_availability=np.asarray([0, 1, 0], dtype=np.uint8),
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4, 8], dtype=np.float32),
    )
    row = {**_row("train", "x"), "path": str(sample), "sha256": file_sha256(sample)}
    dataset = PhysicalOverlapDataset(
        [row], ("H1", "L1", "V1"), (4.0, 8.0), 4, 5
    )
    observed_features, targets, availability = dataset[0]
    assert observed_features.shape == (6, 4, 5)
    assert targets.shape == (2, 6, 4, 5)
    assert availability.tolist() == [0, 1, 0]
    assert int(targets[0].sum()) == 2
    assert int(targets[1].sum()) == 2


def test_overlap_dataset_rejects_nonzero_unavailable_planes(tmp_path) -> None:
    sample = tmp_path / "invalid.npz"
    features = np.zeros((2, 1, 2, 2), dtype=np.float32)
    features[1, 0, 0, 0] = 1
    np.savez(
        sample,
        features=features,
        chirp_mask=np.zeros_like(features),
        glitch_mask=np.zeros_like(features),
        detector_availability=np.asarray([1, 0]),
        ifos=np.asarray(["H1", "L1"]),
        q_values=np.asarray([4]),
    )
    row = {**_row("train", "x"), "path": str(sample), "sha256": file_sha256(sample)}
    dataset = PhysicalOverlapDataset([row], ("H1", "L1"), (4.0,), 2, 2)
    with pytest.raises(ValueError, match="must be zero"):
        dataset[0]
