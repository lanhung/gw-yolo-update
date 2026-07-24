from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.numeric import DetectorSetQNet
from gwyolo.overlap_training import (
    PhysicalOverlapDataset,
    _train_epoch,
    bind_physical_overlap_scaling_hard_endpoints,
    configure_overlap_training_scope,
    glitch_family_sampling_weights,
    overlap_checkpoint_selection_score,
    overlap_training_split_audit,
    promote_overlap_sampling_arm,
    replay_overlap_five_seed_stability,
    resolve_overlap_training_control,
    run_physical_overlap_scaling_hard_endpoint_cell,
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


def test_overlap_checkpoint_selection_score_supports_threshold_free_loss() -> None:
    validation = {"mean_iou": 0.25, "loss": 1.75}
    assert overlap_checkpoint_selection_score(
        validation, "fixed_threshold_mean_iou"
    ) == pytest.approx(0.25)
    assert overlap_checkpoint_selection_score(
        validation, "validation_loss"
    ) == pytest.approx(-1.75)
    with pytest.raises(ValueError, match="checkpoint_selection_metric"):
        overlap_checkpoint_selection_score(validation, "test_metric")
    with pytest.raises(ValueError, match="finite"):
        overlap_checkpoint_selection_score(
            {"mean_iou": np.nan, "loss": 1.0}, "fixed_threshold_mean_iou"
        )


@pytest.mark.skipif(torch is None, reason="torch is optional")
def test_glitch_head_only_scope_preserves_chirp_and_backbone_bit_exact() -> None:
    torch.manual_seed(7)
    student = DetectorSetQNet(ifo_count=3, q_count=2, base_channels=8)
    original = {
        name: value.detach().clone() for name, value in student.state_dict().items()
    }
    parameters, report = configure_overlap_training_scope(
        student,
        "detector_set",
        {"training_scope": "glitch_head_only"},
    )
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)
    features = torch.randn(2, 6, 8, 8)
    availability = torch.ones(2, 3)
    logits = student(features, availability)
    optimizer.zero_grad(set_to_none=True)
    logits.sum().backward()
    optimizer.step()

    current = student.state_dict()
    for name, value in original.items():
        if name == "shared_head.weight":
            assert torch.equal(current[name][:2], value[:2])
            assert not torch.equal(current[name][2:], value[2:])
        elif name == "shared_head.bias":
            assert torch.equal(current[name][:2], value[:2])
            assert not torch.equal(current[name][2:], value[2:])
        else:
            assert torch.equal(current[name], value)
    assert report == {
        "scope": "glitch_head_only",
        "backbone_frozen": True,
        "chirp_output_frozen": True,
        "glitch_output_trainable": True,
        "gradient_mask_policy": "zero_chirp_rows_v1",
        "trainable_parameter_tensors": 2,
        "effective_trainable_parameters": 18,
    }


@pytest.mark.skipif(torch is None, reason="torch is optional")
def test_glitch_head_only_scope_rejects_non_detector_set_teacher() -> None:
    student = DetectorSetQNet(ifo_count=3, q_count=2, base_channels=8)
    with pytest.raises(ValueError, match="exact detector-set"):
        configure_overlap_training_scope(
            student,
            "early_fusion",
            {"training_scope": "glitch_head_only"},
        )


@pytest.mark.skipif(torch is None, reason="torch is optional")
def test_glitch_adapter_scope_preserves_base_and_chirp_bit_exact() -> None:
    torch.manual_seed(11)
    student = DetectorSetQNet(ifo_count=3, q_count=2, base_channels=8)
    student.enable_glitch_adapter(adapter_channels=4)
    original = {
        name: value.detach().clone() for name, value in student.state_dict().items()
    }
    features = torch.randn(2, 6, 8, 8)
    availability = torch.ones(2, 3)
    baseline = student(features, availability).detach().clone()
    parameters, report = configure_overlap_training_scope(
        student,
        "detector_set",
        {"training_scope": "glitch_adapter_only"},
    )
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    student(features, availability)[:, 1].sum().backward()
    optimizer.step()

    current = student.state_dict()
    assert all(
        torch.equal(current[name], value)
        for name, value in original.items()
        if not name.startswith("glitch_adapter")
    )
    assert any(
        not torch.equal(current[name], value)
        for name, value in original.items()
        if name.startswith("glitch_adapter")
    )
    changed = student(features, availability)
    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert not torch.equal(changed[:, 1], baseline[:, 1])
    assert report == {
        "scope": "glitch_adapter_only",
        "backbone_frozen": True,
        "chirp_output_frozen": True,
        "glitch_output_trainable": True,
        "gradient_mask_policy": None,
        "adapter_policy": "zero_initialized_residual_glitch_decoder_v1",
        "adapter_channels": 4,
        "trainable_parameter_tensors": 8,
        "effective_trainable_parameters": 458,
    }


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = TinyDetectorSet().to(device)
    teacher = TinyDetectorSet().to(device)
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
        device,
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
    assert result["component_losses"][
        "clean_chirp_distillation"
    ] == pytest.approx(np.log(2.0))
    assert result["component_losses"][
        "clean_glitch_distillation"
    ] == pytest.approx(np.log(2.0))


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


def test_hard_endpoint_scaling_binder_authorizes_hand_calculated_joint_gain(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "b" * 40)
    subset_manifests = {}
    subsets = []
    for scale in (10, 20):
        path = tmp_path / f"subset-{scale}.jsonl"
        path.write_text(json.dumps({"scale": scale}) + "\n", encoding="utf-8")
        subset_manifests[scale] = path
        subsets.append(
            {
                "scale": scale,
                "manifest_path": str(path),
                "manifest_sha256": file_sha256(path),
            }
        )
    subset_report = tmp_path / "subset-report.json"
    subset_report.write_text(
        json.dumps(
            {
                "status": "frozen_group_safe_physical_overlap_scaling_subsets",
                "passed": True,
                "test_rows_read": 0,
                "validation_manifest_sha256": "validation-hash",
                "subsets": subsets,
            }
        ),
        encoding="utf-8",
    )
    hard_config = tmp_path / "hard-config.yaml"
    hard_config.write_text(
        """
physical_overlap_scaling_hard_subset:
  primary_metric: hard_subset_glitch_iou_at_validation_frozen_threshold
  minimum_material_primary_gain: 0.01
  minimum_clean_chirp_iou_retention: 0.95
  bootstrap_replicates: 1000
  bootstrap_seed: 9
""".lstrip(),
        encoding="utf-8",
    )
    required_strata = [
        "low_network_snr",
        "missing_detector",
        "o3b_transfer",
        "rare_glitch_family",
    ]
    hard_subset = tmp_path / "hard-subset-report.json"
    hard_subset.write_text(
        json.dumps(
            {
                "status": "frozen_score_blind_physical_overlap_scaling_hard_subset",
                "passed": True,
                "candidate_scores_inspected": False,
                "model_outputs_inspected": False,
                "test_rows_read": 0,
                "validation_manifest_sha256": "validation-hash",
                "required_strata": required_strata,
                "strata": {
                    stratum: {"rows": 25, "unique_glitches": 25}
                    for stratum in required_strata
                },
                "config": {
                    "path": str(hard_config),
                    "sha256": file_sha256(hard_config),
                },
            }
        ),
        encoding="utf-8",
    )
    finetune_identities = []
    cells = []
    for control in ("fixed_epochs", "fixed_optimizer_updates"):
        for scale, metric in ((10, 0.20), (20, 0.23)):
            for seed in range(5):
                checkpoint = tmp_path / f"checkpoint-{control}-{scale}-{seed}.pt"
                checkpoint.write_bytes(f"{control}-{scale}-{seed}".encode())
                finetune = tmp_path / f"finetune-{control}-{scale}-{seed}.json"
                finetune.write_text(
                    json.dumps(
                        {
                            "training_control": {"control": control},
                            "overlap_train_manifest_sha256": file_sha256(
                                subset_manifests[scale]
                            ),
                            "seed": seed,
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": file_sha256(checkpoint),
                        }
                    ),
                    encoding="utf-8",
                )
                finetune_identity = {
                    "path": str(finetune.resolve()),
                    "sha256": file_sha256(finetune),
                }
                finetune_identities.append(finetune_identity)
                cell = tmp_path / f"hard-cell-{control}-{scale}-{seed}.json"
                cell.write_text(
                    json.dumps(
                        {
                            "status": (
                                "completed_validation_only_physical_overlap_scaling_"
                                "hard_endpoint_cell"
                            ),
                            "passed": True,
                            "test_rows_read": 0,
                            "test_evaluation": None,
                            "threshold_refits": 0,
                            "endpoint_partition": (
                                "validation_only_predeclared_hard_subset"
                            ),
                            "training_control": control,
                            "scale": scale,
                            "seed": seed,
                            "hard_subset": {
                                "path": str(hard_subset.resolve()),
                                "sha256": file_sha256(hard_subset),
                            },
                            "subset_report": {
                                "path": str(subset_report.resolve()),
                                "sha256": file_sha256(subset_report),
                            },
                            "finetune_report": finetune_identity,
                            "checkpoint": {
                                "path": str(checkpoint),
                                "sha256": file_sha256(checkpoint),
                            },
                            "primary_metric": {
                                "name": (
                                    "hard_subset_glitch_iou_at_validation_frozen_"
                                    "threshold"
                                ),
                                "value": metric + seed * 0.001,
                            },
                            "clean_noninferiority": {
                                "passed": True,
                                "retention": 0.97,
                            },
                            "strata": {
                                stratum: {
                                    "rows": 25,
                                    "unique_glitches": 25,
                                    "glitch_iou": metric,
                                }
                                for stratum in required_strata
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                cells.append(cell)
    scaling_summary = tmp_path / "scaling-diagnostic.json"
    scaling_summary.write_text(
        json.dumps(
            {
                "status": "completed_group_safe_physical_overlap_data_scaling_curve",
                "passed": True,
                "test_rows_read": 0,
                "test_evaluation": None,
                "paired_seeds": list(range(5)),
                "scales": [10, 20],
                "promotion_data_doubling": [10, 20],
                "promote_more_same_distribution_data": True,
                "diagnosis": "data_limited_at_frozen_overlap_endpoint",
                "subset_report_path": str(subset_report),
                "subset_report_sha256": file_sha256(subset_report),
                "finetune_reports": finetune_identities,
                "common_artifact_hashes": {"validation": "hash"},
            }
        ),
        encoding="utf-8",
    )

    result = bind_physical_overlap_scaling_hard_endpoints(
        scaling_summary,
        hard_subset,
        cells,
        tmp_path / "bound-scaling.json",
        next_scale=40,
    )

    assert result["scale_promotion_authorized"] is True
    assert result["authorized_next_physical_scale"] == 40
    assert result["diagnosis"] == "data_limited_on_predeclared_hard_endpoint"
    for control in ("fixed_epochs", "fixed_optimizer_updates"):
        comparison = result["hard_endpoint_comparisons"][control]
        assert comparison["mean_primary_metric_delta"] == pytest.approx(0.03)
        assert comparison["paired_bootstrap_95_interval"] == pytest.approx(
            [0.03, 0.03]
        )
    bundle = result["hard_endpoint_bundle"]
    assert file_sha256(bundle["path"]) == bundle["sha256"]
    bundle_report = json.loads(Path(bundle["path"]).read_text(encoding="utf-8"))
    assert len(bundle_report["cells"]) == 20


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_hard_endpoint_cell_uses_frozen_threshold_without_refit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "c" * 40)
    config = tmp_path / "cell-config.yaml"
    config.write_text(
        """
overlap_training:
  batch_size: 5
  positive_weights: [1.0, 1.0]
  class_weights: [1.0, 1.0]
  focal_gamma: 0.0
  cache_in_memory: false
  model_ifos: [H1, L1]
  q_values: [4]
  tensor:
    frequency_bins: 16
    time_bins: 16
""".lstrip(),
        encoding="utf-8",
    )
    model = DetectorSetQNet(2, 1, 2)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = tmp_path / "cell-checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "detector_set",
            "model_ifos": ["H1", "L1"],
            "q_values": [4.0],
            "input_channels": 2,
            "base_channels": 2,
        },
        checkpoint,
    )
    strata = [
        "low_network_snr",
        "missing_detector",
        "o3b_transfer",
        "rare_glitch_family",
    ]
    hard_rows = []
    for index in range(25):
        sample = tmp_path / f"cell-sample-{index}.npz"
        np.savez(
            sample,
            features=np.zeros((2, 1, 16, 16), dtype=np.float32),
            chirp_mask=np.zeros((2, 1, 16, 16), dtype=np.uint8),
            glitch_mask=np.ones((2, 1, 16, 16), dtype=np.uint8),
            detector_availability=np.ones(2, dtype=np.uint8),
            ifos=np.asarray(["H1", "L1"]),
            q_values=np.asarray([4.0], dtype=np.float32),
        )
        hard_rows.append(
            {
                "split": "val",
                "mixture_id": f"m-{index}",
                "glitch_id": f"g-{index}",
                "ml_label": "Blip",
                "path": str(sample),
                "sha256": file_sha256(sample),
                "hard_subset_strata": strata,
            }
        )
    hard_manifest = tmp_path / "cell-hard.jsonl"
    hard_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in hard_rows), encoding="utf-8"
    )
    train_manifest = tmp_path / "cell-train.jsonl"
    train_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    subset_report = tmp_path / "cell-subsets.json"
    subset_report.write_text(
        json.dumps(
            {
                "status": "frozen_group_safe_physical_overlap_scaling_subsets",
                "passed": True,
                "test_rows_read": 0,
                "validation_manifest_sha256": "validation-hash",
                "subsets": [
                    {
                        "scale": 10,
                        "manifest_path": str(train_manifest),
                        "manifest_sha256": file_sha256(train_manifest),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    hard_report = tmp_path / "cell-hard-report.json"
    hard_report.write_text(
        json.dumps(
            {
                "status": "frozen_score_blind_physical_overlap_scaling_hard_subset",
                "passed": True,
                "candidate_scores_inspected": False,
                "model_outputs_inspected": False,
                "test_rows_read": 0,
                "validation_manifest_sha256": "validation-hash",
                "hard_subset_manifest_path": str(hard_manifest),
                "hard_subset_manifest_sha256": file_sha256(hard_manifest),
                "required_strata": strata,
                "strata": {
                    stratum: {"rows": 25, "unique_glitches": 25}
                    for stratum in strata
                },
            }
        ),
        encoding="utf-8",
    )
    finetune = tmp_path / "cell-finetune.json"
    finetune.write_text(
        json.dumps(
            {
                "status": "validation_selected_real_glitch_overlap_finetune",
                "search_claim_allowed": False,
                "test_evaluation": None,
                "overlap_train_manifest_sha256": file_sha256(train_manifest),
                "overlap_validation_manifest_sha256": "validation-hash",
                "config_file_sha256": file_sha256(config),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "training_control": {"control": "fixed_epochs"},
                "seed": 7,
                "validation_selected_thresholds": {"chirp": 0.4, "glitch": 0.4},
                "best_epoch": 2,
                "minimum_clean_chirp_iou_retention": 0.95,
                "history": [
                    {
                        "epoch": 2,
                        "checkpoint_eligible": True,
                        "clean_chirp_iou_retention": 0.97,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_physical_overlap_scaling_hard_endpoint_cell(
        config,
        subset_report,
        hard_report,
        finetune,
        10,
        tmp_path / "cell-result.json",
    )

    assert result["threshold_refits"] == 0
    assert result["frozen_thresholds"] == {"chirp": 0.4, "glitch": 0.4}
    assert result["primary_metric"]["value"] == pytest.approx(1.0)
    assert all(
        values["glitch_iou"] == pytest.approx(1.0)
        for values in result["strata"].values()
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
    stability_config = tmp_path / "five-seed-stability.yaml"
    stability_config.write_text(
        """overlap_five_seed_stability:
  minimum_passing_seed_fraction: 0.8
  minimum_median_clean_retention: 0.95
  minimum_median_glitch_iou: 0.10
  minimum_median_family_iou: 0.05
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
        stability_config,
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
        "maximum_validation_overlap_mean_iou_among_passing_seeds_then_seed"
    )
    assert summary["five_seed_stability"]["passing_seed_fraction"] == 1.0
    assert summary["five_seed_stability"]["passing_seeds"] == [7, 8, 9, 10, 11]

    legacy_summary = dict(summary)
    legacy_summary.pop("five_seed_stability")
    legacy_path = tmp_path / "legacy-five-seed-summary.json"
    legacy_path.write_text(json.dumps(legacy_summary))
    replayed = replay_overlap_five_seed_stability(
        legacy_path,
        stability_config,
        tmp_path / "replayed-five-seed-summary.json",
    )
    assert replayed["passed"] is True
    assert replayed["five_seed_stability"]["passing_seed_fraction"] == 1.0
    assert replayed["stability_replay_source"]["sha256"] == file_sha256(
        legacy_path
    )

    for path in five_reports[-2:]:
        payload = json.loads(path.read_text())
        payload["calibrated_overlap_validation"]["glitch"]["iou"] = 0.01
        for row in payload["calibrated_overlap_validation"][
            "by_glitch_family"
        ].values():
            row["iou"] = 0.0
        path.write_text(json.dumps(payload))
    failed = summarize_overlap_five_seed_promotion(
        tmp_path / "promotion.json",
        five_reports,
        stability_config,
        tmp_path / "failed-five-seed-summary.json",
    )
    assert failed["passed"] is False
    assert failed["five_seed_stability"]["passing_seed_fraction"] == 0.6
    assert failed["five_seed_stability"]["passing_seeds"] == [7, 8, 9]
    assert failed["selected_checkpoint_path"] is None
    assert failed["selected_checkpoint_sha256"] is None


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
