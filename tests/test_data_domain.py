from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.data_domain import summarize_physical_data_domain_comparison
from gwyolo.injections import audit_paired_data_domain_manifests
from gwyolo.io import file_sha256


def _source_row(index: int, split: str, gps_block: str) -> dict:
    return {
        "split": split,
        "injection_id": f"{split}-i{index}",
        "waveform_id": f"{split}-w{index}",
        "source_family": ("BBH", "BNS", "NSBH")[index % 3],
        "waveform_backend": "validated",
        "waveform_approximant": "A",
        "f_lower_hz": 20.0,
        "mass_1_msun": 30.0 + index,
        "mass_2_msun": 20.0,
        "mass_1_detector_msun": 33.0 + index,
        "mass_2_detector_msun": 22.0,
        "spin_1z": 0.1,
        "spin_2z": 0.2,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
        "inclination": 1.0,
        "right_ascension": 2.0,
        "declination": 0.3,
        "polarization": 0.4,
        "coalescence_phase": 0.5,
        "luminosity_distance_mpc": 1000.0,
        "comoving_distance_mpc": 900.0,
        "redshift": 0.1,
        "maximum_distance_mpc": 5000.0,
        "vt_weight": 3.0,
        "vt_weight_unit": "Mpc^3 yr",
        "vt_measure": "measure",
        "gps_block": gps_block,
        "ifos": ["H1", "L1"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_data_domain_gate_uses_paired_seed_gain_and_both_budget_controls(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    diverse = tmp_path / "diverse.jsonl"
    validation = tmp_path / "validation.jsonl"
    baseline_rows = [_source_row(index, "train", "old") for index in range(3)]
    diverse_rows = [
        {**row, "gps_block": f"new-{index % 2}"}
        for index, row in enumerate(baseline_rows)
    ]
    _write_jsonl(baseline, baseline_rows)
    _write_jsonl(diverse, diverse_rows)
    _write_jsonl(validation, [_source_row(index, "val", "validation") for index in range(3)])
    data_audit_path = tmp_path / "data-audit.json"
    audit_paired_data_domain_manifests(
        baseline, diverse, validation, data_audit_path
    )
    config = tmp_path / "promotion.yaml"
    config.write_text(
        """
physical_data_domain_promotion:
  expected_arms: [baseline_fixed_gps, independent_gps]
  minimum_seeds: 3
  required_training_rows: 3
  required_optimizer_updates: 100
  required_optimizer_examples: 300
  required_epochs: 10
  required_source_families: [BBH, BNS, NSBH]
  minimum_gps_diversity_factor: 1.5
  minimum_overall_iou_gain: 0.005
  maximum_family_iou_regression: 0.005
  bootstrap_replicates: 1000
  seed: 9
""",
        encoding="utf-8",
    )
    train_hashes = {
        "baseline_fixed_gps": file_sha256(baseline),
        "independent_gps": file_sha256(diverse),
    }
    validation_hash = file_sha256(validation)
    training_specs = {"fixed_updates": [], "fixed_epochs": []}
    audit_specs = {"fixed_updates": [], "fixed_epochs": []}
    for mode in training_specs:
        for arm in train_hashes:
            for seed in (1, 2, 3):
                checkpoint_hash = f"checkpoint-{mode}-{arm}-{seed}"
                training_path = tmp_path / f"train-{mode}-{arm}-{seed}.json"
                training_path.write_text(
                    json.dumps(
                        {
                            "status": "physical_real_noise_validation_only_finetune",
                            "seed": seed,
                            "test_evaluation": None,
                            "checkpoint_selection": (
                                "final_update" if mode == "fixed_updates" else "best_validation"
                            ),
                            "training_budget_reached": True,
                            "train_manifest_sha256": train_hashes[arm],
                            "validation_manifest_sha256": validation_hash,
                            "checkpoint_sha256": checkpoint_hash,
                            "config_hash": f"config-{mode}",
                            "selected_chirp_threshold": 0.5,
                            "pretrained_checkpoint_sha256": "pretrained",
                            "code_commit": "abc123",
                            "training_selection": {"selected_rows": 3},
                            "optimizer_updates": 100,
                            "optimizer_examples": 300,
                            "completed_epochs": 10,
                        }
                    ),
                    encoding="utf-8",
                )
                base_metric = 0.10 + seed * 0.001
                gain = 0.02 if arm == "independent_gps" else 0.0
                checkpoint_audit = tmp_path / f"audit-{mode}-{arm}-{seed}.json"
                checkpoint_audit.write_text(
                    json.dumps(
                        {
                            "status": "physical_validation_checkpoint_audit",
                            "seed": seed,
                            "training_code_commit": "abc123",
                            "test_evaluation": None,
                            "validation_manifest_sha256": validation_hash,
                            "checkpoint_sha256": checkpoint_hash,
                            "config_hash": f"config-{mode}",
                            "chirp_threshold": 0.5,
                            "groups": {
                                group: {"iou": base_metric + gain}
                                for group in ("all", "family:BBH", "family:BNS", "family:NSBH")
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                training_specs[mode].append(f"{arm}={training_path}")
                audit_specs[mode].append(f"{arm}={checkpoint_audit}")
    result = summarize_physical_data_domain_comparison(
        config,
        data_audit_path,
        training_specs["fixed_updates"],
        audit_specs["fixed_updates"],
        training_specs["fixed_epochs"],
        audit_specs["fixed_epochs"],
        tmp_path / "comparison.json",
    )
    assert result["promotion_allowed"] is True
    for mode in ("fixed_updates", "fixed_epochs"):
        interval = result["modes"][mode]["paired_bootstrap"]["all"]
        assert interval["mean"] == pytest.approx(0.02)
        assert interval["lower_95"] == pytest.approx(0.02)
        assert result["modes"][mode]["seed_count"] == 3
