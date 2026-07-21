from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, file_sha256, load_yaml
from .runtime import execution_provenance


def _parse_labeled_reports(specs: list[str], kind: str) -> dict[str, list[Path]]:
    parsed: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{kind} reports must use ARM=PATH")
        parsed[label].append(Path(raw_path))
    if not parsed:
        raise ValueError(f"at least one {kind} report is required")
    return dict(parsed)


def _paired_bootstrap_interval(
    differences: list[float], replicates: int, seed: int
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2 or replicates < 100:
        raise ValueError("paired bootstrap requires at least two pairs and 100 replicates")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def summarize_physical_data_domain_comparison(
    config_path: str | Path,
    data_domain_audit_path: str | Path,
    fixed_update_training_specs: list[str],
    fixed_update_audit_specs: list[str],
    fixed_epoch_training_specs: list[str],
    fixed_epoch_audit_specs: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Gate independent-GPS gains under matched update and epoch controls."""
    config = load_yaml(config_path)
    settings = config["physical_data_domain_promotion"]
    arms = tuple(str(value) for value in settings["expected_arms"])
    if len(arms) != 2 or len(set(arms)) != 2:
        raise ValueError("physical data-domain promotion requires exactly two arms")
    baseline_arm, diverse_arm = arms
    minimum_seeds = int(settings["minimum_seeds"])
    replicates = int(settings["bootstrap_replicates"])
    bootstrap_seed = int(settings["seed"])
    required_groups = ("all",) + tuple(
        f"family:{value}" for value in settings["required_source_families"]
    )

    data_audit_path = Path(data_domain_audit_path)
    with data_audit_path.open("r", encoding="utf-8") as handle:
        data_audit = json.load(handle)
    if data_audit.get("status") != "paired_materialized_data_domain_audit_passed":
        raise ValueError("data-domain comparison requires a passed paired manifest audit")
    if data_audit.get("test_evaluation") is not None:
        raise ValueError("data-domain comparison refuses an audit that opened test data")
    if not data_audit.get("source_parameters_identical"):
        raise ValueError("data-domain arms do not preserve the paired source population")
    if int(data_audit.get("cross_arm_gps_block_overlap", -1)) != 0:
        raise ValueError("data-domain arms are not GPS independent")
    manifest_contract = data_audit["manifests"]
    for arm in arms:
        if arm not in manifest_contract:
            raise ValueError(f"paired data audit lacks expected arm: {arm}")
    for item in manifest_contract.values():
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"paired data-domain manifest changed after audit: {path}")
    validation_hash = str(manifest_contract["shared_validation"]["sha256"])

    modes = {
        "fixed_updates": (
            fixed_update_training_specs,
            fixed_update_audit_specs,
            "final_update",
        ),
        "fixed_epochs": (
            fixed_epoch_training_specs,
            fixed_epoch_audit_specs,
            "best_validation",
        ),
    }
    mode_results = {}
    all_checks: dict[str, bool] = {
        "minimum_gps_diversity_factor": float(
            data_audit["independent_gps_diversity_factor"]
        )
        >= float(settings["minimum_gps_diversity_factor"])
    }
    for mode_index, (mode, (training_specs, audit_specs, selection)) in enumerate(
        modes.items()
    ):
        training_paths = _parse_labeled_reports(training_specs, f"{mode} training")
        audit_paths = _parse_labeled_reports(audit_specs, f"{mode} checkpoint audit")
        if set(training_paths) != set(arms) or set(audit_paths) != set(arms):
            raise ValueError(f"{mode} reports must contain exactly the configured arms")
        training_by_key: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
        audit_by_key: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
        for arm in arms:
            for path in training_paths[arm]:
                report = json.loads(path.read_text(encoding="utf-8"))
                key = (arm, int(report["seed"]))
                if key in training_by_key:
                    raise ValueError(f"duplicate {mode} training report: {key}")
                if report.get("test_evaluation") is not None:
                    raise ValueError(f"{mode} training report opened test data: {path}")
                if report.get("status") != "physical_real_noise_validation_only_finetune":
                    raise ValueError(f"{mode} report is not a completed physical fine-tune")
                if report.get("checkpoint_selection") != selection:
                    raise ValueError(f"{mode} checkpoint selection differs from protocol")
                if str(report["train_manifest_sha256"]) != str(
                    manifest_contract[arm]["sha256"]
                ):
                    raise ValueError(f"{mode} training report uses an unaudited {arm} manifest")
                if str(report["validation_manifest_sha256"]) != validation_hash:
                    raise ValueError(f"{mode} training report uses another validation manifest")
                if mode == "fixed_updates" and not report.get("training_budget_reached"):
                    raise ValueError("fixed-update data-domain run did not reach its budget")
                training_by_key[key] = (report, path)
            for path in audit_paths[arm]:
                report = json.loads(path.read_text(encoding="utf-8"))
                seed = int(report["seed"])
                key = (arm, seed)
                if key in audit_by_key:
                    raise ValueError(f"duplicate {mode} checkpoint audit: {key}")
                if report.get("test_evaluation") is not None:
                    raise ValueError(f"{mode} checkpoint audit opened test data: {path}")
                if report.get("status") != "physical_validation_checkpoint_audit":
                    raise ValueError(f"{mode} report is not a physical checkpoint audit")
                if str(report["validation_manifest_sha256"]) != validation_hash:
                    raise ValueError(f"{mode} checkpoint audit uses another validation manifest")
                audit_by_key[key] = (report, path)

        seeds_by_arm = {
            arm: {seed for report_arm, seed in training_by_key if report_arm == arm}
            for arm in arms
        }
        if seeds_by_arm[baseline_arm] != seeds_by_arm[diverse_arm]:
            raise ValueError(f"{mode} arm seed sets are not paired")
        seeds = sorted(seeds_by_arm[baseline_arm])
        if len(seeds) < minimum_seeds:
            raise ValueError(f"{mode} requires at least {minimum_seeds} paired seeds")
        if set(training_by_key) != set(audit_by_key):
            raise ValueError(f"{mode} training and checkpoint-audit reports do not pair")

        controlled = defaultdict(set)
        paired = []
        for seed in seeds:
            seed_record = {"seed": seed, "arms": {}, "differences": {}}
            for arm in arms:
                training, training_path = training_by_key[(arm, seed)]
                audit, audit_path = audit_by_key[(arm, seed)]
                if str(training["checkpoint_sha256"]) != str(audit["checkpoint_sha256"]):
                    raise ValueError(f"{mode} audit checkpoint differs from training report")
                if str(training["config_hash"]) != str(audit["config_hash"]):
                    raise ValueError(f"{mode} audit config differs from training report")
                if str(training.get("code_commit") or "") != str(
                    audit.get("training_code_commit") or ""
                ):
                    raise ValueError(f"{mode} audit code identity differs from training report")
                if float(training["selected_chirp_threshold"]) != float(
                    audit["chirp_threshold"]
                ):
                    raise ValueError(f"{mode} audit did not use the selected threshold")
                for group in required_groups:
                    if group not in audit["groups"]:
                        raise ValueError(f"{mode} checkpoint audit lacks {group}")
                metrics = {
                    group: float(audit["groups"][group]["iou"])
                    for group in required_groups
                }
                seed_record["arms"][arm] = {
                    "metrics": metrics,
                    "checkpoint_sha256": training["checkpoint_sha256"],
                    "training_report_path": str(training_path.resolve()),
                    "training_report_sha256": file_sha256(training_path),
                    "audit_report_path": str(audit_path.resolve()),
                    "audit_report_sha256": file_sha256(audit_path),
                }
                controlled["pretrained_checkpoint_sha256"].add(
                    str(training["pretrained_checkpoint_sha256"])
                )
                controlled["config_hash"].add(str(training["config_hash"]))
                controlled["code_commit"].add(str(training.get("code_commit") or ""))
                controlled["training_rows"].add(
                    int(training["training_selection"]["selected_rows"])
                )
                if mode == "fixed_updates":
                    controlled["optimizer_updates"].add(int(training["optimizer_updates"]))
                    controlled["optimizer_examples"].add(int(training["optimizer_examples"]))
                else:
                    controlled["completed_epochs"].add(int(training["completed_epochs"]))
            for group in required_groups:
                seed_record["differences"][group] = (
                    seed_record["arms"][diverse_arm]["metrics"][group]
                    - seed_record["arms"][baseline_arm]["metrics"][group]
                )
            paired.append(seed_record)
        bad_controls = {key: sorted(values) for key, values in controlled.items() if len(values) != 1}
        if bad_controls or not next(iter(controlled["code_commit"])):
            raise ValueError(f"{mode} reports disagree on controlled fields: {bad_controls}")
        expected_controls = {
            "training_rows": int(settings["required_training_rows"]),
            (
                "optimizer_updates" if mode == "fixed_updates" else "completed_epochs"
            ): int(
                settings[
                    "required_optimizer_updates"
                    if mode == "fixed_updates"
                    else "required_epochs"
                ]
            ),
        }
        if mode == "fixed_updates":
            expected_controls["optimizer_examples"] = int(
                settings["required_optimizer_examples"]
            )
        observed_controls = {
            key: next(iter(controlled[key])) for key in expected_controls
        }
        if observed_controls != expected_controls:
            raise ValueError(
                f"{mode} controlled budget differs from the predeclared gate: "
                f"{observed_controls} != {expected_controls}"
            )

        intervals = {
            group: _paired_bootstrap_interval(
                [record["differences"][group] for record in paired],
                replicates,
                bootstrap_seed + mode_index * 100 + group_index,
            )
            for group_index, group in enumerate(required_groups)
        }
        checks = {
            "minimum_overall_iou_gain": intervals["all"]["mean"]
            >= float(settings["minimum_overall_iou_gain"]),
            "overall_paired_interval_above_zero": intervals["all"]["lower_95"] > 0.0,
            **{
                f"maximum_regression_{group}": intervals[group]["mean"]
                >= -float(settings["maximum_family_iou_regression"])
                for group in required_groups
                if group != "all"
            },
        }
        mode_results[mode] = {
            "seed_count": len(seeds),
            "seeds": seeds,
            "controlled_fields": {
                key: next(iter(values)) for key, values in controlled.items()
            },
            "paired_runs": paired,
            "paired_bootstrap": intervals,
            "promotion_checks": checks,
            "promotion_allowed": all(checks.values()),
        }
        all_checks.update({f"{mode}:{key}": value for key, value in checks.items()})

    promotion_allowed = all(all_checks.values())
    result = {
        "status": "physical_data_domain_validation_promotion_gate",
        "scientific_claim_allowed": False,
        "promotion_allowed": promotion_allowed,
        "scientific_blocker": (
            "validation-only data-axis gate; search claims still require locked FAR/IFAR/VT"
        ),
        "test_evaluation": None,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "data_domain_audit_path": str(data_audit_path.resolve()),
        "data_domain_audit_sha256": file_sha256(data_audit_path),
        "data_domain_summary": {
            "baseline_unique_gps_blocks": data_audit["baseline_unique_gps_blocks"],
            "independent_unique_gps_blocks": data_audit["independent_unique_gps_blocks"],
            "independent_gps_diversity_factor": data_audit[
                "independent_gps_diversity_factor"
            ],
            "paired_population_hash": data_audit["paired_population_hash"],
        },
        "modes": mode_results,
        "promotion_checks": all_checks,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result
