from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .numeric import train_numeric_model


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def _t_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees of freedom must be positive")
    available = sorted(T_CRITICAL_975)
    key = next((value for value in available if value >= degrees_of_freedom), 30)
    return T_CRITICAL_975[key]


def aggregate_numeric_seed_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one numeric seed report is required")
    seeds = [int(report["seed"]) for report in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Numeric seed reports contain duplicate seeds")
    manifest_hashes = {str(report["manifest_sha256"]) for report in reports}
    if len(manifest_hashes) != 1:
        raise ValueError("Numeric seed reports use different manifests")
    family_hashes = {
        str(report["config_family_hash"])
        for report in reports
        if report.get("config_family_hash") is not None
    }
    if len(family_hashes) > 1:
        raise ValueError("Numeric seed reports use different non-seed configurations")
    values = np.asarray(
        [float(report["best_validation_mean_iou"]) for report in reports], dtype=np.float64
    )
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else None
    if values.size > 1:
        half_width = _t_critical(values.size - 1) * standard_deviation / math.sqrt(values.size)
        confidence_interval = [mean - half_width, mean + half_width]
    else:
        confidence_interval = [None, None]
    return {
        "status": "synthetic_validation_multiseed_only",
        "scientific_claim_allowed": False,
        "seed_count": len(seeds),
        "minimum_five_seed_gate_passed": len(seeds) >= 5,
        "seeds": sorted(seeds),
        "manifest_sha256": next(iter(manifest_hashes)),
        "config_family_hash": next(iter(family_hashes)) if family_hashes else None,
        "best_validation_mean_iou": {
            "mean": mean,
            "sample_standard_deviation": standard_deviation,
            "student_t_95_interval": confidence_interval,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        },
        "runs": [
            {
                "seed": int(report["seed"]),
                "best_epoch": int(report["best_epoch"]),
                "best_validation_mean_iou": float(report["best_validation_mean_iou"]),
                "report_path": report.get("report_path"),
                "report_sha256": report.get("report_sha256"),
                "checkpoint_path": report.get("checkpoint_path"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
            }
            for report in sorted(reports, key=lambda item: int(item["seed"]))
        ],
    }


def _load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["report_path"] = str(source)
    report["report_sha256"] = file_sha256(source)
    checkpoint = source.parent / "best_numeric.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Numeric report has no checkpoint alongside it: {source}")
    report["checkpoint_path"] = str(checkpoint)
    report["checkpoint_sha256"] = file_sha256(checkpoint)
    return report


def run_numeric_multiseed(
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    reuse_runs: dict[int, str] | None = None,
) -> dict[str, Any]:
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("seeds must be a non-empty unique list")
    expected_manifest_hash = file_sha256(manifest_path)
    family_config = load_yaml(config_path)
    family_config["numeric_training"].pop("seed", None)
    expected_family_hash = canonical_hash(family_config)
    reuse_runs = reuse_runs or {}
    unknown_reuse = set(reuse_runs) - set(seeds)
    if unknown_reuse:
        raise ValueError(f"Reuse runs contain unrequested seeds: {sorted(unknown_reuse)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    collected = []
    for seed in seeds:
        if seed in reuse_runs:
            report_path = Path(reuse_runs[seed])
        else:
            report_path = output / f"seed-{seed}" / "numeric_training_report.json"
            if not report_path.is_file():
                train_numeric_model(config_path, manifest_path, report_path.parent, seed_override=seed)
        report = _load_report(report_path)
        if int(report["seed"]) != seed:
            raise ValueError(f"Seed mismatch in {report_path}: {report['seed']} != {seed}")
        if str(report["manifest_sha256"]) != expected_manifest_hash:
            raise ValueError(f"Manifest mismatch in reused run {report_path}")
        report_family_hash = report.get("config_family_hash")
        if report_family_hash is not None and str(report_family_hash) != expected_family_hash:
            raise ValueError(f"Non-seed config mismatch in reused run {report_path}")
        if report_family_hash is None:
            report["config_family_hash"] = expected_family_hash
            report["legacy_family_hash_inferred_from_requested_config"] = True
        collected.append(report)
        partial = aggregate_numeric_seed_reports(collected)
        partial["requested_seeds"] = seeds
        partial["completed_seed_count"] = len(collected)
        atomic_write_json(output / "numeric_multiseed_report.json", partial)
    return aggregate_numeric_seed_reports(collected)
