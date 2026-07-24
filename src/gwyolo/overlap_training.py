from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .numeric import (
    DetectorSetQNet,
    _atomic_torch_save,
    initialize_detector_set_from_early_fusion,
    model_from_checkpoint,
)
from .physical_training import PhysicalInjectionDataset, physical_split_audit
from .runtime import execution_provenance

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None
    WeightedRandomSampler = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Overlap fine-tuning requires PyTorch")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def glitch_family_sampling_weights(
    rows: list[dict[str, Any]],
    exponent: float,
    maximum_weight_ratio: float,
    minimum_family_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build bounded family weights without changing the physical-example count."""

    if not rows:
        raise ValueError("Glitch-family sampling requires non-empty rows")
    if not 0 <= exponent <= 1:
        raise ValueError("Glitch-family sampling exponent must lie in [0, 1]")
    if maximum_weight_ratio < 1 or minimum_family_count <= 0:
        raise ValueError("Glitch-family sampling cap/count settings are invalid")
    labels = [str(row.get("ml_label", "")).strip() for row in rows]
    if any(not label for label in labels):
        raise ValueError("Every overlap row requires an ml_label for family sampling")
    counts = Counter(labels)
    reference_count = max(counts.values())
    family_weights = {
        label: (
            min(
                maximum_weight_ratio,
                (reference_count / count) ** exponent,
            )
            if count >= minimum_family_count
            else 1.0
        )
        for label, count in counts.items()
    }
    weights = np.asarray([family_weights[label] for label in labels], dtype=np.float64)
    weights /= float(weights.mean())
    total_mass = float(sum(counts[label] * family_weights[label] for label in counts))
    report = {
        "strategy": "bounded_inverse_glitch_family_frequency_v1",
        "physical_rows": len(rows),
        "sample_draws_per_epoch": len(rows),
        "adds_independent_physical_examples": False,
        "replacement": True,
        "exponent": exponent,
        "maximum_weight_ratio": maximum_weight_ratio,
        "minimum_family_count": minimum_family_count,
        "family_counts": dict(sorted(counts.items())),
        "family_relative_weights": {
            label: family_weights[label] for label in sorted(family_weights)
        },
        "family_expected_draw_fraction": {
            label: counts[label] * family_weights[label] / total_mass
            for label in sorted(counts)
        },
        "families_below_minimum_count_not_boosted": sorted(
            label for label, count in counts.items() if count < minimum_family_count
        ),
        "normalized_minimum_row_weight": float(weights.min()),
        "normalized_maximum_row_weight": float(weights.max()),
    }
    return weights, report


def overlap_training_split_audit(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not train_rows or not validation_rows:
        raise ValueError("Overlap fine-tuning requires non-empty train and validation rows")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Overlap training manifest contains a non-train row")
    if any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("Overlap validation manifest contains a non-validation row")
    fields = (
        "mixture_id",
        "injection_id",
        "waveform_id",
        "glitch_id",
        "injection_gps_block",
        "network_gps_block",
    )
    overlaps = {
        field: sorted(
            {str(row[field]) for row in train_rows}
            & {str(row[field]) for row in validation_rows}
        )
        for field in fields
    }
    if any(overlaps.values()):
        raise ValueError(f"Overlap fine-tuning split leakage: {overlaps}")
    return {
        "passed": True,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_counts": {
            field: len({str(row[field]) for row in train_rows}) for field in fields
        },
        "validation_unique_counts": {
            field: len({str(row[field]) for row in validation_rows}) for field in fields
        },
        "cross_split_overlaps": overlaps,
    }


class PhysicalOverlapDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_ifos: tuple[str, ...],
        q_values: tuple[float, ...],
        frequency_bins: int,
        time_bins: int,
        cache_in_memory: bool = False,
    ):
        self.rows = rows
        self.model_ifos = model_ifos
        self.q_values = q_values
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.input_channels = len(model_ifos) * len(q_values)
        self.cache: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        row = self.rows[index]
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"Overlap sample hash mismatch: {row['mixture_id']}")
        with np.load(row["path"], allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
            chirp = np.asarray(arrays["chirp_mask"], dtype=np.float32)
            glitch = np.asarray(arrays["glitch_mask"], dtype=np.float32)
            availability = np.asarray(arrays["detector_availability"], dtype=np.float32)
            ifos = tuple(str(value) for value in arrays["ifos"].tolist())
            q_values = tuple(float(value) for value in arrays["q_values"].tolist())
        expected = (
            len(self.model_ifos),
            len(self.q_values),
            self.frequency_bins,
            self.time_bins,
        )
        if features.shape != expected or chirp.shape != expected or glitch.shape != expected:
            raise ValueError(f"Overlap tensor shape mismatch: {row['mixture_id']}")
        if ifos != self.model_ifos or not np.allclose(q_values, self.q_values, atol=1e-6):
            raise ValueError("Overlap detector/Q metadata differs from training configuration")
        if availability.shape != (len(self.model_ifos),):
            raise ValueError("Overlap detector availability shape is invalid")
        if np.any((availability != 0) & (availability != 1)) or availability.sum() < 1:
            raise ValueError("Overlap detector availability must be non-empty and binary")
        if not np.isfinite(features).all() or not np.isfinite(chirp).all() or not np.isfinite(glitch).all():
            raise ValueError(f"Overlap tensor contains non-finite values: {row['mixture_id']}")
        if np.any((chirp != 0) & (chirp != 1)) or np.any((glitch != 0) & (glitch != 1)):
            raise ValueError("Overlap masks must be binary")
        unavailable = availability == 0
        if np.any(features[unavailable] != 0) or np.any(chirp[unavailable] != 0):
            raise ValueError("Unavailable overlap detector planes must be zero")
        item = (
            features.reshape(self.input_channels, self.frequency_bins, self.time_bins),
            np.stack([chirp, glitch]).reshape(
                2, self.input_channels, self.frequency_bins, self.time_bins
            ),
            availability,
        )
        if self.cache is not None:
            self.cache[index] = item
        return item


def _forward(model: Any, architecture: str, features: Any, availability: Any) -> Any:
    if architecture == "detector_set":
        return model(features, availability)
    return model(features)


def _availability_mask(availability: Any, q_count: int) -> Any:
    return availability[:, :, None].expand(-1, -1, q_count).reshape(
        availability.shape[0], 1, -1, 1, 1
    )


def _masked_focal_dice(
    logits: Any,
    targets: Any,
    availability: Any,
    q_count: int,
    positive_weights: tuple[float, float],
    class_weights: tuple[float, float],
    gamma: float,
) -> Any:
    mask = _availability_mask(availability, q_count).to(logits)
    positive = torch.as_tensor(positive_weights, device=logits.device).reshape(1, 2, 1, 1, 1)
    raw = torch_functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive, reduction="none"
    )
    if gamma > 0:
        probability = torch.sigmoid(logits)
        correct = probability * targets + (1.0 - probability) * (1.0 - targets)
        raw = raw * ((1.0 - correct) ** gamma)
    weights = torch.as_tensor(class_weights, device=logits.device).reshape(1, 2, 1, 1, 1)
    normalizer = mask.sum() * logits.shape[-2] * logits.shape[-1] * weights.sum()
    bce = (raw * mask * weights).sum() / normalizer.clamp_min(1.0)
    masked_probability = torch.sigmoid(logits) * mask
    masked_target = targets * mask
    axes = (2, 3, 4)
    intersection = (masked_probability * masked_target).sum(dim=axes)
    denominator = masked_probability.sum(dim=axes) + masked_target.sum(dim=axes)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice = (dice * weights.reshape(1, 2)).sum() / (weights.sum() * logits.shape[0])
    return bce + dice


def _counts(logits: Any, targets: Any, availability: Any, q_count: int, thresholds: tuple[float, float]) -> np.ndarray:
    mask = _availability_mask(availability, q_count).to(dtype=torch.bool, device=logits.device)
    threshold = torch.as_tensor(thresholds, device=logits.device).reshape(1, 2, 1, 1, 1)
    predicted = (torch.sigmoid(logits) >= threshold) & mask
    expected = (targets >= 0.5) & mask
    axes = (0, 2, 3, 4)
    tp = (predicted & expected).sum(dim=axes)
    fp = (predicted & ~expected & mask).sum(dim=axes)
    fn = (~predicted & expected).sum(dim=axes)
    return torch.stack([tp, fp, fn], dim=1).cpu().numpy()


def _metrics(counts: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ious = []
    for index, name in enumerate(("chirp", "glitch")):
        tp, fp, fn = (float(value) for value in counts[index])
        iou = tp / max(tp + fp + fn, 1.0)
        result[name] = {
            "precision": tp / max(tp + fp, 1.0),
            "recall": tp / max(tp + fn, 1.0),
            "iou": iou,
            "dice": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
            "counts": {"true_positive": int(tp), "false_positive": int(fp), "false_negative": int(fn)},
        }
        ious.append(iou)
    result["mean_iou"] = float(np.mean(ious))
    return result


def summarize_glitch_family_counts(
    counts_by_family: dict[str, np.ndarray], row_counts: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Report hand-auditable mask counts and metrics for each physical glitch family."""

    if set(counts_by_family) != set(row_counts):
        raise ValueError("Glitch-family metric count keys differ")
    result = {}
    for label in sorted(counts_by_family):
        counts = np.asarray(counts_by_family[label], dtype=np.int64)
        if counts.shape != (2, 3) or np.any(counts < 0) or row_counts[label] <= 0:
            raise ValueError(f"Invalid overlap counts for glitch family {label}")
        result[label] = {
            "physical_rows": int(row_counts[label]),
            **_metrics(counts)["glitch"],
        }
    return result


def promote_overlap_sampling_arm(
    uniform_report_path: str | Path,
    family_balanced_report_path: str | Path,
    overlap_train_manifest: str | Path,
    overlap_validation_manifest: str | Path,
    gravityspy_corpus_audit: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Select a one-seed sampling arm using frozen validation-only criteria."""

    config = load_yaml(config_path)
    settings = config.get("overlap_sampling_promotion")
    if not isinstance(settings, dict):
        raise ValueError("Overlap sampling promotion configuration is missing")
    audit = json.loads(Path(gravityspy_corpus_audit).read_text(encoding="utf-8"))
    if (
        audit.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or not audit.get("passed")
    ):
        raise ValueError("Overlap promotion requires a passed network corpus audit")
    audit_hash = file_sha256(gravityspy_corpus_audit)

    manifest_paths = {
        "train": Path(overlap_train_manifest),
        "val": Path(overlap_validation_manifest),
    }
    manifest_hashes = {split: file_sha256(path) for split, path in manifest_paths.items()}
    for split, path in manifest_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or any(row.get("split") != split for row in rows):
            raise ValueError(f"Overlap promotion received an invalid {split} manifest")
        bound_hashes = {row.get("gravityspy_corpus_audit_sha256") for row in rows}
        if bound_hashes != {audit_hash}:
            raise ValueError(f"Overlap {split} rows are not bound to the corpus audit")

    reports = {
        "uniform": json.loads(Path(uniform_report_path).read_text(encoding="utf-8")),
        "family_balanced": json.loads(
            Path(family_balanced_report_path).read_text(encoding="utf-8")
        ),
    }
    common_fields = (
        "overlap_train_manifest_sha256",
        "overlap_validation_manifest_sha256",
        "clean_train_manifest_sha256",
        "clean_validation_manifest_sha256",
        "pretrained_checkpoint_sha256",
        "seed",
    )
    for name, report in reports.items():
        if report.get("status") != "validation_selected_real_glitch_overlap_finetune":
            raise ValueError(f"Overlap {name} report is not validation-selected")
        if report.get("overlap_train_manifest_sha256") != manifest_hashes["train"]:
            raise ValueError(f"Overlap {name} report uses another training manifest")
        if report.get("overlap_validation_manifest_sha256") != manifest_hashes["val"]:
            raise ValueError(f"Overlap {name} report uses another validation manifest")
    mismatches = {
        field: [reports[name].get(field) for name in reports]
        for field in common_fields
        if len({json.dumps(reports[name].get(field), sort_keys=True) for name in reports}) != 1
    }
    if mismatches:
        raise ValueError(f"Overlap sampling arms are not paired: {mismatches}")

    minimum_clean_retention = float(settings["minimum_clean_chirp_iou_retention"])
    minimum_glitch_iou = float(settings["minimum_glitch_iou"])
    minimum_family_median = float(settings["minimum_family_median_iou"])
    maximum_zero_families = int(settings["maximum_zero_iou_families"])
    minimum_family_rows = int(settings["minimum_validation_rows_per_family"])

    summaries = {}
    for name, report in reports.items():
        best_epoch = int(report["best_epoch"])
        selected_history = [
            row for row in report["history"] if int(row["epoch"]) == best_epoch
        ]
        if len(selected_history) != 1 or not selected_history[0].get("checkpoint_eligible"):
            raise ValueError(f"Overlap {name} checkpoint was not retention-eligible")
        retention = float(selected_history[0]["clean_chirp_iou_retention"])
        metrics = report["calibrated_overlap_validation"]
        families = metrics.get("by_glitch_family", {})
        if not families:
            raise ValueError(f"Overlap {name} report lacks family validation metrics")
        if any(int(row["physical_rows"]) < minimum_family_rows for row in families.values()):
            raise ValueError(f"Overlap {name} has an underpowered validation family")
        family_ious = np.asarray([float(row["iou"]) for row in families.values()])
        absolute_checks = {
            "clean_retention": retention >= minimum_clean_retention,
            "glitch_iou": float(metrics["glitch"]["iou"]) >= minimum_glitch_iou,
            "family_median_iou": float(np.median(family_ious)) >= minimum_family_median,
            "zero_iou_families": int(np.count_nonzero(family_ious == 0))
            <= maximum_zero_families,
        }
        summaries[name] = {
            "clean_chirp_iou_retention": retention,
            "chirp_iou": float(metrics["chirp"]["iou"]),
            "glitch_iou": float(metrics["glitch"]["iou"]),
            "worst_family_iou": float(family_ious.min()),
            "median_family_iou": float(np.median(family_ious)),
            "zero_iou_families": int(np.count_nonzero(family_ious == 0)),
            "family_ious": {label: float(row["iou"]) for label, row in families.items()},
            "absolute_checks": absolute_checks,
            "absolute_passed": all(absolute_checks.values()),
        }

    uniform = summaries["uniform"]
    balanced = summaries["family_balanced"]
    regression_tolerance = float(settings["maximum_family_regression"])
    regressed_families = sorted(
        label
        for label, uniform_iou in uniform["family_ious"].items()
        if balanced["family_ious"].get(label, float("-inf"))
        < uniform_iou - regression_tolerance
    )
    comparison_checks = {
        "overall_glitch_delta": balanced["glitch_iou"] - uniform["glitch_iou"]
        >= float(settings["balanced_minimum_overall_glitch_delta"]),
        "chirp_delta": balanced["chirp_iou"] - uniform["chirp_iou"]
        >= float(settings["balanced_minimum_chirp_delta"]),
        "worst_family_delta": balanced["worst_family_iou"] - uniform["worst_family_iou"]
        >= float(settings["balanced_minimum_worst_family_delta"]),
        "median_family_delta": balanced["median_family_iou"] - uniform["median_family_iou"]
        >= float(settings["balanced_minimum_median_family_delta"]),
        "regressed_family_count": len(regressed_families)
        <= int(settings["maximum_regressed_families"]),
    }
    if balanced["absolute_passed"] and all(comparison_checks.values()):
        promoted = "family_balanced"
    elif uniform["absolute_passed"]:
        promoted = "uniform"
    else:
        promoted = None
    result = {
        "status": "validation_only_overlap_sampling_promotion",
        "passed": promoted is not None,
        "scientific_claim_allowed": False,
        "test_data_opened": False,
        "promoted_arm": promoted,
        "scale_to_five_seeds": promoted is not None,
        "summaries": summaries,
        "family_balanced_comparison_checks": comparison_checks,
        "family_balanced_regressed_families": regressed_families,
        "corpus_audit_path": str(gravityspy_corpus_audit),
        "corpus_audit_sha256": audit_hash,
        "overlap_manifest_hashes": manifest_hashes,
        "input_report_hashes": {
            "uniform": file_sha256(uniform_report_path),
            "family_balanced": file_sha256(family_balanced_report_path),
        },
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def promote_single_overlap_arm(
    finetune_report_path: str | Path,
    promotion_config_path: str | Path,
    output_path: str | Path,
    arm: str,
) -> dict[str, Any]:
    """Gate a predeclared single-arm fallback before five-seed expansion."""

    expected_scopes = {"glitch_adapter": "glitch_adapter_only"}
    if arm not in expected_scopes:
        raise ValueError(f"Unsupported single overlap arm: {arm}")
    report_path = Path(finetune_report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scope = report.get("training_scope", {})
    if (
        report.get("status") != "validation_selected_real_glitch_overlap_finetune"
        or report.get("scientific_claim_allowed") is not False
        or report.get("search_claim_allowed") is not False
        or report.get("checkpoint_selection_metric") != "validation_loss"
        or scope.get("scope") != expected_scopes[arm]
        or scope.get("non_glitch_state_preserved_bit_exact") is not True
    ):
        raise ValueError("Single-arm overlap report violates its validation-only scope")
    checkpoint_path = Path(str(report.get("checkpoint_path", ""))).resolve()
    if (
        not checkpoint_path.is_file()
        or report.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError("Single-arm overlap checkpoint failed hash replay")

    config_path = Path(promotion_config_path).resolve()
    config = load_yaml(config_path)
    settings = config.get("overlap_sampling_promotion")
    if not isinstance(settings, dict):
        raise ValueError("Overlap sampling promotion configuration is missing")
    required_settings = {
        "minimum_clean_chirp_iou_retention",
        "minimum_glitch_iou",
        "minimum_family_median_iou",
        "maximum_zero_iou_families",
        "minimum_validation_rows_per_family",
    }
    missing = sorted(required_settings - set(settings))
    if missing:
        raise ValueError(f"Single-arm promotion config omits settings: {missing}")

    selected_history = [
        row
        for row in report.get("history", [])
        if int(row["epoch"]) == int(report["best_epoch"])
    ]
    if len(selected_history) != 1 or not selected_history[0].get(
        "checkpoint_eligible"
    ):
        raise ValueError("Single-arm overlap checkpoint was not retention-eligible")
    retention = float(selected_history[0]["clean_chirp_iou_retention"])
    metrics = report.get("calibrated_overlap_validation", {})
    families = metrics.get("by_glitch_family", {})
    if not families:
        raise ValueError("Single-arm overlap report lacks family validation metrics")
    minimum_family_rows = int(settings["minimum_validation_rows_per_family"])
    if any(
        int(row.get("physical_rows", -1)) < minimum_family_rows
        for row in families.values()
    ):
        raise ValueError("Single-arm overlap report has an underpowered family")
    family_ious = np.asarray(
        [float(row["iou"]) for row in families.values()], dtype=np.float64
    )
    checks = {
        "clean_retention": retention
        >= float(settings["minimum_clean_chirp_iou_retention"]),
        "glitch_iou": float(metrics["glitch"]["iou"])
        >= float(settings["minimum_glitch_iou"]),
        "family_median_iou": float(np.median(family_ious))
        >= float(settings["minimum_family_median_iou"]),
        "zero_iou_families": int(np.count_nonzero(family_ious == 0))
        <= int(settings["maximum_zero_iou_families"]),
    }
    passed = all(checks.values())
    summary = {
        "clean_chirp_iou_retention": retention,
        "chirp_iou": float(metrics["chirp"]["iou"]),
        "glitch_iou": float(metrics["glitch"]["iou"]),
        "worst_family_iou": float(family_ious.min()),
        "median_family_iou": float(np.median(family_ious)),
        "zero_iou_families": int(np.count_nonzero(family_ious == 0)),
        "family_ious": {
            label: float(row["iou"]) for label, row in sorted(families.items())
        },
        "absolute_checks": checks,
        "absolute_passed": passed,
    }
    result = {
        "status": "validation_only_overlap_single_arm_promotion",
        "passed": passed,
        "scientific_claim_allowed": False,
        "search_claim_allowed": False,
        "test_data_opened": False,
        "promoted_arm": arm if passed else None,
        "candidate_arm": arm,
        "scale_to_five_seeds": passed,
        "summaries": {arm: summary},
        "input_report_hashes": {arm: file_sha256(report_path)},
        "input_report_paths": {arm: str(report_path)},
        "source_training_scope": scope,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def summarize_overlap_five_seed_promotion(
    promotion_report_path: str | Path,
    finetune_report_paths: list[str | Path],
    stability_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate exactly five validation-selected runs of the promoted arm."""

    promotion = json.loads(Path(promotion_report_path).read_text(encoding="utf-8"))
    promoted = promotion.get("promoted_arm")
    status = promotion.get("status")
    accepted = (
        status == "validation_only_overlap_sampling_promotion"
        and promoted in {"uniform", "family_balanced"}
    ) or (
        status == "validation_only_overlap_single_arm_promotion"
        and promoted == "glitch_adapter"
    )
    if (
        not accepted
        or not promotion.get("passed")
        or not promotion.get("scale_to_five_seeds")
    ):
        raise ValueError("Overlap sampling promotion did not authorize five seeds")
    paths = [Path(path) for path in finetune_report_paths]
    if len(paths) != 5:
        raise ValueError("Overlap promotion summary requires exactly five reports")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        report.get("status") != "validation_selected_real_glitch_overlap_finetune"
        for report in reports
    ):
        raise ValueError("A five-seed overlap report is not validation-selected")
    if promoted == "glitch_adapter" and any(
        report.get("checkpoint_selection_metric") != "validation_loss"
        or report.get("training_scope", {}).get("scope") != "glitch_adapter_only"
        or report.get("training_scope", {}).get(
            "non_glitch_state_preserved_bit_exact"
        )
        is not True
        for report in reports
    ):
        raise ValueError("A five-seed glitch-adapter report violates its frozen scope")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != 5:
        raise ValueError("Five-seed overlap reports do not contain five unique seeds")
    report_hashes = {file_sha256(path) for path in paths}
    promoted_hash = str(promotion["input_report_hashes"][promoted])
    if promoted_hash not in report_hashes:
        raise ValueError("Five-seed reports omit the one-seed promoted checkpoint report")
    common_fields = (
        "config_hash",
        "config_file_sha256",
        "overlap_train_manifest_sha256",
        "overlap_validation_manifest_sha256",
        "clean_train_manifest_sha256",
        "clean_validation_manifest_sha256",
        "pretrained_checkpoint_sha256",
    )
    for field in common_fields:
        if len({str(report.get(field)) for report in reports}) != 1:
            raise ValueError(f"Five-seed overlap reports differ in {field}")

    family_labels = [
        set(report["calibrated_overlap_validation"]["by_glitch_family"])
        for report in reports
    ]
    if any(labels != family_labels[0] for labels in family_labels[1:]):
        raise ValueError("Five-seed overlap reports differ in validation families")

    promotion_config_path = Path(str(promotion.get("config_path", "")))
    if not promotion_config_path.is_file():
        raise ValueError("Five-seed promotion config is absent")
    promotion_config = load_yaml(promotion_config_path)
    if canonical_hash(promotion_config) != promotion.get("config_hash"):
        raise ValueError("Five-seed promotion config changed after one-seed selection")
    promotion_settings = promotion_config.get("overlap_sampling_promotion", {})
    required_promotion_settings = {
        "minimum_clean_chirp_iou_retention",
        "minimum_glitch_iou",
        "minimum_family_median_iou",
        "maximum_zero_iou_families",
        "minimum_validation_rows_per_family",
    }
    missing_promotion = sorted(required_promotion_settings - set(promotion_settings))
    if missing_promotion:
        raise ValueError(
            f"Five-seed promotion config omits per-seed settings: {missing_promotion}"
        )
    stability_path = Path(stability_config_path).resolve()
    stability_config = load_yaml(stability_path)
    stability_settings = stability_config.get("overlap_five_seed_stability", {})
    required_stability_settings = {
        "minimum_passing_seed_fraction",
        "minimum_median_clean_retention",
        "minimum_median_glitch_iou",
        "minimum_median_family_iou",
    }
    missing_stability = sorted(
        required_stability_settings - set(stability_settings)
    )
    if missing_stability:
        raise ValueError(
            f"Five-seed stability config omits settings: {missing_stability}"
        )
    minimum_passing_fraction = float(stability_settings["minimum_passing_seed_fraction"])
    if not 0 < minimum_passing_fraction <= 1:
        raise ValueError("Five-seed passing fraction must lie in (0, 1]")

    def selected_retention(report: dict[str, Any]) -> float:
        rows = [
            row
            for row in report["history"]
            if int(row["epoch"]) == int(report["best_epoch"])
        ]
        if len(rows) != 1 or not rows[0].get("checkpoint_eligible"):
            raise ValueError("A five-seed overlap checkpoint is not retention-eligible")
        return float(rows[0]["clean_chirp_iou_retention"])

    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "sample_standard_deviation": float(array.std(ddof=1)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }

    metrics = {
        "clean_chirp_iou_retention": summary(
            [selected_retention(report) for report in reports]
        ),
        "overlap_chirp_iou": summary(
            [
                float(report["calibrated_overlap_validation"]["chirp"]["iou"])
                for report in reports
            ]
        ),
        "overlap_glitch_iou": summary(
            [
                float(report["calibrated_overlap_validation"]["glitch"]["iou"])
                for report in reports
            ]
        ),
    }
    minimum_family_rows = int(
        promotion_settings["minimum_validation_rows_per_family"]
    )
    seed_audits = []
    for report in reports:
        overlap_metrics = report["calibrated_overlap_validation"]
        families = overlap_metrics["by_glitch_family"]
        if any(
            int(row.get("physical_rows", -1)) < minimum_family_rows
            for row in families.values()
        ):
            raise ValueError("A five-seed report has an underpowered validation family")
        family_ious = np.asarray(
            [float(row["iou"]) for row in families.values()], dtype=np.float64
        )
        clean_retention = selected_retention(report)
        checks = {
            "clean_retention": clean_retention
            >= float(promotion_settings["minimum_clean_chirp_iou_retention"]),
            "glitch_iou": float(overlap_metrics["glitch"]["iou"])
            >= float(promotion_settings["minimum_glitch_iou"]),
            "family_median_iou": float(np.median(family_ious))
            >= float(promotion_settings["minimum_family_median_iou"]),
            "zero_iou_families": int(np.count_nonzero(family_ious == 0))
            <= int(promotion_settings["maximum_zero_iou_families"]),
        }
        seed_audits.append(
            {
                "seed": int(report["seed"]),
                "clean_chirp_iou_retention": clean_retention,
                "glitch_iou": float(overlap_metrics["glitch"]["iou"]),
                "median_family_iou": float(np.median(family_ious)),
                "zero_iou_families": int(np.count_nonzero(family_ious == 0)),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passing_seed_ids = sorted(
        row["seed"] for row in seed_audits if row["passed"]
    )
    passing_fraction = len(passing_seed_ids) / len(seed_audits)
    median_clean_retention = float(
        np.median([row["clean_chirp_iou_retention"] for row in seed_audits])
    )
    median_glitch_iou = float(
        np.median([row["glitch_iou"] for row in seed_audits])
    )
    median_family_iou = float(
        np.median([row["median_family_iou"] for row in seed_audits])
    )
    stability_checks = {
        "minimum_passing_seed_fraction": passing_fraction
        >= minimum_passing_fraction,
        "minimum_median_clean_retention": median_clean_retention
        >= float(stability_settings["minimum_median_clean_retention"]),
        "minimum_median_glitch_iou": median_glitch_iou
        >= float(stability_settings["minimum_median_glitch_iou"]),
        "minimum_median_family_iou": median_family_iou
        >= float(stability_settings["minimum_median_family_iou"]),
    }
    stability_passed = all(stability_checks.values())
    by_family = {
        label: summary(
            [
                float(
                    report["calibrated_overlap_validation"]["by_glitch_family"][label][
                        "iou"
                    ]
                )
                for report in reports
            ]
        )
        for label in sorted(family_labels[0])
    }
    passing_reports = [
        report
        for report, audit in zip(reports, seed_audits)
        if audit["passed"]
    ]
    ranked = sorted(
        passing_reports,
        key=lambda report: (
            -float(report["calibrated_overlap_validation"]["mean_iou"]),
            int(report["seed"]),
        ),
    )
    selected = ranked[0] if stability_passed else None
    selected_checkpoint = Path(selected["checkpoint_path"]) if selected else None
    if selected_checkpoint is not None and file_sha256(selected_checkpoint) != str(
        selected["checkpoint_sha256"]
    ):
        raise ValueError("Five-seed selected checkpoint hash differs from its report")
    result = {
        "status": "completed_five_seed_source_safe_overlap_validation",
        "passed": stability_passed,
        "scientific_claim_allowed": False,
        "test_data_opened": False,
        "promoted_arm": promoted,
        "seeds": sorted(seeds),
        "metrics": metrics,
        "by_glitch_family": by_family,
        "five_seed_stability": {
            "status": "five_seed_reproducibility_gate_v1",
            "passed": stability_passed,
            "config_path": str(stability_path),
            "config_sha256": file_sha256(stability_path),
            "config_hash": canonical_hash(stability_config),
            "minimum_required_passing_fraction": minimum_passing_fraction,
            "passing_seed_fraction": passing_fraction,
            "passing_seeds": passing_seed_ids,
            "median_clean_chirp_iou_retention": median_clean_retention,
            "median_glitch_iou": median_glitch_iou,
            "median_family_iou": median_family_iou,
            "checks": stability_checks,
            "seed_audits": sorted(seed_audits, key=lambda row: row["seed"]),
        },
        "checkpoint_selection": (
            "maximum_validation_overlap_mean_iou_among_passing_seeds_then_seed"
            if selected
            else None
        ),
        "selected_seed": int(selected["seed"]) if selected else None,
        "selected_validation_overlap_mean_iou": (
            float(selected["calibrated_overlap_validation"]["mean_iou"])
            if selected
            else None
        ),
        "selected_checkpoint_path": str(selected_checkpoint) if selected else None,
        "selected_checkpoint_sha256": (
            str(selected["checkpoint_sha256"]) if selected else None
        ),
        "promotion_report_path": str(promotion_report_path),
        "promotion_report_sha256": file_sha256(promotion_report_path),
        "finetune_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "common_artifact_hashes": {
            field: reports[0].get(field) for field in common_fields
        },
        "required_next_gates": [
            "continuous_background_far_ifar_vt",
            "paired_real_glitch_functional_mask_endpoints",
            "locked_o4a_then_o4b_evaluation",
        ],
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def bind_glitch_adapter_five_seed_gate(
    original_report_path: str | Path,
    promotion_report_path: str | Path,
    five_seed_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind a positive adapter five-seed result for downstream consumers."""

    paths = {
        "original_adapter_report": Path(original_report_path).resolve(),
        "one_seed_promotion": Path(promotion_report_path).resolve(),
        "five_seed_summary": Path(five_seed_summary_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Adapter five-seed gate input is absent")
    original = json.loads(
        paths["original_adapter_report"].read_text(encoding="utf-8")
    )
    promotion = json.loads(paths["one_seed_promotion"].read_text(encoding="utf-8"))
    summary = json.loads(paths["five_seed_summary"].read_text(encoding="utf-8"))
    original_hash = file_sha256(paths["original_adapter_report"])
    promotion_hash = file_sha256(paths["one_seed_promotion"])
    finetune_hashes = {
        str(identity.get("sha256"))
        for identity in summary.get("finetune_reports", [])
    }
    if (
        original.get("status")
        != "validation_selected_real_glitch_overlap_finetune"
        or original.get("scientific_claim_allowed") is not False
        or original.get("search_claim_allowed") is not False
        or original.get("checkpoint_selection_metric") != "validation_loss"
        or original.get("training_scope", {}).get("scope")
        != "glitch_adapter_only"
        or original.get("training_scope", {}).get(
            "non_glitch_state_preserved_bit_exact"
        )
        is not True
        or promotion.get("status")
        != "validation_only_overlap_single_arm_promotion"
        or promotion.get("passed") is not True
        or promotion.get("promoted_arm") != "glitch_adapter"
        or promotion.get("test_data_opened") is not False
        or promotion.get("input_report_hashes", {}).get("glitch_adapter")
        != original_hash
        or summary.get("status")
        != "completed_five_seed_source_safe_overlap_validation"
        or summary.get("passed") is not True
        or summary.get("promoted_arm") != "glitch_adapter"
        or summary.get("test_data_opened") is not False
        or summary.get("five_seed_stability", {}).get("status")
        != "five_seed_reproducibility_gate_v1"
        or summary.get("five_seed_stability", {}).get("passed") is not True
        or len(summary.get("seeds", [])) != 5
        or original_hash not in finetune_hashes
        or summary.get("promotion_report_sha256") != promotion_hash
    ):
        raise ValueError("Adapter five-seed gate failed provenance replay")
    checkpoint = Path(str(summary.get("selected_checkpoint_path", ""))).resolve()
    if (
        not checkpoint.is_file()
        or file_sha256(checkpoint) != summary.get("selected_checkpoint_sha256")
    ):
        raise ValueError("Adapter five-seed selected checkpoint failed hash replay")
    result = {
        "status": "completed_glitch_adapter_five_seed_gate",
        "execution_passed": True,
        "five_seed_promoted": True,
        "scientific_claim_allowed": False,
        "search_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "artifacts": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in paths.items()
        },
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def replay_overlap_five_seed_stability(
    source_summary_path: str | Path,
    stability_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute a frozen five-seed decision without repeating model training."""

    source_path = Path(source_summary_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("status")
        != "completed_five_seed_source_safe_overlap_validation"
        or source.get("test_data_opened") is not False
        or source.get("scientific_claim_allowed") is not False
    ):
        raise ValueError("Five-seed stability replay source is not validation-only")
    promotion_path = Path(str(source.get("promotion_report_path", ""))).resolve()
    if (
        not promotion_path.is_file()
        or source.get("promotion_report_sha256") != file_sha256(promotion_path)
    ):
        raise ValueError("Five-seed stability replay promotion report changed")
    identities = source.get("finetune_reports", [])
    if len(identities) != 5:
        raise ValueError("Five-seed stability replay requires exactly five reports")
    report_paths = []
    for identity in identities:
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or identity.get("sha256") != file_sha256(path):
            raise ValueError("Five-seed stability replay report changed")
        report_paths.append(path)
    result = summarize_overlap_five_seed_promotion(
        promotion_path,
        report_paths,
        stability_config_path,
        output_path,
    )
    result["stability_replay_source"] = {
        "path": str(source_path),
        "sha256": file_sha256(source_path),
    }
    atomic_write_json(output_path, result)
    return result


def summarize_physical_overlap_data_scaling(
    subset_report_path: str | Path,
    finetune_report_paths: list[str | Path],
    output_path: str | Path,
    minimum_seeds: int = 5,
    minimum_material_glitch_iou_gain: float = 0.01,
    minimum_clean_chirp_iou_retention: float = 0.95,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    """Summarize paired-seed fixed-epoch/fixed-update overlap scaling curves."""

    if minimum_seeds < 5:
        raise ValueError("Publication overlap scaling requires at least five seeds")
    if minimum_material_glitch_iou_gain <= 0:
        raise ValueError("Material overlap-scaling gain must be positive")
    if not 0 < minimum_clean_chirp_iou_retention <= 1:
        raise ValueError("Clean chirp retention threshold must lie in (0, 1]")
    if bootstrap_replicates < 100:
        raise ValueError("Overlap scaling requires at least 100 bootstrap replicates")
    subset_path = Path(subset_report_path)
    subset_report = json.loads(subset_path.read_text(encoding="utf-8"))
    if (
        subset_report.get("status")
        != "frozen_group_safe_physical_overlap_scaling_subsets"
        or subset_report.get("passed") is not True
        or subset_report.get("test_rows_read") != 0
        or subset_report.get("required_training_controls")
        != ["fixed_epochs", "fixed_optimizer_updates"]
    ):
        raise ValueError("Physical overlap scaling subset report did not pass")
    audit_identity = subset_report.get("train_validation_group_audit", {})
    audit_path = Path(str(audit_identity.get("path", "")))
    if not audit_path.is_file() or file_sha256(audit_path) != audit_identity.get(
        "sha256"
    ):
        raise ValueError("Physical overlap scaling group audit hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed_physical_overlap_group_audit" or not audit.get(
        "passed"
    ):
        raise ValueError("Physical overlap scaling group audit did not pass")
    corpus_identity = subset_report.get("gravityspy_corpus_audit", {})
    corpus_path = Path(str(corpus_identity.get("path", "")))
    if not corpus_path.is_file() or file_sha256(corpus_path) != corpus_identity.get(
        "sha256"
    ):
        raise ValueError("Physical overlap scaling corpus audit hash mismatch")
    corpus_audit = json.loads(corpus_path.read_text(encoding="utf-8"))
    if (
        corpus_audit.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or corpus_audit.get("passed") is not True
    ):
        raise ValueError("Physical overlap scaling corpus audit did not pass")

    subset_by_hash = {}
    for identity in subset_report.get("subsets", []):
        manifest = Path(str(identity.get("manifest_path", "")))
        if not manifest.is_file() or file_sha256(manifest) != identity.get(
            "manifest_sha256"
        ):
            raise ValueError("Physical overlap scaling subset hash mismatch")
        subset_by_hash[str(identity["manifest_sha256"])] = int(identity["scale"])
    declared_scales = [int(value) for value in subset_report.get("scales", [])]
    if sorted(set(subset_by_hash.values())) != declared_scales:
        raise ValueError("Physical overlap scaling subset identities are incomplete")

    controls = ("fixed_epochs", "fixed_optimizer_updates")
    cells: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
        control: {scale: {} for scale in declared_scales} for control in controls
    }
    paths = [Path(path) for path in finetune_report_paths]
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "validation_selected_real_glitch_overlap_finetune":
            raise ValueError("Overlap scaling input is not validation-selected")
        control = str(report.get("training_control", {}).get("control"))
        if control not in cells:
            raise ValueError("Overlap scaling input lacks a declared training control")
        scale = subset_by_hash.get(str(report.get("overlap_train_manifest_sha256")))
        if scale is None:
            raise ValueError("Overlap scaling input uses an undeclared training subset")
        if (
            report.get("overlap_validation_manifest_sha256")
            != subset_report.get("validation_manifest_sha256")
            or report.get("split_audit", {}).get("passed") is not True
            or any(
                report.get("split_audit", {})
                .get("cross_split_overlaps", {})
                .get(field, [])
                for field in report.get("split_audit", {}).get(
                    "cross_split_overlaps", {}
                )
            )
        ):
            raise ValueError("Overlap scaling input differs from the frozen validation endpoint")
        seed = int(report["seed"])
        if seed in cells[control][scale]:
            raise ValueError("Overlap scaling cell repeats a seed")
        checkpoint = Path(str(report.get("checkpoint_path", "")))
        if not checkpoint.is_file() or file_sha256(checkpoint) != report.get(
            "checkpoint_sha256"
        ):
            raise ValueError("Overlap scaling checkpoint hash mismatch")
        if control == "fixed_optimizer_updates" and (
            int(report.get("completed_optimizer_updates", -1))
            != int(report["training_control"].get("target_optimizer_updates", -2))
        ):
            raise ValueError("Fixed-update overlap scaling run missed its update target")
        cells[control][scale][seed] = report

    seed_sets = []
    for control in controls:
        config_hashes = set()
        for scale in declared_scales:
            reports = cells[control][scale]
            if len(reports) < minimum_seeds:
                raise ValueError("Overlap scaling cell has too few independent seeds")
            seed_sets.append(set(reports))
            config_hashes.update(str(row.get("config_file_sha256")) for row in reports.values())
        if len(config_hashes) != 1:
            raise ValueError("Overlap scaling control changes configuration across scales")
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Overlap scaling cells do not use paired seed identities")
    paired_seeds = sorted(seed_sets[0])

    common_fields = (
        "overlap_validation_manifest_sha256",
        "clean_train_manifest_sha256",
        "clean_validation_manifest_sha256",
        "pretrained_checkpoint_sha256",
    )
    all_reports = [
        report
        for control in controls
        for scale in declared_scales
        for report in cells[control][scale].values()
    ]
    for field in common_fields:
        if len({str(report.get(field)) for report in all_reports}) != 1:
            raise ValueError(f"Overlap scaling inputs differ in {field}")

    def selected_retention(report: dict[str, Any]) -> float:
        selected = [
            row
            for row in report.get("history", [])
            if int(row.get("epoch", -1)) == int(report.get("best_epoch", -2))
        ]
        if len(selected) != 1 or selected[0].get("checkpoint_eligible") is not True:
            raise ValueError("Overlap scaling checkpoint is not clean-retention eligible")
        return float(selected[0]["clean_chirp_iou_retention"])

    def metrics(report: dict[str, Any]) -> dict[str, float]:
        overlap = report["calibrated_overlap_validation"]
        return {
            "overlap_mean_iou": float(overlap["mean_iou"]),
            "overlap_chirp_iou": float(overlap["chirp"]["iou"]),
            "overlap_glitch_iou": float(overlap["glitch"]["iou"]),
            "clean_chirp_iou_retention": selected_retention(report),
        }

    def scalar_summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "sample_standard_deviation": float(array.std(ddof=1)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }

    summaries = {}
    for control in controls:
        summaries[control] = {}
        for scale in declared_scales:
            by_seed = {
                seed: metrics(cells[control][scale][seed]) for seed in paired_seeds
            }
            summaries[control][str(scale)] = {
                key: scalar_summary([by_seed[seed][key] for seed in paired_seeds])
                for key in next(iter(by_seed.values()))
            }

    rng = np.random.default_rng(bootstrap_seed)
    comparisons: dict[str, list[dict[str, Any]]] = {control: [] for control in controls}
    for control in controls:
        for lower, upper in zip(declared_scales, declared_scales[1:]):
            deltas = np.asarray(
                [
                    metrics(cells[control][upper][seed])["overlap_glitch_iou"]
                    - metrics(cells[control][lower][seed])["overlap_glitch_iou"]
                    for seed in paired_seeds
                ],
                dtype=np.float64,
            )
            sampled = deltas[
                rng.integers(0, deltas.size, size=(bootstrap_replicates, deltas.size))
            ].mean(axis=1)
            lower_ci, upper_ci = np.quantile(sampled, [0.025, 0.975])
            comparisons[control].append(
                {
                    "lower_scale": lower,
                    "upper_scale": upper,
                    "scale_ratio": upper / lower,
                    "paired_seed_deltas": {
                        str(seed): float(delta)
                        for seed, delta in zip(paired_seeds, deltas)
                    },
                    "mean_glitch_iou_delta": float(deltas.mean()),
                    "paired_bootstrap_95_interval": [
                        float(lower_ci),
                        float(upper_ci),
                    ],
                    "material_positive_gain": bool(
                        deltas.mean() >= minimum_material_glitch_iou_gain
                        and lower_ci > 0
                    ),
                }
            )

    doubling_pairs = [
        (lower, upper)
        for lower, upper in zip(declared_scales, declared_scales[1:])
        if upper / lower >= 1.8
    ]
    if not doubling_pairs:
        raise ValueError("Overlap scaling curve lacks an approximately doubled data step")
    promotion_pair = doubling_pairs[-1]
    control_checks = {}
    for control in controls:
        comparison = next(
            row
            for row in comparisons[control]
            if (row["lower_scale"], row["upper_scale"]) == promotion_pair
        )
        retention = summaries[control][str(promotion_pair[1])][
            "clean_chirp_iou_retention"
        ]["mean"]
        control_checks[control] = {
            "material_positive_gain": comparison["material_positive_gain"],
            "clean_non_inferiority": retention
            >= minimum_clean_chirp_iou_retention,
        }
    promote = all(all(checks.values()) for checks in control_checks.values())
    if promote:
        diagnosis = "data_limited_at_frozen_overlap_endpoint"
    elif control_checks["fixed_epochs"]["material_positive_gain"] and not control_checks[
        "fixed_optimizer_updates"
    ]["material_positive_gain"]:
        diagnosis = "epoch_budget_coupling_or_optimization_limited"
    else:
        diagnosis = "no_joint_control_evidence_for_more_same_distribution_data"
    result = {
        "status": "completed_group_safe_physical_overlap_data_scaling_curve",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation mask curve; continuous-background FAR/IFAR/<VT> and locked transfer "
            "remain required"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "scales": declared_scales,
        "paired_seeds": paired_seeds,
        "minimum_seeds": minimum_seeds,
        "summaries": summaries,
        "adjacent_scale_comparisons": comparisons,
        "promotion_data_doubling": list(promotion_pair),
        "promotion_checks": control_checks,
        "promote_more_same_distribution_data": promote,
        "diagnosis": diagnosis,
        "minimum_material_glitch_iou_gain": minimum_material_glitch_iou_gain,
        "minimum_clean_chirp_iou_retention": minimum_clean_chirp_iou_retention,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "subset_report_path": str(subset_path.resolve()),
        "subset_report_sha256": file_sha256(subset_path),
        "finetune_reports": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in paths
        ],
        "common_artifact_hashes": {
            field: all_reports[0].get(field) for field in common_fields
        },
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def _require_publication_commit() -> dict[str, Any]:
    provenance = execution_provenance(torch)
    commit = provenance.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
    ) is None:
        raise ValueError("Scaling hard endpoints require a full GWYOLO_CODE_COMMIT hash")
    return provenance


def run_physical_overlap_scaling_hard_endpoint_cell(
    config_path: str | Path,
    subset_report_path: str | Path,
    hard_subset_report_path: str | Path,
    finetune_report_path: str | Path,
    scale: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one scale/control/seed on a predeclared hard subset without refitting."""

    _require_torch()
    provenance = _require_publication_commit()
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Scaling hard-endpoint cell outputs are immutable")
    subset_path = Path(subset_report_path).resolve()
    subset_report = json.loads(subset_path.read_text(encoding="utf-8"))
    if (
        subset_report.get("status")
        != "frozen_group_safe_physical_overlap_scaling_subsets"
        or subset_report.get("passed") is not True
        or subset_report.get("test_rows_read") != 0
    ):
        raise ValueError("Scaling subset report did not pass")
    subset_identity = next(
        (
            row
            for row in subset_report.get("subsets", [])
            if int(row.get("scale", -1)) == int(scale)
        ),
        None,
    )
    if subset_identity is None:
        raise ValueError("Hard-endpoint cell scale is not declared")
    subset_manifest = Path(str(subset_identity.get("manifest_path", "")))
    if (
        not subset_manifest.is_file()
        or file_sha256(subset_manifest) != subset_identity.get("manifest_sha256")
    ):
        raise ValueError("Hard-endpoint cell training subset failed replay")

    hard_path = Path(hard_subset_report_path).resolve()
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if (
        hard.get("status")
        != "frozen_score_blind_physical_overlap_scaling_hard_subset"
        or hard.get("passed") is not True
        or hard.get("candidate_scores_inspected") is not False
        or hard.get("model_outputs_inspected") is not False
        or hard.get("test_rows_read") != 0
        or hard.get("validation_manifest_sha256")
        != subset_report.get("validation_manifest_sha256")
    ):
        raise ValueError("Scaling hard-subset report failed its score-blind gate")
    hard_manifest = Path(str(hard.get("hard_subset_manifest_path", "")))
    if (
        not hard_manifest.is_file()
        or file_sha256(hard_manifest) != hard.get("hard_subset_manifest_sha256")
    ):
        raise ValueError("Scaling hard-subset manifest failed replay")
    hard_rows = _read_rows(hard_manifest)
    required_strata = [str(value) for value in hard.get("required_strata", [])]
    if any(row.get("split") != "val" for row in hard_rows) or any(
        not set(row.get("hard_subset_strata", [])) <= set(required_strata)
        or not row.get("hard_subset_strata")
        for row in hard_rows
    ):
        raise ValueError("Scaling hard-subset rows are not valid frozen validation rows")

    finetune_path = Path(finetune_report_path).resolve()
    finetune = json.loads(finetune_path.read_text(encoding="utf-8"))
    if (
        finetune.get("status") != "validation_selected_real_glitch_overlap_finetune"
        or finetune.get("overlap_train_manifest_sha256")
        != subset_identity.get("manifest_sha256")
        or finetune.get("overlap_validation_manifest_sha256")
        != subset_report.get("validation_manifest_sha256")
        or finetune.get("test_evaluation") is not None
        or finetune.get("search_claim_allowed") is not False
    ):
        raise ValueError("Scaling hard-endpoint cell finetune report failed replay")
    if file_sha256(config_path) != finetune.get("config_file_sha256"):
        raise ValueError("Scaling hard-endpoint config differs from the finetune report")
    checkpoint_path = Path(str(finetune.get("checkpoint_path", "")))
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != finetune.get("checkpoint_sha256")
    ):
        raise ValueError("Scaling hard-endpoint checkpoint failed replay")

    config = load_yaml(config_path)
    settings = config.get("overlap_training")
    if not isinstance(settings, dict):
        raise ValueError("Scaling hard endpoint requires overlap_training config")
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    tensor = settings["tensor"]
    batch_size = int(settings["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, architecture = model_from_checkpoint(checkpoint, model_ifos, q_values)
    if architecture != "detector_set":
        raise ValueError("Scaling hard endpoint requires the detector-set model")
    model = model.to(device)
    thresholds = finetune.get("validation_selected_thresholds", {})
    frozen_thresholds = (float(thresholds["chirp"]), float(thresholds["glitch"]))

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        dataset = PhysicalOverlapDataset(
            rows,
            model_ifos,
            q_values,
            int(tensor["frequency_bins"]),
            int(tensor["time_bins"]),
            bool(settings.get("cache_in_memory", False)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return _overlap_epoch(
            model,
            loader,
            device,
            len(q_values),
            tuple(float(value) for value in settings["positive_weights"]),
            tuple(float(value) for value in settings["class_weights"]),
            float(settings.get("focal_gamma", 0.0)),
            frozen_thresholds,
            [str(row["ml_label"]) for row in rows],
        )

    overall = evaluate(hard_rows)
    stratum_metrics = {}
    for stratum in required_strata:
        selected = [
            row for row in hard_rows if stratum in row.get("hard_subset_strata", [])
        ]
        expected = int(hard.get("strata", {}).get(stratum, {}).get("rows", -1))
        if len(selected) != expected or len(selected) < 25:
            raise ValueError("Scaling hard-endpoint stratum count differs from its freeze")
        metrics = evaluate(selected)
        stratum_metrics[stratum] = {
            "rows": len(selected),
            "unique_glitches": len({str(row["glitch_id"]) for row in selected}),
            "glitch_iou": float(metrics["glitch"]["iou"]),
            "chirp_iou": float(metrics["chirp"]["iou"]),
            "mean_iou": float(metrics["mean_iou"]),
        }

    selected_history = [
        row
        for row in finetune.get("history", [])
        if int(row.get("epoch", -1)) == int(finetune.get("best_epoch", -2))
    ]
    if len(selected_history) != 1 or selected_history[0].get(
        "checkpoint_eligible"
    ) is not True:
        raise ValueError("Scaling hard endpoint lacks a clean-eligible selected epoch")
    clean_retention = float(selected_history[0]["clean_chirp_iou_retention"])
    minimum_retention = float(finetune["minimum_clean_chirp_iou_retention"])
    result = {
        "status": "completed_validation_only_physical_overlap_scaling_hard_endpoint_cell",
        "passed": True,
        "scientific_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "threshold_refits": 0,
        "endpoint_partition": "validation_only_predeclared_hard_subset",
        "training_control": str(finetune["training_control"]["control"]),
        "scale": int(scale),
        "seed": int(finetune["seed"]),
        "primary_metric": {
            "name": "hard_subset_glitch_iou_at_validation_frozen_threshold",
            "value": float(overall["glitch"]["iou"]),
        },
        "overall_metrics": overall,
        "strata": stratum_metrics,
        "clean_noninferiority": {
            "passed": clean_retention >= minimum_retention,
            "retention": clean_retention,
            "minimum_retention": minimum_retention,
        },
        "frozen_thresholds": {
            "chirp": frozen_thresholds[0],
            "glitch": frozen_thresholds[1],
        },
        "subset_report": {"path": str(subset_path), "sha256": file_sha256(subset_path)},
        "hard_subset": {"path": str(hard_path), "sha256": file_sha256(hard_path)},
        "finetune_report": {
            "path": str(finetune_path),
            "sha256": file_sha256(finetune_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
        },
        **provenance,
    }
    atomic_write_json(output, result)
    return result


def bind_physical_overlap_scaling_hard_endpoints(
    scaling_summary_path: str | Path,
    hard_subset_report_path: str | Path,
    hard_endpoint_report_paths: list[str | Path],
    output_path: str | Path,
    next_scale: int,
) -> dict[str, Any]:
    """Authorize one next scale only when both controls improve a frozen hard endpoint."""

    provenance = _require_publication_commit()
    output = Path(output_path)
    bundle_path = output.with_name(f"{output.stem}_hard_endpoint_bundle.json")
    if output.exists() or bundle_path.exists():
        raise FileExistsError("Bound overlap scaling outputs are immutable")
    summary_path = Path(scaling_summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status")
        != "completed_group_safe_physical_overlap_data_scaling_curve"
        or summary.get("passed") is not True
        or summary.get("test_rows_read") != 0
        or summary.get("test_evaluation") is not None
        or len(summary.get("paired_seeds", [])) < 5
    ):
        raise ValueError("Overlap scaling diagnostic is incomplete")
    subset_path = Path(str(summary.get("subset_report_path", ""))).resolve()
    if (
        not subset_path.is_file()
        or file_sha256(subset_path) != summary.get("subset_report_sha256")
    ):
        raise ValueError("Overlap scaling subset report changed")
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    subset_by_hash = {
        str(row["manifest_sha256"]): int(row["scale"])
        for row in subset.get("subsets", [])
    }

    hard_path = Path(hard_subset_report_path).resolve()
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    if (
        hard.get("status")
        != "frozen_score_blind_physical_overlap_scaling_hard_subset"
        or hard.get("passed") is not True
        or hard.get("candidate_scores_inspected") is not False
        or hard.get("model_outputs_inspected") is not False
        or hard.get("test_rows_read") != 0
        or hard.get("validation_manifest_sha256")
        != subset.get("validation_manifest_sha256")
    ):
        raise ValueError("Overlap scaling hard-subset freeze failed replay")
    hard_config_path = Path(str(hard.get("config", {}).get("path", "")))
    if (
        not hard_config_path.is_file()
        or file_sha256(hard_config_path) != hard.get("config", {}).get("sha256")
    ):
        raise ValueError("Overlap scaling hard-subset config changed")
    hard_settings = load_yaml(hard_config_path).get(
        "physical_overlap_scaling_hard_subset", {}
    )
    minimum_gain = float(hard_settings.get("minimum_material_primary_gain", 0.0))
    minimum_retention = float(
        hard_settings.get("minimum_clean_chirp_iou_retention", 0.0)
    )
    bootstrap_replicates = int(hard_settings.get("bootstrap_replicates", 0))
    bootstrap_seed = int(hard_settings.get("bootstrap_seed", 0))
    primary_metric_name = str(hard_settings.get("primary_metric", ""))
    if (
        minimum_gain <= 0
        or not 0 < minimum_retention <= 1
        or bootstrap_replicates < 1000
        or not primary_metric_name
    ):
        raise ValueError("Overlap scaling hard-endpoint decision config is invalid")

    expected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for identity in summary.get("finetune_reports", []):
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != identity.get("sha256"):
            raise ValueError("Overlap scaling finetune report changed")
        report = json.loads(path.read_text(encoding="utf-8"))
        scale = subset_by_hash.get(str(report.get("overlap_train_manifest_sha256")))
        key = (
            str(report.get("training_control", {}).get("control")),
            int(scale) if scale is not None else -1,
            int(report.get("seed", -1)),
        )
        if key in expected or key[0] not in {
            "fixed_epochs",
            "fixed_optimizer_updates",
        } or key[1] < 1 or key[2] < 0:
            raise ValueError("Overlap scaling diagnostic cell identity is invalid")
        expected[key] = {
            "finetune": {"path": str(path), "sha256": file_sha256(path)},
            "checkpoint_sha256": str(report.get("checkpoint_sha256")),
        }

    observed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw_path in hard_endpoint_report_paths:
        path = Path(raw_path).resolve()
        cell = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(cell.get("training_control")),
            int(cell.get("scale", -1)),
            int(cell.get("seed", -1)),
        )
        if key not in expected or key in observed:
            raise ValueError("Hard-endpoint cell is missing, duplicate or undeclared")
        checkpoint_path = Path(str(cell.get("checkpoint", {}).get("path", "")))
        if (
            cell.get("status")
            != "completed_validation_only_physical_overlap_scaling_hard_endpoint_cell"
            or cell.get("passed") is not True
            or cell.get("test_rows_read") != 0
            or cell.get("test_evaluation") is not None
            or cell.get("threshold_refits") != 0
            or cell.get("endpoint_partition")
            != "validation_only_predeclared_hard_subset"
            or cell.get("hard_subset")
            != {"path": str(hard_path), "sha256": file_sha256(hard_path)}
            or cell.get("subset_report")
            != {"path": str(subset_path), "sha256": file_sha256(subset_path)}
            or cell.get("finetune_report") != expected[key]["finetune"]
            or cell.get("checkpoint", {}).get("sha256")
            != expected[key]["checkpoint_sha256"]
            or not checkpoint_path.is_file()
            or file_sha256(checkpoint_path)
            != cell.get("checkpoint", {}).get("sha256")
            or cell.get("primary_metric", {}).get("name") != primary_metric_name
        ):
            raise ValueError("Hard-endpoint cell failed frozen artifact replay")
        metric = float(cell.get("primary_metric", {}).get("value"))
        retention = float(cell.get("clean_noninferiority", {}).get("retention"))
        if not np.isfinite(metric) or not np.isfinite(retention):
            raise ValueError("Hard-endpoint cell metrics must be finite")
        if set(cell.get("strata", {})) != set(hard.get("required_strata", [])):
            raise ValueError("Hard-endpoint cell omitted a required robustness stratum")
        for stratum, values in cell["strata"].items():
            if (
                int(values.get("rows", 0))
                != int(hard["strata"][stratum]["rows"])
                or int(values.get("unique_glitches", 0))
                != int(hard["strata"][stratum]["unique_glitches"])
                or not np.isfinite(float(values.get("glitch_iou")))
            ):
                raise ValueError("Hard-endpoint stratum failed count/metric replay")
        observed[key] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "metric": metric,
            "retention": retention,
            "clean_passed": cell.get("clean_noninferiority", {}).get("passed") is True,
            "report": cell,
        }
    if set(observed) != set(expected):
        raise ValueError("Hard-endpoint reports do not cover every scaling cell")

    lower, upper = (int(value) for value in summary["promotion_data_doubling"])
    declared_scales = [int(value) for value in summary["scales"]]
    if (
        lower not in declared_scales
        or upper not in declared_scales
        or upper / lower < 1.8
        or next_scale <= max(declared_scales)
        or next_scale > int(2.5 * upper)
    ):
        raise ValueError("Requested next scale is not one bounded continuation step")
    paired_seeds = [int(value) for value in summary["paired_seeds"]]
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = {}
    checks = {}
    for control in ("fixed_epochs", "fixed_optimizer_updates"):
        deltas = np.asarray(
            [
                observed[(control, upper, seed)]["metric"]
                - observed[(control, lower, seed)]["metric"]
                for seed in paired_seeds
            ],
            dtype=np.float64,
        )
        sampled = deltas[
            rng.integers(0, len(deltas), size=(bootstrap_replicates, len(deltas)))
        ].mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975])
        clean_passed = all(
            observed[(control, upper, seed)]["clean_passed"]
            and observed[(control, upper, seed)]["retention"] >= minimum_retention
            for seed in paired_seeds
        )
        material_gain = bool(deltas.mean() >= minimum_gain and interval[0] > 0)
        comparisons[control] = {
            "lower_scale": lower,
            "upper_scale": upper,
            "paired_seed_deltas": {
                str(seed): float(delta)
                for seed, delta in zip(paired_seeds, deltas)
            },
            "mean_primary_metric_delta": float(deltas.mean()),
            "paired_bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        }
        checks[control] = {
            "material_hard_endpoint_gain": material_gain,
            "clean_noninferiority": clean_passed,
        }
    authorized = all(all(value.values()) for value in checks.values())
    raw_in_domain_gain = bool(summary.get("promote_more_same_distribution_data"))
    if authorized:
        diagnosis = "data_limited_on_predeclared_hard_endpoint"
    elif any(not value["clean_noninferiority"] for value in checks.values()):
        diagnosis = "clean_noninferiority_failed_do_not_scale"
    elif checks["fixed_epochs"]["material_hard_endpoint_gain"] and not checks[
        "fixed_optimizer_updates"
    ]["material_hard_endpoint_gain"]:
        diagnosis = "optimization_budget_limited_do_not_scale"
    elif raw_in_domain_gain:
        diagnosis = "domain_transfer_limited_do_not_scale_same_distribution"
    else:
        diagnosis = "representation_or_label_limited_do_not_scale_same_distribution"

    ordered_observed = sorted(observed.items())
    bundle = {
        "status": "frozen_physical_overlap_scaling_hard_endpoint_bundle",
        "schema": "physical_overlap_scaling_hard_endpoint_bundle_v1",
        "test_rows_read": 0,
        "source_scaling_summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "hard_subset": {"path": str(hard_path), "sha256": file_sha256(hard_path)},
        "cells": [
            {
                "identity": {
                    "training_control": key[0],
                    "scale": key[1],
                    "seed": key[2],
                },
                "source": {"path": value["path"], "sha256": value["sha256"]},
                "report": value["report"],
            }
            for key, value in ordered_observed
        ],
        **provenance,
    }
    atomic_write_json(bundle_path, bundle)
    result = {
        "status": "completed_group_safe_physical_overlap_data_scaling_curve",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation hard-subset scaling cannot replace continuous-background FAR/IFAR/<VT> "
            "or the one-time locked evaluation"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "hard_endpoint_binding": {
            "passed": True,
            "endpoint_partition": "validation_only_predeclared_hard_subset",
            "required_strata": list(hard["required_strata"]),
            "all_scaling_cells_replayed": len(observed) == len(expected),
        },
        "hard_endpoint_kind": "predeclared_validation_hard_subset",
        "primary_metric": primary_metric_name,
        "scales": declared_scales,
        "paired_seeds": paired_seeds,
        "minimum_seeds": len(paired_seeds),
        "promotion_data_doubling": [lower, upper],
        "promotion_checks": checks,
        "hard_endpoint_comparisons": comparisons,
        "scale_promotion_authorized": authorized,
        "authorized_next_physical_scale": next_scale if authorized else None,
        "diagnosis": diagnosis,
        "raw_in_domain_mask_diagnostic": {
            "promote_more_same_distribution_data": raw_in_domain_gain,
            "diagnosis": summary.get("diagnosis"),
        },
        "minimum_material_primary_gain": minimum_gain,
        "minimum_clean_chirp_iou_retention": minimum_retention,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "subset_report_path": str(subset_path),
        "subset_report_sha256": file_sha256(subset_path),
        "scaling_diagnostic": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "hard_subset": {"path": str(hard_path), "sha256": file_sha256(hard_path)},
        "hard_endpoint_bundle": {
            "path": str(bundle_path.resolve()),
            "sha256": file_sha256(bundle_path),
        },
        "hard_endpoint_reports": [
            {"path": value["path"], "sha256": value["sha256"]}
            for _, value in ordered_observed
        ],
        "common_artifact_hashes": summary.get("common_artifact_hashes", {}),
        **provenance,
    }
    atomic_write_json(output, result)
    return result


def _overlap_epoch(
    model: Any,
    loader: Any,
    device: Any,
    q_count: int,
    positive_weights: tuple[float, float],
    class_weights: tuple[float, float],
    gamma: float,
    thresholds: tuple[float, float] = (0.5, 0.5),
    row_labels: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    counts = np.zeros((2, 3), dtype=np.int64)
    family_counts: dict[str, np.ndarray] = {}
    family_rows: Counter[str] = Counter()
    row_offset = 0
    losses = []
    with torch.no_grad():
        for features, targets, availability in loader:
            features, targets, availability = (
                features.to(device), targets.to(device), availability.to(device)
            )
            logits = model(features, availability)
            losses.append(float(_masked_focal_dice(
                logits, targets, availability, q_count, positive_weights, class_weights, gamma
            ).cpu()))
            counts += _counts(logits, targets, availability, q_count, thresholds)
            if row_labels is not None:
                batch_labels = row_labels[row_offset : row_offset + logits.shape[0]]
                if len(batch_labels) != logits.shape[0]:
                    raise ValueError("Glitch-family labels do not align with validation rows")
                for index, label in enumerate(batch_labels):
                    family_rows[label] += 1
                    family_counts.setdefault(
                        label, np.zeros((2, 3), dtype=np.int64)
                    )
                    family_counts[label] += _counts(
                        logits[index : index + 1],
                        targets[index : index + 1],
                        availability[index : index + 1],
                        q_count,
                        thresholds,
                    )
                row_offset += logits.shape[0]
    if row_labels is not None and row_offset != len(row_labels):
        raise ValueError("Glitch-family labels were not fully evaluated")
    result = {"loss": float(np.mean(losses)), **_metrics(counts)}
    if row_labels is not None:
        result["by_glitch_family"] = summarize_glitch_family_counts(
            family_counts, dict(family_rows)
        )
    return result


def _clean_metrics(
    model: Any,
    architecture: str,
    loader: Any,
    device: Any,
    q_count: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    counts = np.zeros((2, 3), dtype=np.int64)
    with torch.no_grad():
        for features, chirp, availability in loader:
            features, chirp, availability = (
                features.to(device), chirp.to(device), availability.to(device)
            )
            logits = _forward(model, architecture, features, availability)[:, 0:1]
            targets = chirp[:, None]
            two_logits = torch.cat([logits, torch.full_like(logits, -20.0)], dim=1)
            two_targets = torch.cat([targets, torch.zeros_like(targets)], dim=1)
            counts += _counts(
                two_logits, two_targets, availability, q_count, (threshold, 1.0)
            )
    return _metrics(counts)["chirp"]


def _train_epoch(
    model: Any,
    teacher: Any,
    teacher_architecture: str,
    overlap_loader: Any,
    clean_loader: Any,
    device: Any,
    optimizer: Any,
    q_count: int,
    settings: dict[str, Any],
    maximum_optimizer_updates: int | None = None,
) -> dict[str, Any]:
    model.train()
    teacher.eval()
    clean_batches = iter(clean_loader)
    losses = []
    component_losses: dict[str, list[float]] = {
        "overlap": [],
        "clean_chirp": [],
        "clean_chirp_distillation": [],
        "clean_glitch_distillation": [],
    }
    for overlap_features, overlap_targets, overlap_availability in overlap_loader:
        if maximum_optimizer_updates is not None and len(losses) >= maximum_optimizer_updates:
            break
        try:
            clean_features, clean_chirp, clean_availability = next(clean_batches)
        except StopIteration:
            clean_batches = iter(clean_loader)
            clean_features, clean_chirp, clean_availability = next(clean_batches)
        overlap_features, overlap_targets, overlap_availability = (
            overlap_features.to(device),
            overlap_targets.to(device),
            overlap_availability.to(device),
        )
        clean_features, clean_chirp, clean_availability = (
            clean_features.to(device), clean_chirp.to(device), clean_availability.to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        overlap_logits = model(overlap_features, overlap_availability)
        overlap_loss = _masked_focal_dice(
            overlap_logits,
            overlap_targets,
            overlap_availability,
            q_count,
            tuple(float(value) for value in settings["positive_weights"]),
            tuple(float(value) for value in settings["class_weights"]),
            float(settings.get("focal_gamma", 0.0)),
        )
        clean_logits = model(clean_features, clean_availability)
        clean_target = clean_chirp[:, None]
        clean_mask = _availability_mask(clean_availability, q_count).to(clean_logits)
        clean_positive = torch.as_tensor(
            [float(settings["clean_chirp_positive_weight"])], device=device
        ).reshape(1, 1, 1, 1, 1)
        clean_chirp_loss = torch_functional.binary_cross_entropy_with_logits(
            clean_logits[:, 0:1], clean_target, pos_weight=clean_positive, reduction="none"
        )
        clean_chirp_loss = (clean_chirp_loss * clean_mask).sum() / (
            clean_mask.sum() * clean_logits.shape[-2] * clean_logits.shape[-1]
        ).clamp_min(1.0)
        with torch.no_grad():
            teacher_logits = _forward(
                teacher, teacher_architecture, clean_features, clean_availability
            )
            teacher_chirp = torch.sigmoid(teacher_logits[:, 0:1])
            teacher_glitch = torch.sigmoid(teacher_logits[:, 1:2])
        chirp_distillation = torch_functional.binary_cross_entropy_with_logits(
            clean_logits[:, 0:1], teacher_chirp, reduction="none"
        )
        chirp_distillation = (chirp_distillation * clean_mask).sum() / (
            clean_mask.sum()
            * clean_logits.shape[-2]
            * clean_logits.shape[-1]
        ).clamp_min(1.0)
        glitch_distillation = torch_functional.binary_cross_entropy_with_logits(
            clean_logits[:, 1:2], teacher_glitch, reduction="none"
        )
        glitch_distillation = (glitch_distillation * clean_mask).sum() / (
            clean_mask.sum() * clean_logits.shape[-2] * clean_logits.shape[-1]
        ).clamp_min(1.0)
        loss = (
            overlap_loss
            + float(settings.get("clean_chirp_weight", 1.0)) * clean_chirp_loss
            + float(
                settings.get("clean_chirp_distillation_weight", 0.0)
            )
            * chirp_distillation
            + float(settings.get("clean_glitch_distillation_weight", 0.25))
            * glitch_distillation
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, value in (
            ("overlap", overlap_loss),
            ("clean_chirp", clean_chirp_loss),
            ("clean_chirp_distillation", chirp_distillation),
            ("clean_glitch_distillation", glitch_distillation),
        ):
            component_losses[name].append(float(value.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "optimizer_updates": len(losses),
        "component_losses": {
            name: float(np.mean(values))
            for name, values in component_losses.items()
        },
    }


def configure_overlap_training_scope(
    student: Any,
    teacher_architecture: str,
    settings: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Configure full-model or bit-exact chirp-preserving overlap updates."""

    scope = str(settings.get("training_scope", "full_model"))
    if scope == "full_model":
        student.requires_grad_(True)
        parameters = [
            parameter for parameter in student.parameters() if parameter.requires_grad
        ]
        return parameters, {
            "scope": scope,
            "backbone_frozen": False,
            "chirp_output_frozen": False,
            "glitch_output_trainable": True,
            "gradient_mask_policy": None,
            "trainable_parameter_tensors": len(parameters),
            "effective_trainable_parameters": sum(
                parameter.numel() for parameter in parameters
            ),
        }
    if scope not in {"glitch_head_only", "glitch_adapter_only"}:
        raise ValueError(
            "Overlap training_scope must be full_model, glitch_head_only, "
            "or glitch_adapter_only"
        )
    if teacher_architecture != "detector_set":
        raise ValueError(
            f"{scope} requires an exact detector-set pretrained checkpoint"
        )
    if scope == "glitch_adapter_only":
        adapter = getattr(student, "glitch_adapter", None)
        adapter_head = getattr(student, "glitch_adapter_head", None)
        adapter_channels = int(getattr(student, "glitch_adapter_channels", 0))
        if adapter is None or adapter_head is None or adapter_channels <= 0:
            raise ValueError(
                "glitch_adapter_only requires an enabled zero-residual glitch adapter"
            )
        student.requires_grad_(False)
        adapter.requires_grad_(True)
        adapter_head.requires_grad_(True)
        parameters = [
            parameter
            for module in (adapter, adapter_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return parameters, {
            "scope": scope,
            "backbone_frozen": True,
            "chirp_output_frozen": True,
            "glitch_output_trainable": True,
            "gradient_mask_policy": None,
            "adapter_policy": "zero_initialized_residual_glitch_decoder_v1",
            "adapter_channels": adapter_channels,
            "trainable_parameter_tensors": len(parameters),
            "effective_trainable_parameters": sum(
                parameter.numel() for parameter in parameters
            ),
        }
    head = getattr(student, "shared_head", None)
    q_count = int(getattr(student, "q_count", 0))
    if (
        head is None
        or q_count <= 0
        or int(head.weight.shape[0]) != 2 * q_count
        or (head.bias is not None and int(head.bias.shape[0]) != 2 * q_count)
    ):
        raise ValueError("glitch_head_only requires the detector-set two-class head")
    student.requires_grad_(False)
    head.weight.requires_grad_(True)
    weight_mask = torch.zeros_like(head.weight)
    weight_mask[q_count:] = 1
    head.weight.register_hook(lambda gradient, mask=weight_mask: gradient * mask)
    parameters = [head.weight]
    if head.bias is not None:
        head.bias.requires_grad_(True)
        bias_mask = torch.zeros_like(head.bias)
        bias_mask[q_count:] = 1
        head.bias.register_hook(lambda gradient, mask=bias_mask: gradient * mask)
        parameters.append(head.bias)
    glitch_parameters = int(head.weight[q_count:].numel())
    if head.bias is not None:
        glitch_parameters += int(head.bias[q_count:].numel())
    return parameters, {
        "scope": scope,
        "backbone_frozen": True,
        "chirp_output_frozen": True,
        "glitch_output_trainable": True,
        "gradient_mask_policy": "zero_chirp_rows_v1",
        "trainable_parameter_tensors": len(parameters),
        "effective_trainable_parameters": glitch_parameters,
    }


def resolve_overlap_training_control(
    settings: dict[str, Any], batches_per_epoch: int
) -> dict[str, Any]:
    """Resolve the predeclared fixed-epoch or fixed-update scaling control."""

    if batches_per_epoch <= 0:
        raise ValueError("Overlap training requires at least one optimizer batch per epoch")
    control = str(settings.get("training_control", "fixed_epochs"))
    epochs = int(settings.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("Overlap training epochs must be positive")
    if control == "fixed_epochs":
        if settings.get("max_optimizer_updates") is not None:
            raise ValueError("Fixed-epoch overlap training cannot set max_optimizer_updates")
        return {
            "control": control,
            "maximum_epochs": epochs,
            "target_optimizer_updates": None,
        }
    if control != "fixed_optimizer_updates":
        raise ValueError("Overlap training_control must be fixed_epochs or fixed_optimizer_updates")
    target = int(settings.get("max_optimizer_updates", 0))
    if target <= 0:
        raise ValueError("Fixed-update overlap training requires max_optimizer_updates > 0")
    required_epochs = int(math.ceil(target / batches_per_epoch))
    if epochs < required_epochs:
        raise ValueError(
            "Overlap training epoch safety cap cannot reach max_optimizer_updates"
        )
    return {
        "control": control,
        "maximum_epochs": epochs,
        "target_optimizer_updates": target,
        "minimum_epochs_required": required_epochs,
    }


def overlap_checkpoint_selection_score(
    overlap_validation: dict[str, Any], metric: str
) -> float:
    """Return a higher-is-better, validation-only checkpoint score."""
    if metric == "fixed_threshold_mean_iou":
        score = float(overlap_validation["mean_iou"])
    elif metric == "validation_loss":
        score = -float(overlap_validation["loss"])
    else:
        raise ValueError(
            "Overlap checkpoint_selection_metric must be "
            "fixed_threshold_mean_iou or validation_loss"
        )
    if not np.isfinite(score):
        raise ValueError("Overlap checkpoint selection score must be finite")
    return score


def _calibrate_overlap_thresholds(
    model: Any, loader: Any, device: Any, q_count: int, grid: tuple[float, ...]
) -> tuple[tuple[float, float], dict[str, Any]]:
    curves: dict[str, list[dict[str, float]]] = {"chirp": [], "glitch": []}
    selected = []
    for class_index, class_name in enumerate(("chirp", "glitch")):
        best = (-1.0, grid[0])
        for threshold in grid:
            thresholds = (threshold, 1.0) if class_index == 0 else (1.0, threshold)
            metrics = _overlap_epoch(
                model, loader, device, q_count, (1.0, 1.0), (1.0, 1.0), 0.0, thresholds
            )[class_name]
            curves[class_name].append({"threshold": threshold, **{k: metrics[k] for k in ("precision", "recall", "iou", "dice")}})
            if metrics["iou"] > best[0]:
                best = (float(metrics["iou"]), threshold)
        selected.append(float(best[1]))
    return (selected[0], selected[1]), curves


def run_physical_overlap_finetune(
    config_path: str | Path,
    overlap_train_manifest: str | Path,
    overlap_validation_manifest: str | Path,
    clean_train_manifest: str | Path,
    clean_validation_manifest: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
    clean_validation_feature_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    _require_torch()
    config = load_yaml(config_path)
    settings = config.get("overlap_training")
    if not isinstance(settings, dict):
        raise ValueError("Configuration requires overlap_training")
    clean_loss_weights = {
        "clean_chirp_weight": float(
            settings.get("clean_chirp_weight", 1.0)
        ),
        "clean_chirp_distillation_weight": float(
            settings.get("clean_chirp_distillation_weight", 0.0)
        ),
        "clean_glitch_distillation_weight": float(
            settings.get("clean_glitch_distillation_weight", 0.25)
        ),
    }
    if (
        any(
            not np.isfinite(value) or value < 0
            for value in clean_loss_weights.values()
        )
        or clean_loss_weights["clean_chirp_weight"]
        + clean_loss_weights["clean_chirp_distillation_weight"]
        <= 0
    ):
        raise ValueError(
            "Overlap clean/distillation loss weights must be non-negative "
            "with a positive chirp anchor"
        )
    seed = int(settings.get("seed", 0) if seed_override is None else seed_override)
    run_identity = {
        "config_hash": canonical_hash(config),
        "config_file_sha256": file_sha256(config_path),
        "overlap_train_manifest_sha256": file_sha256(overlap_train_manifest),
        "overlap_validation_manifest_sha256": file_sha256(overlap_validation_manifest),
        "clean_train_manifest_sha256": file_sha256(clean_train_manifest),
        "clean_validation_manifest_sha256": file_sha256(clean_validation_manifest),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "seed": seed,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed_report_path = output / "overlap_finetune_report.json"
    if completed_report_path.is_file():
        with completed_report_path.open("r", encoding="utf-8") as handle:
            completed = json.load(handle)
        if completed.get("run_identity") != run_identity:
            raise ValueError("Completed overlap fine-tune output belongs to a different run")
        return completed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_rows = _read_rows(overlap_train_manifest)
    validation_rows = _read_rows(overlap_validation_manifest)
    clean_train_rows = _read_rows(clean_train_manifest)
    clean_validation_rows = _read_rows(clean_validation_manifest)
    split_audit = overlap_training_split_audit(train_rows, validation_rows)
    if any(row.get("split") != "train" for row in clean_train_rows):
        raise ValueError("Clean training manifest contains a non-train row")
    if any(row.get("split") != "val" for row in clean_validation_rows):
        raise ValueError("Clean validation manifest contains a non-validation row")
    clean_split_audit = physical_split_audit(clean_train_rows, clean_validation_rows)
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    tensor = settings["tensor"]
    overlap_datasets = {
        "train": PhysicalOverlapDataset(
            train_rows, model_ifos, q_values, int(tensor["frequency_bins"]), int(tensor["time_bins"]), bool(settings.get("cache_in_memory", False))
        ),
        "val": PhysicalOverlapDataset(
            validation_rows, model_ifos, q_values, int(tensor["frequency_bins"]), int(tensor["time_bins"]), bool(settings.get("cache_in_memory", False))
        ),
    }
    clean_datasets = {
        "train": PhysicalInjectionDataset(
            clean_train_rows, tensor, model_ifos, q_values, int(settings["target_sample_rate"]), bool(settings.get("cache_in_memory", False)), return_detector_availability=True
        ),
        "val": PhysicalInjectionDataset(
            clean_validation_rows,
            tensor,
            model_ifos,
            q_values,
            int(settings["target_sample_rate"]),
            bool(settings.get("cache_in_memory", False)),
            tensor_cache_dir=clean_validation_feature_cache_dir,
            return_detector_availability=True,
        ),
    }
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(settings["batch_size"])
    family_sampling = settings.get("glitch_family_sampling", {})
    family_sampling_enabled = bool(family_sampling.get("enabled", False))
    sampling_report: dict[str, Any] = {
        "strategy": "uniform_row_shuffle_v1",
        "enabled": False,
        "physical_rows": len(train_rows),
        "sample_draws_per_epoch": len(train_rows),
        "adds_independent_physical_examples": False,
        "replacement": False,
        "family_counts": dict(
            sorted(Counter(str(row.get("ml_label")) for row in train_rows).items())
        ),
    }
    train_sampler = None
    if family_sampling_enabled:
        weights, sampling_report = glitch_family_sampling_weights(
            train_rows,
            float(family_sampling.get("exponent", 0.5)),
            float(family_sampling.get("maximum_weight_ratio", 4.0)),
            int(family_sampling.get("minimum_family_count", 5)),
        )
        sampling_report["enabled"] = True
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_rows),
            replacement=True,
            generator=generator,
        )
    overlap_loaders = {
        "train": DataLoader(
            overlap_datasets["train"],
            batch_size=batch_size,
            shuffle=not family_sampling_enabled,
            sampler=train_sampler,
            num_workers=0,
            generator=generator if not family_sampling_enabled else None,
        ),
        "val": DataLoader(
            overlap_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    training_control = resolve_overlap_training_control(
        settings, len(overlap_loaders["train"])
    )
    clean_loaders = {
        key: DataLoader(value, batch_size=batch_size, shuffle=key == "train", num_workers=0, generator=generator if key == "train" else None)
        for key, value in clean_datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    teacher, teacher_architecture = model_from_checkpoint(pretrained, model_ifos, q_values)
    teacher = teacher.to(device).requires_grad_(False)
    student = DetectorSetQNet(
        len(model_ifos), len(q_values), int(pretrained["base_channels"])
    )
    if teacher_architecture == "detector_set":
        student.load_state_dict(pretrained["model"])
        warm_start = {"status": "exact_detector_set_state_dict"}
    else:
        warm_start = initialize_detector_set_from_early_fusion(student, pretrained)
    requested_scope = str(settings.get("training_scope", "full_model"))
    if requested_scope == "glitch_adapter_only":
        adapter_report = student.enable_glitch_adapter(
            int(settings.get("glitch_adapter_channels", 0))
        )
        warm_start = {
            **warm_start,
            "glitch_adapter": adapter_report,
        }
    student = student.to(device)
    optimized_parameters, training_scope = configure_overlap_training_scope(
        student, teacher_architecture, settings
    )
    effective_weight_decay = (
        0.0
        if training_scope["scope"] == "glitch_head_only"
        else float(settings["weight_decay"])
    )
    training_scope["configured_weight_decay"] = float(settings["weight_decay"])
    training_scope["effective_weight_decay"] = effective_weight_decay
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=effective_weight_decay,
    )
    checkpoint_selection_metric = str(
        settings.get("checkpoint_selection_metric", "fixed_threshold_mean_iou")
    )
    if (
        checkpoint_selection_metric == "validation_loss"
        and training_scope["scope"]
        not in {"glitch_head_only", "glitch_adapter_only"}
    ):
        raise ValueError(
            "validation_loss checkpoint selection is restricted to frozen-chirp "
            "glitch_head_only or glitch_adapter_only scopes"
        )
    overlap_checkpoint_selection_score(
        {"mean_iou": 0.0, "loss": 0.0}, checkpoint_selection_metric
    )
    q_count = len(q_values)
    frozen_non_glitch_state = None
    if training_scope["scope"] in {"glitch_head_only", "glitch_adapter_only"}:
        frozen_non_glitch_state = {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
        }
    teacher_clean = _clean_metrics(
        teacher, teacher_architecture, clean_loaders["val"], device, q_count
    )
    checkpoint_path = output / "best_overlap_finetune.pt"
    resume_path = output / "last_overlap_finetune.pt"
    history = []
    best_selection_score = -float("inf")
    best_validation_overlap_mean_iou = None
    best_epoch = None
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != run_identity:
            raise ValueError("Overlap fine-tune resume checkpoint belongs to a different run")
        student.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_selection_score = float(
            resume.get(
                "best_checkpoint_selection_score",
                resume["best_validation_overlap_mean_iou"],
            )
        )
        resumed_best_mean_iou = resume.get("best_validation_overlap_mean_iou")
        best_validation_overlap_mean_iou = (
            None
            if resumed_best_mean_iou is None
            else float(resumed_best_mean_iou)
        )
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
    completed_optimizer_updates = sum(
        int(row.get("train", {}).get("optimizer_updates", 0)) for row in history
    )
    retention_fraction = float(settings.get("minimum_clean_chirp_iou_retention", 0.95))
    started = time.time()
    for epoch in range(start_epoch, training_control["maximum_epochs"] + 1):
        target_updates = training_control["target_optimizer_updates"]
        if target_updates is not None and completed_optimizer_updates >= target_updates:
            break
        remaining_updates = (
            None
            if target_updates is None
            else target_updates - completed_optimizer_updates
        )
        train_metrics = _train_epoch(
            student,
            teacher,
            teacher_architecture,
            overlap_loaders["train"],
            clean_loaders["train"],
            device,
            optimizer,
            q_count,
            settings,
            remaining_updates,
        )
        completed_optimizer_updates += int(train_metrics["optimizer_updates"])
        overlap_validation = _overlap_epoch(
            student,
            overlap_loaders["val"],
            device,
            q_count,
            tuple(float(value) for value in settings["positive_weights"]),
            tuple(float(value) for value in settings["class_weights"]),
            float(settings.get("focal_gamma", 0.0)),
        )
        clean_validation = _clean_metrics(
            student, "detector_set", clean_loaders["val"], device, q_count
        )
        retention = clean_validation["iou"] / max(float(teacher_clean["iou"]), 1e-12)
        eligible = retention >= retention_fraction
        selection_score = overlap_checkpoint_selection_score(
            overlap_validation, checkpoint_selection_metric
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "cumulative_optimizer_updates": completed_optimizer_updates,
                "overlap_validation": overlap_validation,
                "clean_validation": clean_validation,
                "clean_chirp_iou_retention": retention,
                "checkpoint_eligible": eligible,
                "checkpoint_selection_metric": checkpoint_selection_metric,
                "checkpoint_selection_score": selection_score,
            }
        )
        if eligible and selection_score > best_selection_score:
            best_selection_score = selection_score
            best_validation_overlap_mean_iou = float(overlap_validation["mean_iou"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": student.state_dict(),
                    "architecture": "detector_set",
                    "model_ifos": list(model_ifos),
                    "q_values": list(q_values),
                    "input_channels": len(model_ifos) * len(q_values),
                    "base_channels": int(pretrained["base_channels"]),
                    "glitch_adapter_channels": int(
                        getattr(student, "glitch_adapter_channels", 0)
                    ),
                    "epoch": epoch,
                    "validation_overlap_mean_iou": float(
                        overlap_validation["mean_iou"]
                    ),
                    "checkpoint_selection_metric": checkpoint_selection_metric,
                    "checkpoint_selection_score": selection_score,
                    "clean_chirp_iou_retention": retention,
                    "training_scope": training_scope,
                    "config_hash": canonical_hash(config),
                    "seed": seed,
                    "run_identity": run_identity,
                },
            )
        _atomic_torch_save(
            resume_path,
            {
                "run_identity": run_identity,
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_generator_state": generator.get_state(),
                "epoch": epoch,
                "history": history,
                "checkpoint_selection_metric": checkpoint_selection_metric,
                "best_checkpoint_selection_score": best_selection_score,
                "best_validation_overlap_mean_iou": (
                    best_validation_overlap_mean_iou
                ),
                "best_epoch": best_epoch,
            },
        )
        atomic_write_json(output / "history.json", history)
    if (
        training_control["target_optimizer_updates"] is not None
        and completed_optimizer_updates
        != training_control["target_optimizer_updates"]
    ):
        raise RuntimeError("Overlap training did not reach the fixed optimizer-update target")
    if best_epoch is None:
        raise RuntimeError("No overlap checkpoint passed the clean-chirp retention gate")
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(selected["model"])
    chirp_head_preserved_bit_exact = None
    backbone_preserved_bit_exact = None
    non_glitch_state_preserved_bit_exact = None
    if frozen_non_glitch_state is not None:
        selected_state = student.state_dict()
        chirp_head_preserved_bit_exact = bool(
            torch.equal(
                selected_state["shared_head.weight"][:q_count].detach().cpu(),
                frozen_non_glitch_state["shared_head.weight"][:q_count],
            )
            and (
                "shared_head.bias" not in frozen_non_glitch_state
                or torch.equal(
                    selected_state["shared_head.bias"][:q_count].detach().cpu(),
                    frozen_non_glitch_state["shared_head.bias"][:q_count],
                )
            )
        )
        if training_scope["scope"] == "glitch_adapter_only":
            backbone_preserved_bit_exact = all(
                torch.equal(value.detach().cpu(), frozen_non_glitch_state[name])
                for name, value in selected_state.items()
                if not name.startswith("glitch_adapter")
            )
        else:
            backbone_preserved_bit_exact = all(
                torch.equal(value.detach().cpu(), frozen_non_glitch_state[name])
                for name, value in selected_state.items()
                if name not in {"shared_head.weight", "shared_head.bias"}
            )
        non_glitch_state_preserved_bit_exact = bool(
            chirp_head_preserved_bit_exact and backbone_preserved_bit_exact
        )
        if not non_glitch_state_preserved_bit_exact:
            raise RuntimeError(
                "glitch-head-only training changed frozen non-glitch state"
            )
    grid = tuple(float(value) for value in settings["threshold_grid"])
    thresholds, curves = _calibrate_overlap_thresholds(
        student, overlap_loaders["val"], device, q_count, grid
    )
    calibrated_overlap = _overlap_epoch(
        student,
        overlap_loaders["val"],
        device,
        q_count,
        tuple(float(value) for value in settings["positive_weights"]),
        tuple(float(value) for value in settings["class_weights"]),
        float(settings.get("focal_gamma", 0.0)),
        thresholds,
        [str(row["ml_label"]) for row in validation_rows],
    )
    calibrated_clean = _clean_metrics(
        student, "detector_set", clean_loaders["val"], device, q_count, thresholds[0]
    )
    report = {
        "status": "validation_selected_real_glitch_overlap_finetune",
        "scientific_claim_allowed": False,
        "search_claim_allowed": False,
        "scientific_blockers": [
            "automatic_mask_policy_replay",
            "aligned_multi_ifo_glitch_contexts",
            "continuous_background_far_ifar_vt",
            "five_seed_locked_evaluation",
        ],
        "seed": seed,
        "run_identity": run_identity,
        "split_audit": split_audit,
        "clean_split_audit": clean_split_audit,
        "glitch_family_sampling": sampling_report,
        "training_scope": {
            **training_scope,
            "chirp_head_preserved_bit_exact": chirp_head_preserved_bit_exact,
            "backbone_preserved_bit_exact": backbone_preserved_bit_exact,
            "non_glitch_state_preserved_bit_exact": (
                non_glitch_state_preserved_bit_exact
            ),
        },
        "training_control": training_control,
        "completed_optimizer_updates": completed_optimizer_updates,
        "warm_start": warm_start,
        "teacher_architecture": teacher_architecture,
        "teacher_clean_validation": teacher_clean,
        "minimum_clean_chirp_iou_retention": retention_fraction,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "best_checkpoint_selection_score": best_selection_score,
        "best_epoch": best_epoch,
        "best_validation_overlap_mean_iou": best_validation_overlap_mean_iou,
        "validation_selected_thresholds": {"chirp": thresholds[0], "glitch": thresholds[1]},
        "threshold_curves": curves,
        "calibrated_overlap_validation": calibrated_overlap,
        "calibrated_clean_validation": calibrated_clean,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_hash": canonical_hash(config),
        "config_file_sha256": file_sha256(config_path),
        "overlap_train_manifest_sha256": file_sha256(overlap_train_manifest),
        "overlap_validation_manifest_sha256": file_sha256(overlap_validation_manifest),
        "clean_train_manifest_sha256": file_sha256(clean_train_manifest),
        "clean_validation_manifest_sha256": file_sha256(clean_validation_manifest),
        "clean_validation_feature_cache_dir": (
            str(clean_validation_feature_cache_dir)
            if clean_validation_feature_cache_dir is not None
            else None
        ),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "elapsed_seconds": time.time() - started,
        "history": history,
        **execution_provenance(torch),
    }
    atomic_write_json(completed_report_path, report)
    return report
