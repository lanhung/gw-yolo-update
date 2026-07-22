from __future__ import annotations

import json
import math
import os
import platform
import random
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    load_yaml,
)
from .metrics import wilson_interval
from .runtime import execution_provenance

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - dependency-minimal installations
    torch = None
    torch_functional = None
    DataLoader = None


def calibrate_known_only_abstention(
    known_scores: Iterable[float],
    maximum_known_abstention_rate: float,
) -> dict[str, Any]:
    """Freeze an OOD threshold using known validation artifacts only."""
    if not 0 <= maximum_known_abstention_rate < 1:
        raise ValueError("maximum known abstention rate must be in [0, 1)")
    scores = np.asarray(list(known_scores), dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("known validation OOD scores must be non-empty and finite")
    maximum_count = int(math.floor(maximum_known_abstention_rate * scores.size))
    candidates = [math.nextafter(float(scores.max()), math.inf), *sorted(set(scores), reverse=True)]
    allowed = []
    for threshold in candidates:
        count = int(np.count_nonzero(scores >= threshold))
        if count <= maximum_count:
            allowed.append((float(threshold), count))
    if not allowed:
        raise AssertionError("zero-count OOD threshold must always satisfy calibration")
    threshold, count = min(allowed, key=lambda item: item[0])
    return {
        "threshold": threshold,
        "known_validation_rows": int(scores.size),
        "maximum_known_abstention_rate": maximum_known_abstention_rate,
        "maximum_known_abstentions": maximum_count,
        "observed_known_abstentions": count,
        "observed_known_abstention_rate": count / scores.size,
        "selection_data": "known_validation_only",
        "unknown_scores_used_for_selection": False,
        "tie_safe": True,
    }


def ood_auc(rows: list[dict[str, Any]], score_field: str = "ood_score") -> float:
    """Pair-count AUROC where larger scores indicate unknown artifacts."""
    known = [float(row[score_field]) for row in rows if not bool(row["is_unknown"])]
    unknown = [float(row[score_field]) for row in rows if bool(row["is_unknown"])]
    if not known or not unknown:
        raise ValueError("OOD AUROC requires known and unknown evaluation rows")
    wins = 0.0
    for unknown_score in unknown:
        for known_score in known:
            wins += float(unknown_score > known_score) + 0.5 * float(
                unknown_score == known_score
            )
    return wins / (len(known) * len(unknown))


def fit_class_conditional_mahalanobis(
    embeddings: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    shrinkage: float = 0.1,
    epsilon: float = 1e-4,
) -> dict[str, np.ndarray | float | int]:
    """Fit known-train class centers and one regularized within-class precision matrix."""
    values = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("Mahalanobis fit requires [rows, features] and one target per row")
    if values.shape[0] <= class_count or values.shape[1] < 1:
        raise ValueError("Mahalanobis fit requires more rows than known classes")
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("Mahalanobis fit inputs must be finite")
    if not 0 <= shrinkage <= 1 or epsilon <= 0:
        raise ValueError("Mahalanobis shrinkage/epsilon are invalid")
    if set(labels.tolist()) != set(range(class_count)):
        raise ValueError("Mahalanobis fit requires every contiguous known class")
    centers = np.stack([values[labels == index].mean(axis=0) for index in range(class_count)])
    residuals = values - centers[labels]
    covariance = residuals.T @ residuals / max(values.shape[0] - class_count, 1)
    diagonal = np.diag(np.diag(covariance))
    regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    regularized += epsilon * np.eye(values.shape[1], dtype=np.float64)
    precision = np.linalg.pinv(regularized, hermitian=True)
    if not np.isfinite(precision).all():
        raise ValueError("Mahalanobis precision is non-finite")
    return {
        "centers": centers,
        "precision": precision,
        "shrinkage": float(shrinkage),
        "epsilon": float(epsilon),
        "known_train_rows": int(values.shape[0]),
    }


def class_conditional_mahalanobis_scores(
    embeddings: np.ndarray,
    fit: dict[str, np.ndarray | float | int],
) -> np.ndarray:
    """Return the minimum squared distance to any known-train class center."""
    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(fit["centers"], dtype=np.float64)
    precision = np.asarray(fit["precision"], dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Mahalanobis score dimensions do not agree")
    if precision.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Mahalanobis precision has the wrong shape")
    differences = values[:, None, :] - centers[None, :, :]
    distances = np.einsum("ncd,df,ncf->nc", differences, precision, differences)
    scores = distances.min(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Mahalanobis scores are non-finite")
    return np.maximum(scores, 0.0)


def supervised_contrastive_loss(
    embeddings: Any,
    targets: Any,
    temperature: float = 0.1,
) -> Any:
    """Supervised contrastive loss over normalized known-family embeddings."""
    if torch is None:
        raise RuntimeError("supervised contrastive loss requires torch")
    if embeddings.ndim != 2 or targets.shape != (embeddings.shape[0],):
        raise ValueError("contrastive loss requires [batch, features] and one target per row")
    if embeddings.shape[0] < 2 or temperature <= 0:
        raise ValueError("contrastive loss requires at least two rows and positive temperature")
    normalized = torch_functional.normalize(embeddings, p=2, dim=1)
    logits = normalized @ normalized.T / float(temperature)
    identity = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = ~identity
    log_denominator = torch.logsumexp(
        logits.masked_fill(~denominator_mask, -torch.inf), dim=1
    )
    positive_mask = targets[:, None].eq(targets[None, :]) & denominator_mask
    positive_counts = positive_mask.sum(dim=1)
    usable = positive_counts > 0
    if not bool(usable.any()):
        return embeddings.sum() * 0.0
    mean_positive_log_probability = (
        ((logits - log_denominator[:, None]) * positive_mask).sum(dim=1)
        / positive_counts.clamp_min(1)
    )
    return -mean_positive_log_probability[usable].mean()


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("OOD rate requires a valid non-empty binomial count")
    return {
        "count": successes,
        "total": total,
        "rate": successes / total,
        "wilson_95": list(wilson_interval(successes, total)),
    }


def _network_source_ids(row: dict[str, Any]) -> set[str]:
    sources = row.get("network_strain_sources")
    if not isinstance(sources, dict):
        return set()
    identities = set()
    for record in sources.values():
        if not isinstance(record, dict):
            continue
        identity = record.get("hdf5_url") or record.get("detail_url")
        if identity:
            identities.add(str(identity))
    return identities


def evaluate_frozen_ood_threshold(
    calibration_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    maximum_known_abstention_rate: float = 0.05,
    score_field: str = "ood_score",
) -> dict[str, Any]:
    if not calibration_rows or not evaluation_rows:
        raise ValueError("OOD calibration and evaluation rows must be non-empty")
    required = {"glitch_id", "gps_block", "glitch_family", "observing_run", score_field}
    for label, rows in (("calibration", calibration_rows), ("evaluation", evaluation_rows)):
        missing = [index for index, row in enumerate(rows) if required - set(row)]
        if missing:
            raise ValueError(f"OOD {label} rows lack required fields at {missing[:10]}")
        scores = np.asarray([float(row[score_field]) for row in rows])
        if not np.isfinite(scores).all():
            raise ValueError(f"OOD {label} scores must be finite")
    if any(bool(row.get("is_unknown", False)) for row in calibration_rows):
        raise ValueError("OOD threshold calibration cannot contain unknown artifacts")
    if any(str(row.get("split")) != "val" for row in calibration_rows):
        raise ValueError("OOD threshold calibration must be validation-only")
    if any("is_unknown" not in row for row in evaluation_rows):
        raise ValueError("OOD evaluation rows require explicit is_unknown labels")
    overlaps = {}
    for field in ("glitch_id", "gps_block"):
        calibration_ids = {str(row[field]) for row in calibration_rows}
        evaluation_ids = {str(row[field]) for row in evaluation_rows}
        overlaps[field] = sorted(calibration_ids & evaluation_ids)
    if any(overlaps.values()):
        raise ValueError(f"OOD calibration/evaluation group leakage: {overlaps}")
    calibration = calibrate_known_only_abstention(
        (float(row[score_field]) for row in calibration_rows),
        maximum_known_abstention_rate,
    )
    threshold = float(calibration["threshold"])
    evaluated = [
        {
            **row,
            "abstained": float(row[score_field]) >= threshold,
        }
        for row in evaluation_rows
    ]
    known = [row for row in evaluated if not bool(row["is_unknown"])]
    unknown = [row for row in evaluated if bool(row["is_unknown"])]
    if not known or not unknown:
        raise ValueError("OOD evaluation requires both known and unknown rows")
    known_false_abstention = _rate(sum(row["abstained"] for row in known), len(known))
    unknown_true_abstention = _rate(sum(row["abstained"] for row in unknown), len(unknown))
    unknown_false_acceptance = _rate(sum(not row["abstained"] for row in unknown), len(unknown))

    def strata(field: str) -> dict[str, Any]:
        output = {}
        for value in sorted({str(row[field]) for row in evaluated}):
            selected = [row for row in evaluated if str(row[field]) == value]
            selected_unknown = [row for row in selected if bool(row["is_unknown"])]
            selected_known = [row for row in selected if not bool(row["is_unknown"])]
            output[value] = {
                "rows": len(selected),
                "unknown_rows": len(selected_unknown),
                "known_rows": len(selected_known),
                "unknown_true_abstention": (
                    _rate(sum(row["abstained"] for row in selected_unknown), len(selected_unknown))
                    if selected_unknown
                    else None
                ),
                "known_false_abstention": (
                    _rate(sum(row["abstained"] for row in selected_known), len(selected_known))
                    if selected_known
                    else None
                ),
            }
        return output

    return {
        "status": "frozen_known_only_ood_abstention_evaluation",
        "scientific_claim_allowed": False,
        "protocol": (
            "threshold frozen from known validation artifacts only; held-out families and runs "
            "are evaluated without threshold adjustment"
        ),
        "score_field": score_field,
        "higher_score_means": "more_unknown",
        "calibration": calibration,
        "split_audit": {"passed": True, "cross_split_overlaps": overlaps},
        "evaluation_rows": len(evaluated),
        "known_rows": len(known),
        "unknown_rows": len(unknown),
        "known_false_abstention": known_false_abstention,
        "unknown_true_abstention": unknown_true_abstention,
        "unknown_false_acceptance": unknown_false_acceptance,
        "auroc_diagnostic": ood_auc(evaluated, score_field),
        "family_strata": strata("glitch_family"),
        "observing_run_strata": strata("observing_run"),
        "unknown_family_counts": dict(
            sorted(Counter(str(row["glitch_family"]) for row in unknown).items())
        ),
    }


def run_ood_abstention_evaluation(
    calibration_manifest: str | Path,
    evaluation_manifest: str | Path,
    output: str | Path,
    maximum_known_abstention_rate: float = 0.05,
    score_field: str = "ood_score",
) -> dict[str, Any]:
    def load(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    result = evaluate_frozen_ood_threshold(
        load(calibration_manifest),
        load(evaluation_manifest),
        maximum_known_abstention_rate,
        score_field,
    )
    result.update(
        {
            "calibration_manifest_path": str(calibration_manifest),
            "calibration_manifest_sha256": file_sha256(calibration_manifest),
            "evaluation_manifest_path": str(evaluation_manifest),
            "evaluation_manifest_sha256": file_sha256(evaluation_manifest),
            **execution_provenance(),
        }
    )
    atomic_write_json(output, result)
    return result


def run_frozen_glitch_ood_scoring(
    config_path: str | Path,
    validation_ood_report: str | Path,
    evaluation_manifest: str | Path,
    output_manifest: str | Path,
    output_report: str | Path,
    required_split: str = "test",
    locked_suite_plan: str | Path | None = None,
    access_log: str | Path | None = None,
) -> dict[str, Any]:
    """Score a new detector-set corpus with one validation-frozen OOD model."""

    if torch is None:
        raise RuntimeError("frozen glitch OOD scoring requires torch")
    from .numeric import DetectorSetGlitchEmbeddingNet

    manifest_output = Path(output_manifest).resolve()
    report_output = Path(output_report).resolve()
    if report_output.exists():
        raise FileExistsError("frozen OOD score reports are immutable")
    if manifest_output == report_output or not required_split:
        raise ValueError("frozen OOD score output paths and split are invalid")
    suite_values = (locked_suite_plan, access_log)
    if any(value is not None for value in suite_values) and not all(
        value is not None for value in suite_values
    ):
        raise ValueError("locked suite plan and access log must be supplied together")
    locked_suite_access = None
    locked_suite_inputs = None
    if required_split == "test":
        if locked_suite_plan is None:
            raise ValueError("test OOD scoring requires the one-time locked suite receipt")
        from .evaluation_lock import (
            validate_locked_evaluation_suite_access,
            validate_locked_evaluation_suite_input,
        )

        plan = json.loads(Path(locked_suite_plan).read_text(encoding="utf-8"))
        locked_suite_access = validate_locked_evaluation_suite_access(
            locked_suite_plan,
            access_log,
            "locked_ood_transfer",
            plan.get("outputs", {}).get("locked_ood_transfer", ""),
        )
        locked_suite_inputs = {
            "source_manifest": validate_locked_evaluation_suite_input(
                locked_suite_plan,
                "locked_ood_source_manifest",
                evaluation_manifest,
            ),
            "score_manifest": validate_locked_evaluation_suite_input(
                locked_suite_plan,
                "locked_ood_score_manifest",
                manifest_output,
            ),
            "score_report": validate_locked_evaluation_suite_input(
                locked_suite_plan,
                "locked_ood_score_report",
                report_output,
            ),
        }
    validation_path = Path(validation_ood_report).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if locked_suite_access is not None:
        validation_identity = locked_suite_access["frozen_artifacts"].get(
            "validation_ood_report", {}
        )
        if (
            Path(str(validation_identity.get("path", ""))).resolve()
            != validation_path
            or validation_identity.get("sha256") != file_sha256(validation_path)
        ):
            raise ValueError("test OOD scorer validation model differs from the access receipt")
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    settings = config.get("glitch_ood_embedding")
    if not isinstance(settings, dict):
        raise ValueError("frozen OOD scoring requires glitch_ood_embedding settings")
    checkpoint_path = Path(str(validation.get("checkpoint_path", ""))).resolve()
    if (
        validation.get("status")
        != "known_family_embedding_heldout_ood_validation"
        or validation.get("architecture") != "detector_set"
        or validation.get("ood_score_method") != "logit_energy"
        or validation.get("test_evaluation") is not None
        or validation.get("run_identity", {}).get("config_hash")
        != canonical_hash(config)
        or validation.get("run_identity", {}).get("config_file_sha256")
        != file_sha256(config_file)
        or not checkpoint_path.is_file()
        or validation.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError("frozen OOD scorer requires an exact detector-set validation model")
    source_path = Path(evaluation_manifest).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    required_fields = {
        "glitch_id",
        "gps_block",
        "glitch_family",
        "observing_run",
        "is_unknown",
        "available_ifos",
        "split",
        "aligned_network_context",
        "path",
        "sha256",
        "detector_availability",
        "ifo",
    }
    missing = [index for index, row in enumerate(rows) if required_fields - set(row)]
    if not rows or missing:
        raise ValueError(f"frozen OOD scoring rows are empty or incomplete: {missing[:10]}")
    if any(str(row["split"]) != required_split for row in rows):
        raise ValueError("frozen OOD scoring manifest mixes data outside the required split")
    if len({str(row["glitch_id"]) for row in rows}) != len(rows):
        raise ValueError("frozen OOD scoring requires unique glitch IDs")

    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    q_values = tuple(float(value) for value in settings["q_values"])
    labels = [str(value) for value in validation.get("labels", [])]
    if len(labels) < 2:
        raise ValueError("frozen OOD validation report has too few known labels")
    label_to_index = {label: index for index, label in enumerate(labels)}
    dataset = DetectorSetGlitchOODDataset(
        rows,
        model_ifos,
        q_values,
        label_to_index,
        allow_unknown=True,
        cache_in_memory=bool(settings.get("cache_in_memory", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        selected.get("run_identity") != validation.get("run_identity")
        or selected.get("architecture") != "detector_set"
        or [str(value) for value in selected.get("labels", [])] != labels
        or tuple(str(value) for value in selected.get("model_ifos", ())) != model_ifos
        or tuple(float(value) for value in selected.get("q_values", ())) != q_values
    ):
        raise ValueError("frozen OOD checkpoint metadata differs from its validation report")
    model = DetectorSetGlitchEmbeddingNet(
        ifo_count=len(model_ifos),
        q_count=len(q_values),
        class_count=len(labels),
        base_channels=int(selected["base_channels"]),
        embedding_dim=int(selected["embedding_dim"]),
    ).to(device)
    model.load_state_dict(selected["model"])
    model.eval()
    score_values = []
    predicted_indices = []
    confidences = []
    with torch.no_grad():
        for features, availability, _ in loader:
            logits, _ = model(features.to(device), availability.to(device))
            probabilities = torch.softmax(logits, dim=1)
            score_values.extend((-torch.logsumexp(logits, dim=1)).cpu().tolist())
            predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
            confidences.extend(probabilities.max(dim=1).values.cpu().tolist())
    if len(score_values) != len(rows):
        raise AssertionError("frozen OOD scorer returned an incomplete batch")
    scored_rows = [
        {
            **row,
            "ood_score": float(score_values[index]),
            "ood_score_method": "logit_energy",
            "predicted_known_family": labels[int(predicted_indices[index])],
            "known_classifier_confidence": float(confidences[index]),
            "embedding_checkpoint_sha256": file_sha256(checkpoint_path),
            "ood_validation_report_sha256": file_sha256(validation_path),
            "ood_config_sha256": file_sha256(config_file),
        }
        for index, row in enumerate(rows)
    ]
    rendered_manifest = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in scored_rows
    )
    if manifest_output.exists():
        if manifest_output.read_text(encoding="utf-8") != rendered_manifest:
            raise ValueError("existing frozen OOD score manifest differs from deterministic replay")
    else:
        atomic_write_text(manifest_output, rendered_manifest)
    result = {
        "status": "frozen_glitch_ood_scores_complete",
        "scientific_claim_allowed": False,
        "selection_data": "validation_model_only",
        "test_scores_used_for_model_threshold_or_method_selection": False,
        "required_split": required_split,
        "rows": len(scored_rows),
        "architecture": "detector_set",
        "ood_score_method": "logit_energy",
        "manifest_path": str(manifest_output),
        "manifest_sha256": file_sha256(manifest_output),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "validation_ood_report_path": str(validation_path),
        "validation_ood_report_sha256": file_sha256(validation_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "device": str(device),
        "locked_suite_access": locked_suite_access,
        "locked_suite_inputs": locked_suite_inputs,
        **execution_provenance(),
    }
    atomic_write_json(report_output, result)
    return result


def run_locked_ood_transfer_evaluation(
    validation_ood_report: str | Path,
    locked_score_report: str | Path,
    locked_score_manifest: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
    score_field: str = "ood_score",
) -> dict[str, Any]:
    """Apply one validation-frozen OOD threshold to a disjoint locked O4b set."""

    from .evaluation_lock import (
        validate_locked_evaluation_suite_access,
        validate_locked_evaluation_suite_input,
    )

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked OOD transfer outputs are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, "locked_ood_transfer", output_path
    )
    suite_inputs = {
        "score_report": validate_locked_evaluation_suite_input(
            locked_suite_plan,
            "locked_ood_score_report",
            locked_score_report,
        ),
        "score_manifest": validate_locked_evaluation_suite_input(
            locked_suite_plan,
            "locked_ood_score_manifest",
            locked_score_manifest,
        ),
    }
    validation_path = Path(validation_ood_report).resolve()
    validation_identity = suite_access["frozen_artifacts"].get(
        "validation_ood_report", {}
    )
    if (
        Path(str(validation_identity.get("path", ""))).resolve() != validation_path
        or validation_identity.get("sha256") != file_sha256(validation_path)
    ):
        raise ValueError("locked OOD validation report differs from the access receipt")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(validation.get("checkpoint_path", ""))).resolve()
    calibration_path = Path(
        str(validation.get("known_calibration_scores_path", ""))
    ).resolve()
    heldout_path = Path(
        str(validation.get("heldout_evaluation_scores_path", ""))
    ).resolve()
    frozen_evaluation = validation.get("ood_evaluation", {})
    frozen_calibration = frozen_evaluation.get("calibration", {})
    if (
        validation.get("status")
        != "known_family_embedding_heldout_ood_validation"
        or validation.get("architecture") != "detector_set"
        or validation.get("ood_score_method") != "logit_energy"
        or validation.get("test_evaluation") is not None
        or validation.get("ood_score_fit", {}).get(
            "heldout_scores_used_for_method_or_fit_selection"
        )
        is not False
        or frozen_evaluation.get("status")
        != "frozen_known_only_ood_abstention_evaluation"
        or frozen_calibration.get("selection_data") != "known_validation_only"
        or frozen_calibration.get("unknown_scores_used_for_selection") is not False
        or not checkpoint_path.is_file()
        or validation.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or not calibration_path.is_file()
        or validation.get("known_calibration_scores_sha256")
        != file_sha256(calibration_path)
        or not heldout_path.is_file()
        or validation.get("heldout_evaluation_scores_sha256")
        != file_sha256(heldout_path)
    ):
        raise ValueError("locked OOD endpoint requires a replayable detector-set validation gate")

    score_report_path = Path(locked_score_report).resolve()
    score_report = json.loads(score_report_path.read_text(encoding="utf-8"))
    locked_path = Path(locked_score_manifest).resolve()
    source_path = Path(str(score_report.get("source_manifest_path", ""))).resolve()
    suite_inputs["source_manifest"] = validate_locked_evaluation_suite_input(
        locked_suite_plan,
        "locked_ood_source_manifest",
        source_path,
    )
    if (
        score_report.get("status") != "frozen_glitch_ood_scores_complete"
        or score_report.get("selection_data") != "validation_model_only"
        or score_report.get("test_scores_used_for_model_threshold_or_method_selection")
        is not False
        or score_report.get("required_split") != "test"
        or score_report.get("architecture") != "detector_set"
        or score_report.get("ood_score_method") != "logit_energy"
        or Path(str(score_report.get("manifest_path", ""))).resolve() != locked_path
        or score_report.get("manifest_sha256") != file_sha256(locked_path)
        or Path(str(score_report.get("validation_ood_report_path", ""))).resolve()
        != validation_path
        or score_report.get("validation_ood_report_sha256")
        != file_sha256(validation_path)
        or Path(str(score_report.get("checkpoint_path", ""))).resolve()
        != checkpoint_path
        or score_report.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or not source_path.is_file()
        or score_report.get("source_manifest_sha256") != file_sha256(source_path)
        or score_report.get("locked_suite_access") != suite_access
        or score_report.get("locked_suite_inputs") != suite_inputs
    ):
        raise ValueError("locked OOD score report failed frozen-model replay")

    def load(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"OOD score manifest is empty: {path}")
        return rows

    calibration_rows = load(calibration_path)
    heldout_rows = load(heldout_path)
    locked_rows = load(locked_path)
    if len(locked_rows) != int(score_report.get("rows", -1)):
        raise ValueError("locked OOD score row count differs from its report")
    if len(locked_rows) < int(suite_access["endpoints"]["minimum_locked_ood_rows"]):
        raise ValueError("locked OOD transfer set is smaller than the predeclared minimum")
    required = {
        "glitch_id",
        "gps_block",
        "glitch_family",
        "observing_run",
        "is_unknown",
        "available_ifos",
        "embedding_checkpoint_sha256",
        "ood_score_method",
        score_field,
    }
    missing = [index for index, row in enumerate(locked_rows) if required - set(row)]
    if missing:
        raise ValueError(f"locked OOD rows lack required fields at {missing[:10]}")
    if any(str(row.get("split")) != "test" for row in locked_rows):
        raise ValueError("locked OOD transfer rows must use the test split")
    scores = np.asarray([float(row[score_field]) for row in locked_rows])
    if not np.isfinite(scores).all():
        raise ValueError("locked OOD scores must be finite")
    if any(
        row["embedding_checkpoint_sha256"] != validation["checkpoint_sha256"]
        or row["ood_score_method"] != validation["ood_score_method"]
        for row in locked_rows
    ):
        raise ValueError("locked OOD rows differ from the validation-frozen model or score")
    overlaps = {}
    for label, rows in (
        ("known_calibration", calibration_rows),
        ("heldout_validation", heldout_rows),
    ):
        for field in ("glitch_id", "gps_block"):
            overlaps[f"{label}_{field}"] = sorted(
                {str(row[field]) for row in rows}
                & {str(row[field]) for row in locked_rows}
            )
    if any(overlaps.values()):
        raise ValueError(f"locked OOD transfer overlaps validation groups: {overlaps}")
    known = [row for row in locked_rows if not bool(row["is_unknown"])]
    unknown = [row for row in locked_rows if bool(row["is_unknown"])]
    if not known or not unknown:
        raise ValueError("locked OOD transfer requires known and unknown artifacts")
    threshold = float(frozen_calibration["threshold"])
    evaluated = [
        {**row, "abstained": float(row[score_field]) >= threshold}
        for row in locked_rows
    ]
    known_evaluated = [row for row in evaluated if not bool(row["is_unknown"])]
    unknown_evaluated = [row for row in evaluated if bool(row["is_unknown"])]

    def strata(field: str) -> dict[str, Any]:
        output_rows = {}
        for value in sorted({str(row[field]) for row in evaluated}):
            selected = [row for row in evaluated if str(row[field]) == value]
            selected_unknown = [row for row in selected if bool(row["is_unknown"])]
            selected_known = [row for row in selected if not bool(row["is_unknown"])]
            output_rows[value] = {
                "rows": len(selected),
                "unknown_false_acceptance": (
                    _rate(
                        sum(not row["abstained"] for row in selected_unknown),
                        len(selected_unknown),
                    )
                    if selected_unknown
                    else None
                ),
                "known_false_abstention": (
                    _rate(
                        sum(row["abstained"] for row in selected_known),
                        len(selected_known),
                    )
                    if selected_known
                    else None
                ),
            }
        return output_rows

    detector_rows = [
        {**row, "detector_subset": "+".join(str(ifo) for ifo in row["available_ifos"])}
        for row in evaluated
    ]
    evaluated = detector_rows
    result = {
        "status": "locked_detector_set_ood_transfer_evaluation",
        "endpoint_complete": True,
        "scientific_claim_allowed": False,
        "threshold_refits_on_test": 0,
        "threshold": threshold,
        "threshold_source": "known_validation_only",
        "score_field": score_field,
        "evaluation_rows": len(evaluated),
        "known_rows": len(known_evaluated),
        "unknown_rows": len(unknown_evaluated),
        "known_false_abstention": _rate(
            sum(row["abstained"] for row in known_evaluated), len(known_evaluated)
        ),
        "unknown_true_abstention": _rate(
            sum(row["abstained"] for row in unknown_evaluated), len(unknown_evaluated)
        ),
        "unknown_false_acceptance": _rate(
            sum(not row["abstained"] for row in unknown_evaluated),
            len(unknown_evaluated),
        ),
        "auroc_diagnostic": ood_auc(evaluated, score_field),
        "glitch_family_strata": strata("glitch_family"),
        "detector_subset_strata": strata("detector_subset"),
        "observing_run_strata": strata("observing_run"),
        "split_audit": {"passed": True, "cross_split_overlaps": overlaps},
        "validation_ood_report": {
            "path": str(validation_path),
            "sha256": file_sha256(validation_path),
        },
        "locked_score_manifest": {
            "path": str(locked_path),
            "sha256": file_sha256(locked_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "locked_suite_access": suite_access,
        "locked_suite_inputs": suite_inputs,
        "locked_score_report": {
            "path": str(score_report_path),
            "sha256": file_sha256(score_report_path),
        },
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def freeze_ood_held_family_protocol(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output: str | Path,
    excluded_families: Iterable[str] = (),
    minimum_train_rows: int = 20,
    minimum_validation_rows: int = 20,
    minimum_validation_gps_blocks: int = 5,
) -> dict[str, Any]:
    """Choose the next held family from labels/group counts before model scores exist."""

    def load(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    train = load(train_manifest)
    validation = load(validation_manifest)
    if not train or not validation:
        raise ValueError("OOD held-family protocol requires non-empty train/validation")
    if minimum_train_rows < 1 or minimum_validation_rows < 1:
        raise ValueError("OOD held-family row minima must be positive")
    if minimum_validation_gps_blocks < 1:
        raise ValueError("OOD held-family GPS-block minimum must be positive")
    if any(str(row.get("split")) != "train" for row in train):
        raise ValueError("OOD held-family training input must be train-only")
    if any(str(row.get("split")) != "val" for row in validation):
        raise ValueError("OOD held-family validation input must be val-only")
    required = {"glitch_id", "network_gps_block", "ml_label", "observing_run"}
    if any(required - set(row) for row in train + validation):
        raise ValueError("OOD held-family inputs lack physical group/family fields")
    group_overlaps = {
        field: sorted(
            {str(row[field]) for row in train}
            & {str(row[field]) for row in validation}
        )
        for field in ("glitch_id", "network_gps_block")
    }
    train_sources = set().union(*(_network_source_ids(row) for row in train))
    validation_sources = set().union(
        *(_network_source_ids(row) for row in validation)
    )
    if train_sources or validation_sources:
        group_overlaps["network_source"] = sorted(train_sources & validation_sources)
    if any(group_overlaps.values()):
        raise ValueError(f"OOD held-family base split leakage: {group_overlaps}")
    excluded = sorted({str(value) for value in excluded_families if str(value)})
    train_counts = Counter(str(row["ml_label"]) for row in train)
    validation_counts = Counter(str(row["ml_label"]) for row in validation)
    validation_blocks = {
        family: len(
            {
                str(row["network_gps_block"])
                for row in validation
                if str(row["ml_label"]) == family
            }
        )
        for family in validation_counts
    }
    candidates = []
    for family in sorted(set(train_counts) & set(validation_counts)):
        eligible = (
            family not in excluded
            and train_counts[family] >= minimum_train_rows
            and validation_counts[family] >= minimum_validation_rows
            and validation_blocks[family] >= minimum_validation_gps_blocks
        )
        candidates.append(
            {
                "glitch_family": family,
                "train_rows": train_counts[family],
                "validation_rows": validation_counts[family],
                "validation_gps_blocks": validation_blocks[family],
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise ValueError("no unexamined glitch family satisfies the frozen OOD minima")
    selected = min(
        eligible,
        key=lambda row: (
            -int(row["validation_rows"]),
            -int(row["validation_gps_blocks"]),
            -int(row["train_rows"]),
            str(row["glitch_family"]),
        ),
    )
    identity = {
        "method": "largest_validation_support_score_blind_v1",
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "excluded_families": excluded,
        "minimum_train_rows": minimum_train_rows,
        "minimum_validation_rows": minimum_validation_rows,
        "minimum_validation_gps_blocks": minimum_validation_gps_blocks,
        "selected_held_out_family": selected["glitch_family"],
    }
    result = {
        "status": "frozen_score_blind_held_glitch_family_protocol",
        "scientific_claim_allowed": False,
        "protocol_id": canonical_hash(identity, 32),
        "selection_method": identity["method"],
        "selection_data": "family labels, row counts and GPS-block counts only",
        "model_scores_used_for_selection": False,
        "unknown_scores_opened_before_selection": False,
        "identity": identity,
        "base_split_audit": {
            "passed": True,
            "cross_split_overlaps": group_overlaps,
        },
        "candidates": candidates,
        "selected": selected,
        **execution_provenance(),
    }
    output_path = Path(output)
    if output_path.is_file():
        completed = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.get("identity") != identity:
            raise ValueError("frozen OOD held-family output belongs to another protocol")
        if completed.get("protocol_id") != result["protocol_id"]:
            raise ValueError("frozen OOD held-family protocol identity is corrupted")
        return completed
    atomic_write_json(output_path, result)
    return result


def build_leave_one_family_out_split(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    held_out_family: str,
    output_dir: str | Path,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Freeze group-disjoint known training/calibration and held-family evaluation rows."""
    def load(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    train = load(train_manifest)
    validation = load(validation_manifest)
    if not train or not validation or not held_out_family:
        raise ValueError("leave-one-family-out split requires non-empty inputs and family")
    if any(row.get("split") != "train" for row in train):
        raise ValueError("leave-one-family-out training input must be train-only")
    if any(row.get("split") != "val" for row in validation):
        raise ValueError("leave-one-family-out validation input must be val-only")
    required = {"glitch_id", "network_gps_block", "ml_label", "observing_run"}
    if any(required - set(row) for row in train + validation):
        raise ValueError("Gravity Spy OOD split inputs lack group/family/run metadata")
    base_overlaps = {
        field: sorted(
            {str(row[field]) for row in train}
            & {str(row[field]) for row in validation}
        )
        for field in ("glitch_id", "network_gps_block")
    }
    train_sources = set().union(*(_network_source_ids(row) for row in train))
    validation_sources = set().union(
        *(_network_source_ids(row) for row in validation)
    )
    if train_sources or validation_sources:
        base_overlaps["network_source"] = sorted(train_sources & validation_sources)
    if any(base_overlaps.values()):
        raise ValueError(f"Gravity Spy OOD base split leakage: {base_overlaps}")
    if held_out_family not in {str(row["ml_label"]) for row in train + validation}:
        raise ValueError("held-out glitch family is absent from input manifests")
    held_train_blocks = {
        str(row["network_gps_block"])
        for row in train
        if str(row["ml_label"]) == held_out_family
    }
    known_train = [
        row
        for row in train
        if str(row["network_gps_block"]) not in held_train_blocks
        and str(row["ml_label"]) != held_out_family
    ]
    held_validation_blocks = {
        str(row["network_gps_block"])
        for row in validation
        if str(row["ml_label"]) == held_out_family
    }
    if not held_validation_blocks:
        raise ValueError("held-out family has no validation GPS blocks")
    evaluation = [
        row
        for row in validation
        if str(row["network_gps_block"]) in held_validation_blocks
    ]
    remaining_known_blocks = sorted(
        {
            str(row["network_gps_block"])
            for row in validation
            if str(row["network_gps_block"]) not in held_validation_blocks
            and str(row["ml_label"]) != held_out_family
        },
        key=lambda block: canonical_hash(
            {"gps_block": block, "seed": seed, "purpose": "ood_known_evaluation"}, 32
        ),
    )
    if not any(str(row["ml_label"]) != held_out_family for row in evaluation):
        if not remaining_known_blocks:
            raise ValueError("no group-disjoint known validation block is available for evaluation")
        selected_known_block = remaining_known_blocks.pop(0)
        evaluation.extend(
            row
            for row in validation
            if str(row["network_gps_block"]) == selected_known_block
        )
    evaluation_blocks = {str(row["network_gps_block"]) for row in evaluation}
    calibration = [
        row
        for row in validation
        if str(row["network_gps_block"]) not in evaluation_blocks
        and str(row["ml_label"]) != held_out_family
    ]
    if not known_train or not calibration:
        raise ValueError("leave-one-family-out split leaves empty known training/calibration data")

    def normalize(row: dict[str, Any], role: str) -> dict[str, Any]:
        return {
            **row,
            "gps_block": row["network_gps_block"],
            "glitch_family": row["ml_label"],
            "ood_role": role,
            "is_unknown": str(row["ml_label"]) == held_out_family,
            "held_out_family": held_out_family,
        }

    outputs = {
        "known_train": [normalize(row, "known_train") for row in known_train],
        "known_calibration": [
            normalize(row, "known_calibration") for row in calibration
        ],
        "heldout_evaluation": [
            normalize(row, "heldout_evaluation") for row in evaluation
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in outputs.items():
        path = output / f"{name}.jsonl"
        atomic_write_text(
            path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
        artifacts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(rows),
            "unique_glitches": len({str(row["glitch_id"]) for row in rows}),
            "unique_gps_blocks": len({str(row["gps_block"]) for row in rows}),
        }
    role_blocks = {
        name: {str(row["gps_block"]) for row in rows} for name, rows in outputs.items()
    }
    overlaps = {
        "train_calibration": sorted(role_blocks["known_train"] & role_blocks["known_calibration"]),
        "train_evaluation": sorted(role_blocks["known_train"] & role_blocks["heldout_evaluation"]),
        "calibration_evaluation": sorted(
            role_blocks["known_calibration"] & role_blocks["heldout_evaluation"]
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"leave-one-family-out GPS overlap after construction: {overlaps}")
    result = {
        "status": "frozen_leave_one_glitch_family_out_split",
        "scientific_claim_allowed": False,
        "held_out_family": held_out_family,
        "seed": seed,
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "excluded_train_gps_blocks_with_held_family": len(held_train_blocks),
        "held_validation_gps_blocks": len(held_validation_blocks),
        "split_audit": {"passed": True, "gps_block_overlaps": overlaps},
        "base_split_audit": {
            "passed": True,
            "cross_split_overlaps": base_overlaps,
        },
        "artifacts": artifacts,
        "evaluation_unknown_rows": sum(row["is_unknown"] for row in outputs["heldout_evaluation"]),
        "evaluation_known_rows": sum(not row["is_unknown"] for row in outputs["heldout_evaluation"]),
        **execution_provenance(),
    }
    atomic_write_json(output / "leave_one_family_out_report.json", result)
    return result


class GlitchOODDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_ifos: tuple[str, ...],
        q_count: int,
        label_to_index: dict[str, int],
        allow_unknown: bool = False,
        cache_in_memory: bool = True,
    ):
        self.rows = rows
        self.model_ifos = model_ifos
        self.q_count = q_count
        self.label_to_index = label_to_index
        self.allow_unknown = allow_unknown
        self.cache: list[tuple[np.ndarray, np.int64] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        row = self.rows[index]
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"Gravity Spy OOD sample hash mismatch: {row['glitch_id']}")
        ifo = str(row["ifo"])
        if ifo not in self.model_ifos:
            raise ValueError(f"Gravity Spy OOD sample uses unconfigured IFO: {ifo}")
        with np.load(row["path"], allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
        if features.ndim != 4 or features.shape[:2] != (
            len(self.model_ifos),
            self.q_count,
        ):
            raise ValueError(f"Gravity Spy OOD tensor shape mismatch: {row['glitch_id']}")
        label = str(row["glitch_family"])
        if label not in self.label_to_index and not self.allow_unknown:
            raise ValueError(f"unknown family entered known-only OOD data: {label}")
        item = features[self.model_ifos.index(ifo)], np.int64(
            self.label_to_index.get(label, -1)
        )
        if self.cache is not None:
            self.cache[index] = item
        return item


class DetectorSetGlitchOODDataset:
    """Aligned numeric H1/L1/V1 contexts with explicit detector availability."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_ifos: tuple[str, ...],
        q_values: tuple[float, ...],
        label_to_index: dict[str, int],
        allow_unknown: bool = False,
        cache_in_memory: bool = True,
    ):
        self.rows = rows
        self.model_ifos = model_ifos
        self.q_values = q_values
        self.label_to_index = label_to_index
        self.allow_unknown = allow_unknown
        self.cache: list[tuple[np.ndarray, np.ndarray, np.int64] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.int64]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        row = self.rows[index]
        if row.get("aligned_network_context") is not True:
            raise ValueError(
                f"network OOD sample lacks aligned context: {row['glitch_id']}"
            )
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"Gravity Spy OOD sample hash mismatch: {row['glitch_id']}")
        with np.load(row["path"], allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
            availability = np.asarray(
                arrays["detector_availability"], dtype=np.float32
            )
            ifos = tuple(str(value) for value in arrays["ifos"].tolist())
            q_values = tuple(float(value) for value in arrays["q_values"].tolist())
        expected_prefix = (len(self.model_ifos), len(self.q_values))
        if features.ndim != 4 or features.shape[:2] != expected_prefix:
            raise ValueError(
                f"network Gravity Spy OOD tensor shape mismatch: {row['glitch_id']}"
            )
        if ifos != self.model_ifos or not np.allclose(
            q_values, self.q_values, atol=1e-6
        ):
            raise ValueError("network OOD detector/Q metadata differs from configuration")
        if availability.shape != (len(self.model_ifos),) or np.any(
            (availability != 0) & (availability != 1)
        ):
            raise ValueError("network OOD detector availability must be binary [IFO]")
        if availability.sum() < 1:
            raise ValueError("network OOD sample has no available detector")
        declared = np.asarray(row.get("detector_availability"), dtype=np.float32)
        if declared.shape != availability.shape or not np.array_equal(
            declared, availability
        ):
            raise ValueError("network OOD row/array detector availability differs")
        available_ifos = tuple(
            ifo for ifo, valid in zip(self.model_ifos, availability) if valid
        )
        if tuple(row.get("available_ifos", ())) != available_ifos:
            raise ValueError("network OOD available IFO identities differ")
        if str(row["ifo"]) not in available_ifos:
            raise ValueError("network OOD event IFO is marked unavailable")
        if not np.isfinite(features).all():
            raise ValueError(f"network OOD tensor is non-finite: {row['glitch_id']}")
        if np.any(features[availability == 0] != 0):
            raise ValueError("unavailable network OOD detector planes must be zero")
        label = str(row["glitch_family"])
        if label not in self.label_to_index and not self.allow_unknown:
            raise ValueError(f"unknown family entered known-only OOD data: {label}")
        item = (
            features,
            availability,
            np.int64(self.label_to_index.get(label, -1)),
        )
        if self.cache is not None:
            self.cache[index] = item
        return item


def run_glitch_ood_embedding(
    config_path: str | Path,
    known_train_manifest: str | Path,
    known_calibration_manifest: str | Path,
    heldout_evaluation_manifest: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Train a known-family embedding and score held families without tuning on them."""
    if torch is None:
        raise RuntimeError("glitch OOD embedding training requires torch")
    from .numeric import (
        DetectorSetGlitchEmbeddingNet,
        GlitchEmbeddingNet,
        _atomic_torch_save,
    )
    from .io import load_yaml

    config = load_yaml(config_path)
    settings = config["glitch_ood_embedding"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "config_hash": canonical_hash(config),
        "config_file_sha256": file_sha256(config_path),
        "known_train_manifest_sha256": file_sha256(known_train_manifest),
        "known_calibration_manifest_sha256": file_sha256(known_calibration_manifest),
        "heldout_evaluation_manifest_sha256": file_sha256(heldout_evaluation_manifest),
        "seed": seed,
    }
    completed_report_path = output / "glitch_ood_embedding_report.json"
    if completed_report_path.is_file():
        completed = json.loads(completed_report_path.read_text(encoding="utf-8"))
        if completed.get("run_identity") != run_identity:
            raise ValueError("completed glitch OOD output belongs to another run")
        if file_sha256(completed["checkpoint_path"]) != completed["checkpoint_sha256"]:
            raise ValueError("completed glitch OOD checkpoint hash mismatch")
        return completed

    def load(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    train_rows = load(known_train_manifest)
    calibration_rows = load(known_calibration_manifest)
    evaluation_rows = load(heldout_evaluation_manifest)
    if not train_rows or not calibration_rows or not evaluation_rows:
        raise ValueError("glitch OOD embedding manifests must be non-empty")
    expected_roles = (
        (train_rows, "known_train"),
        (calibration_rows, "known_calibration"),
        (evaluation_rows, "heldout_evaluation"),
    )
    for rows, role in expected_roles:
        if any(str(row.get("ood_role")) != role for row in rows):
            raise ValueError(f"glitch OOD manifest mixes rows outside {role}")
    if any(bool(row["is_unknown"]) for row in train_rows + calibration_rows):
        raise ValueError("known-only OOD training/calibration contains held-out artifacts")
    if not any(bool(row["is_unknown"]) for row in evaluation_rows):
        raise ValueError("OOD evaluation contains no held-out artifacts")
    overlaps = {}
    for first_name, first_rows, second_name, second_rows in (
        ("train", train_rows, "calibration", calibration_rows),
        ("train", train_rows, "evaluation", evaluation_rows),
        ("calibration", calibration_rows, "evaluation", evaluation_rows),
    ):
        for field in ("glitch_id", "gps_block"):
            key = f"{first_name}_{second_name}_{field}"
            overlaps[key] = sorted(
                {str(row[field]) for row in first_rows}
                & {str(row[field]) for row in second_rows}
            )
    if any(overlaps.values()):
        raise ValueError(f"glitch OOD embedding split leakage: {overlaps}")
    labels = sorted({str(row["glitch_family"]) for row in train_rows})
    if len(labels) < 2:
        raise ValueError("glitch OOD embedding requires at least two known families")
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_calibration_labels = {
        str(row["glitch_family"]) for row in calibration_rows
    } - set(labels)
    if unknown_calibration_labels:
        raise ValueError(
            f"calibration contains families absent from known training: {unknown_calibration_labels}"
        )
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    architecture = str(settings.get("architecture", "single_ifo"))
    if architecture not in {"single_ifo", "detector_set"}:
        raise ValueError(f"unsupported glitch OOD architecture: {architecture}")
    dataset_class = (
        DetectorSetGlitchOODDataset
        if architecture == "detector_set"
        else GlitchOODDataset
    )

    def dataset(rows: list[dict[str, Any]], allow_unknown: bool = False) -> Any:
        common = {
            "rows": rows,
            "model_ifos": model_ifos,
            "label_to_index": label_to_index,
            "allow_unknown": allow_unknown,
            "cache_in_memory": bool(settings.get("cache_in_memory", True)),
        }
        if architecture == "detector_set":
            return dataset_class(q_values=q_values, **common)
        return dataset_class(q_count=len(q_values), **common)

    datasets = {
        "train": dataset(train_rows),
        "calibration": dataset(calibration_rows),
        "evaluation": dataset(evaluation_rows, allow_unknown=True),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_settings = {
        "q_count": len(q_values),
        "class_count": len(labels),
        "base_channels": int(settings.get("base_channels", 24)),
        "embedding_dim": int(settings.get("embedding_dim", 32)),
    }
    model = (
        DetectorSetGlitchEmbeddingNet(
            ifo_count=len(model_ifos), **model_settings
        )
        if architecture == "detector_set"
        else GlitchEmbeddingNet(**model_settings)
    ).to(device)
    counts = Counter(str(row["glitch_family"]) for row in train_rows)
    class_weights = torch.as_tensor(
        [len(train_rows) / (len(labels) * counts[label]) for label in labels],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )

    contrastive_weight = float(settings.get("supervised_contrastive_weight", 0.0))
    contrastive_temperature = float(
        settings.get("supervised_contrastive_temperature", 0.1)
    )
    if contrastive_weight < 0 or contrastive_temperature <= 0:
        raise ValueError("supervised contrastive configuration is invalid")

    def forward_batch(batch: Any) -> tuple[Any, Any, Any]:
        if architecture == "detector_set":
            features, availability, targets = batch
            logits, embeddings = model(
                features.to(device), availability.to(device)
            )
        else:
            features, targets = batch
            logits, embeddings = model(features.to(device))
        return logits, embeddings, targets.to(device)

    def epoch(loader: Any, training: bool) -> dict[str, float]:
        model.train(training)
        losses = []
        cross_entropy_losses = []
        contrastive_losses = []
        correct = total = 0
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits, embeddings, targets = forward_batch(batch)
                cross_entropy = torch_functional.cross_entropy(
                    logits, targets, weight=class_weights
                )
                contrastive = (
                    supervised_contrastive_loss(
                        embeddings, targets, contrastive_temperature
                    )
                    if training and contrastive_weight > 0
                    else embeddings.sum() * 0.0
                )
                loss = cross_entropy + contrastive_weight * contrastive
                if training:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            cross_entropy_losses.append(float(cross_entropy.detach().cpu()))
            contrastive_losses.append(float(contrastive.detach().cpu()))
            correct += int((logits.argmax(dim=1) == targets).sum().cpu())
            total += int(targets.numel())
        return {
            "loss": float(np.mean(losses)),
            "cross_entropy_loss": float(np.mean(cross_entropy_losses)),
            "supervised_contrastive_loss": float(np.mean(contrastive_losses)),
            "accuracy": correct / total,
        }

    checkpoint_path = output / "best_glitch_ood_embedding.pt"
    history = []
    best_accuracy = -1.0
    best_epoch = None
    started = time.time()
    for epoch_index in range(1, int(settings["epochs"]) + 1):
        train_metrics = epoch(loaders["train"], True)
        calibration_metrics = epoch(loaders["calibration"], False)
        history.append(
            {
                "epoch": epoch_index,
                "train": train_metrics,
                "known_calibration": calibration_metrics,
            }
        )
        if calibration_metrics["accuracy"] > best_accuracy:
            best_accuracy = calibration_metrics["accuracy"]
            best_epoch = epoch_index
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch_index,
                    "known_calibration_accuracy": best_accuracy,
                    "model_ifos": list(model_ifos),
                    "q_values": list(q_values),
                    "labels": labels,
                    "architecture": architecture,
                    "base_channels": int(settings.get("base_channels", 24)),
                    "embedding_dim": int(settings.get("embedding_dim", 32)),
                    "run_identity": run_identity,
                },
            )
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    model.eval()

    def embed(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = []
        logits = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch_logits, batch_embeddings, batch_targets = forward_batch(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
                logits.append(batch_logits.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
        return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(targets)

    train_embeddings, _, train_targets = embed(
        DataLoader(datasets["train"], batch_size=int(settings["batch_size"]), shuffle=False)
    )
    prototypes = np.stack(
        [train_embeddings[train_targets == index].mean(axis=0) for index in range(len(labels))]
    )
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    mahalanobis_fit = fit_class_conditional_mahalanobis(
        train_embeddings,
        train_targets,
        len(labels),
        float(settings.get("mahalanobis_shrinkage", 0.1)),
        float(settings.get("mahalanobis_epsilon", 1e-4)),
    )
    score_method = str(settings.get("ood_score_method", "prototype_cosine"))
    supported_score_methods = {
        "prototype_cosine": "prototype_cosine_ood_score",
        "class_conditional_mahalanobis": "class_conditional_mahalanobis_ood_score",
        "logit_energy": "logit_energy_ood_score",
    }
    if score_method not in supported_score_methods:
        raise ValueError(f"unsupported OOD score method: {score_method}")

    def score(rows: list[dict[str, Any]], loader: Any) -> list[dict[str, Any]]:
        embeddings, logits, _ = embed(loader)
        similarities = embeddings @ prototypes.T
        mahalanobis_scores = class_conditional_mahalanobis_scores(
            embeddings, mahalanobis_fit
        )
        logit_energy_scores = -np.logaddexp.reduce(logits, axis=1)
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        scored = []
        for index, row in enumerate(rows):
            diagnostics = {
                **row,
                "prototype_cosine_ood_score": float(1.0 - similarities[index].max()),
                "class_conditional_mahalanobis_ood_score": float(
                    mahalanobis_scores[index]
                ),
                "logit_energy_ood_score": float(logit_energy_scores[index]),
                "predicted_known_family": labels[int(similarities[index].argmax())],
                "known_classifier_confidence": float(probabilities[index].max()),
                "embedding_checkpoint_sha256": file_sha256(checkpoint_path),
            }
            scored.append(
                {
                    **diagnostics,
                    "ood_score": diagnostics[supported_score_methods[score_method]],
                    "ood_score_method": score_method,
                }
            )
        return scored

    scored_calibration = score(calibration_rows, loaders["calibration"])
    scored_evaluation = score(evaluation_rows, loaders["evaluation"])
    calibration_path = output / "known_calibration_scores.jsonl"
    evaluation_path = output / "heldout_evaluation_scores.jsonl"
    atomic_write_text(
        calibration_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_calibration),
    )
    atomic_write_text(
        evaluation_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_evaluation),
    )
    evaluation = evaluate_frozen_ood_threshold(
        scored_calibration,
        scored_evaluation,
        float(settings.get("maximum_known_abstention_rate", 0.05)),
    )
    report = {
        "status": "known_family_embedding_heldout_ood_validation",
        "scientific_claim_allowed": False,
        "auxiliary_policy": "attribution_or_review_only; cannot veto a strain-coherent candidate",
        "run_identity": run_identity,
        "device": str(device),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "labels": labels,
        "label_counts": dict(sorted(counts.items())),
        "architecture": architecture,
        "detector_context": {
            "model_ifos": list(model_ifos),
            "explicit_detector_identity": architecture == "detector_set",
            "explicit_detector_availability": architecture == "detector_set",
            "aligned_network_context_required": architecture == "detector_set",
            "train_detector_subsets": dict(
                sorted(
                    Counter(
                        "".join(row.get("available_ifos", ()))
                        if architecture == "detector_set"
                        else str(row["ifo"])
                        for row in train_rows
                    ).items()
                )
            ),
            "calibration_detector_subsets": dict(
                sorted(
                    Counter(
                        "".join(row.get("available_ifos", ()))
                        if architecture == "detector_set"
                        else str(row["ifo"])
                        for row in calibration_rows
                    ).items()
                )
            ),
            "evaluation_detector_subsets": dict(
                sorted(
                    Counter(
                        "".join(row.get("available_ifos", ()))
                        if architecture == "detector_set"
                        else str(row["ifo"])
                        for row in evaluation_rows
                    ).items()
                )
            ),
        },
        "ood_score_method": score_method,
        "supervised_contrastive": {
            "weight": contrastive_weight,
            "temperature": contrastive_temperature,
            "training_only": True,
        },
        "ood_score_fit": {
            "selection_data": "known_train_only",
            "known_train_rows": mahalanobis_fit["known_train_rows"],
            "mahalanobis_shrinkage": mahalanobis_fit["shrinkage"],
            "mahalanobis_epsilon": mahalanobis_fit["epsilon"],
            "diagnostic_score_fields": list(supported_score_methods.values()),
            "heldout_scores_used_for_method_or_fit_selection": False,
        },
        "best_epoch": best_epoch,
        "best_known_calibration_accuracy": best_accuracy,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "known_calibration_scores_path": str(calibration_path),
        "known_calibration_scores_sha256": file_sha256(calibration_path),
        "heldout_evaluation_scores_path": str(evaluation_path),
        "heldout_evaluation_scores_sha256": file_sha256(evaluation_path),
        "ood_evaluation": evaluation,
        "elapsed_seconds": time.time() - started,
        "test_evaluation": None,
    }
    atomic_write_json(completed_report_path, report)
    return report
