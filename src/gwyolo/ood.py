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

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
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


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("OOD rate requires a valid non-empty binomial count")
    return {
        "count": successes,
        "total": total,
        "rate": successes / total,
        "wilson_95": list(wilson_interval(successes, total)),
    }


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
    from .numeric import GlitchEmbeddingNet, _atomic_torch_save
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
    datasets = {
        "train": GlitchOODDataset(
            train_rows,
            model_ifos,
            len(q_values),
            label_to_index,
            cache_in_memory=bool(settings.get("cache_in_memory", True)),
        ),
        "calibration": GlitchOODDataset(
            calibration_rows,
            model_ifos,
            len(q_values),
            label_to_index,
            cache_in_memory=bool(settings.get("cache_in_memory", True)),
        ),
        "evaluation": GlitchOODDataset(
            evaluation_rows,
            model_ifos,
            len(q_values),
            label_to_index,
            allow_unknown=True,
            cache_in_memory=bool(settings.get("cache_in_memory", True)),
        ),
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
    model = GlitchEmbeddingNet(
        len(q_values),
        len(labels),
        int(settings.get("base_channels", 24)),
        int(settings.get("embedding_dim", 32)),
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

    def epoch(loader: Any, training: bool) -> dict[str, float]:
        model.train(training)
        losses = []
        correct = total = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits, _ = model(features)
                loss = torch_functional.cross_entropy(
                    logits, targets, weight=class_weights
                )
                if training:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == targets).sum().cpu())
            total += int(targets.numel())
        return {"loss": float(np.mean(losses)), "accuracy": correct / total}

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
            for features, batch_targets in loader:
                batch_logits, batch_embeddings = model(features.to(device))
                embeddings.append(batch_embeddings.cpu().numpy())
                logits.append(batch_logits.cpu().numpy())
                targets.append(batch_targets.numpy())
        return np.concatenate(embeddings), np.concatenate(logits), np.concatenate(targets)

    train_embeddings, _, train_targets = embed(
        DataLoader(datasets["train"], batch_size=int(settings["batch_size"]), shuffle=False)
    )
    prototypes = np.stack(
        [train_embeddings[train_targets == index].mean(axis=0) for index in range(len(labels))]
    )
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    def score(rows: list[dict[str, Any]], loader: Any) -> list[dict[str, Any]]:
        embeddings, logits, _ = embed(loader)
        similarities = embeddings @ prototypes.T
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return [
            {
                **row,
                "ood_score": float(1.0 - similarities[index].max()),
                "predicted_known_family": labels[int(similarities[index].argmax())],
                "known_classifier_confidence": float(probabilities[index].max()),
                "embedding_checkpoint_sha256": file_sha256(checkpoint_path),
            }
            for index, row in enumerate(rows)
        ]

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
