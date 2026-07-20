from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_refiner import (
    candidate_average_precision,
    candidate_interval_pair_features,
    candidate_pair_truth_support,
)
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .numeric import _atomic_torch_save
from .physical_training import physical_split_audit
from .runtime import execution_provenance

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    torch_functional = None
    DataLoader = None
    TensorDataset = None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def candidate_pair_feature_vector(
    first: dict[str, Any],
    second: dict[str, Any],
    physical_delay_limit_seconds: float,
    width_scale_seconds: float,
) -> np.ndarray:
    base = candidate_interval_pair_features(
        first, second, physical_delay_limit_seconds, width_scale_seconds
    )
    widths = [
        float(first["gps_end"]) - float(first["gps_start"]),
        float(second["gps_end"]) - float(second["gps_start"]),
    ]
    proposal = [float(first["proposal_score"]), float(second["proposal_score"])]
    relative_peaks = [
        (float(row["gps_peak"]) - float(row["gps_start"])) / width
        for row, width in zip((first, second), widths)
    ]
    peak_separation = abs(float(first["gps_peak"]) - float(second["gps_peak"]))
    overlap = max(
        min(float(first["gps_end"]), float(second["gps_end"]))
        - max(float(first["gps_start"]), float(second["gps_start"])),
        0.0,
    )
    values = np.asarray(
        [
            proposal[0],
            proposal[1],
            min(proposal),
            max(proposal),
            abs(proposal[0] - proposal[1]),
            widths[0] / width_scale_seconds,
            widths[1] / width_scale_seconds,
            min(widths) / width_scale_seconds,
            max(widths) / width_scale_seconds,
            abs(widths[0] - widths[1]) / width_scale_seconds,
            float(base["interval_gap_seconds"]) / physical_delay_limit_seconds,
            float(base["center_excess_normalized"]),
            min(peak_separation / width_scale_seconds, 32.0),
            overlap / width_scale_seconds,
            relative_peaks[0],
            relative_peaks[1],
        ],
        dtype=np.float32,
    )
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("candidate pair feature vector is invalid")
    return values


def candidate_parent_top1_metrics(
    parent_ids: list[str],
    example_parent_ids: list[str],
    scores: np.ndarray,
    padded_labels: np.ndarray,
    exact_labels: np.ndarray,
    peak_errors_seconds: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    padded = np.asarray(padded_labels, dtype=bool)
    exact = np.asarray(exact_labels, dtype=bool)
    errors = np.asarray(peak_errors_seconds, dtype=np.float64)
    if (
        not parent_ids
        or len(set(parent_ids)) != len(parent_ids)
        or values.shape != padded.shape
        or values.shape != exact.shape
        or values.shape != errors.shape
        or values.shape != (len(example_parent_ids),)
        or not np.isfinite(values).all()
        or not np.isfinite(errors).all()
    ):
        raise ValueError("candidate parent top1 metric inputs are invalid")
    indices: dict[str, list[int]] = defaultdict(list)
    for index, parent_id in enumerate(example_parent_ids):
        if parent_id not in set(parent_ids):
            raise ValueError("candidate pair example has an unknown parent")
        indices[parent_id].append(index)
    selected = []
    for parent_id in parent_ids:
        choices = indices.get(parent_id, [])
        if not choices:
            continue
        selected.append(max(choices, key=lambda index: (values[index], -index)))
    selected_array = np.asarray(selected, dtype=np.int64)
    found = len(selected)
    padded_count = int(np.count_nonzero(padded[selected_array])) if found else 0
    exact_count = int(np.count_nonzero(exact[selected_array])) if found else 0
    selected_errors = errors[selected_array] if found else np.asarray([], dtype=np.float64)
    return {
        "eligible_parents": len(parent_ids),
        "parents_with_compatible_pair": found,
        "compatible_pair_fraction": found / len(parent_ids),
        "top1_padded_truth_pair_fraction": padded_count / len(parent_ids),
        "top1_padded_truth_pair_wilson_95": list(
            wilson_interval(padded_count, len(parent_ids))
        ),
        "top1_exact_interval_truth_pair_fraction": exact_count / len(parent_ids),
        "top1_peak_error_seconds_quantiles": (
            {
                str(q): float(np.quantile(selected_errors, q))
                for q in (0.5, 0.9, 0.99, 1.0)
            }
            if found
            else None
        ),
    }


def _build_examples(
    parents: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    first_ifo: str,
    second_ifo: str,
    physical_delay_limit_seconds: float,
    width_scale_seconds: float,
    padding_seconds: float,
    maximum_negative_pairs_per_parent: int | None,
    seed: int,
) -> dict[str, Any]:
    parent_map = {str(row["injection_id"]): row for row in parents}
    by_parent: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in candidates:
        injection_id = str(row["injection_id"])
        if injection_id not in parent_map:
            raise ValueError("candidate pair training row has an unknown parent")
        by_parent[injection_id][str(row["ifo"])].append(row)
    eligible = []
    features = []
    padded_labels = []
    exact_labels = []
    peak_errors = []
    example_parent_ids = []
    retained_negative_pairs = 0
    available_negative_pairs = 0
    for injection_id, parent in sorted(parent_map.items()):
        arrivals = {
            str(ifo): float(value)
            for ifo, value in parent.get("detector_arrival_gps", {}).items()
        }
        if first_ifo not in arrivals or second_ifo not in arrivals:
            continue
        eligible.append(injection_id)
        rows = []
        for first in by_parent[injection_id].get(first_ifo, []):
            for second in by_parent[injection_id].get(second_ifo, []):
                pair_features = candidate_interval_pair_features(
                    first,
                    second,
                    physical_delay_limit_seconds,
                    width_scale_seconds,
                )
                if not pair_features["compatible"]:
                    continue
                support = candidate_pair_truth_support(
                    first, second, arrivals, padding_seconds
                )
                rows.append(
                    {
                        "features": candidate_pair_feature_vector(
                            first,
                            second,
                            physical_delay_limit_seconds,
                            width_scale_seconds,
                        ),
                        "padded": bool(support["padded"]),
                        "exact": bool(support["exact"]),
                        "peak_error": float(support["maximum_peak_error_seconds"]),
                        "pair_id": f'{first["candidate_id"]}|{second["candidate_id"]}',
                    }
                )
        positives = [row for row in rows if row["padded"]]
        negatives = [row for row in rows if not row["padded"]]
        available_negative_pairs += len(negatives)
        if maximum_negative_pairs_per_parent is not None:
            negatives.sort(
                key=lambda row: canonical_hash(
                    {
                        "pair_id": row["pair_id"],
                        "seed": seed,
                        "purpose": "candidate_pair_training_negative_v1",
                    }
                )
            )
            negatives = negatives[:maximum_negative_pairs_per_parent]
        retained_negative_pairs += len(negatives)
        for row in positives + negatives:
            features.append(row["features"])
            padded_labels.append(row["padded"])
            exact_labels.append(row["exact"])
            peak_errors.append(row["peak_error"])
            example_parent_ids.append(injection_id)
    if not eligible or not features or not any(padded_labels) or all(padded_labels):
        raise ValueError("candidate pair training examples lack parents or class diversity")
    return {
        "parent_ids": eligible,
        "features": np.stack(features),
        "padded_labels": np.asarray(padded_labels, dtype=bool),
        "exact_labels": np.asarray(exact_labels, dtype=bool),
        "peak_errors_seconds": np.asarray(peak_errors, dtype=np.float64),
        "example_parent_ids": example_parent_ids,
        "available_negative_pairs": available_negative_pairs,
        "retained_negative_pairs": retained_negative_pairs,
    }


if nn is not None:

    class CandidatePairMLP(nn.Module):
        def __init__(self, input_features: int = 16, hidden_features: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, 1),
            )

        def forward(self, value: Any) -> Any:
            return self.network(value)[:, 0]

else:

    class CandidatePairMLP:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            raise RuntimeError("Candidate pair training requires torch")


def _evaluate_pair_model(
    model: Any,
    examples: dict[str, Any],
    device: Any,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    features = torch.from_numpy(examples["features"])
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            scores.append(torch.sigmoid(model(features[start : start + batch_size].to(device))).cpu())
    values = torch.cat(scores).numpy()
    metrics = candidate_parent_top1_metrics(
        examples["parent_ids"],
        examples["example_parent_ids"],
        values,
        examples["padded_labels"],
        examples["exact_labels"],
        examples["peak_errors_seconds"],
    )
    metrics["pair_average_precision"] = candidate_average_precision(
        examples["padded_labels"], values
    )
    metrics["pairs"] = len(values)
    return metrics, values


def run_candidate_pair_ranker_training(
    config_path: str | Path,
    train_injection_manifest: str | Path,
    train_candidate_manifest: str | Path,
    validation_injection_manifest: str | Path,
    validation_selection_candidate_manifest: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Candidate pair ranker training requires torch")
    config = load_yaml(config_path)
    settings = config["candidate_pair_ranker"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "train_injection_manifest_sha256": file_sha256(train_injection_manifest),
        "train_candidate_manifest_sha256": file_sha256(train_candidate_manifest),
        "validation_injection_manifest_sha256": file_sha256(validation_injection_manifest),
        "validation_selection_candidate_manifest_sha256": file_sha256(
            validation_selection_candidate_manifest
        ),
        "seed": seed,
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_pair_ranker_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_identity") != identity:
            raise ValueError("completed candidate pair ranker has another identity")
        return report
    if any(output.iterdir()):
        raise FileExistsError("candidate pair ranker output must be empty")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_parents = _read_jsonl(train_injection_manifest)
    validation_all = _read_jsonl(validation_injection_manifest)
    train_candidates = _read_jsonl(train_candidate_manifest)
    validation_candidates = _read_jsonl(validation_selection_candidate_manifest)
    if any(row.get("refiner_role") != "train" for row in train_candidates) or any(
        row.get("refiner_role") != "selection" for row in validation_candidates
    ):
        raise ValueError("candidate pair ranker candidate roles differ")
    selection_ids = {str(row["injection_id"]) for row in validation_candidates}
    validation_parents = [
        row for row in validation_all if str(row["injection_id"]) in selection_ids
    ]
    split_audit = physical_split_audit(train_parents, validation_parents)
    first_ifo, second_ifo = (str(value) for value in settings["detector_pair"])
    common = (
        first_ifo,
        second_ifo,
        float(settings["physical_delay_limit_seconds"]),
        float(settings["width_scale_seconds"]),
        float(settings["positive_padding_seconds"]),
    )
    train_examples = _build_examples(
        train_parents,
        train_candidates,
        *common,
        int(settings["maximum_negative_pairs_per_parent"]),
        seed,
    )
    validation_examples = _build_examples(
        validation_parents,
        validation_candidates,
        *common,
        None,
        seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidatePairMLP(16, int(settings["hidden_features"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    features = torch.from_numpy(train_examples["features"])
    labels = torch.from_numpy(train_examples["padded_labels"].astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    positives = int(np.count_nonzero(train_examples["padded_labels"]))
    negatives = len(labels) - positives
    positive_weight = torch.as_tensor(negatives / positives, device=device)
    checkpoint_path = output / "best_candidate_pair_ranker.pt"
    resume_path = output / "last_candidate_pair_ranker.pt"
    history = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = None
    updates = 0
    maximum_updates = int(settings["max_optimizer_updates"])
    if maximum_updates <= 0 or maximum_updates > int(settings["epochs"]) * len(loader):
        raise ValueError("candidate pair ranker optimizer budget is invalid")
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != identity:
            raise ValueError("candidate pair ranker resume identity differs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_key = tuple(float(value) for value in resume["best_key"])
        best_epoch = resume["best_epoch"]
        updates = int(resume["optimizer_updates"])
        start_epoch = int(resume["epoch"]) + 1
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        model.train()
        losses = []
        for batch_features, batch_labels in loader:
            if updates >= maximum_updates:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = torch_functional.binary_cross_entropy_with_logits(
                logits, batch_labels.to(device), pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            updates += 1
        validation_metrics, _ = _evaluate_pair_model(
            model,
            validation_examples,
            device,
            int(settings["evaluation_batch_size"]),
        )
        validation_metrics["loss"] = float(np.mean(losses))
        history.append({"epoch": epoch, "validation_selection": validation_metrics})
        key = (
            -float(validation_metrics["top1_padded_truth_pair_fraction"]),
            float(validation_metrics["top1_peak_error_seconds_quantiles"]["0.9"]),
            -float(validation_metrics["pair_average_precision"]),
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "architecture": "candidate_pair_mlp_v1",
                    "model": model.state_dict(),
                    "input_features": 16,
                    "hidden_features": int(settings["hidden_features"]),
                    "epoch": epoch,
                    "validation_selection_key": key,
                    "run_identity": identity,
                },
            )
        _atomic_torch_save(
            resume_path,
            {
                "run_identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_generator_state": generator.get_state(),
                "history": history,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "optimizer_updates": updates,
                "epoch": epoch,
            },
        )
        atomic_write_json(output / "history.json", history)
        if updates >= maximum_updates:
            break
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    selected_metrics, scores = _evaluate_pair_model(
        model,
        validation_examples,
        device,
        int(settings["evaluation_batch_size"]),
    )
    scores_path = output / "validation_selection_pair_scores.npz"
    with scores_path.open("wb") as handle:
        np.savez_compressed(handle, scores=scores.astype(np.float32))
    gates = {
        "minimum_top1_padded_pair_fraction": float(
            selected_metrics["top1_padded_truth_pair_fraction"]
        )
        >= float(settings["minimum_top1_padded_pair_fraction"]),
        "maximum_top1_peak_p90": float(
            selected_metrics["top1_peak_error_seconds_quantiles"]["0.9"]
        )
        <= float(settings["maximum_top1_peak_p90_seconds"]),
    }
    result = {
        "status": "validation_selection_candidate_pair_ranker",
        "scientific_claim_allowed": False,
        "search_promotion_allowed": False,
        "scientific_blocker": (
            "fresh group-disjoint calibration, continuous background FAR/IFAR and locked-test VT "
            "remain required"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "split_audit": split_audit,
        "architecture": "candidate_pair_mlp_v1",
        "detector_pair": [first_ifo, second_ifo],
        "physical_delay_limit_seconds": common[2],
        "all_validation_compatible_pairs_scored": True,
        "top_k_pruning": None,
        "training_negative_sampling": {
            "maximum_per_parent": int(settings["maximum_negative_pairs_per_parent"]),
            "available": train_examples["available_negative_pairs"],
            "retained": train_examples["retained_negative_pairs"],
            "all_positive_pairs_retained": True,
        },
        "train_pairs": len(train_examples["features"]),
        "validation_selection_pairs": len(validation_examples["features"]),
        "validation_selection_parents": len(validation_examples["parent_ids"]),
        "best_epoch": best_epoch,
        "selection_metric": "maximum parent top1 padded pair fraction, then minimum peak p90",
        "selected_validation_metrics": selected_metrics,
        "selection_gate_checks": gates,
        "selection_gate_passed": all(gates.values()),
        "optimizer_updates": updates,
        "max_optimizer_updates": maximum_updates,
        "training_budget_reached": updates == maximum_updates,
        "completed_epochs": len(history),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "scores_path": str(scores_path),
        "scores_sha256": file_sha256(scores_path),
        "history": history,
        "elapsed_seconds": time.time() - started,
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result
