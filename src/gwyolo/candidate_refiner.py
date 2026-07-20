from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .arrival_timing import DetectorArrivalDataset
from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    load_yaml,
)
from .metrics import wilson_interval
from .numeric import (
    CandidateEndpointWarmRefiner,
    CandidateLocalSpectrogramRefiner,
    _atomic_torch_save,
)
from .physical_training import physical_split_audit
from .runtime import execution_provenance

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _validation_role(injection_id: str, selection_fraction: float, seed: int) -> str:
    draw = int(
        canonical_hash(
            {
                "injection_id": injection_id,
                "seed": seed,
                "purpose": "candidate_refiner_validation_selection_v1",
            },
            16,
        ),
        16,
    ) / float(16**16 - 1)
    return "selection" if draw < selection_fraction else "calibration"


def label_candidate_refiner_rows(
    injection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    split: str,
    positive_padding_seconds: float,
    validation_selection_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label every candidate without top-k pruning and preserve parent-level roles."""

    if split not in {"train", "val"} or positive_padding_seconds < 0:
        raise ValueError("candidate refiner split or positive padding is invalid")
    if not 0 < validation_selection_fraction < 1:
        raise ValueError("candidate refiner validation selection fraction is invalid")
    parents = {}
    roles = {}
    for row in injection_rows:
        if row.get("split") != split:
            raise ValueError("candidate refiner injection manifest has the wrong split")
        injection_id = str(row["injection_id"])
        if injection_id in parents:
            raise ValueError(f"candidate refiner repeats injection: {injection_id}")
        parents[injection_id] = row
        roles[injection_id] = (
            "train"
            if split == "train"
            else _validation_role(injection_id, validation_selection_fraction, seed)
        )
    if not parents:
        raise ValueError("candidate refiner requires injection parents")
    candidate_ids = set()
    output = []
    by_arrival: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"candidate refiner repeats candidate: {candidate_id}")
        candidate_ids.add(candidate_id)
        injection_id = str(row["injection_id"])
        if injection_id not in parents or row.get("split") != split:
            raise ValueError("candidate refiner candidate has the wrong parent or split")
        ifo = str(row["ifo"])
        arrivals = parents[injection_id].get("detector_arrival_gps", {})
        if ifo not in arrivals:
            raise ValueError("candidate refiner candidate uses an unavailable detector")
        start = float(row["gps_start"])
        stop = float(row["gps_end"])
        peak = float(row["gps_peak"])
        arrival = float(arrivals[ifo])
        if not np.isfinite([start, stop, peak, arrival]).all() or stop <= start:
            raise ValueError("candidate refiner candidate geometry is invalid")
        distance = max(start - arrival, 0.0, arrival - stop)
        positive = distance <= positive_padding_seconds
        enriched = {
            **row,
            "refiner_role": roles[injection_id],
            "refiner_positive": bool(positive),
            "target_detector_arrival_gps": arrival,
            "interval_distance_to_arrival_seconds": distance,
            "peak_error_seconds": abs(peak - arrival),
            "positive_padding_seconds": positive_padding_seconds,
            "top_k_pruned": False,
        }
        output.append(enriched)
        by_arrival[(injection_id, ifo)].append(bool(positive))
    expected_arrivals = {
        (injection_id, str(ifo))
        for injection_id, parent in parents.items()
        for ifo in parent.get("detector_arrival_gps", {})
    }
    missing_candidate_arrivals = sorted(expected_arrivals - set(by_arrival))
    if missing_candidate_arrivals:
        raise ValueError(
            f"candidate refiner input lacks candidates for arrivals: {missing_candidate_arrivals[:10]}"
        )
    covered = sum(any(by_arrival[key]) for key in expected_arrivals)
    positive_count = sum(bool(row["refiner_positive"]) for row in output)
    role_counts = Counter(str(row["refiner_role"]) for row in output)
    parent_role_counts = Counter(roles.values())
    report = {
        "split": split,
        "injections": len(parents),
        "waveforms": len({str(row["waveform_id"]) for row in parents.values()}),
        "gps_blocks": len({str(row["gps_block"]) for row in parents.values()}),
        "candidates": len(output),
        "positive_candidates": positive_count,
        "negative_candidates": len(output) - positive_count,
        "positive_candidate_fraction": positive_count / max(len(output), 1),
        "expected_detector_arrivals": len(expected_arrivals),
        "arrivals_with_positive_candidate": covered,
        "positive_candidate_coverage_fraction": covered / len(expected_arrivals),
        "positive_candidate_coverage_wilson_95": list(
            wilson_interval(covered, len(expected_arrivals))
        ),
        "candidate_counts_by_role": dict(sorted(role_counts.items())),
        "parent_counts_by_role": dict(sorted(parent_role_counts.items())),
        "candidate_counts_by_ifo": dict(
            sorted(Counter(str(row["ifo"]) for row in output).items())
        ),
        "positive_counts_by_ifo": dict(
            sorted(
                Counter(
                    str(row["ifo"]) for row in output if row["refiner_positive"]
                ).items()
            )
        ),
        "all_connected_candidates_retained": len(output) == len(candidate_rows),
        "top_k_pruning": None,
    }
    return output, report


def run_candidate_refiner_plan(
    train_injection_manifest: str | Path,
    train_candidate_manifest: str | Path,
    validation_injection_manifest: str | Path,
    validation_candidate_manifest: str | Path,
    output_dir: str | Path,
    positive_padding_seconds: float = 0.5,
    validation_selection_fraction: float = 0.2,
    seed: int = 20260720,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "train_injection_manifest_sha256": file_sha256(train_injection_manifest),
        "train_candidate_manifest_sha256": file_sha256(train_candidate_manifest),
        "validation_injection_manifest_sha256": file_sha256(
            validation_injection_manifest
        ),
        "validation_candidate_manifest_sha256": file_sha256(
            validation_candidate_manifest
        ),
        "positive_padding_seconds": float(positive_padding_seconds),
        "validation_selection_fraction": float(validation_selection_fraction),
        "seed": int(seed),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_refiner_plan_report.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed candidate refiner plan has another identity")
        return result
    if any(output.iterdir()):
        raise FileExistsError("candidate refiner plan output must be empty")
    train_injections = _read_jsonl(train_injection_manifest)
    validation_injections = _read_jsonl(validation_injection_manifest)
    split_audit = physical_split_audit(train_injections, validation_injections)
    train_rows, train_summary = label_candidate_refiner_rows(
        train_injections,
        _read_jsonl(train_candidate_manifest),
        "train",
        positive_padding_seconds,
        validation_selection_fraction,
        seed,
    )
    validation_rows, validation_summary = label_candidate_refiner_rows(
        validation_injections,
        _read_jsonl(validation_candidate_manifest),
        "val",
        positive_padding_seconds,
        validation_selection_fraction,
        seed,
    )
    destinations = {
        "train": output / "candidate_refiner_train.jsonl",
        "selection": output / "candidate_refiner_validation_selection.jsonl",
        "calibration": output / "candidate_refiner_validation_calibration.jsonl",
    }
    atomic_write_text(
        destinations["train"],
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
    )
    for role in ("selection", "calibration"):
        rows = [row for row in validation_rows if row["refiner_role"] == role]
        atomic_write_text(
            destinations[role],
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    result = {
        "status": "candidate_local_refiner_group_safe_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "labeled candidate plans are train/validation supervision, not search recall; "
            "continuous background and locked-test VT remain required"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "split_audit": split_audit,
        "train": train_summary,
        "validation": validation_summary,
        "manifests": {role: str(path) for role, path in destinations.items()},
        "manifest_sha256": {
            role: file_sha256(path) for role, path in destinations.items()
        },
        "validation_parent_roles_are_group_safe": True,
        "all_connected_candidates_retained": True,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def candidate_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    expected = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if expected.ndim != 1 or values.shape != expected.shape or not values.size:
        raise ValueError("candidate average precision inputs must be aligned vectors")
    if not np.isfinite(values).all() or not np.any(expected):
        raise ValueError("candidate average precision requires finite scores and positives")
    order = np.argsort(-values, kind="stable")
    ranked = expected[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / np.count_nonzero(ranked))


def candidate_crop_contains_arrival(
    candidate_row: dict[str, Any], local_duration_seconds: float
) -> bool:
    """Return whether the physical arrival is inside a peak-centred local crop."""

    peak = float(candidate_row["gps_peak"])
    arrival = float(candidate_row["target_detector_arrival_gps"])
    if not np.isfinite([peak, arrival]).all() or local_duration_seconds <= 0:
        raise ValueError("candidate local crop geometry is invalid")
    crop_start = peak - local_duration_seconds / 2
    return crop_start <= arrival < crop_start + local_duration_seconds


def _candidate_timing_prediction(logits: Any, estimator: str) -> Any:
    if estimator == "argmax":
        return torch.argmax(logits, dim=1).to(logits.dtype)
    if estimator == "expected_probability":
        bins = torch.arange(
            logits.shape[1], dtype=logits.dtype, device=logits.device
        )[None]
        return torch.sum(torch.softmax(logits, dim=1) * bins, dim=1)
    raise ValueError("candidate timing estimator is unknown")


def candidate_arrival_threshold_metrics(
    prediction_rows: list[dict[str, Any]],
    thresholds: list[float],
    timing_tolerances_seconds: list[float],
) -> list[dict[str, Any]]:
    """Measure abstention and refined top-score timing for each physical arrival."""

    if not prediction_rows or not thresholds or not timing_tolerances_seconds:
        raise ValueError("candidate arrival metrics require predictions and grids")
    if thresholds != sorted(set(thresholds)) or any(
        not 0 <= value <= 1 for value in thresholds
    ):
        raise ValueError("candidate presence thresholds must be sorted unique probabilities")
    if timing_tolerances_seconds != sorted(set(timing_tolerances_seconds)) or any(
        value <= 0 for value in timing_tolerances_seconds
    ):
        raise ValueError("candidate timing tolerances must be sorted and positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in prediction_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError("candidate arrival metrics repeat a candidate")
        seen.add(candidate_id)
        score = float(row["presence_score"])
        error = float(row["refined_timing_error_seconds"])
        if not np.isfinite([score, error]).all() or not 0 <= score <= 1 or error < 0:
            raise ValueError("candidate arrival metrics contain invalid predictions")
        groups[(str(row["injection_id"]), str(row["ifo"]))].append(row)
    arrival_count = len(groups)
    output = []
    for threshold in thresholds:
        accepted = 0
        retained_total = 0
        correct = {tolerance: 0 for tolerance in timing_tolerances_seconds}
        retained_counts = []
        for rows in groups.values():
            retained = [
                row for row in rows if float(row["presence_score"]) >= threshold
            ]
            retained_total += len(retained)
            retained_counts.append(len(retained))
            if not retained:
                continue
            accepted += 1
            selected = min(
                retained,
                key=lambda row: (
                    -float(row["presence_score"]),
                    str(row["candidate_id"]),
                ),
            )
            error = float(selected["refined_timing_error_seconds"])
            for tolerance in timing_tolerances_seconds:
                correct[tolerance] += error <= tolerance
        count_values = np.asarray(retained_counts, dtype=np.float64)
        output.append(
            {
                "presence_threshold": threshold,
                "physical_arrivals": arrival_count,
                "accepted_arrivals": accepted,
                "arrival_acceptance_fraction": accepted / arrival_count,
                "arrival_acceptance_wilson_95": list(
                    wilson_interval(accepted, arrival_count)
                ),
                "retained_candidates": retained_total,
                "retained_candidates_per_arrival_quantiles": {
                    str(q): float(np.quantile(count_values, q))
                    for q in (0.5, 0.9, 0.99, 1.0)
                },
                "top_score_refined_timing": {
                    str(tolerance): {
                        "correct_arrivals": correct[tolerance],
                        "unconditional_fraction": correct[tolerance] / arrival_count,
                        "unconditional_wilson_95": list(
                            wilson_interval(correct[tolerance], arrival_count)
                        ),
                        "conditional_on_acceptance_fraction": (
                            correct[tolerance] / accepted if accepted else None
                        ),
                    }
                    for tolerance in timing_tolerances_seconds
                },
            }
        )
    return output


def candidate_positive_timing_error_quantiles(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, float]:
    errors = np.asarray(
        [
            float(row["refined_timing_error_seconds"])
            for row in prediction_rows
            if bool(row["local_crop_contains_arrival"])
        ],
        dtype=np.float64,
    )
    if not errors.size or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("candidate timing summary requires finite localizable errors")
    return {
        str(q): float(np.quantile(errors, q)) for q in (0.5, 0.9, 0.99, 1.0)
    }


class CandidateLocalDataset:
    def __init__(
        self,
        injection_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        model_ifos: tuple[str, ...],
        target_sample_rate: int,
        analysis_duration_seconds: float,
        parent_output_bins: int,
        local_duration_seconds: float,
        local_output_bins: int,
        cache_parents: bool,
    ):
        self.injection_rows = injection_rows
        self.candidate_rows = candidate_rows
        self.model_ifos = model_ifos
        self.target_sample_rate = int(target_sample_rate)
        self.local_duration_seconds = float(local_duration_seconds)
        self.local_output_bins = int(local_output_bins)
        self.local_samples = int(round(local_duration_seconds * target_sample_rate))
        if (
            self.local_samples <= 0
            or self.local_output_bins < 2
            or self.local_samples % self.local_output_bins
        ):
            raise ValueError("candidate local crop geometry is invalid")
        self.parent_indices = {
            str(row["injection_id"]): index
            for index, row in enumerate(injection_rows)
        }
        if len(self.parent_indices) != len(injection_rows):
            raise ValueError("candidate local dataset repeats injection parents")
        for row in candidate_rows:
            if str(row["injection_id"]) not in self.parent_indices:
                raise ValueError("candidate local dataset has an unknown parent")
            if str(row["ifo"]) not in model_ifos:
                raise ValueError("candidate local dataset has an unknown detector")
        self.parents = DetectorArrivalDataset(
            injection_rows,
            model_ifos,
            target_sample_rate,
            analysis_duration_seconds,
            parent_output_bins,
            cache_parents,
        )

    def __len__(self) -> int:
        return len(self.candidate_rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        row = self.candidate_rows[index]
        parent_index = self.parent_indices[str(row["injection_id"])]
        strain, availability, _, offsets = self.parents[parent_index]
        ifo_index = self.model_ifos.index(str(row["ifo"]))
        if not availability[ifo_index] or not np.isfinite(offsets[ifo_index]):
            raise ValueError("candidate local crop uses an unavailable detector")
        arrival = float(row["target_detector_arrival_gps"])
        analysis_start = arrival - float(offsets[ifo_index])
        crop_start_gps = float(row["gps_peak"]) - self.local_duration_seconds / 2
        source_start = int(round((crop_start_gps - analysis_start) * self.target_sample_rate))
        source_stop = source_start + self.local_samples
        crop = np.zeros((len(self.model_ifos), self.local_samples), dtype=np.float32)
        copy_start = max(source_start, 0)
        copy_stop = min(source_stop, strain.shape[-1])
        if copy_stop > copy_start:
            destination_start = copy_start - source_start
            crop[:, destination_start : destination_start + copy_stop - copy_start] = strain[
                :, copy_start:copy_stop
            ]
        local_offset = arrival - crop_start_gps
        positive = candidate_crop_contains_arrival(row, self.local_duration_seconds)
        timing_target = -1
        if positive:
            if not 0 <= local_offset < self.local_duration_seconds:
                raise ValueError("positive candidate arrival lies outside the local crop")
            timing_target = min(
                int(
                    np.floor(
                        local_offset
                        * self.local_output_bins
                        / self.local_duration_seconds
                    )
                ),
                self.local_output_bins - 1,
            )
        return (
            crop,
            availability.astype(bool),
            np.int64(ifo_index),
            np.float32(positive),
            np.int64(timing_target),
            np.float64(local_offset),
        )


def _candidate_refiner_epoch(
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    positive_weight: float,
    focal_gamma: float,
    timing_loss_weight: float,
    label_smoothing: float,
    timing_loss_mode: str,
    gaussian_sigma_bins: float,
    coordinate_loss_weight: float,
    timing_estimator: str,
    local_duration_seconds: float,
    local_output_bins: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses = []
    labels = []
    scores = []
    timing_errors = []
    examples = 0
    batches = 0
    weight = torch.as_tensor(positive_weight, device=device)
    for crop, availability, ifo_index, presence, timing_target, local_offset in loader:
        crop = crop.to(device)
        availability = availability.to(device)
        ifo_index = ifo_index.to(device)
        presence = presence.to(device)
        timing_target = timing_target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            presence_logits, timing_logits = model(crop, availability, ifo_index)
            raw = torch_functional.binary_cross_entropy_with_logits(
                presence_logits, presence, pos_weight=weight, reduction="none"
            )
            probability = torch.sigmoid(presence_logits)
            correct = probability * presence + (1.0 - probability) * (1.0 - presence)
            presence_loss = torch.mean(raw * ((1.0 - correct) ** focal_gamma))
            positive = timing_target >= 0
            if torch.any(positive) and timing_loss_mode == "categorical":
                timing_loss = torch_functional.cross_entropy(
                    timing_logits[positive],
                    timing_target[positive],
                    label_smoothing=label_smoothing,
                )
            elif torch.any(positive) and timing_loss_mode == "gaussian_coordinate":
                selected_logits = timing_logits[positive]
                selected_targets = timing_target[positive]
                bins = torch.arange(
                    local_output_bins,
                    dtype=selected_logits.dtype,
                    device=selected_logits.device,
                )[None]
                soft_targets = torch.exp(
                    -0.5
                    * ((bins - selected_targets[:, None]) / gaussian_sigma_bins) ** 2
                )
                soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
                distribution_loss = torch.mean(
                    torch.sum(
                        -soft_targets
                        * torch_functional.log_softmax(selected_logits, dim=1),
                        dim=1,
                    )
                )
                probabilities = torch.softmax(selected_logits, dim=1)
                expected_bin = torch.sum(probabilities * bins, dim=1)
                denominator = max(local_output_bins - 1, 1)
                coordinate_loss = torch_functional.smooth_l1_loss(
                    expected_bin / denominator,
                    selected_targets.to(expected_bin.dtype) / denominator,
                    beta=0.01,
                )
                timing_loss = (
                    distribution_loss + coordinate_loss_weight * coordinate_loss
                )
            else:
                timing_loss = timing_logits.sum() * 0.0
            loss = presence_loss + timing_loss_weight * timing_loss
            if training:
                loss.backward()
                optimizer.step()
        labels.extend(presence.detach().cpu().numpy().astype(bool).tolist())
        scores.extend(probability.detach().cpu().numpy().tolist())
        if torch.any(positive):
            predicted = _candidate_timing_prediction(
                timing_logits[positive], timing_estimator
            ).cpu().numpy()
            predicted_offset = (
                predicted.astype(np.float64) + 0.5
            ) * local_duration_seconds / local_output_bins
            exact = local_offset.numpy()[timing_target.cpu().numpy() >= 0]
            timing_errors.extend(np.abs(predicted_offset - exact).tolist())
        losses.append(float(loss.detach().cpu()))
        examples += int(crop.shape[0])
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    error_values = np.asarray(timing_errors, dtype=np.float64)
    return {
        "loss": float(np.mean(losses)),
        "average_precision": candidate_average_precision(
            np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
        ),
        "positive_candidates": int(np.count_nonzero(labels)),
        "negative_candidates": int(len(labels) - np.count_nonzero(labels)),
        "positive_timing_error_seconds_quantiles": {
            str(q): float(np.quantile(error_values, q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
        "within_10ms_positive_fraction": float(np.mean(error_values <= 0.01)),
        "examples": examples,
        "batches": batches,
    }


def run_candidate_local_refiner_training(
    config_path: str | Path,
    train_injection_manifest: str | Path,
    train_candidate_manifest: str | Path,
    validation_injection_manifest: str | Path,
    validation_selection_candidate_manifest: str | Path,
    validation_calibration_candidate_manifest: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
    pretrained_endpoint_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Candidate local refiner training requires torch")
    config = load_yaml(config_path)
    settings = config["candidate_local_refiner"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "train_injection_manifest_sha256": file_sha256(train_injection_manifest),
        "train_candidate_manifest_sha256": file_sha256(train_candidate_manifest),
        "validation_injection_manifest_sha256": file_sha256(
            validation_injection_manifest
        ),
        "validation_selection_candidate_manifest_sha256": file_sha256(
            validation_selection_candidate_manifest
        ),
        "validation_calibration_candidate_manifest_sha256": file_sha256(
            validation_calibration_candidate_manifest
        ),
        "seed": seed,
        "pretrained_endpoint_checkpoint_sha256": (
            file_sha256(pretrained_endpoint_checkpoint)
            if pretrained_endpoint_checkpoint is not None
            else None
        ),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_local_refiner_report.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed candidate local refiner has another identity")
        return result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_injections = _read_jsonl(train_injection_manifest)
    validation_injections = _read_jsonl(validation_injection_manifest)
    split_audit = physical_split_audit(train_injections, validation_injections)
    train_candidates = _read_jsonl(train_candidate_manifest)
    selection_candidates = _read_jsonl(validation_selection_candidate_manifest)
    calibration_candidates = _read_jsonl(validation_calibration_candidate_manifest)
    selection_ids = {str(row["injection_id"]) for row in selection_candidates}
    calibration_ids = {str(row["injection_id"]) for row in calibration_candidates}
    if not selection_ids or not calibration_ids or selection_ids & calibration_ids:
        raise ValueError("candidate refiner validation parent roles overlap or are empty")
    if selection_ids | calibration_ids != {
        str(row["injection_id"]) for row in validation_injections
    }:
        raise ValueError("candidate refiner validation roles do not cover every parent")
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    target_rate = int(settings["target_sample_rate"])
    analysis_duration = float(settings["analysis_duration_seconds"])
    parent_output_bins = int(settings["parent_output_bins"])
    local_duration = float(settings["local_duration_seconds"])
    local_output_bins = int(settings["local_output_bins"])
    datasets = {
        "train": CandidateLocalDataset(
            train_injections,
            train_candidates,
            model_ifos,
            target_rate,
            analysis_duration,
            parent_output_bins,
            local_duration,
            local_output_bins,
            bool(settings.get("cache_parents", True)),
        ),
        "selection": CandidateLocalDataset(
            validation_injections,
            selection_candidates,
            model_ifos,
            target_rate,
            analysis_duration,
            parent_output_bins,
            local_duration,
            local_output_bins,
            bool(settings.get("cache_parents", True)),
        ),
        "calibration": CandidateLocalDataset(
            validation_injections,
            calibration_candidates,
            model_ifos,
            target_rate,
            analysis_duration,
            parent_output_bins,
            local_duration,
            local_output_bins,
            bool(settings.get("cache_parents", True)),
        ),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=split == "train",
            num_workers=0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    positives = sum(
        candidate_crop_contains_arrival(row, local_duration)
        for row in train_candidates
    )
    negatives = len(train_candidates) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("candidate refiner training needs positive and negative candidates")
    positive_weight = negatives / positives
    timing_loss_mode = str(settings.get("timing_loss_mode", "categorical"))
    gaussian_sigma_bins = float(settings.get("gaussian_sigma_bins", 2.0))
    coordinate_loss_weight = float(settings.get("coordinate_loss_weight", 0.0))
    timing_estimator = str(settings.get("timing_estimator", "argmax"))
    if timing_loss_mode not in {"categorical", "gaussian_coordinate"}:
        raise ValueError("candidate refiner timing loss mode is unknown")
    if gaussian_sigma_bins <= 0 or coordinate_loss_weight < 0:
        raise ValueError("candidate refiner soft timing loss settings are invalid")
    if timing_estimator not in {"argmax", "expected_probability"}:
        raise ValueError("candidate refiner timing estimator is unknown")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = str(
        settings.get("architecture", "candidate_local_spectrogram_refiner_v1")
    )
    model_class = {
        "candidate_local_spectrogram_refiner_v1": CandidateLocalSpectrogramRefiner,
        "candidate_endpoint_warm_refiner_v2": CandidateEndpointWarmRefiner,
    }.get(architecture)
    if model_class is None:
        raise ValueError("candidate refiner architecture is unknown")
    model = model_class(
        len(model_ifos), local_output_bins, int(settings["base_channels"])
    ).to(device)
    if architecture == "candidate_endpoint_warm_refiner_v2":
        if pretrained_endpoint_checkpoint is None:
            raise ValueError("endpoint-warm refiner requires a pretrained checkpoint")
        warm = torch.load(
            pretrained_endpoint_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            warm.get("architecture") != "detector_endpoint_spectrogram_dense_v1"
            or list(warm.get("model_ifos", [])) != list(model_ifos)
            or int(warm.get("base_channels", -1)) != int(settings["base_channels"])
            or int(warm.get("output_bins", -1)) != 1024
        ):
            raise ValueError("endpoint-warm refiner checkpoint geometry differs")
        model.load_endpoint_backbone(warm["model"])
    elif pretrained_endpoint_checkpoint is not None:
        raise ValueError("local refiner architecture does not accept endpoint warm start")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_path = output / "best_candidate_local_refiner.pt"
    resume_path = output / "last_candidate_local_refiner.pt"
    history = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = None
    start_epoch = 1
    updates = 0
    examples = 0
    steps = len(loaders["train"])
    maximum_updates = int(settings["max_optimizer_updates"])
    if maximum_updates <= 0 or maximum_updates > int(settings["epochs"]) * steps:
        raise ValueError("candidate refiner optimizer budget is invalid")
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != identity:
            raise ValueError("candidate local refiner resume identity differs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_key = tuple(float(value) for value in resume["best_key"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
        updates = int(resume["optimizer_updates"])
        examples = int(resume["optimizer_examples"])
    started = time.time()
    epoch_arguments = (
        positive_weight,
        float(settings["focal_gamma"]),
        float(settings["timing_loss_weight"]),
        float(settings["label_smoothing"]),
        timing_loss_mode,
        gaussian_sigma_bins,
        coordinate_loss_weight,
        timing_estimator,
        local_duration,
        local_output_bins,
    )
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        remaining = maximum_updates - updates
        if remaining <= 0:
            break
        train_metrics = _candidate_refiner_epoch(
            model,
            loaders["train"],
            device,
            optimizer,
            *epoch_arguments,
            max_batches=min(remaining, steps),
        )
        updates += int(train_metrics["batches"])
        examples += int(train_metrics["examples"])
        selection_metrics = _candidate_refiner_epoch(
            model, loaders["selection"], device, None, *epoch_arguments
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "selection": selection_metrics}
        )
        timing_p90 = float(
            selection_metrics["positive_timing_error_seconds_quantiles"]["0.9"]
        )
        key = (
            -float(selection_metrics["average_precision"]),
            timing_p90,
            float(selection_metrics["loss"]),
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "architecture": architecture,
                    "model": model.state_dict(),
                    "model_ifos": list(model_ifos),
                    "target_sample_rate": target_rate,
                    "local_duration_seconds": local_duration,
                    "local_output_bins": local_output_bins,
                    "base_channels": int(settings["base_channels"]),
                    "timing_estimator": timing_estimator,
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
                "epoch": epoch,
                "history": history,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "optimizer_updates": updates,
                "optimizer_examples": examples,
            },
        )
        atomic_write_json(output / "history.json", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    calibration_evaluation_allowed = bool(
        settings.get("calibration_evaluation_allowed", True)
    )
    calibration_metrics = (
        _candidate_refiner_epoch(
            model, loaders["calibration"], device, None, *epoch_arguments
        )
        if calibration_evaluation_allowed
        else None
    )
    result = {
        "status": "validation_selected_candidate_local_timing_abstention_refiner",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "candidate-level calibration, continuous background FAR/VT, multi-seed evidence "
            "and locked-test evaluation remain required"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": split_audit,
        "validation_parent_partition": {
            "selection_injections": len(selection_ids),
            "calibration_injections": len(calibration_ids),
            "overlap": 0,
        },
        "architecture": architecture,
        "model_ifos": list(model_ifos),
        "all_candidates_scored": True,
        "top_k_pruning": None,
        "train_candidates": len(train_candidates),
        "selection_candidates": len(selection_candidates),
        "calibration_candidates": len(calibration_candidates),
        "positive_weight": positive_weight,
        "presence_supervision": "physical_arrival_inside_peak_centered_local_crop",
        "local_crop_positive_candidates": positives,
        "proposal_interval_positive_candidates": sum(
            bool(row["refiner_positive"]) for row in train_candidates
        ),
        "timing_loss_mode": timing_loss_mode,
        "timing_estimator": timing_estimator,
        "best_epoch": best_epoch,
        "selection_metric": (
            "maximum validation-selection candidate average precision, then minimum positive "
            "timing p90 and loss"
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "calibration_candidate_metrics": calibration_metrics,
        "calibration_evaluation_allowed": calibration_evaluation_allowed,
        "calibration_preexposure_note": settings.get("calibration_preexposure_note"),
        "epochs": int(settings["epochs"]),
        "completed_epochs": len(history),
        "steps_per_full_epoch": steps,
        "max_optimizer_updates": maximum_updates,
        "optimizer_updates": updates,
        "optimizer_examples": examples,
        "training_budget_reached": updates == maximum_updates,
        "history": history,
        "elapsed_seconds": time.time() - started,
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result


def run_candidate_local_refiner_validation(
    config_path: str | Path,
    checkpoint_path: str | Path,
    validation_injection_manifest: str | Path,
    validation_candidate_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze a validation-only abstention threshold and retain every prediction."""

    if torch is None:
        raise RuntimeError("Candidate local refiner validation requires torch")
    config = load_yaml(config_path)
    settings = config["candidate_refiner_validation"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_injection_manifest_sha256": file_sha256(
            validation_injection_manifest
        ),
        "validation_candidate_manifest_sha256": file_sha256(
            validation_candidate_manifest
        ),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_local_refiner_validation_report.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed candidate validation has another identity")
        return result
    if any(output.iterdir()):
        raise FileExistsError("candidate validation output must be empty")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = str(checkpoint.get("architecture"))
    model_class = {
        "candidate_local_spectrogram_refiner_v1": CandidateLocalSpectrogramRefiner,
        "candidate_endpoint_warm_refiner_v2": CandidateEndpointWarmRefiner,
    }.get(architecture)
    if model_class is None:
        raise ValueError("candidate validation checkpoint architecture differs")
    injections = _read_jsonl(validation_injection_manifest)
    candidates = _read_jsonl(validation_candidate_manifest)
    parent_ids = {str(row["injection_id"]) for row in candidates}
    if not parent_ids or any(
        row.get("refiner_role") != "calibration" for row in candidates
    ):
        raise ValueError("candidate validation requires only calibration-role candidates")
    selected_injections = [
        row for row in injections if str(row["injection_id"]) in parent_ids
    ]
    if len(selected_injections) != len(parent_ids):
        raise ValueError("candidate validation lacks calibration parents")
    if any(row.get("split") != "val" for row in selected_injections):
        raise ValueError("candidate validation may only use the validation split")
    model_ifos = tuple(str(value) for value in checkpoint["model_ifos"])
    target_rate = int(checkpoint["target_sample_rate"])
    local_duration = float(checkpoint["local_duration_seconds"])
    local_output_bins = int(checkpoint["local_output_bins"])
    if (
        model_ifos != tuple(str(value) for value in settings["model_ifos"])
        or target_rate != int(settings["target_sample_rate"])
        or local_duration != float(settings["local_duration_seconds"])
        or local_output_bins != int(settings["local_output_bins"])
    ):
        raise ValueError("candidate validation geometry differs from checkpoint")
    dataset = CandidateLocalDataset(
        selected_injections,
        candidates,
        model_ifos,
        target_rate,
        float(settings["analysis_duration_seconds"]),
        int(settings["parent_output_bins"]),
        local_duration,
        local_output_bins,
        bool(settings.get("cache_parents", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(
        len(model_ifos), local_output_bins, int(checkpoint["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prediction_rows = []
    cursor = 0
    with torch.no_grad():
        for crop, availability, ifo_index, _, _, _ in loader:
            presence_logits, timing_logits = model(
                crop.to(device), availability.to(device), ifo_index.to(device)
            )
            scores = torch.sigmoid(presence_logits).cpu().numpy()
            timing_bins = _candidate_timing_prediction(
                timing_logits, str(settings["timing_estimator"])
            ).cpu().numpy()
            for batch_index, (score, timing_bin) in enumerate(
                zip(scores, timing_bins, strict=True)
            ):
                source = candidates[cursor + batch_index]
                crop_start = float(source["gps_peak"]) - local_duration / 2
                refined_gps = crop_start + (
                    float(timing_bin) + 0.5
                ) * local_duration / local_output_bins
                target_gps = float(source["target_detector_arrival_gps"])
                prediction_rows.append(
                    {
                        "candidate_id": str(source["candidate_id"]),
                        "injection_id": str(source["injection_id"]),
                        "ifo": str(source["ifo"]),
                        "presence_score": float(score),
                        "proposal_interval_positive": bool(
                            source["refiner_positive"]
                        ),
                        "local_crop_contains_arrival": candidate_crop_contains_arrival(
                            source, local_duration
                        ),
                        "predicted_local_timing_bin": float(timing_bin),
                        "refined_arrival_gps": refined_gps,
                        "target_detector_arrival_gps": target_gps,
                        "refined_timing_error_seconds": abs(refined_gps - target_gps),
                        "parent_split": "val",
                        "validation_role": "calibration",
                    }
                )
            cursor += len(scores)
    if cursor != len(candidates):
        raise RuntimeError("candidate validation did not score every candidate")
    predictions_path = output / "candidate_local_refiner_predictions.jsonl"
    atomic_write_text(
        predictions_path,
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows
        ),
    )
    labels = np.asarray(
        [row["local_crop_contains_arrival"] for row in prediction_rows], dtype=bool
    )
    scores = np.asarray(
        [row["presence_score"] for row in prediction_rows], dtype=np.float64
    )
    thresholds = [float(value) for value in settings["presence_thresholds"]]
    tolerances = [float(value) for value in settings["timing_tolerances_seconds"]]
    threshold_metrics = candidate_arrival_threshold_metrics(
        prediction_rows, thresholds, tolerances
    )
    primary_tolerance = float(settings["primary_timing_tolerance_seconds"])
    if primary_tolerance not in tolerances:
        raise ValueError("primary candidate timing tolerance is absent from the grid")
    minimum_acceptance = float(settings["minimum_arrival_acceptance_fraction"])
    eligible = [
        row
        for row in threshold_metrics
        if row["arrival_acceptance_fraction"] >= minimum_acceptance
    ]
    selected_threshold = eligible[-1] if eligible else None
    average_precision = candidate_average_precision(labels, scores)
    timing_quantiles = candidate_positive_timing_error_quantiles(prediction_rows)
    gate_checks = {
        "candidate_average_precision": average_precision
        >= float(settings["minimum_candidate_average_precision"]),
        "positive_candidate_timing_p90": timing_quantiles["0.9"]
        <= float(settings["maximum_positive_timing_p90_seconds"]),
        "threshold_with_required_arrival_acceptance": selected_threshold is not None,
        "selected_top_score_primary_timing": (
            selected_threshold is not None
            and selected_threshold["top_score_refined_timing"][str(primary_tolerance)][
                "unconditional_fraction"
            ]
            >= float(settings["minimum_top_score_primary_timing_fraction"])
        ),
    }
    result = {
        "status": "validation_calibrated_candidate_timing_abstention_threshold",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "the selected validation threshold has not been tested on signal-free continuous "
            "background, so false acceptance, FAR/IFAR and VT remain unknown"
        ),
        "search_promotion_allowed": False,
        "test_evaluation": None,
        "run_identity": identity,
        "checkpoint_training_identity": checkpoint.get("run_identity"),
        "checkpoint_selected_epoch": checkpoint.get("epoch"),
        "calibration_injections": len(parent_ids),
        "calibration_candidates": len(candidates),
        "physical_detector_arrivals": len(
            {(row["injection_id"], row["ifo"]) for row in prediction_rows}
        ),
        "all_candidates_scored": True,
        "top_k_pruning": None,
        "candidate_average_precision": average_precision,
        "timing_estimator": str(settings["timing_estimator"]),
        "positive_candidate_timing_error_seconds_quantiles": timing_quantiles,
        "threshold_selection_rule": (
            "highest predeclared presence threshold retaining at least the configured "
            "fraction of calibration detector arrivals"
        ),
        "minimum_arrival_acceptance_fraction": minimum_acceptance,
        "selected_presence_threshold": (
            selected_threshold["presence_threshold"] if selected_threshold else None
        ),
        "selected_threshold_metrics": selected_threshold,
        "threshold_metrics": threshold_metrics,
        "validation_timing_gate_checks": gate_checks,
        "validation_timing_gate_passed": all(gate_checks.values()),
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result
