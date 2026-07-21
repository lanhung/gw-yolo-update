from __future__ import annotations

import json
import random
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
) -> dict[str, float]:
    model.train()
    teacher.eval()
    clean_batches = iter(clean_loader)
    losses = []
    for overlap_features, overlap_targets, overlap_availability in overlap_loader:
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
            teacher_glitch = torch.sigmoid(teacher_logits[:, 1:2])
        glitch_distillation = torch_functional.binary_cross_entropy_with_logits(
            clean_logits[:, 1:2], teacher_glitch, reduction="none"
        )
        glitch_distillation = (glitch_distillation * clean_mask).sum() / (
            clean_mask.sum() * clean_logits.shape[-2] * clean_logits.shape[-1]
        ).clamp_min(1.0)
        loss = (
            overlap_loss
            + float(settings.get("clean_chirp_weight", 1.0)) * clean_chirp_loss
            + float(settings.get("clean_glitch_distillation_weight", 0.25))
            * glitch_distillation
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)), "optimizer_updates": len(losses)}


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
    clean_loaders = {
        key: DataLoader(value, batch_size=batch_size, shuffle=key == "train", num_workers=0, generator=generator if key == "train" else None)
        for key, value in clean_datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    teacher, teacher_architecture = model_from_checkpoint(pretrained, model_ifos, q_values)
    teacher = teacher.to(device).requires_grad_(False)
    student = DetectorSetQNet(len(model_ifos), len(q_values), int(pretrained["base_channels"])).to(device)
    if teacher_architecture == "detector_set":
        student.load_state_dict(pretrained["model"])
        warm_start = {"status": "exact_detector_set_state_dict"}
    else:
        warm_start = initialize_detector_set_from_early_fusion(student, pretrained)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    q_count = len(q_values)
    teacher_clean = _clean_metrics(
        teacher, teacher_architecture, clean_loaders["val"], device, q_count
    )
    checkpoint_path = output / "best_overlap_finetune.pt"
    resume_path = output / "last_overlap_finetune.pt"
    history = []
    best_metric = -1.0
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
        best_metric = float(resume["best_validation_overlap_mean_iou"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
    retention_fraction = float(settings.get("minimum_clean_chirp_iou_retention", 0.95))
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_metrics = _train_epoch(
            student, teacher, teacher_architecture, overlap_loaders["train"], clean_loaders["train"], device, optimizer, q_count, settings
        )
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
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "overlap_validation": overlap_validation,
                "clean_validation": clean_validation,
                "clean_chirp_iou_retention": retention,
                "checkpoint_eligible": eligible,
            }
        )
        metric = float(overlap_validation["mean_iou"])
        if eligible and metric > best_metric:
            best_metric = metric
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
                    "epoch": epoch,
                    "validation_overlap_mean_iou": metric,
                    "clean_chirp_iou_retention": retention,
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
                "best_validation_overlap_mean_iou": best_metric,
                "best_epoch": best_epoch,
            },
        )
        atomic_write_json(output / "history.json", history)
    if best_epoch is None:
        raise RuntimeError("No overlap checkpoint passed the clean-chirp retention gate")
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(selected["model"])
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
            "weak_glitch_mask_human_audit",
            "aligned_multi_ifo_glitch_contexts",
            "continuous_background_far_ifar_vt",
            "five_seed_locked_evaluation",
        ],
        "seed": seed,
        "run_identity": run_identity,
        "split_audit": split_audit,
        "clean_split_audit": clean_split_audit,
        "glitch_family_sampling": sampling_report,
        "warm_start": warm_start,
        "teacher_architecture": teacher_architecture,
        "teacher_clean_validation": teacher_clean,
        "minimum_clean_chirp_iou_retention": retention_fraction,
        "best_epoch": best_epoch,
        "best_validation_overlap_mean_iou": best_metric,
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
