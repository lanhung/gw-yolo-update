from __future__ import annotations

import json
import os
import platform
import random
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .numeric import MultiIFOQNet, _atomic_torch_save, _dice_loss
from .physical_training import PhysicalInjectionDataset, _chirp_epoch, focal_binary_cross_entropy

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None
    WeightedRandomSampler = None


def gravityspy_numeric_split_audit(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not train_rows or not validation_rows:
        raise ValueError("Gravity Spy fine-tuning requires non-empty train and validation rows")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Gravity Spy train manifest contains a non-train row")
    if any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("Gravity Spy validation manifest contains a non-validation row")
    overlaps = {}
    for field in ("glitch_id", "network_gps_block"):
        train_values = {str(row[field]) for row in train_rows}
        validation_values = {str(row[field]) for row in validation_rows}
        overlaps[field] = sorted(train_values & validation_values)
    if any(overlaps.values()):
        raise ValueError(f"Gravity Spy numeric split leakage: {overlaps}")
    return {
        "passed": True,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_unique_glitches": len({row["glitch_id"] for row in train_rows}),
        "validation_unique_glitches": len(
            {row["glitch_id"] for row in validation_rows}
        ),
        "train_unique_network_gps_blocks": len(
            {row["network_gps_block"] for row in train_rows}
        ),
        "validation_unique_network_gps_blocks": len(
            {row["network_gps_block"] for row in validation_rows}
        ),
        "cross_split_overlaps": overlaps,
    }


def inverse_label_sampling_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("label-balanced sampling requires non-empty rows")
    counts = Counter(str(row["ml_label"]) for row in rows)
    weights = np.asarray([1.0 / counts[str(row["ml_label"])] for row in rows])
    return weights / weights.mean()


class GravitySpyNumericDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        input_channels: int,
        frequency_bins: int,
        time_bins: int,
        cache_in_memory: bool = True,
    ):
        self.rows = rows
        self.input_channels = input_channels
        self.frequency_bins = frequency_bins
        self.time_bins = time_bins
        self.cache: list[tuple[np.ndarray, np.ndarray] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        row = self.rows[index]
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"Gravity Spy numeric sample hash mismatch: {row['glitch_id']}")
        with np.load(row["path"], allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
            target = np.asarray(arrays["glitch_mask"], dtype=np.float32)
        expected = (self.input_channels, self.frequency_bins, self.time_bins)
        features = features.reshape(-1, *features.shape[-2:])
        target = target.reshape(-1, *target.shape[-2:])
        if features.shape != expected or target.shape != expected:
            raise ValueError(
                f"Gravity Spy tensor shape mismatch for {row['glitch_id']}: "
                f"{features.shape}/{target.shape} != {expected}"
            )
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError(f"Gravity Spy tensor is non-finite: {row['glitch_id']}")
        if np.any((target < 0) | (target > 1)):
            raise ValueError(f"Gravity Spy mask is outside [0,1]: {row['glitch_id']}")
        item = features, target
        if self.cache is not None:
            self.cache[index] = item
        return item


def _glitch_epoch(
    model: Any,
    teacher: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    positive_weight: float,
    focal_gamma: float,
    chirp_distillation_weight: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    teacher.eval()
    losses = []
    true_positive = false_positive = false_negative = 0
    positive = torch.as_tensor([positive_weight], device=device).reshape(1, 1, 1, 1, 1)
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            glitch_logits = logits[:, 1:2]
            bce = focal_binary_cross_entropy(
                glitch_logits, target[:, None], positive, focal_gamma
            )
            dice = _dice_loss(glitch_logits, target[:, None])
            with torch.no_grad():
                teacher_chirp = torch.sigmoid(teacher(features)[:, 0])
            chirp_distillation = torch_functional.binary_cross_entropy_with_logits(
                logits[:, 0], teacher_chirp
            )
            loss = bce + dice + chirp_distillation_weight * chirp_distillation
            if training:
                loss.backward()
                optimizer.step()
        predicted = torch.sigmoid(glitch_logits.detach()) >= threshold
        expected = target[:, None] >= 0.5
        true_positive += int((predicted & expected).sum().cpu())
        false_positive += int((predicted & ~expected).sum().cpu())
        false_negative += int((~predicted & expected).sum().cpu())
        losses.append(float(loss.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "glitch_iou": true_positive
        / max(true_positive + false_positive + false_negative, 1),
        "glitch_precision": true_positive / max(true_positive + false_positive, 1),
        "glitch_recall": true_positive / max(true_positive + false_negative, 1),
    }


def _calibrate_glitch_threshold(
    model: Any, loader: Any, device: Any, grid: tuple[float, ...]
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            probabilities.append(
                torch.sigmoid(model(features.to(device))[:, 1]).cpu().numpy()
            )
            targets.append(target.numpy())
    probability = np.concatenate(probabilities)
    expected = np.concatenate(targets) >= 0.5
    curve = []
    for threshold in grid:
        predicted = probability >= threshold
        tp = int(np.count_nonzero(predicted & expected))
        fp = int(np.count_nonzero(predicted & ~expected))
        fn = int(np.count_nonzero(~predicted & expected))
        curve.append(
            {
                "threshold": threshold,
                "iou": tp / max(tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
            }
        )
    selected = max(curve, key=lambda row: row["iou"])
    return float(selected["threshold"]), curve


def _glitch_label_metrics(
    model: Any,
    loader: Any,
    rows: list[dict[str, Any]],
    device: Any,
    threshold: float,
) -> dict[str, Any]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    offset = 0
    model.eval()
    with torch.no_grad():
        for features, target in loader:
            probability = torch.sigmoid(model(features.to(device))[:, 1]).cpu().numpy()
            expected = target.numpy() >= 0.5
            for index in range(features.shape[0]):
                label = str(rows[offset + index]["ml_label"])
                predicted = probability[index] >= threshold
                counts[label][0] += int(np.count_nonzero(predicted & expected[index]))
                counts[label][1] += int(np.count_nonzero(predicted & ~expected[index]))
                counts[label][2] += int(np.count_nonzero(~predicted & expected[index]))
                counts[label][3] += 1
            offset += features.shape[0]
    result = {}
    for label, (tp, fp, fn, samples) in sorted(counts.items()):
        result[label] = {
            "samples": samples,
            "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
    return result


def run_gravityspy_glitch_finetune(
    config_path: str | Path,
    glitch_train_manifest: str | Path,
    glitch_validation_manifest: str | Path,
    chirp_validation_manifest: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Gravity Spy glitch fine-tuning requires torch")
    config = load_yaml(config_path)
    settings = config["gravityspy_glitch_training"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "config_hash": canonical_hash(config),
        "glitch_train_manifest_sha256": file_sha256(glitch_train_manifest),
        "glitch_validation_manifest_sha256": file_sha256(glitch_validation_manifest),
        "chirp_validation_manifest_sha256": file_sha256(chirp_validation_manifest),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "seed": seed,
    }
    report_path = output / "gravityspy_glitch_finetune_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_identity") != run_identity:
            raise ValueError("Completed Gravity Spy fine-tune belongs to another run")
        return report
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with Path(glitch_train_manifest).open("r", encoding="utf-8") as handle:
        train_rows = [json.loads(line) for line in handle if line.strip()]
    with Path(glitch_validation_manifest).open("r", encoding="utf-8") as handle:
        validation_rows = [json.loads(line) for line in handle if line.strip()]
    with Path(chirp_validation_manifest).open("r", encoding="utf-8") as handle:
        chirp_rows = [json.loads(line) for line in handle if line.strip()]
    split_audit = gravityspy_numeric_split_audit(train_rows, validation_rows)
    if not chirp_rows or any(row.get("split") != "val" for row in chirp_rows):
        raise ValueError("chirp retention manifest must be non-empty and validation-only")
    if any(bool(row.get("human_pixel_mask")) for row in train_rows + validation_rows):
        raise ValueError("weak-mask fine-tune cannot mix an undeclared human-mask protocol")
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    channels = len(model_ifos) * len(q_values)
    frequency_bins = int(settings["tensor"]["frequency_bins"])
    time_bins = int(settings["tensor"]["time_bins"])
    datasets = {
        "glitch_train": GravitySpyNumericDataset(
            train_rows,
            channels,
            frequency_bins,
            time_bins,
            bool(settings.get("cache_in_memory", True)),
        ),
        "glitch_val": GravitySpyNumericDataset(
            validation_rows,
            channels,
            frequency_bins,
            time_bins,
            bool(settings.get("cache_in_memory", True)),
        ),
        "chirp_val": PhysicalInjectionDataset(
            chirp_rows,
            settings["tensor"],
            model_ifos,
            q_values,
            int(settings["target_sample_rate"]),
            bool(settings.get("cache_in_memory", True)),
        ),
    }
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(inverse_label_sampling_weights(train_rows), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_rows),
        replacement=True,
        generator=generator,
    )
    loaders = {
        "glitch_train": DataLoader(
            datasets["glitch_train"],
            batch_size=int(settings["batch_size"]),
            sampler=sampler,
            num_workers=0,
        ),
        "glitch_val": DataLoader(
            datasets["glitch_val"],
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            num_workers=0,
        ),
        "chirp_val": DataLoader(
            datasets["chirp_val"],
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            num_workers=0,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    if int(pretrained["input_channels"]) != channels:
        raise ValueError("pretrained checkpoint differs from Gravity Spy input channels")
    base_channels = int(pretrained["base_channels"])
    model = MultiIFOQNet(channels, base_channels).to(device)
    teacher = MultiIFOQNet(channels, base_channels).to(device)
    model.load_state_dict(pretrained["model"])
    teacher.load_state_dict(pretrained["model"])
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    started = time.time()

    def chirp_validation() -> dict[str, Any]:
        return _chirp_epoch(
            model,
            teacher,
            loaders["chirp_val"],
            device,
            None,
            float(settings["chirp_positive_weight"]),
            float(settings["glitch_distillation_weight"]),
            float(settings.get("focal_gamma", 0.0)),
            threshold=float(settings.get("chirp_threshold", 0.5)),
        )

    baseline_glitch = _glitch_epoch(
        model,
        teacher,
        loaders["glitch_val"],
        device,
        None,
        float(settings["glitch_positive_weight"]),
        float(settings.get("focal_gamma", 0.0)),
        float(settings["chirp_distillation_weight"]),
    )
    baseline_chirp = chirp_validation()
    minimum_retention = float(settings.get("minimum_chirp_iou_retention", 0.95))
    if not 0 < minimum_retention <= 1:
        raise ValueError("minimum chirp IoU retention must be in (0,1]")
    checkpoint_path = output / "best_gravityspy_glitch_finetune.pt"
    resume_path = output / "last_gravityspy_glitch_finetune.pt"
    if not checkpoint_path.is_file():
        _atomic_torch_save(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "epoch": 0,
                "validation_glitch_iou": baseline_glitch["glitch_iou"],
                "validation_chirp_iou": baseline_chirp["chirp_iou"],
                "input_channels": channels,
                "base_channels": base_channels,
                "run_identity": run_identity,
            },
        )
    else:
        prior_checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if prior_checkpoint.get("run_identity") != run_identity:
            raise ValueError("Gravity Spy best checkpoint belongs to another run")
    history: list[dict[str, Any]] = []
    best_glitch_iou = float(baseline_glitch["glitch_iou"])
    best_epoch = 0
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != run_identity:
            raise ValueError("Gravity Spy fine-tune resume belongs to another run")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_glitch_iou = float(resume["best_glitch_iou"])
        best_epoch = int(resume["best_epoch"])
        start_epoch = int(resume["epoch"]) + 1
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        training = _glitch_epoch(
            model,
            teacher,
            loaders["glitch_train"],
            device,
            optimizer,
            float(settings["glitch_positive_weight"]),
            float(settings.get("focal_gamma", 0.0)),
            float(settings["chirp_distillation_weight"]),
        )
        glitch_validation = _glitch_epoch(
            model,
            teacher,
            loaders["glitch_val"],
            device,
            None,
            float(settings["glitch_positive_weight"]),
            float(settings.get("focal_gamma", 0.0)),
            float(settings["chirp_distillation_weight"]),
        )
        chirp_metrics = chirp_validation()
        chirp_retention = float(chirp_metrics["chirp_iou"]) / max(
            float(baseline_chirp["chirp_iou"]), 1e-12
        )
        eligible = chirp_retention >= minimum_retention
        history.append(
            {
                "epoch": epoch,
                "train": training,
                "glitch_validation": glitch_validation,
                "chirp_validation": chirp_metrics,
                "chirp_iou_retention": chirp_retention,
                "selection_eligible": eligible,
            }
        )
        if eligible and float(glitch_validation["glitch_iou"]) > best_glitch_iou:
            best_glitch_iou = float(glitch_validation["glitch_iou"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_glitch_iou": best_glitch_iou,
                    "validation_chirp_iou": chirp_metrics["chirp_iou"],
                    "input_channels": channels,
                    "base_channels": base_channels,
                    "run_identity": run_identity,
                },
            )
        _atomic_torch_save(
            resume_path,
            {
                "run_identity": run_identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_generator_state": generator.get_state(),
                "epoch": epoch,
                "history": history,
                "best_glitch_iou": best_glitch_iou,
                "best_epoch": best_epoch,
            },
        )
        atomic_write_json(output / "history.json", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    threshold, threshold_curve = _calibrate_glitch_threshold(
        model,
        loaders["glitch_val"],
        device,
        tuple(float(item) for item in settings["threshold_grid"]),
    )
    calibrated_glitch = _glitch_epoch(
        model,
        teacher,
        loaders["glitch_val"],
        device,
        None,
        float(settings["glitch_positive_weight"]),
        float(settings.get("focal_gamma", 0.0)),
        float(settings["chirp_distillation_weight"]),
        threshold,
    )
    final_chirp = chirp_validation()
    report = {
        "status": "gravityspy_weak_mask_validation_only_finetune",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "all Gravity Spy masks are metadata-derived weak supervision; a frozen human pixel-mask "
            "audit, physical mixtures, fixed-FAR search and locked test remain required"
        ),
        "test_evaluation": None,
        "run_identity": run_identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": split_audit,
        "mask_provenance": "weak_gravityspy_duration_peak_frequency_q_geometry_v1",
        "human_pixel_masks": 0,
        "label_sampling": "inverse train label frequency with replacement",
        "train_label_counts": dict(
            sorted(Counter(str(row["ml_label"]) for row in train_rows).items())
        ),
        "validation_label_counts": dict(
            sorted(Counter(str(row["ml_label"]) for row in validation_rows).items())
        ),
        "selection_protocol": (
            "maximize fixed-threshold validation glitch IoU subject to physical chirp IoU "
            f"retention >= {minimum_retention:.6g}"
        ),
        "baseline": {"glitch_validation": baseline_glitch, "chirp_validation": baseline_chirp},
        "best_epoch": best_epoch,
        "selected_chirp_threshold": float(settings.get("chirp_threshold", 0.5)),
        "selected_glitch_threshold": threshold,
        "threshold_curve": threshold_curve,
        "calibrated_glitch_validation": calibrated_glitch,
        "selected_chirp_validation": final_chirp,
        "validation_label_metrics": _glitch_label_metrics(
            model, loaders["glitch_val"], validation_rows, device, threshold
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "epochs": int(settings["epochs"]),
        "history": history,
        "seed": seed,
        "device": str(device),
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(report_path, report)
    return report
