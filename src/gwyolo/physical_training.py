from __future__ import annotations

import json
import os
import platform
import random
import shlex
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .numeric import MultiIFOQNet, _atomic_torch_save, _dice_loss
from .waveforms import load_materialized_context

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None


def relative_component_mask(power: np.ndarray, fraction: float = 0.08) -> np.ndarray:
    values = np.asarray(power)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError("component power must be finite [IFO, Q, frequency, time]")
    if not 0 < fraction < 1:
        raise ValueError("component mask fraction must be between zero and one")
    flattened = values.reshape(*values.shape[:2], -1)
    peaks = np.max(flattened, axis=-1)[..., None, None]
    return ((peaks > 0) & (values >= peaks * fraction)).astype(np.float32)


def physical_split_audit(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not train_rows or not validation_rows:
        raise ValueError("Physical fine-tuning requires non-empty train and validation manifests")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Training manifest contains a non-train row")
    if any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("Validation manifest contains a non-val row")
    overlaps = {}
    for field in ("injection_id", "waveform_id", "gps_block"):
        train_ids = {str(row[field]) for row in train_rows}
        validation_ids = {str(row[field]) for row in validation_rows}
        overlaps[field] = sorted(train_ids & validation_ids)
    if any(overlaps.values()):
        raise ValueError(f"Physical fine-tuning split leakage: {overlaps}")
    return {
        "passed": True,
        "train_injections": len(train_rows),
        "validation_injections": len(validation_rows),
        "train_waveforms": len({row["waveform_id"] for row in train_rows}),
        "validation_waveforms": len({row["waveform_id"] for row in validation_rows}),
        "train_gps_blocks": len({row["gps_block"] for row in train_rows}),
        "validation_gps_blocks": len({row["gps_block"] for row in validation_rows}),
        "cross_split_overlaps": overlaps,
    }


class PhysicalInjectionDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tensor_config: dict[str, Any],
        model_ifos: tuple[str, ...],
        q_values: tuple[float, ...],
        target_sample_rate: int,
        cache_in_memory: bool,
    ):
        self.rows = rows
        self.tensor_config = tensor_config
        self.model_ifos = model_ifos
        self.q_values = q_values
        self.target_sample_rate = target_sample_rate
        self.verified_background_hashes: dict[str, str] = {}
        self.cache: list[tuple[np.ndarray, np.ndarray] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        context = load_materialized_context(self.rows[index], self.verified_background_hashes)
        source_rate = int(context["sample_rate"])
        if source_rate < self.target_sample_rate or source_rate % self.target_sample_rate:
            raise ValueError("Physical sample rate must be an integer multiple of target")
        ifos = [str(ifo) for ifo in context["ifos"]]
        source_start = int(context["analysis_start_index"])
        source_stop = int(context["analysis_stop_index"])
        duration = (source_stop - source_start) / source_rate
        output_samples = int(round(duration * self.target_sample_rate))
        target_start = int(
            round(
                (float(context["analysis_gps_start"]) - float(context["context_gps_start"]))
                * self.target_sample_rate
            )
        )
        mixture_planes = []
        signal_planes = []
        mixture = np.asarray(context["mixture"], dtype=np.float64)
        signal = np.asarray(context["signal"], dtype=np.float64)
        for ifo in self.model_ifos:
            if ifo not in ifos:
                mixture_planes.append(np.zeros(output_samples, dtype=np.float32))
                signal_planes.append(np.zeros(output_samples, dtype=np.float32))
                continue
            ifo_index = ifos.index(ifo)
            mixture_context = _fft_downsample(
                mixture[ifo_index], source_rate, self.target_sample_rate
            )
            signal_context = _fft_downsample(
                signal[ifo_index], source_rate, self.target_sample_rate
            )
            mixture_planes.append(
                _whiten(mixture_context)[target_start : target_start + output_samples]
            )
            signal_planes.append(signal_context[target_start : target_start + output_samples])
        mixture_array = np.stack(mixture_planes)
        signal_array = np.stack(signal_planes)
        settings = self.tensor_config
        feature_power = multiresolution_power(
            mixture_array,
            self.target_sample_rate,
            self.q_values,
            int(settings["frequency_bins"]),
            int(settings["time_bins"]),
            float(settings["fmin"]),
            float(settings["fmax"]),
        )
        signal_power = multiresolution_power(
            signal_array,
            self.target_sample_rate,
            self.q_values,
            int(settings["frequency_bins"]),
            int(settings["time_bins"]),
            float(settings["fmin"]),
            float(settings["fmax"]),
        )
        features = _normalize_power(feature_power).reshape(
            -1, feature_power.shape[-2], feature_power.shape[-1]
        )
        target = relative_component_mask(signal_power).reshape(
            -1, signal_power.shape[-2], signal_power.shape[-1]
        )
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("Physical tensor construction produced non-finite values")
        item = features.astype(np.float32), target.astype(np.float32)
        if self.cache is not None:
            self.cache[index] = item
        return item


def _chirp_epoch(
    model: Any,
    teacher: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    positive_weight: float,
    distillation_weight: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    teacher.eval()
    total_loss = 0.0
    true_positive = false_positive = false_negative = 0
    batches = 0
    positive = torch.as_tensor([positive_weight], device=device).reshape(1, 1, 1, 1)
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            chirp_logits = logits[:, 0:1]
            bce = torch_functional.binary_cross_entropy_with_logits(
                chirp_logits, target[:, None], pos_weight=positive
            )
            dice = _dice_loss(chirp_logits, target[:, None])
            with torch.no_grad():
                teacher_glitch = torch.sigmoid(teacher(features)[:, 1])
            distillation = torch_functional.binary_cross_entropy_with_logits(
                logits[:, 1], teacher_glitch
            )
            loss = bce + dice + distillation_weight * distillation
            if training:
                loss.backward()
                optimizer.step()
        predicted = torch.sigmoid(chirp_logits.detach()) >= threshold
        expected = target[:, None] >= 0.5
        true_positive += int((predicted & expected).sum().cpu())
        false_positive += int((predicted & ~expected).sum().cpu())
        false_negative += int((~predicted & expected).sum().cpu())
        total_loss += float(loss.detach().cpu())
        batches += 1
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "loss": total_loss / max(batches, 1),
        "chirp_iou": iou,
        "chirp_precision": true_positive / max(true_positive + false_positive, 1),
        "chirp_recall": true_positive / max(true_positive + false_negative, 1),
    }


def _calibrate_chirp_threshold(
    model: Any, loader: Any, device: Any, grid: tuple[float, ...]
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            probabilities.append(torch.sigmoid(model(features.to(device))[:, 0]).cpu().numpy())
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


def run_physical_finetune(
    config_path: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Physical fine-tuning requires torch")
    config = load_yaml(config_path)
    settings = deepcopy(config["physical_training"])
    seed = int(seed_override if seed_override is not None else settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with Path(train_manifest).open("r", encoding="utf-8") as handle:
        train_rows = [json.loads(line) for line in handle if line.strip()]
    with Path(validation_manifest).open("r", encoding="utf-8") as handle:
        validation_rows = [json.loads(line) for line in handle if line.strip()]
    audit = physical_split_audit(train_rows, validation_rows)
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    target_sample_rate = int(settings["target_sample_rate"])
    datasets = {
        "train": PhysicalInjectionDataset(
            train_rows,
            settings["tensor"],
            model_ifos,
            q_values,
            target_sample_rate,
            bool(settings.get("cache_in_memory", True)),
        ),
        "val": PhysicalInjectionDataset(
            validation_rows,
            settings["tensor"],
            model_ifos,
            q_values,
            target_sample_rate,
            bool(settings.get("cache_in_memory", True)),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != channels:
        raise ValueError("Pretrained checkpoint channel count differs from physical configuration")
    base_channels = int(checkpoint["base_channels"])
    model = MultiIFOQNet(channels, base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    teacher = MultiIFOQNet(channels, base_channels).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_physical_finetune.pt"
    history = []
    best_iou = -1.0
    best_epoch = None
    started = time.time()
    for epoch in range(1, int(settings["epochs"]) + 1):
        train_metrics = _chirp_epoch(
            model,
            teacher,
            loaders["train"],
            device,
            optimizer,
            float(settings["chirp_positive_weight"]),
            float(settings["glitch_distillation_weight"]),
        )
        validation_metrics = _chirp_epoch(
            model,
            teacher,
            loaders["val"],
            device,
            None,
            float(settings["chirp_positive_weight"]),
            float(settings["glitch_distillation_weight"]),
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        if float(validation_metrics["chirp_iou"]) > best_iou:
            best_iou = float(validation_metrics["chirp_iou"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_chirp_iou": best_iou,
                    "input_channels": channels,
                    "base_channels": base_channels,
                    "seed": seed,
                    "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
                    "train_manifest_sha256": file_sha256(train_manifest),
                    "validation_manifest_sha256": file_sha256(validation_manifest),
                },
            )
        atomic_write_json(output / "history.json", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    threshold, threshold_curve = _calibrate_chirp_threshold(
        model,
        loaders["val"],
        device,
        tuple(float(item) for item in settings["threshold_grid"]),
    )
    calibrated = _chirp_epoch(
        model,
        teacher,
        loaders["val"],
        device,
        None,
        float(settings["chirp_positive_weight"]),
        float(settings["glitch_distillation_weight"]),
        threshold,
    )
    report = {
        "status": "physical_real_noise_validation_only_finetune",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "waveform backend equivalence, real glitch labels, multi-seed evidence, fixed-FAR "
            "background and locked-test sensitivity remain required"
        ),
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
        "seed": seed,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": audit,
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "best_epoch": best_epoch,
        "best_validation_chirp_iou_precalibration": best_iou,
        "selected_chirp_threshold": threshold,
        "threshold_curve": threshold_curve,
        "calibrated_validation": calibrated,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "epochs": int(settings["epochs"]),
        "elapsed_seconds": time.time() - started,
        "test_evaluation": None,
    }
    atomic_write_json(output / "physical_finetune_report.json", report)
    return report
