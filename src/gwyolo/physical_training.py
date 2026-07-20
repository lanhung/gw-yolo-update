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
from .gwosc import _fft_downsample, _whiten, _whiten_with_reference
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
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


def scale_component_for_transform(component: np.ndarray) -> np.ndarray:
    """Scale each IFO before power construction so physical strain cannot underflow float32."""
    values = np.asarray(component, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("component must be finite [IFO, time]")
    peaks = np.max(np.abs(values), axis=-1, keepdims=True)
    return np.divide(values, peaks, out=np.zeros_like(values), where=peaks > 0)


def gate_component_by_ifo_snr(
    component: np.ndarray,
    ifos: list[str],
    snr_by_ifo: dict[str, float],
    minimum_ifo_snr: float,
    signal_scale: float = 1.0,
) -> np.ndarray:
    if minimum_ifo_snr < 0 or signal_scale <= 0:
        raise ValueError("IFO visibility SNR and signal scale are invalid")
    values = np.asarray(component, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != len(ifos):
        raise ValueError("component and IFO axes differ")
    missing = [ifo for ifo in ifos if ifo not in snr_by_ifo]
    if missing:
        raise ValueError(f"Missing per-IFO optimal SNR for {missing}")
    for index, ifo in enumerate(ifos):
        effective_snr = float(snr_by_ifo[ifo]) * signal_scale
        if not np.isfinite(effective_snr) or effective_snr < 0:
            raise ValueError(f"Invalid effective optimal SNR for {ifo}")
        if effective_snr < minimum_ifo_snr:
            values[index] = 0.0
    return values


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
        signal_scale = float(self.rows[index].get("training_signal_scale", 1.0))
        if not np.isfinite(signal_scale) or signal_scale <= 0:
            raise ValueError("training signal scale must be finite and positive")
        signal = np.asarray(context["signal"], dtype=np.float64) * signal_scale
        mixture = np.asarray(context["noise"], dtype=np.float64) + signal
        noise = np.asarray(context["noise"], dtype=np.float64)
        target_signal = signal
        minimum_ifo_mask_snr = self.tensor_config.get("minimum_ifo_mask_snr")
        if minimum_ifo_mask_snr is not None:
            if "optimal_snr_by_ifo" not in self.rows[index]:
                raise ValueError("Visibility-gated masks require per-IFO optimal SNR annotation")
            target_signal = gate_component_by_ifo_snr(
                signal,
                ifos,
                self.rows[index]["optimal_snr_by_ifo"],
                float(minimum_ifo_mask_snr),
                signal_scale=signal_scale,
            )
        for ifo in self.model_ifos:
            if ifo not in ifos:
                mixture_planes.append(np.zeros(output_samples, dtype=np.float32))
                signal_planes.append(np.zeros(output_samples, dtype=np.float32))
                continue
            ifo_index = ifos.index(ifo)
            noise_context = _fft_downsample(
                noise[ifo_index], source_rate, self.target_sample_rate
            )
            mixture_context = _fft_downsample(
                mixture[ifo_index], source_rate, self.target_sample_rate
            )
            signal_context = _fft_downsample(
                target_signal[ifo_index], source_rate, self.target_sample_rate
            )
            whitening = str(self.tensor_config.get("whitening", "self"))
            if whitening == "self":
                whitened_mixture = _whiten(mixture_context)
            elif whitening == "noise_reference":
                whitened_mixture = _whiten_with_reference(noise_context, mixture_context)
            else:
                raise ValueError("physical whitening must be self or noise_reference")
            mixture_planes.append(
                whitened_mixture[target_start : target_start + output_samples]
            )
            target_whitening = str(
                self.tensor_config.get("target_whitening", "morphology")
            )
            if target_whitening == "morphology":
                transformed_signal = signal_context
            elif target_whitening == "noise_reference":
                transformed_signal = _whiten_with_reference(
                    noise_context, signal_context, component=True
                )
            else:
                raise ValueError(
                    "physical target whitening must be morphology or noise_reference"
                )
            signal_planes.append(
                transformed_signal[target_start : target_start + output_samples]
            )
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
            scale_component_for_transform(signal_array),
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
        target = relative_component_mask(
            signal_power, float(settings.get("mask_fraction", 0.08))
        ).reshape(
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
    focal_gamma: float = 0.0,
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
            bce = focal_binary_cross_entropy(
                chirp_logits, target[:, None], positive, focal_gamma
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


def focal_binary_cross_entropy(
    logits: Any, target: Any, positive_weight: Any, gamma: float
) -> Any:
    if gamma < 0:
        raise ValueError("focal gamma cannot be negative")
    raw = torch_functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight, reduction="none"
    )
    if gamma == 0:
        return raw.mean()
    probability = torch.sigmoid(logits)
    correct_probability = probability * target + (1.0 - probability) * (1.0 - target)
    return (((1.0 - correct_probability) ** gamma) * raw).mean()


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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "config_hash": canonical_hash(config),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "seed": seed,
    }
    completed_report_path = output / "physical_finetune_report.json"
    if completed_report_path.is_file():
        with completed_report_path.open("r", encoding="utf-8") as handle:
            completed_report = json.load(handle)
        if completed_report.get("run_identity") != run_identity:
            raise ValueError("Completed physical fine-tune output belongs to a different run")
        return completed_report
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with Path(train_manifest).open("r", encoding="utf-8") as handle:
        all_train_rows = [json.loads(line) for line in handle if line.strip()]
    with Path(validation_manifest).open("r", encoding="utf-8") as handle:
        validation_rows = [json.loads(line) for line in handle if line.strip()]
    minimum_training_snr = settings.get("minimum_training_network_snr")
    if minimum_training_snr is not None:
        missing_snr = [
            row["injection_id"]
            for row in all_train_rows
            if "network_optimal_snr" not in row
            and "training_network_optimal_snr" not in row
        ]
        if missing_snr:
            raise ValueError(
                "SNR-filtered physical training requires an annotated manifest; missing "
                f"{missing_snr[:10]}"
            )
        train_rows = [
            row
            for row in all_train_rows
            if float(
                row["training_network_optimal_snr"]
                if "training_network_optimal_snr" in row
                else row["network_optimal_snr"]
            )
            >= float(minimum_training_snr)
        ]
    else:
        train_rows = all_train_rows
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
    pretrained = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    channels = len(model_ifos) * len(q_values)
    if int(pretrained["input_channels"]) != channels:
        raise ValueError("Pretrained checkpoint channel count differs from physical configuration")
    base_channels = int(pretrained["base_channels"])
    model = MultiIFOQNet(channels, base_channels).to(device)
    model.load_state_dict(pretrained["model"])
    teacher = MultiIFOQNet(channels, base_channels).to(device)
    teacher.load_state_dict(pretrained["model"])
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_path = output / "best_physical_finetune.pt"
    resume_path = output / "last_physical_finetune.pt"
    history = []
    best_iou = -1.0
    best_epoch = None
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != run_identity:
            raise ValueError("Physical fine-tune resume checkpoint belongs to a different run")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_iou = float(resume["best_validation_chirp_iou"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_metrics = _chirp_epoch(
            model,
            teacher,
            loaders["train"],
            device,
            optimizer,
            float(settings["chirp_positive_weight"]),
            float(settings["glitch_distillation_weight"]),
            float(settings.get("focal_gamma", 0.0)),
        )
        validation_metrics = _chirp_epoch(
            model,
            teacher,
            loaders["val"],
            device,
            None,
            float(settings["chirp_positive_weight"]),
            float(settings["glitch_distillation_weight"]),
            float(settings.get("focal_gamma", 0.0)),
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
                    **run_identity,
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
                "best_validation_chirp_iou": best_iou,
                "best_epoch": best_epoch,
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
        float(settings.get("focal_gamma", 0.0)),
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
        "run_identity": run_identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": audit,
        "training_selection": {
            "input_rows": len(all_train_rows),
            "selected_rows": len(train_rows),
            "excluded_rows": len(all_train_rows) - len(train_rows),
            "minimum_network_optimal_snr": minimum_training_snr,
        },
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "best_epoch": best_epoch,
        "resumed_from_epoch": start_epoch - 1,
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
    atomic_write_json(completed_report_path, report)
    return report


def build_snr_curriculum_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    minimum_snr: float = 4.0,
    rescale_upper_snr: float = 8.0,
    seed: int = 20260720,
) -> dict[str, Any]:
    if minimum_snr <= 0 or rescale_upper_snr <= minimum_snr:
        raise ValueError("SNR curriculum bounds are invalid")
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("SNR curriculum accepts a non-empty train-only manifest")
    if any("network_optimal_snr" not in row for row in rows):
        raise ValueError("SNR curriculum requires an optimal-SNR annotated manifest")
    curriculum = []
    rescaled = 0
    for row in rows:
        original_snr = float(row["network_optimal_snr"])
        if not np.isfinite(original_snr) or original_snr <= 0:
            raise ValueError(f"Invalid network optimal SNR for {row['injection_id']}")
        if original_snr < minimum_snr:
            uniform = int(
                canonical_hash(f"{seed}:{row['injection_id']}:snr-curriculum", 16), 16
            ) / 16**16
            target_snr = minimum_snr * (rescale_upper_snr / minimum_snr) ** uniform
            signal_scale = target_snr / original_snr
            rescaled += 1
        else:
            target_snr = original_snr
            signal_scale = 1.0
        curriculum.append(
            {
                **row,
                "training_original_network_optimal_snr": original_snr,
                "training_network_optimal_snr": target_snr,
                "training_signal_scale": signal_scale,
                "training_snr_curriculum": (
                    "below-floor signals deterministically log-uniformly rescaled into floor band"
                ),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "physical_train_snr_curriculum.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in curriculum)
    )
    report = {
        "status": "train_only_snr_curriculum",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "rescaling improves training coverage but does not add independent waveforms or "
            "background; validation/test populations and VT weights must remain untouched"
        ),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "manifest_path": str(target),
        "manifest_sha256": file_sha256(target),
        "seed": seed,
        "minimum_snr": minimum_snr,
        "rescale_upper_snr": rescale_upper_snr,
        "rows": len(curriculum),
        "rescaled_rows": rescaled,
        "unchanged_rows": len(curriculum) - rescaled,
        "unique_injection_ids": len({row["injection_id"] for row in curriculum}),
        "unique_waveform_ids": len({row["waveform_id"] for row in curriculum}),
        "unique_gps_blocks": len({row["gps_block"] for row in curriculum}),
        "target_snr_quantiles": {
            str(q): float(
                np.quantile([row["training_network_optimal_snr"] for row in curriculum], q)
            )
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    atomic_write_json(output / "snr_curriculum_report.json", report)
    return report


def summarize_binary_mask_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    if min(tp, fp, fn) < 0:
        raise ValueError("mask counts cannot be negative")
    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def audit_physical_checkpoint(
    config_path: str | Path,
    validation_manifest: str | Path,
    checkpoint_path: str | Path,
    chirp_threshold: float,
    output_path: str | Path,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Physical checkpoint audit requires torch")
    if not 0 < chirp_threshold < 1:
        raise ValueError("chirp audit threshold must be between zero and one")
    config = load_yaml(config_path)
    settings = config["physical_training"]
    with Path(validation_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split") != "val" for row in rows):
        raise ValueError("Physical checkpoint audit accepts a non-empty validation-only manifest")
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    dataset = PhysicalInjectionDataset(
        rows,
        settings["tensor"],
        model_ifos,
        q_values,
        int(settings["target_sample_rate"]),
        False,
    )
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != channels:
        raise ValueError("Checkpoint channel count differs from physical audit configuration")
    model = MultiIFOQNet(channels, int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    groups: dict[str, dict[str, Any]] = {}

    def accumulator(key: str) -> dict[str, Any]:
        return groups.setdefault(
            key,
            {"injections": 0, "pixels": 0, "target_pixels": 0, "tp": 0, "fp": 0, "fn": 0, "max_probabilities": []},
        )

    offset = 0
    started = time.time()
    with torch.no_grad():
        for features, target in loader:
            probabilities = torch.sigmoid(model(features.to(device))[:, 0]).cpu().numpy()
            expected = target.numpy() >= 0.5
            for index in range(features.shape[0]):
                row = rows[offset + index]
                predicted = probabilities[index] >= chirp_threshold
                tp = int(np.count_nonzero(predicted & expected[index]))
                fp = int(np.count_nonzero(predicted & ~expected[index]))
                fn = int(np.count_nonzero(~predicted & expected[index]))
                keys = (
                    "all",
                    f"family:{row['source_family']}",
                    f"snr:{row.get('optimal_snr_stratum', 'unassigned')}",
                )
                for key in keys:
                    item = accumulator(key)
                    item["injections"] += 1
                    item["pixels"] += int(expected[index].size)
                    item["target_pixels"] += int(np.count_nonzero(expected[index]))
                    item["tp"] += tp
                    item["fp"] += fp
                    item["fn"] += fn
                    item["max_probabilities"].append(float(np.max(probabilities[index])))
            offset += features.shape[0]
    summaries = {}
    for key, item in sorted(groups.items()):
        maximums = item.pop("max_probabilities")
        counts = summarize_binary_mask_counts(item.pop("tp"), item.pop("fp"), item.pop("fn"))
        summaries[key] = {
            **item,
            "target_pixel_fraction": item["target_pixels"] / max(item["pixels"], 1),
            **counts,
            "maximum_probability_quantiles": {
                str(q): float(np.quantile(maximums, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)
            },
        }
    report = {
        "status": "physical_validation_checkpoint_audit",
        "scientific_claim_allowed": False,
        "test_evaluation": None,
        "protocol": "frozen threshold applied to validation-only physical injections",
        "chirp_threshold": chirp_threshold,
        "validation_manifest_path": str(validation_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "device": str(device),
        "groups": summaries,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_path, report)
    return report
