from __future__ import annotations

import json
import os
import platform
import random
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .numeric import CoalescenceTimingNet, _atomic_torch_save
from .physical_training import PhysicalInjectionDataset, physical_split_audit

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None


def timing_errors_seconds(
    predicted_bins: np.ndarray,
    exact_offsets_seconds: np.ndarray,
    analysis_duration_seconds: float,
    time_bins: int,
) -> np.ndarray:
    """Return absolute errors using categorical bin centers and exact GPS offsets."""
    predicted = np.asarray(predicted_bins, dtype=np.int64).reshape(-1)
    offsets = np.asarray(exact_offsets_seconds, dtype=np.float64).reshape(-1)
    if predicted.shape != offsets.shape or predicted.size == 0:
        raise ValueError("timing predictions and exact offsets must be non-empty and aligned")
    if analysis_duration_seconds <= 0 or time_bins < 2:
        raise ValueError("timing duration and bins are invalid")
    if (
        np.any(predicted < 0)
        or np.any(predicted >= time_bins)
        or not np.isfinite(offsets).all()
        or np.any(offsets < 0)
        or np.any(offsets >= analysis_duration_seconds)
    ):
        raise ValueError("timing prediction or exact offset lies outside the analysis window")
    centers = (predicted.astype(np.float64) + 0.5) * analysis_duration_seconds / time_bins
    return np.abs(centers - offsets)


def _timing_summary(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("timing summary requires finite errors")
    within = int(np.count_nonzero(values <= 0.01))
    interval = wilson_interval(within, int(values.size))
    return {
        "injections": int(values.size),
        "mean_absolute_error_seconds": float(values.mean()),
        "absolute_error_seconds_quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "within_10ms": within,
        "within_10ms_fraction": within / values.size,
        "within_10ms_wilson_95": list(interval),
    }


def _timing_epoch(
    model: Any,
    loader: Any,
    device: Any,
    analysis_duration: float,
    time_bins: int,
    optimizer: Any | None,
    label_smoothing: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses = []
    errors: list[float] = []
    for features, _, target_bins, exact_offsets in loader:
        features = features.to(device)
        target_bins = target_bins.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = torch_functional.cross_entropy(
                logits, target_bins, label_smoothing=label_smoothing
            )
            if training:
                loss.backward()
                optimizer.step()
        predicted = torch.argmax(logits.detach(), dim=-1).cpu().numpy()
        batch_errors = timing_errors_seconds(
            predicted,
            exact_offsets.numpy(),
            analysis_duration,
            time_bins,
        )
        errors.extend(float(value) for value in batch_errors)
        losses.append(float(loss.detach().cpu()))
    summary = _timing_summary(errors)
    summary["loss"] = float(np.mean(losses))
    return summary


def _grouped_timing_evaluation(
    model: Any,
    loader: Any,
    rows: list[dict[str, Any]],
    device: Any,
    analysis_duration: float,
    time_bins: int,
) -> dict[str, Any]:
    model.eval()
    groups: dict[str, list[float]] = defaultdict(list)
    offset = 0
    with torch.no_grad():
        for features, _, _, exact_offsets in loader:
            predicted = torch.argmax(model(features.to(device)), dim=-1).cpu().numpy()
            errors = timing_errors_seconds(
                predicted,
                exact_offsets.numpy(),
                analysis_duration,
                time_bins,
            )
            for index, error in enumerate(errors):
                row = rows[offset + index]
                for key in (
                    "all",
                    f"family:{row['source_family']}",
                    f"snr:{row.get('optimal_snr_stratum', 'unassigned')}",
                ):
                    groups[key].append(float(error))
            offset += len(errors)
    if offset != len(rows):
        raise RuntimeError("timing evaluation did not consume the full validation manifest")
    return {key: _timing_summary(values) for key, values in sorted(groups.items())}


def run_physical_timing_training(
    config_path: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Physical timing training requires torch")
    config = load_yaml(config_path)
    settings = config["physical_timing"]
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
    report_path = output / "physical_timing_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_identity") != run_identity:
            raise ValueError("Completed physical timing output belongs to another run")
        return report
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
    split_audit = physical_split_audit(train_rows, validation_rows)
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    time_bins = int(settings["tensor"]["time_bins"])
    analysis_duration = float(settings["analysis_duration"])
    if time_bins < 2 or analysis_duration <= 0:
        raise ValueError("Physical timing grid is invalid")
    datasets = {
        "train": PhysicalInjectionDataset(
            train_rows,
            settings["tensor"],
            model_ifos,
            q_values,
            int(settings["target_sample_rate"]),
            bool(settings.get("cache_in_memory", True)),
            time_bins,
        ),
        "val": PhysicalInjectionDataset(
            validation_rows,
            settings["tensor"],
            model_ifos,
            q_values,
            int(settings["target_sample_rate"]),
            bool(settings.get("cache_in_memory", True)),
            time_bins,
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
        raise ValueError("Pretrained mask checkpoint differs from timing input channels")
    base_channels = int(pretrained["base_channels"])
    model = CoalescenceTimingNet(channels, base_channels).to(device)
    model.backbone.load_state_dict(pretrained["model"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_path = output / "best_physical_timing.pt"
    resume_path = output / "last_physical_timing.pt"
    history: list[dict[str, Any]] = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = None
    start_epoch = 1
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != run_identity:
            raise ValueError("Physical timing resume checkpoint belongs to another run")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_key = tuple(float(value) for value in resume["best_key"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_metrics = _timing_epoch(
            model,
            loaders["train"],
            device,
            analysis_duration,
            time_bins,
            optimizer,
            float(settings.get("label_smoothing", 0.0)),
        )
        validation_metrics = _timing_epoch(
            model,
            loaders["val"],
            device,
            analysis_duration,
            time_bins,
            None,
            0.0,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        quantiles = validation_metrics["absolute_error_seconds_quantiles"]
        selection_key = (
            float(quantiles["0.9"]),
            float(quantiles["0.5"]),
            float(validation_metrics["loss"]),
        )
        if selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "architecture": "coalescence_timing_net_v1",
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_selection_key": best_key,
                    "input_channels": channels,
                    "base_channels": base_channels,
                    "time_bins": time_bins,
                    "analysis_duration": analysis_duration,
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
                "best_key": best_key,
                "best_epoch": best_epoch,
            },
        )
        atomic_write_json(output / "history.json", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    grouped = _grouped_timing_evaluation(
        model,
        loaders["val"],
        validation_rows,
        device,
        analysis_duration,
        time_bins,
    )
    overall_p90 = float(grouped["all"]["absolute_error_seconds_quantiles"]["0.9"])
    report = {
        "status": "physical_validation_exact_gps_timing_refiner",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation timing is architecture-development evidence; multi-seed frozen-background "
            "candidate timing and locked-test evaluation remain required"
        ),
        "test_evaluation": None,
        "run_identity": run_identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": split_audit,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "validation p90 absolute exact-GPS timing error",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "timing": {
            "target": "exact geocentric injection GPS relative to analysis start",
            "time_bins": time_bins,
            "analysis_duration_seconds": analysis_duration,
            "bin_width_seconds": analysis_duration / time_bins,
            "representation_gate_passed": analysis_duration / time_bins <= 0.01,
            "accuracy_gate_passed": (
                analysis_duration / time_bins <= 0.01 and overall_p90 <= 0.01
            ),
        },
        "validation_groups": grouped,
        "epochs": int(settings["epochs"]),
        "history": history,
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
