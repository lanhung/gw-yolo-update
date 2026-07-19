from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .factory import synthesize_scene
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .provenance import SceneRecipe, read_recipe_manifest

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised in dependency-minimal installations
    torch = None
    nn = None
    torch_functional = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Numeric training requires the optional train dependencies, including torch")


class NumericRecipeDataset:
    def __init__(
        self,
        recipes: list[SceneRecipe],
        tensor_config: dict[str, Any],
        cache_in_memory: bool = False,
    ):
        self.recipes = recipes
        self.tensor_config = tensor_config
        self.cache: list[tuple[np.ndarray, np.ndarray] | None] | None = (
            [None] * len(recipes) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        arrays = synthesize_scene(self.recipes[index], self.tensor_config)
        features = arrays["features"].reshape(-1, *arrays["features"].shape[-2:])
        masks = np.stack([arrays["chirp_mask"], arrays["glitch_mask"]])
        masks = masks.reshape(2, -1, *masks.shape[-2:])
        item = features.astype(np.float32), masks.astype(np.float32)
        if self.cache is not None:
            self.cache[index] = item
        return item


if nn is not None:

    class _ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            groups = min(8, output_channels)
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)


    class MultiIFOQNet(nn.Module):
        """Compact numeric baseline retaining per-IFO/per-Q class masks."""

        def __init__(self, input_channels: int, base_channels: int = 24):
            super().__init__()
            self.input_channels = input_channels
            self.encoder = _ConvBlock(input_channels, base_channels)
            self.bottleneck = _ConvBlock(base_channels, base_channels * 2)
            self.decoder = _ConvBlock(base_channels * 3, base_channels)
            self.head = nn.Conv2d(base_channels, 2 * input_channels, 1)

        def forward(self, value: Any) -> Any:
            encoded = self.encoder(value)
            low = self.bottleneck(torch_functional.max_pool2d(encoded, 2))
            up = torch_functional.interpolate(low, size=encoded.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.decoder(torch.cat([encoded, up], dim=1))
            logits = self.head(decoded)
            return logits.reshape(value.shape[0], 2, self.input_channels, *value.shape[-2:])

else:

    class MultiIFOQNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()


def _dice_loss(logits: Any, targets: Any, class_weights: Any | None = None) -> Any:
    probabilities = torch.sigmoid(logits)
    axes = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
    losses = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    if class_weights is None:
        return losses.mean()
    weights = class_weights.reshape(1, -1)
    return (losses * weights).sum() / (weights.sum() * losses.shape[0])


def _batch_counts(
    logits: Any,
    targets: Any,
    thresholds: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    threshold_tensor = torch.as_tensor(thresholds, device=logits.device).reshape(1, 2, 1, 1, 1)
    predicted = torch.sigmoid(logits) >= threshold_tensor
    expected = targets >= 0.5
    axes = tuple(range(2, predicted.ndim))
    true_positive = (predicted & expected).sum(dim=axes)
    false_positive = (predicted & ~expected).sum(dim=axes)
    false_negative = (~predicted & expected).sum(dim=axes)
    return torch.stack([true_positive, false_positive, false_negative], dim=-1).sum(dim=0).cpu().numpy()


def _metrics_from_counts(counts: np.ndarray) -> dict[str, Any]:
    class_names = ("chirp", "glitch")
    result: dict[str, Any] = {}
    ious = []
    for index, name in enumerate(class_names):
        true_positive, false_positive, false_negative = (float(item) for item in counts[index])
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
        dice = 2.0 * true_positive / max(2.0 * true_positive + false_positive + false_negative, 1.0)
        result[name] = {"precision": precision, "recall": recall, "iou": iou, "dice": dice}
        ious.append(iou)
    result["mean_iou"] = float(np.mean(ious))
    return result


def _run_epoch(
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    positive_weights: tuple[float, float],
    dice_weights: tuple[float, float],
    thresholds: tuple[float, float] = (0.5, 0.5),
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    counts = np.zeros((2, 3), dtype=np.int64)
    positive_weight_tensor = torch.as_tensor(positive_weights, device=device).reshape(2, 1, 1, 1)
    dice_weight_tensor = torch.as_tensor(dice_weights, device=device)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            bce = torch_functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weight_tensor
            )
            loss = bce + _dice_loss(logits, targets, dice_weight_tensor)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        counts += _batch_counts(logits.detach(), targets, thresholds)
    return {
        "loss": total_loss / max(batches, 1),
        **_metrics_from_counts(counts),
    }


def _calibrate_thresholds(
    model: Any,
    loader: Any,
    device: Any,
    grid: tuple[float, ...],
) -> tuple[tuple[float, float], dict[str, Any]]:
    model.eval()
    logits_batches = []
    target_batches = []
    with torch.no_grad():
        for features, targets in loader:
            logits_batches.append(model(features.to(device)).cpu())
            target_batches.append(targets.cpu())
    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches)
    selected = []
    curves: dict[str, Any] = {"chirp": [], "glitch": []}
    for class_index, class_name in enumerate(("chirp", "glitch")):
        best_threshold = grid[0]
        best_iou = -1.0
        for threshold in grid:
            thresholds = (threshold, 1.0) if class_index == 0 else (1.0, threshold)
            counts = _batch_counts(logits, targets, thresholds)
            class_metrics = _metrics_from_counts(counts)[class_name]
            curves[class_name].append({"threshold": threshold, **class_metrics})
            if float(class_metrics["iou"]) > best_iou:
                best_iou = float(class_metrics["iou"])
                best_threshold = threshold
        selected.append(best_threshold)
    return (float(selected[0]), float(selected[1])), curves


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def train_numeric_model(
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    _require_torch()
    config = load_yaml(config_path)
    settings = config["numeric_training"]
    seed = int(settings.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    recipes = read_recipe_manifest(manifest_path)
    by_split = {split: [recipe for recipe in recipes if recipe.split == split] for split in ("train", "val", "test")}
    if any(not items for items in by_split.values()):
        raise ValueError(f"Manifest must contain non-empty train/val/test splits: { {key: len(value) for key, value in by_split.items()} }")
    tensor_config = settings["tensor"]
    batch_size = int(settings.get("batch_size", 8))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            NumericRecipeDataset(
                items,
                tensor_config,
                cache_in_memory=bool(settings.get("cache_in_memory", False)),
            ),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=int(settings.get("workers", 0)),
            generator=generator if split == "train" else None,
        )
        for split, items in by_split.items()
    }
    first_recipe = recipes[0]
    input_channels = len(first_recipe.ifos) * len(first_recipe.q_values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiIFOQNet(input_channels, int(settings.get("base_channels", 24))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_numeric.pt"
    history = []
    positive_weights = tuple(float(item) for item in settings.get("positive_weights", [1.0, 1.0]))
    dice_weights = tuple(float(item) for item in settings.get("dice_weights", [1.0, 1.0]))
    if len(positive_weights) != 2 or len(dice_weights) != 2:
        raise ValueError("positive_weights and dice_weights require [chirp, glitch]")
    best_metric = -1.0
    best_epoch = None
    started = time.time()
    for epoch in range(1, int(settings.get("epochs", 10)) + 1):
        train_metrics = _run_epoch(
            model, loaders["train"], device, optimizer, positive_weights, dice_weights
        )
        validation_metrics = _run_epoch(
            model, loaders["val"], device, None, positive_weights, dice_weights
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if float(validation_metrics["mean_iou"]) > best_metric:
            best_metric = float(validation_metrics["mean_iou"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_mean_iou": best_metric,
                    "input_channels": input_channels,
                    "base_channels": int(settings.get("base_channels", 24)),
                    "config_hash": canonical_hash(config),
                    "manifest_sha256": file_sha256(manifest_path),
                    "seed": seed,
                },
            )
        atomic_write_json(output / "history.json", history)

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    threshold_grid = tuple(float(item) for item in settings.get("threshold_grid", [0.5]))
    thresholds, threshold_curves = _calibrate_thresholds(
        model, loaders["val"], device, threshold_grid
    )
    calibrated_validation_metrics = _run_epoch(
        model,
        loaders["val"],
        device,
        None,
        positive_weights,
        dice_weights,
        thresholds,
    )
    evaluate_test = bool(settings.get("evaluate_test", False))
    test_metrics = None
    if evaluate_test:
        test_metrics = _run_epoch(
            model,
            loaders["test"],
            device,
            None,
            positive_weights,
            dice_weights,
            thresholds,
        )
    report = {
        "status": "synthetic_engineering_baseline",
        "scientific_claim_allowed": False,
        "device": str(device),
        "seed": seed,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "cache_in_memory": bool(settings.get("cache_in_memory", False)),
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_metric,
        "positive_weights": positive_weights,
        "dice_weights": dice_weights,
        "validation_selected_thresholds": {"chirp": thresholds[0], "glitch": thresholds[1]},
        "threshold_curves": threshold_curves,
        "calibrated_validation": calibrated_validation_metrics,
        "test_evaluated": evaluate_test,
        "test": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    atomic_write_json(output / "numeric_training_report.json", report)
    return report


def evaluate_numeric_checkpoint(
    config_path: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    thresholds: tuple[float, float],
    output_path: str | Path,
) -> dict[str, Any]:
    _require_torch()
    if split not in {"val", "test"}:
        raise ValueError("numeric evaluation split must be val or test")
    config = load_yaml(config_path)
    settings = config["numeric_training"]
    recipes = [recipe for recipe in read_recipe_manifest(manifest_path) if recipe.split == split]
    if not recipes:
        raise ValueError(f"No recipes found for split {split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiIFOQNet(
        int(checkpoint["input_channels"]), int(checkpoint["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        NumericRecipeDataset(recipes, settings["tensor"]),
        batch_size=int(settings.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(settings.get("workers", 0)),
    )
    positive_weights = tuple(float(item) for item in settings.get("positive_weights", [1.0, 1.0]))
    dice_weights = tuple(float(item) for item in settings.get("dice_weights", [1.0, 1.0]))
    metrics = _run_epoch(
        model,
        loader,
        device,
        None,
        positive_weights,
        dice_weights,
        thresholds,
    )
    report = {
        "status": "synthetic_engineering_baseline",
        "scientific_claim_allowed": False,
        "split": split,
        "scene_count": len(recipes),
        "thresholds": {"chirp": thresholds[0], "glitch": thresholds[1]},
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": file_sha256(manifest_path),
        "config_hash": canonical_hash(config),
    }
    atomic_write_json(output_path, report)
    return report
