from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .gwosc import _fft_downsample, _whiten
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .numeric import (
    DetectorArrivalTimingContextNet,
    DetectorArrivalTimingNet,
    _atomic_torch_save,
)
from .physical_training import physical_split_audit
from .runtime import execution_provenance
from .waveforms import load_materialized_context

try:
    import torch
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    torch_functional = None
    DataLoader = None


def _build_detector_arrival_model(
    architecture: str, detector_count: int, base_channels: int
) -> Any:
    if architecture == "detector_arrival_timing_net_v1":
        return DetectorArrivalTimingNet(detector_count, base_channels)
    if architecture == "detector_arrival_timing_context_net_v2":
        return DetectorArrivalTimingContextNet(detector_count, base_channels)
    raise ValueError(f"unsupported detector arrival timing architecture: {architecture}")


def detector_arrival_receptive_field_samples(architecture: str) -> int:
    """Return the analytically derived raw-sample receptive field of a timing logit."""

    if architecture == "detector_arrival_timing_net_v1":
        return 129
    if architecture == "detector_arrival_timing_context_net_v2":
        return 8257
    raise ValueError(f"unsupported detector arrival timing architecture: {architecture}")


def detector_arrival_bin_targets(
    detector_arrivals_gps: dict[str, float],
    model_ifos: tuple[str, ...],
    analysis_start_gps: float,
    analysis_duration_seconds: float,
    output_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map exact per-detector arrivals to categorical bins without zero-fill ambiguity."""

    if analysis_duration_seconds <= 0 or output_bins < 2:
        raise ValueError("detector arrival timing grid is invalid")
    targets = np.full(len(model_ifos), -1, dtype=np.int64)
    offsets = np.full(len(model_ifos), np.nan, dtype=np.float64)
    availability = np.zeros(len(model_ifos), dtype=bool)
    for index, ifo in enumerate(model_ifos):
        if ifo not in detector_arrivals_gps:
            continue
        offset = float(detector_arrivals_gps[ifo]) - float(analysis_start_gps)
        if not np.isfinite(offset) or not 0 <= offset < analysis_duration_seconds:
            raise ValueError(f"detector arrival for {ifo} lies outside the analysis window")
        targets[index] = min(
            int(np.floor(offset * output_bins / analysis_duration_seconds)),
            output_bins - 1,
        )
        offsets[index] = offset
        availability[index] = True
    if np.count_nonzero(availability) < 2:
        raise ValueError("detector arrival timing requires at least two available IFOs")
    return targets, offsets, availability


def detector_arrival_errors_seconds(
    predicted_bins: np.ndarray,
    exact_offsets_seconds: np.ndarray,
    availability: np.ndarray,
    analysis_duration_seconds: float,
    output_bins: int,
) -> np.ndarray:
    predicted = np.asarray(predicted_bins, dtype=np.int64)
    offsets = np.asarray(exact_offsets_seconds, dtype=np.float64)
    valid = np.asarray(availability, dtype=bool)
    if predicted.shape != offsets.shape or valid.shape != offsets.shape:
        raise ValueError("detector arrival predictions, offsets and availability must align")
    if not np.any(valid) or analysis_duration_seconds <= 0 or output_bins < 2:
        raise ValueError("detector arrival error inputs are invalid")
    if (
        np.any(predicted[valid] < 0)
        or np.any(predicted[valid] >= output_bins)
        or not np.isfinite(offsets[valid]).all()
    ):
        raise ValueError("available detector arrival prediction is invalid")
    centers = (
        predicted[valid].astype(np.float64) + 0.5
    ) * analysis_duration_seconds / output_bins
    return np.abs(centers - offsets[valid])


def detector_network_arrival_errors_seconds(
    predicted_bins: np.ndarray,
    exact_offsets_seconds: np.ndarray,
    availability: np.ndarray,
    analysis_duration_seconds: float,
    output_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-example worst-IFO and pairwise relative-delay errors."""

    predicted = np.asarray(predicted_bins, dtype=np.int64)
    offsets = np.asarray(exact_offsets_seconds, dtype=np.float64)
    valid = np.asarray(availability, dtype=bool)
    if predicted.ndim != 2 or predicted.shape != offsets.shape or valid.shape != offsets.shape:
        raise ValueError("network arrival timing arrays must share shape [example, IFO]")
    if analysis_duration_seconds <= 0 or output_bins < 2:
        raise ValueError("network arrival timing grid is invalid")
    centers = (
        predicted.astype(np.float64) + 0.5
    ) * analysis_duration_seconds / output_bins
    maximum_errors = []
    pairwise_delay_errors = []
    for index in range(predicted.shape[0]):
        indices = np.flatnonzero(valid[index])
        if indices.size < 2:
            raise ValueError("network arrival timing requires two available IFOs per example")
        if (
            np.any(predicted[index, indices] < 0)
            or np.any(predicted[index, indices] >= output_bins)
            or not np.isfinite(offsets[index, indices]).all()
        ):
            raise ValueError("available network arrival prediction is invalid")
        maximum_errors.append(
            float(np.max(np.abs(centers[index, indices] - offsets[index, indices])))
        )
        for left_offset, left in enumerate(indices[:-1]):
            for right in indices[left_offset + 1 :]:
                predicted_delay = centers[index, left] - centers[index, right]
                exact_delay = offsets[index, left] - offsets[index, right]
                pairwise_delay_errors.append(abs(predicted_delay - exact_delay))
    return (
        np.asarray(maximum_errors, dtype=np.float64),
        np.asarray(pairwise_delay_errors, dtype=np.float64),
    )


def _summary(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("arrival timing summary requires finite errors")
    within = int(np.count_nonzero(values <= 0.01))
    return {
        "detector_arrivals": int(values.size),
        "mean_absolute_error_seconds": float(values.mean()),
        "absolute_error_seconds_quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "within_10ms": within,
        "within_10ms_fraction": within / values.size,
        "within_10ms_wilson_95": list(wilson_interval(within, int(values.size))),
    }


def _network_summary(
    maximum_errors: list[float], pairwise_delay_errors: list[float]
) -> dict[str, Any]:
    maximum = np.asarray(maximum_errors, dtype=np.float64)
    pairwise = np.asarray(pairwise_delay_errors, dtype=np.float64)
    if maximum.size == 0 or pairwise.size == 0:
        raise ValueError("network timing summary requires examples and detector pairs")
    if not np.isfinite(maximum).all() or not np.isfinite(pairwise).all():
        raise ValueError("network timing summary requires finite errors")
    within = int(np.count_nonzero(maximum <= 0.01))
    quantiles = (0.0, 0.5, 0.9, 0.99, 1.0)
    return {
        "network_examples": int(maximum.size),
        "pairwise_delays": int(pairwise.size),
        "all_available_ifos_within_10ms": within,
        "all_available_ifos_within_10ms_fraction": within / maximum.size,
        "all_available_ifos_within_10ms_wilson_95": list(
            wilson_interval(within, int(maximum.size))
        ),
        "maximum_ifo_absolute_error_seconds_quantiles": {
            str(q): float(np.quantile(maximum, q)) for q in quantiles
        },
        "pairwise_delay_absolute_error_seconds_quantiles": {
            str(q): float(np.quantile(pairwise, q)) for q in quantiles
        },
    }


class DetectorArrivalDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model_ifos: tuple[str, ...],
        target_sample_rate: int,
        analysis_duration_seconds: float,
        output_bins: int,
        cache_in_memory: bool,
    ):
        self.rows = rows
        self.model_ifos = model_ifos
        self.target_sample_rate = target_sample_rate
        self.analysis_duration_seconds = analysis_duration_seconds
        self.output_bins = output_bins
        self.cache: list[tuple[np.ndarray, ...] | None] | None = (
            [None] * len(rows) if cache_in_memory else None
        )
        self.verified_background_hashes: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        row = self.rows[index]
        context = load_materialized_context(row, self.verified_background_hashes)
        source_rate = int(context["sample_rate"])
        if source_rate < self.target_sample_rate or source_rate % self.target_sample_rate:
            raise ValueError("arrival timing sample rates must divide exactly")
        source_start = int(context["analysis_start_index"])
        source_stop = int(context["analysis_stop_index"])
        duration = (source_stop - source_start) / source_rate
        if not np.isclose(duration, self.analysis_duration_seconds, rtol=0, atol=1e-9):
            raise ValueError("arrival timing analysis duration differs from configuration")
        target_samples = int(round(duration * self.target_sample_rate))
        target_start = int(
            round(
                (float(context["analysis_gps_start"]) - float(context["context_gps_start"]))
                * self.target_sample_rate
            )
        )
        source_ifos = [str(value) for value in context["ifos"]]
        mixture = np.asarray(context["noise"], dtype=np.float64) + np.asarray(
            context["signal"], dtype=np.float64
        ) * float(row.get("training_signal_scale", 1.0))
        strain = []
        present_arrivals = {
            str(ifo): float(value)
            for ifo, value in row.get("detector_arrival_gps", {}).items()
            if str(ifo) in source_ifos
        }
        targets, offsets, availability = detector_arrival_bin_targets(
            present_arrivals,
            self.model_ifos,
            float(context["analysis_gps_start"]),
            duration,
            self.output_bins,
        )
        for ifo in self.model_ifos:
            if ifo not in source_ifos:
                strain.append(np.zeros(target_samples, dtype=np.float32))
                continue
            values = _fft_downsample(
                mixture[source_ifos.index(ifo)], source_rate, self.target_sample_rate
            )
            whitened = _whiten(values)
            strain.append(whitened[target_start : target_start + target_samples])
        item = (
            np.stack(strain).astype(np.float32),
            availability.astype(bool),
            targets,
            offsets,
        )
        if self.cache is not None:
            self.cache[index] = item
        return item


def _epoch(
    model: Any,
    loader: Any,
    device: Any,
    duration: float,
    output_bins: int,
    optimizer: Any | None,
    label_smoothing: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses = []
    errors = []
    examples = 0
    batches = 0
    if max_batches is not None and max_batches <= 0:
        raise ValueError("arrival timing max batches must be positive")
    for strain, availability, targets, offsets in loader:
        strain = strain.to(device)
        availability = availability.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(strain, availability)
            selected_logits = logits[availability]
            selected_targets = targets[availability]
            loss = torch_functional.cross_entropy(
                selected_logits, selected_targets, label_smoothing=label_smoothing
            )
            if training:
                loss.backward()
                optimizer.step()
        predicted = torch.argmax(logits.detach(), dim=-1).cpu().numpy()
        errors.extend(
            detector_arrival_errors_seconds(
                predicted,
                offsets.numpy(),
                availability.cpu().numpy(),
                duration,
                output_bins,
            ).tolist()
        )
        losses.append(float(loss.detach().cpu()))
        examples += int(strain.shape[0])
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    result = _summary(errors)
    result.update(
        {"loss": float(np.mean(losses)), "examples": examples, "batches": batches}
    )
    return result


def _grouped_evaluation(
    model: Any,
    loader: Any,
    rows: list[dict[str, Any]],
    model_ifos: tuple[str, ...],
    device: Any,
    duration: float,
    output_bins: int,
    ifo_snr_thresholds: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    groups: dict[str, list[float]] = defaultdict(list)
    network_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"maximum_errors": [], "pairwise_delay_errors": []}
    )
    prediction_rows = []
    offset = 0
    with torch.no_grad():
        for strain, availability, _, exact_offsets in loader:
            predicted = torch.argmax(
                model(strain.to(device), availability.to(device)), dim=-1
            ).cpu().numpy()
            available_values = availability.numpy().astype(bool)
            exact_values = exact_offsets.numpy()
            for batch_index in range(strain.shape[0]):
                row = rows[offset + batch_index]
                row_snr = {
                    str(ifo): float(value)
                    for ifo, value in row.get("optimal_snr_by_ifo", {}).items()
                }
                maximum_errors, pairwise_errors = detector_network_arrival_errors_seconds(
                    predicted[batch_index : batch_index + 1],
                    exact_values[batch_index : batch_index + 1],
                    available_values[batch_index : batch_index + 1],
                    duration,
                    output_bins,
                )
                network_keys = ["all"]
                available_ifos = [
                    ifo
                    for ifo_index, ifo in enumerate(model_ifos)
                    if available_values[batch_index, ifo_index]
                ]
                predicted_offsets = (
                    predicted[batch_index].astype(np.float64) + 0.5
                ) * duration / output_bins
                detector_predictions = {}
                for ifo_index, ifo in enumerate(model_ifos):
                    if not available_values[batch_index, ifo_index]:
                        continue
                    detector_predictions[ifo] = {
                        "predicted_bin": int(predicted[batch_index, ifo_index]),
                        "predicted_offset_seconds": float(predicted_offsets[ifo_index]),
                        "exact_offset_seconds": float(exact_values[batch_index, ifo_index]),
                        "absolute_error_seconds": float(
                            abs(
                                predicted_offsets[ifo_index]
                                - exact_values[batch_index, ifo_index]
                            )
                        ),
                        "optimal_snr": row_snr.get(ifo),
                    }
                prediction_rows.append(
                    {
                        "row_index": offset + batch_index,
                        "split": str(row["split"]),
                        "injection_id": str(row["injection_id"]),
                        "waveform_id": str(row["waveform_id"]),
                        "background_window_id": str(row["background_window_id"]),
                        "source_family": str(row["source_family"]),
                        "network_optimal_snr": float(row["network_optimal_snr"]),
                        "optimal_snr_stratum": str(row["optimal_snr_stratum"]),
                        "minimum_available_ifo_optimal_snr": (
                            min(row_snr[ifo] for ifo in available_ifos)
                            if row_snr and all(ifo in row_snr for ifo in available_ifos)
                            else None
                        ),
                        "detector_predictions": detector_predictions,
                        "maximum_ifo_absolute_error_seconds": float(maximum_errors[0]),
                        "maximum_pairwise_delay_absolute_error_seconds": float(
                            np.max(pairwise_errors)
                        ),
                    }
                )
                if row_snr and all(ifo in row_snr for ifo in available_ifos):
                    minimum_ifo_snr = min(row_snr[ifo] for ifo in available_ifos)
                    network_keys.extend(
                        f"minimum_ifo_snr_ge_{threshold:g}"
                        for threshold in ifo_snr_thresholds
                        if minimum_ifo_snr >= threshold
                    )
                for key in network_keys:
                    network_groups[key]["maximum_errors"].extend(
                        maximum_errors.tolist()
                    )
                    network_groups[key]["pairwise_delay_errors"].extend(
                        pairwise_errors.tolist()
                    )
                for ifo_index, ifo in enumerate(model_ifos):
                    if not available_values[batch_index, ifo_index]:
                        continue
                    error = float(
                        detector_arrival_errors_seconds(
                            predicted[batch_index, ifo_index : ifo_index + 1],
                            exact_values[batch_index, ifo_index : ifo_index + 1],
                            np.ones(1, dtype=bool),
                            duration,
                            output_bins,
                        )[0]
                    )
                    for key in (
                        "all",
                        f"ifo:{ifo}",
                        f"family:{row['source_family']}",
                        f"snr:{row.get('optimal_snr_stratum', 'unassigned')}",
                    ):
                        groups[key].append(error)
                    if ifo in row_snr:
                        for threshold in ifo_snr_thresholds:
                            if row_snr[ifo] >= threshold:
                                groups[f"ifo_snr_ge_{threshold:g}"].append(error)
            offset += int(strain.shape[0])
    if offset != len(rows):
        raise RuntimeError("arrival timing evaluation did not consume every validation row")
    return (
        {key: _summary(values) for key, values in sorted(groups.items())},
        {
            key: _network_summary(
                values["maximum_errors"], values["pairwise_delay_errors"]
            )
            for key, values in sorted(network_groups.items())
        },
        prediction_rows,
    )


def run_detector_arrival_timing_training(
    config_path: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Detector arrival timing training requires torch")
    config = load_yaml(config_path)
    settings = config["detector_arrival_timing"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "seed": seed,
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "detector_arrival_timing_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_identity") != identity:
            raise ValueError("completed detector arrival timing run has another identity")
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
    minimum_snr = settings.get("minimum_training_network_snr")
    if minimum_snr is not None:
        train_rows = [
            row
            for row in train_rows
            if float(
                row["training_network_optimal_snr"]
                if "training_network_optimal_snr" in row
                else row["network_optimal_snr"]
            )
            >= float(minimum_snr)
        ]
    split_audit = physical_split_audit(train_rows, validation_rows)
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    target_rate = int(settings["target_sample_rate"])
    duration = float(settings["analysis_duration"])
    output_bins = int(settings["output_bins"])
    if int(round(duration * target_rate)) // output_bins != 8:
        raise ValueError("arrival timing v1 requires exactly eight input samples per output bin")
    datasets = {
        "train": DetectorArrivalDataset(
            train_rows,
            model_ifos,
            target_rate,
            duration,
            output_bins,
            bool(settings.get("cache_in_memory", True)),
        ),
        "val": DetectorArrivalDataset(
            validation_rows,
            model_ifos,
            target_rate,
            duration,
            output_bins,
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
    architecture = str(
        settings.get("architecture", "detector_arrival_timing_net_v1")
    )
    model = _build_detector_arrival_model(
        architecture, len(model_ifos), int(settings.get("base_channels", 32))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_path = output / "best_detector_arrival_timing.pt"
    resume_path = output / "last_detector_arrival_timing.pt"
    history = []
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = None
    start_epoch = 1
    optimizer_updates = 0
    optimizer_examples = 0
    steps_per_full_epoch = len(loaders["train"])
    max_optimizer_updates = settings.get("max_optimizer_updates")
    if max_optimizer_updates is not None:
        max_optimizer_updates = int(max_optimizer_updates)
        maximum_possible_updates = int(settings["epochs"]) * steps_per_full_epoch
        if max_optimizer_updates <= 0 or max_optimizer_updates > maximum_possible_updates:
            raise ValueError(
                "arrival timing max optimizer updates must be positive and no larger "
                f"than epochs * steps_per_full_epoch ({maximum_possible_updates})"
            )
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != identity:
            raise ValueError("detector arrival timing resume identity differs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_key = tuple(float(value) for value in resume["best_key"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
        optimizer_updates = int(resume.get("optimizer_updates", 0))
        optimizer_examples = int(resume.get("optimizer_examples", 0))
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        remaining_updates = (
            None
            if max_optimizer_updates is None
            else max_optimizer_updates - optimizer_updates
        )
        if remaining_updates is not None and remaining_updates <= 0:
            break
        train_metrics = _epoch(
            model,
            loaders["train"],
            device,
            duration,
            output_bins,
            optimizer,
            float(settings.get("label_smoothing", 0.0)),
            max_batches=(
                None
                if remaining_updates is None
                else min(remaining_updates, steps_per_full_epoch)
            ),
        )
        optimizer_updates += int(train_metrics["batches"])
        optimizer_examples += int(train_metrics["examples"])
        validation_metrics = _epoch(
            model, loaders["val"], device, duration, output_bins, None, 0.0
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        quantiles = validation_metrics["absolute_error_seconds_quantiles"]
        key = (float(quantiles["0.9"]), float(quantiles["0.5"]), validation_metrics["loss"])
        if key < best_key:
            best_key = key
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "architecture": architecture,
                    "model": model.state_dict(),
                    "model_ifos": model_ifos,
                    "target_sample_rate": target_rate,
                    "analysis_duration": duration,
                    "output_bins": output_bins,
                    "base_channels": int(settings.get("base_channels", 32)),
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
                "optimizer_updates": optimizer_updates,
                "optimizer_examples": optimizer_examples,
            },
        )
        atomic_write_json(output / "history.json", history)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    ifo_snr_thresholds = tuple(
        float(value)
        for value in settings.get(
            "validation_ifo_snr_thresholds", (4.0, 6.0, 8.0, 10.0)
        )
    )
    if (
        not ifo_snr_thresholds
        or any(value <= 0 for value in ifo_snr_thresholds)
        or tuple(sorted(set(ifo_snr_thresholds))) != ifo_snr_thresholds
    ):
        raise ValueError("validation IFO SNR thresholds must be positive and increasing")
    groups, network_groups, _ = _grouped_evaluation(
        model,
        loaders["val"],
        validation_rows,
        model_ifos,
        device,
        duration,
        output_bins,
        ifo_snr_thresholds,
    )
    bin_width = duration / output_bins
    p90 = float(groups["all"]["absolute_error_seconds_quantiles"]["0.9"])
    result = {
        "status": "validation_only_detector_arrival_timing_head",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "requires candidate integration, fixed-update/fixed-epoch controls, multi-seed "
            "validation and independent locked-test evaluation"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": split_audit,
        "model_ifos": list(model_ifos),
        "architecture": architecture,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "receptive_field_samples": detector_arrival_receptive_field_samples(
            architecture
        ),
        "receptive_field_seconds": detector_arrival_receptive_field_samples(
            architecture
        )
        / target_rate,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "validation p90 per-detector absolute arrival-time error",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "timing": {
            "target": "exact geometric detector arrival GPS relative to analysis start",
            "analysis_duration_seconds": duration,
            "target_sample_rate": target_rate,
            "output_bins": output_bins,
            "bin_width_seconds": bin_width,
            "representation_gate_passed": bin_width <= 0.01,
            "accuracy_gate_passed": bin_width <= 0.01 and p90 <= 0.01,
        },
        "validation_groups": groups,
        "validation_network_groups": network_groups,
        "validation_ifo_snr_thresholds": list(ifo_snr_thresholds),
        "epochs": int(settings["epochs"]),
        "completed_epochs": len(history),
        "steps_per_full_epoch": steps_per_full_epoch,
        "max_optimizer_updates": max_optimizer_updates,
        "optimizer_updates": optimizer_updates,
        "optimizer_examples": optimizer_examples,
        "training_budget_reached": (
            max_optimizer_updates is not None and optimizer_updates == max_optimizer_updates
        ),
        "history": history,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result


def run_detector_arrival_timing_validation_stratification(
    config_path: str | Path,
    validation_manifest: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    predictions_output_path: str | Path,
) -> dict[str, Any]:
    """Stratify a frozen timing checkpoint without redefining its all-sample gate."""

    if torch is None:
        raise RuntimeError("Detector arrival timing validation requires torch")
    config = load_yaml(config_path)
    settings = config["detector_arrival_timing"]
    with Path(validation_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or {str(row.get("split")) for row in rows} != {"val"}:
        raise ValueError("arrival timing stratification accepts validation rows only")
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    target_rate = int(settings["target_sample_rate"])
    duration = float(settings["analysis_duration"])
    output_bins = int(settings["output_bins"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = str(
        settings.get("architecture", "detector_arrival_timing_net_v1")
    )
    expected = {
        "architecture": architecture,
        "model_ifos": model_ifos,
        "target_sample_rate": target_rate,
        "analysis_duration": duration,
        "output_bins": output_bins,
    }
    for key, value in expected.items():
        observed = checkpoint.get(key)
        if key == "model_ifos":
            observed = tuple(observed or ())
        if observed != value:
            raise ValueError(f"arrival timing checkpoint {key} differs from configuration")
    dataset = DetectorArrivalDataset(
        rows,
        model_ifos,
        target_rate,
        duration,
        output_bins,
        bool(settings.get("cache_in_memory", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_detector_arrival_model(
        architecture, len(model_ifos), int(checkpoint["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    thresholds = tuple(
        float(value)
        for value in settings.get(
            "validation_ifo_snr_thresholds", (4.0, 6.0, 8.0, 10.0)
        )
    )
    if (
        not thresholds
        or any(value <= 0 for value in thresholds)
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ValueError("validation IFO SNR thresholds must be positive and increasing")
    groups, network_groups, prediction_rows = _grouped_evaluation(
        model,
        loader,
        rows,
        model_ifos,
        device,
        duration,
        output_bins,
        thresholds,
    )
    bin_width = duration / output_bins
    all_p90 = float(groups["all"]["absolute_error_seconds_quantiles"]["0.9"])
    identity = {
        "config_sha256": file_sha256(config_path),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "code_commit": execution_provenance()["code_commit"],
    }
    prediction_destination = Path(predictions_output_path)
    if prediction_destination.is_file():
        raise ValueError("arrival timing prediction output already exists")
    atomic_write_text(
        prediction_destination,
        "\n".join(
            json.dumps(row, sort_keys=True, allow_nan=False)
            for row in prediction_rows
        )
        + "\n",
    )
    report = {
        "status": "validation_only_detector_arrival_timing_stratification",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "SNR-conditioned strata diagnose detectability but may not replace candidate-level "
            "coverage and timing calibration at a frozen search threshold"
        ),
        "run_identity": identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "validation_manifest": str(validation_manifest),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_training_identity": checkpoint.get("run_identity"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "predictions_path": str(prediction_destination),
        "predictions_sha256": file_sha256(prediction_destination),
        "prediction_rows": len(prediction_rows),
        "validation_rows": len(rows),
        "model_ifos": list(model_ifos),
        "architecture": architecture,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "receptive_field_samples": detector_arrival_receptive_field_samples(
            architecture
        ),
        "receptive_field_seconds": detector_arrival_receptive_field_samples(
            architecture
        )
        / target_rate,
        "validation_ifo_snr_thresholds": list(thresholds),
        "timing": {
            "bin_width_seconds": bin_width,
            "representation_gate_passed": bin_width <= 0.01,
            "all_validation_accuracy_gate_passed": bin_width <= 0.01
            and all_p90 <= 0.01,
        },
        "validation_groups": groups,
        "validation_network_groups": network_groups,
        "device": str(device),
        **execution_provenance(torch),
    }
    destination = Path(output_path)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("run_identity") != identity:
            raise ValueError("arrival timing stratification output has another identity")
        return existing
    atomic_write_json(destination, report)
    return report


def compare_detector_arrival_prediction_rows(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    ifo_snr_thresholds: tuple[float, ...],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired comparison of timing predictions on identical validation injections."""

    if not reference_rows or bootstrap_replicates <= 0:
        raise ValueError("timing comparison requires rows and bootstrap replicates")
    if (
        not ifo_snr_thresholds
        or any(value <= 0 for value in ifo_snr_thresholds)
        or tuple(sorted(set(ifo_snr_thresholds))) != ifo_snr_thresholds
    ):
        raise ValueError("timing comparison SNR thresholds must be unique and increasing")

    def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        result = {}
        for row in rows:
            injection_id = str(row["injection_id"])
            if injection_id in result:
                raise ValueError(f"duplicate {label} timing injection: {injection_id}")
            result[injection_id] = row
        return result

    reference = keyed(reference_rows, "reference")
    candidate = keyed(candidate_rows, "candidate")
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate timing injections differ")
    paired = []
    for injection_id in sorted(reference):
        left = reference[injection_id]
        right = candidate[injection_id]
        for field in ("waveform_id", "background_window_id", "source_family"):
            if str(left[field]) != str(right[field]):
                raise ValueError(f"paired timing {field} differs for {injection_id}")
        for field in ("network_optimal_snr", "minimum_available_ifo_optimal_snr"):
            if not np.isclose(
                float(left[field]), float(right[field]), rtol=0, atol=1e-12
            ):
                raise ValueError(f"paired timing {field} differs for {injection_id}")
        left_detectors = left["detector_predictions"]
        right_detectors = right["detector_predictions"]
        if set(left_detectors) != set(right_detectors):
            raise ValueError(f"paired timing detector set differs for {injection_id}")
        for ifo in left_detectors:
            if not np.isclose(
                float(left_detectors[ifo]["exact_offset_seconds"]),
                float(right_detectors[ifo]["exact_offset_seconds"]),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(f"paired timing target differs for {injection_id}/{ifo}")
        paired.append((left, right))

    group_indices: dict[str, list[int]] = {"all": list(range(len(paired)))}
    for index, (left, _) in enumerate(paired):
        family_key = f"family:{left['source_family']}"
        group_indices.setdefault(family_key, []).append(index)
        minimum_snr = left.get("minimum_available_ifo_optimal_snr")
        if minimum_snr is not None:
            for threshold in ifo_snr_thresholds:
                if float(minimum_snr) >= threshold:
                    group_indices.setdefault(
                        f"minimum_ifo_snr_ge_{threshold:g}", []
                    ).append(index)

    rng = np.random.default_rng(seed)
    groups = {}
    for group, indices in sorted(group_indices.items()):
        if not indices:
            continue
        reference_max = np.asarray(
            [
                float(paired[index][0]["maximum_ifo_absolute_error_seconds"])
                for index in indices
            ]
        )
        candidate_max = np.asarray(
            [
                float(paired[index][1]["maximum_ifo_absolute_error_seconds"])
                for index in indices
            ]
        )
        reference_pair = np.asarray(
            [
                float(
                    paired[index][0][
                        "maximum_pairwise_delay_absolute_error_seconds"
                    ]
                )
                for index in indices
            ]
        )
        candidate_pair = np.asarray(
            [
                float(
                    paired[index][1][
                        "maximum_pairwise_delay_absolute_error_seconds"
                    ]
                )
                for index in indices
            ]
        )
        reference_within = reference_max <= 0.01
        candidate_within = candidate_max <= 0.01
        bootstrap = {
            "mean_maximum_ifo_error_delta": [],
            "p90_maximum_ifo_error_delta": [],
            "within_10ms_fraction_delta": [],
            "p90_pairwise_delay_error_delta": [],
        }
        for _ in range(bootstrap_replicates):
            sampled = rng.integers(0, len(indices), size=len(indices))
            bootstrap["mean_maximum_ifo_error_delta"].append(
                float(candidate_max[sampled].mean() - reference_max[sampled].mean())
            )
            bootstrap["p90_maximum_ifo_error_delta"].append(
                float(
                    np.quantile(candidate_max[sampled], 0.9)
                    - np.quantile(reference_max[sampled], 0.9)
                )
            )
            bootstrap["within_10ms_fraction_delta"].append(
                float(
                    candidate_within[sampled].mean()
                    - reference_within[sampled].mean()
                )
            )
            bootstrap["p90_pairwise_delay_error_delta"].append(
                float(
                    np.quantile(candidate_pair[sampled], 0.9)
                    - np.quantile(reference_pair[sampled], 0.9)
                )
            )

        def endpoint(maximum: np.ndarray, pairwise: np.ndarray) -> dict[str, float]:
            return {
                "mean_maximum_ifo_error_seconds": float(maximum.mean()),
                "p90_maximum_ifo_error_seconds": float(np.quantile(maximum, 0.9)),
                "within_10ms_fraction": float(np.mean(maximum <= 0.01)),
                "p90_pairwise_delay_error_seconds": float(np.quantile(pairwise, 0.9)),
            }

        groups[group] = {
            "injections": len(indices),
            "reference": endpoint(reference_max, reference_pair),
            "candidate": endpoint(candidate_max, candidate_pair),
            "delta_candidate_minus_reference": {
                "mean_maximum_ifo_error_seconds": float(
                    candidate_max.mean() - reference_max.mean()
                ),
                "p90_maximum_ifo_error_seconds": float(
                    np.quantile(candidate_max, 0.9)
                    - np.quantile(reference_max, 0.9)
                ),
                "within_10ms_fraction": float(
                    candidate_within.mean() - reference_within.mean()
                ),
                "p90_pairwise_delay_error_seconds": float(
                    np.quantile(candidate_pair, 0.9)
                    - np.quantile(reference_pair, 0.9)
                ),
            },
            "paired_bootstrap_95": {
                key: [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ]
                for key, values in bootstrap.items()
            },
        }
    return {
        "paired_injections": len(paired),
        "ifo_snr_thresholds": list(ifo_snr_thresholds),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "groups": groups,
    }


def run_detector_arrival_timing_validation_comparison(
    config_path: str | Path,
    reference_predictions_path: str | Path,
    candidate_predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    settings = config["detector_arrival_timing_promotion"]

    def load_rows(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    thresholds = tuple(float(value) for value in settings["ifo_snr_thresholds"])
    required_thresholds = {
        float(settings["primary_ifo_snr"]),
        float(settings["high_ifo_snr"]),
    }
    if not required_thresholds.issubset(thresholds):
        raise ValueError("promotion SNR thresholds must include primary and high strata")
    comparison = compare_detector_arrival_prediction_rows(
        load_rows(reference_predictions_path),
        load_rows(candidate_predictions_path),
        thresholds,
        int(settings["bootstrap_replicates"]),
        int(settings["seed"]),
    )
    all_group = comparison["groups"]["all"]
    conditional_key = f"minimum_ifo_snr_ge_{float(settings['primary_ifo_snr']):g}"
    high_key = f"minimum_ifo_snr_ge_{float(settings['high_ifo_snr']):g}"
    conditional = comparison["groups"][conditional_key]
    high = comparison["groups"][high_key]
    all_relative = -float(
        all_group["delta_candidate_minus_reference"]["p90_maximum_ifo_error_seconds"]
    ) / float(all_group["reference"]["p90_maximum_ifo_error_seconds"])
    conditional_relative = -float(
        conditional["delta_candidate_minus_reference"][
            "p90_maximum_ifo_error_seconds"
        ]
    ) / float(conditional["reference"]["p90_maximum_ifo_error_seconds"])
    checks = {
        "all_p90_relative_improvement": all_relative
        >= float(settings["minimum_all_p90_relative_improvement"]),
        "all_p90_bootstrap_upper_below_zero": float(
            all_group["paired_bootstrap_95"]["p90_maximum_ifo_error_delta"][1]
        )
        < 0,
        "conditional_p90_relative_improvement": conditional_relative
        >= float(settings["minimum_conditional_p90_relative_improvement"]),
        "conditional_coverage_gain": float(
            conditional["delta_candidate_minus_reference"]["within_10ms_fraction"]
        )
        >= float(settings["minimum_conditional_within_10ms_gain"]),
        "conditional_coverage_bootstrap_lower_above_zero": float(
            conditional["paired_bootstrap_95"]["within_10ms_fraction_delta"][0]
        )
        > 0,
        "high_snr_worst_ifo_p90": float(
            high["candidate"]["p90_maximum_ifo_error_seconds"]
        )
        <= float(settings["maximum_high_snr_worst_ifo_p90_seconds"]),
        "high_snr_pairwise_p90": float(
            high["candidate"]["p90_pairwise_delay_error_seconds"]
        )
        <= float(settings["maximum_high_snr_pairwise_p90_seconds"]),
    }
    report = {
        "status": "validation_only_detector_arrival_timing_paired_comparison",
        "scientific_claim_allowed": False,
        "promotion_allowed": all(checks.values()),
        "promotion_checks": checks,
        "all_p90_relative_improvement": all_relative,
        "conditional_p90_relative_improvement": conditional_relative,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "reference_predictions_sha256": file_sha256(reference_predictions_path),
        "candidate_predictions_sha256": file_sha256(candidate_predictions_path),
        "comparison": comparison,
        **execution_provenance(),
    }
    destination = Path(output_path)
    if destination.is_file():
        raise ValueError("arrival timing comparison output already exists")
    atomic_write_json(destination, report)
    return report
