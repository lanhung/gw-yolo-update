from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .arrival_timing import DetectorArrivalDataset
from .candidates import candidate_proposal_coverage
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .numeric import DetectorArrivalSpectrogramNet, _atomic_torch_save
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


def dense_endpoint_targets(
    arrival_offsets_by_ifo: dict[str, float | Iterable[float]],
    model_ifos: tuple[str, ...],
    duration_seconds: float,
    output_bins: int,
    half_width_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a multi-peak dense target without collapsing endpoint instances."""

    if duration_seconds <= 0 or output_bins < 2 or half_width_seconds < 0:
        raise ValueError("dense endpoint target geometry is invalid")
    target = np.zeros((len(model_ifos), output_bins), dtype=np.float32)
    availability = np.zeros(len(model_ifos), dtype=bool)
    bin_width = duration_seconds / output_bins
    radius = int(np.ceil(half_width_seconds / bin_width))
    for ifo_index, ifo in enumerate(model_ifos):
        if ifo not in arrival_offsets_by_ifo:
            continue
        raw = arrival_offsets_by_ifo[ifo]
        values = [raw] if np.isscalar(raw) else list(raw)
        if not values:
            raise ValueError(f"dense endpoint target for {ifo} is empty")
        availability[ifo_index] = True
        for value in values:
            offset = float(value)
            if not np.isfinite(offset) or not 0 <= offset < duration_seconds:
                raise ValueError(f"dense endpoint target for {ifo} lies outside the window")
            center = min(int(np.floor(offset / bin_width)), output_bins - 1)
            start = max(center - radius, 0)
            stop = min(center + radius + 1, output_bins)
            target[ifo_index, start:stop] = 1.0
    if np.count_nonzero(availability) < 2:
        raise ValueError("dense endpoint target requires at least two available detectors")
    return target, availability


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(changes == 1).astype(int).tolist(),
            np.flatnonzero(changes == -1).astype(int).tolist(),
        )
    )


def extract_dense_endpoint_candidates(
    probabilities: np.ndarray,
    availability: np.ndarray,
    exact_offsets_seconds: np.ndarray,
    injection_rows: list[dict[str, Any]],
    model_ifos: tuple[str, ...],
    duration_seconds: float,
    threshold: float,
    minimum_bins: int = 1,
) -> list[dict[str, Any]]:
    """Retain every connected endpoint proposal for every available detector."""

    values = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(availability, dtype=bool)
    offsets = np.asarray(exact_offsets_seconds, dtype=np.float64)
    expected_shape = (len(injection_rows), len(model_ifos))
    if values.ndim != 3 or values.shape[:2] != expected_shape:
        raise ValueError("dense endpoint probabilities must have shape [injection, IFO, time]")
    if valid.shape != expected_shape or offsets.shape != expected_shape:
        raise ValueError("dense endpoint metadata does not align with probabilities")
    if values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("dense endpoint probabilities are invalid")
    if not 0 <= threshold <= 1 or minimum_bins <= 0 or duration_seconds <= 0:
        raise ValueError("dense endpoint extraction settings are invalid")
    bin_width = duration_seconds / values.shape[-1]
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(injection_rows):
        starts = []
        arrivals = row.get("detector_arrival_gps", {})
        for ifo_index, ifo in enumerate(model_ifos):
            if not valid[row_index, ifo_index]:
                continue
            if ifo not in arrivals or not np.isfinite(offsets[row_index, ifo_index]):
                raise ValueError(f"available proposal detector lacks arrival metadata: {ifo}")
            starts.append(float(arrivals[ifo]) - float(offsets[row_index, ifo_index]))
        if not starts or not np.allclose(starts, starts[0], rtol=0, atol=2e-6):
            raise ValueError("proposal analysis-start reconstruction is inconsistent")
        analysis_start = starts[0]
        for ifo_index, ifo in enumerate(model_ifos):
            if not valid[row_index, ifo_index]:
                continue
            profile = values[row_index, ifo_index]
            for run_index, (start, stop) in enumerate(_active_runs(profile >= threshold)):
                if stop - start < minimum_bins:
                    continue
                peak = start + int(np.argmax(profile[start:stop]))
                identity = {
                    "injection": str(row["injection_id"]),
                    "ifo": ifo,
                    "run": run_index,
                    "threshold": threshold,
                }
                output.append(
                    {
                        "candidate_id": f"endpoint-proposal-{canonical_hash(identity, 24)}",
                        "injection_id": str(row["injection_id"]),
                        "waveform_id": str(row["waveform_id"]),
                        "split": str(row["split"]),
                        "source_family": str(row["source_family"]),
                        "gps_block": str(row["gps_block"]),
                        "ifo": ifo,
                        "gps_start": analysis_start + start * bin_width,
                        "gps_end": analysis_start + stop * bin_width,
                        "gps_peak": analysis_start + (peak + 0.5) * bin_width,
                        "proposal_score": float(profile[peak]),
                        "start_bin": start,
                        "stop_bin_exclusive": stop,
                        "peak_bin": peak,
                        "time_bins": values.shape[-1],
                        "bin_width_seconds": bin_width,
                        "proposal_method": "dense_sigmoid_endpoint_map",
                    }
                )
    return output


def proposal_gate_record(
    coverage: dict[str, Any], threshold: float, settings: dict[str, Any]
) -> dict[str, Any]:
    required_groups = tuple(str(value) for value in settings["required_groups"])
    groups = coverage["groups"]
    missing = [key for key in ("all", *required_groups) if key not in groups]
    if missing:
        raise ValueError(f"dense proposal coverage lacks required groups: {missing}")
    all_group = groups["all"]
    group_checks = {
        key: float(groups[key]["padded_coverage_fraction"])
        >= float(settings["minimum_required_group_padded_coverage"])
        for key in required_groups
    }
    widths = all_group.get("minimum_containing_proposal_width_seconds_quantiles", {})
    checks = {
        "all_padded_coverage": float(all_group["padded_coverage_fraction"])
        >= float(settings["minimum_all_padded_coverage"]),
        "required_group_padded_coverage": all(group_checks.values()),
        "median_union_fraction": float(
            all_group["proposal_union_fraction_of_analysis_quantiles"]["0.5"]
        )
        <= float(settings["maximum_median_union_fraction"]),
        "p90_union_fraction": float(
            all_group["proposal_union_fraction_of_analysis_quantiles"]["0.9"]
        )
        <= float(settings["maximum_p90_union_fraction"]),
        "median_containing_width": bool(widths)
        and float(widths["0.5"])
        <= float(settings["maximum_median_containing_width_seconds"]),
    }
    return {
        "threshold": float(threshold),
        "candidates": int(coverage["candidates"]),
        "padded_coverage_fraction": float(all_group["padded_coverage_fraction"]),
        "median_union_fraction": float(
            all_group["proposal_union_fraction_of_analysis_quantiles"]["0.5"]
        ),
        "p90_union_fraction": float(
            all_group["proposal_union_fraction_of_analysis_quantiles"]["0.9"]
        ),
        "median_containing_width_seconds": (
            float(widths["0.5"]) if widths else None
        ),
        "required_group_coverage_checks": group_checks,
        "checks": checks,
        "qualified": all(checks.values()),
    }


def select_dense_proposal_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    qualified = [record for record in records if record["qualified"]]
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda record: (
            record["median_union_fraction"],
            record["p90_union_fraction"],
            record["candidates"],
            -record["threshold"],
        ),
    )


def _dense_target_from_batch(
    targets: Any, availability: Any, output_bins: int, half_width_bins: int
) -> Any:
    batch, detectors = targets.shape
    dense = torch.zeros(
        (batch, detectors, output_bins), dtype=torch.float32, device=targets.device
    )
    for radius_offset in range(-half_width_bins, half_width_bins + 1):
        indices = torch.clamp(targets + radius_offset, 0, output_bins - 1)
        dense.scatter_(2, indices[:, :, None], availability[:, :, None].to(dense.dtype))
    return dense


def _proposal_epoch(
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    output_bins: int,
    half_width_bins: int,
    positive_weight: float,
    focal_gamma: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses = []
    examples = 0
    batches = 0
    positive = torch.as_tensor(positive_weight, device=device)
    for strain, availability, targets, _ in loader:
        strain = strain.to(device)
        availability = availability.to(device)
        targets = targets.to(device)
        dense = _dense_target_from_batch(
            targets, availability, output_bins, half_width_bins
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(strain, availability)
            # The network deliberately writes -inf into unavailable detector slots.
            # Index those slots out before BCE: BCEWithLogits(-inf, 0) is NaN, and
            # multiplying that NaN by a zero validity mask would not remove it.
            selected_logits = logits[availability]
            selected_dense = dense[availability]
            raw = torch_functional.binary_cross_entropy_with_logits(
                selected_logits,
                selected_dense,
                pos_weight=positive,
                reduction="none",
            )
            probability = torch.sigmoid(selected_logits)
            correct = (
                probability * selected_dense
                + (1.0 - probability) * (1.0 - selected_dense)
            )
            loss_map = raw * ((1.0 - correct) ** focal_gamma)
            loss = loss_map.mean()
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        examples += int(strain.shape[0])
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return {
        "loss": float(np.mean(losses)),
        "examples": examples,
        "batches": batches,
    }


def _predict_proposals(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, ...]:
    model.eval()
    probabilities = []
    availability_rows = []
    offsets = []
    with torch.no_grad():
        for strain, availability, _, exact_offsets in loader:
            logits = model(strain.to(device), availability.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            availability_rows.append(availability.numpy())
            offsets.append(exact_offsets.numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(availability_rows).astype(bool),
        np.concatenate(offsets),
    )


def _evaluate_threshold_grid(
    probabilities: np.ndarray,
    availability: np.ndarray,
    offsets: np.ndarray,
    validation_rows: list[dict[str, Any]],
    model_ifos: tuple[str, ...],
    duration: float,
    checkpoint_path: str | Path,
    output: Path,
    settings: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    threshold_records = []
    audit_hashes = {}
    for threshold in (float(value) for value in settings["threshold_grid"]):
        tag = f"{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        threshold_dir = output / f"threshold-{tag}"
        threshold_dir.mkdir(parents=True, exist_ok=False)
        candidates = extract_dense_endpoint_candidates(
            probabilities,
            availability,
            offsets,
            validation_rows,
            model_ifos,
            duration,
            threshold,
            int(settings.get("minimum_bins", 1)),
        )
        candidate_path = threshold_dir / "endpoint_injection_candidates.jsonl"
        atomic_write_text(
            candidate_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        )
        coverage = candidate_proposal_coverage(
            validation_rows, candidates, float(gates["padding_seconds"])
        )
        audit = {
            "status": "validation_only_dense_endpoint_proposal_coverage",
            "scientific_claim_allowed": False,
            "threshold": threshold,
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": file_sha256(candidate_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            **coverage,
            **execution_provenance(torch),
        }
        audit_path = threshold_dir / "proposal_coverage.json"
        atomic_write_json(audit_path, audit)
        audit_hashes[str(threshold)] = file_sha256(audit_path)
        record = proposal_gate_record(coverage, threshold, gates)
        record["audit_path"] = str(audit_path)
        record["audit_sha256"] = audit_hashes[str(threshold)]
        record["candidate_manifest_sha256"] = file_sha256(candidate_path)
        threshold_records.append(record)
    return threshold_records, audit_hashes


def run_detector_endpoint_proposal_training(
    config_path: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("Detector endpoint proposal training requires torch")
    config = load_yaml(config_path)
    settings = config["detector_endpoint_proposal"]
    gates = config["candidate_proposal_threshold_selection"]
    seed = int(seed_override if seed_override is not None else settings["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "train_manifest_sha256": file_sha256(train_manifest),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "pretrained_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "seed": seed,
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "detector_endpoint_proposal_report.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed detector endpoint proposal run has another identity")
        return result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_rows = [json.loads(line) for line in Path(train_manifest).read_text().splitlines() if line]
    validation_rows = [
        json.loads(line) for line in Path(validation_manifest).read_text().splitlines() if line
    ]
    minimum_snr = settings.get("minimum_training_network_snr")
    if minimum_snr is not None:
        train_rows = [
            row
            for row in train_rows
            if float(row.get("training_network_optimal_snr", row["network_optimal_snr"]))
            >= float(minimum_snr)
        ]
    split_audit = physical_split_audit(train_rows, validation_rows)
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    target_rate = int(settings["target_sample_rate"])
    duration = float(settings["analysis_duration"])
    output_bins = int(settings["output_bins"])
    half_width_seconds = float(settings["target_half_width_seconds"])
    bin_width = duration / output_bins
    half_width_bins = int(np.ceil(half_width_seconds / bin_width))
    datasets = {
        "train": DetectorArrivalDataset(
            train_rows, model_ifos, target_rate, duration, output_bins,
            bool(settings.get("cache_in_memory", True)),
        ),
        "val": DetectorArrivalDataset(
            validation_rows, model_ifos, target_rate, duration, output_bins,
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
    base_channels = int(settings["base_channels"])
    model = DetectorArrivalSpectrogramNet(len(model_ifos), base_channels).to(device)
    warm = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    for key, expected in (
        ("model_ifos", list(model_ifos)),
        ("output_bins", output_bins),
        ("base_channels", base_channels),
    ):
        observed = warm.get(key)
        if key == "model_ifos":
            observed = list(observed)
        if observed != expected:
            raise ValueError(f"endpoint proposal warm start {key} differs from config")
    if warm.get("architecture") != "detector_arrival_spectrogram_net_v3":
        raise ValueError("endpoint proposal requires the v3 numeric spectrogram checkpoint")
    model.load_state_dict(warm["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    checkpoint_path = output / "best_detector_endpoint_proposal.pt"
    resume_path = output / "last_detector_endpoint_proposal.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = None
    start_epoch = 1
    updates = 0
    examples = 0
    steps = len(loaders["train"])
    maximum_updates = int(settings["max_optimizer_updates"])
    if maximum_updates <= 0 or maximum_updates > int(settings["epochs"]) * steps:
        raise ValueError("endpoint proposal optimizer budget is invalid")
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("run_identity") != identity:
            raise ValueError("endpoint proposal resume identity differs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["data_generator_state"])
        history = list(resume["history"])
        best_loss = float(resume["best_validation_loss"])
        best_epoch = resume["best_epoch"]
        start_epoch = int(resume["epoch"]) + 1
        updates = int(resume["optimizer_updates"])
        examples = int(resume["optimizer_examples"])
    started = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        remaining = maximum_updates - updates
        if remaining <= 0:
            break
        train_metrics = _proposal_epoch(
            model, loaders["train"], device, optimizer, output_bins, half_width_bins,
            float(settings["positive_weight"]), float(settings["focal_gamma"]),
            min(remaining, steps),
        )
        updates += int(train_metrics["batches"])
        examples += int(train_metrics["examples"])
        validation_metrics = _proposal_epoch(
            model, loaders["val"], device, None, output_bins, half_width_bins,
            float(settings["positive_weight"]), float(settings["focal_gamma"]),
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "architecture": "detector_endpoint_spectrogram_dense_v1",
                    "model": model.state_dict(),
                    "model_ifos": list(model_ifos),
                    "target_sample_rate": target_rate,
                    "analysis_duration": duration,
                    "output_bins": output_bins,
                    "base_channels": base_channels,
                    "epoch": epoch,
                    "validation_loss": best_loss,
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
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "optimizer_updates": updates,
                "optimizer_examples": examples,
            },
        )
        atomic_write_json(output / "history.json", history)
    selected_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_checkpoint["model"])
    probabilities, availability, offsets = _predict_proposals(model, loaders["val"], device)
    threshold_records, audit_hashes = _evaluate_threshold_grid(
        probabilities,
        availability,
        offsets,
        validation_rows,
        model_ifos,
        duration,
        checkpoint_path,
        output,
        settings,
        gates,
    )
    selected = select_dense_proposal_record(threshold_records)
    result = {
        "status": "validation_only_dense_detector_endpoint_proposal",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "requires a passing frozen proposal gate, all-instance overlap training, continuous "
            "background, timing calibration and locked-test VT"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "split_audit": split_audit,
        "seed": seed,
        "architecture": "detector_endpoint_spectrogram_dense_v1",
        "mask_preservation": (
            "independent sidecar output; the source chirp/glitch segmentation checkpoint is not "
            "loaded or modified"
        ),
        "multi_instance_contract": "one sigmoid heatmap may retain every disconnected peak per IFO",
        "warm_start_checkpoint_sha256": file_sha256(pretrained_checkpoint),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "selection_metric": "minimum validation dense focal BCE loss",
        "target_half_width_seconds": half_width_seconds,
        "target_half_width_bins": half_width_bins,
        "bin_width_seconds": bin_width,
        "threshold_records": threshold_records,
        "proposal_gate_passed": selected is not None,
        "selected_threshold": selected,
        "audit_hashes": audit_hashes,
        "epochs": int(settings["epochs"]),
        "completed_epochs": len(history),
        "steps_per_full_epoch": steps,
        "max_optimizer_updates": maximum_updates,
        "optimizer_updates": updates,
        "optimizer_examples": examples,
        "training_budget_reached": updates == maximum_updates,
        "candidate_counts_by_threshold": {
            str(record["threshold"]): record["candidates"] for record in threshold_records
        },
        "history": history,
        "elapsed_seconds": time.time() - started,
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result


def run_detector_endpoint_proposal_evaluation(
    config_path: str | Path,
    validation_manifest: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Refine a validation-only proposal threshold without retraining or test access."""

    if torch is None:
        raise RuntimeError("Detector endpoint proposal evaluation requires torch")
    config = load_yaml(config_path)
    settings = config["detector_endpoint_proposal"]
    gates = config["candidate_proposal_threshold_selection"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": file_sha256(config_path),
        "validation_manifest_sha256": file_sha256(validation_manifest),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "detector_endpoint_proposal_evaluation.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed endpoint proposal evaluation has another identity")
        return result
    if any(output.iterdir()):
        raise FileExistsError("endpoint proposal evaluation output must be empty")
    validation_rows = [
        json.loads(line)
        for line in Path(validation_manifest).read_text().splitlines()
        if line
    ]
    if not validation_rows or any(row.get("split") != "val" for row in validation_rows):
        raise ValueError("endpoint proposal threshold refinement accepts validation rows only")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "detector_endpoint_spectrogram_dense_v1":
        raise ValueError("endpoint proposal evaluation checkpoint has the wrong architecture")
    model_ifos = tuple(str(value) for value in settings["model_ifos"])
    duration = float(settings["analysis_duration"])
    output_bins = int(settings["output_bins"])
    base_channels = int(settings["base_channels"])
    expected = {
        "model_ifos": list(model_ifos),
        "target_sample_rate": int(settings["target_sample_rate"]),
        "analysis_duration": duration,
        "output_bins": output_bins,
        "base_channels": base_channels,
    }
    for key, value in expected.items():
        observed = checkpoint.get(key)
        if key == "model_ifos" and observed is not None:
            observed = list(observed)
        if observed != value:
            raise ValueError(f"endpoint proposal checkpoint {key} differs from evaluation config")
    dataset = DetectorArrivalDataset(
        validation_rows,
        model_ifos,
        int(settings["target_sample_rate"]),
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
    model = DetectorArrivalSpectrogramNet(len(model_ifos), base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    started = time.time()
    probabilities, availability, offsets = _predict_proposals(model, loader, device)
    threshold_records, audit_hashes = _evaluate_threshold_grid(
        probabilities,
        availability,
        offsets,
        validation_rows,
        model_ifos,
        duration,
        checkpoint_path,
        output,
        settings,
        gates,
    )
    selected = select_dense_proposal_record(threshold_records)
    result = {
        "status": "validation_only_dense_endpoint_proposal_threshold_refinement",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation proposal selection is not search recall and still requires timing, "
            "continuous background and locked-test VT"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_injections": len(validation_rows),
        "threshold_records": threshold_records,
        "proposal_gate_passed": selected is not None,
        "selected_threshold": selected,
        "audit_hashes": audit_hashes,
        "elapsed_seconds": time.time() - started,
        **execution_provenance(torch),
    }
    atomic_write_json(report_path, result)
    return result
