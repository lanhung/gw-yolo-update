from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .arrival_timing import DetectorArrivalDataset
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


def candidate_pair_strain_feature_vector(
    first: dict[str, Any],
    second: dict[str, Any],
    strain: np.ndarray,
    model_ifos: tuple[str, ...],
    analysis_start_gps: float,
    sample_rate: int,
    physical_delay_limit_seconds: float,
    width_scale_seconds: float,
) -> np.ndarray:
    """Summarize local cross-IFO strain coherence inside a compatible interval pair."""

    values = np.asarray(strain, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] != len(model_ifos)
        or str(first["ifo"]) not in model_ifos
        or str(second["ifo"]) not in model_ifos
        or sample_rate <= 0
        or physical_delay_limit_seconds <= 0
        or width_scale_seconds <= 0
    ):
        raise ValueError("candidate pair strain feature inputs are invalid")
    analysis_stop = analysis_start_gps + values.shape[1] / sample_rate
    roi_start = max(
        float(first["gps_start"]) - physical_delay_limit_seconds,
        float(second["gps_start"]) - physical_delay_limit_seconds,
        analysis_start_gps,
    )
    roi_stop = min(
        float(first["gps_end"]) + physical_delay_limit_seconds,
        float(second["gps_end"]) + physical_delay_limit_seconds,
        analysis_stop,
    )
    start = max(int(np.floor((roi_start - analysis_start_gps) * sample_rate)), 0)
    stop = min(int(np.ceil((roi_stop - analysis_start_gps) * sample_rate)), values.shape[1])
    if stop - start < 4:
        return np.zeros(7, dtype=np.float32)
    first_values = values[model_ifos.index(str(first["ifo"])), start:stop].astype(
        np.float64
    )
    second_values = values[model_ifos.index(str(second["ifo"])), start:stop].astype(
        np.float64
    )
    maximum_lag = max(int(np.ceil(physical_delay_limit_seconds * sample_rate)), 1)
    correlations = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = first_values[-lag:], second_values[:lag]
        elif lag > 0:
            left, right = first_values[:-lag], second_values[lag:]
        else:
            left, right = first_values, second_values
        if left.size < 4:
            continue
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        correlations.append(float(np.dot(left, right) / denominator) if denominator else 0.0)
    if not correlations:
        correlations = [0.0]
    correlation = max(correlations, key=abs)
    result = np.asarray(
        [
            abs(correlation),
            correlation,
            np.log1p(np.sqrt(np.mean(first_values**2))),
            np.log1p(np.sqrt(np.mean(second_values**2))),
            np.log1p(np.max(np.abs(first_values))),
            np.log1p(np.max(np.abs(second_values))),
            (stop - start) / sample_rate / width_scale_seconds,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("candidate pair strain features are non-finite")
    return result


def candidate_pair_aligned_strain_crop(
    first: dict[str, Any],
    second: dict[str, Any],
    strain: np.ndarray,
    model_ifos: tuple[str, ...],
    analysis_start_gps: float,
    sample_rate: int,
    crop_duration_seconds: float,
    clip_amplitude: float,
) -> np.ndarray:
    """Extract a truth-free H1/L1 crop on one shared GPS time axis."""

    values = np.asarray(strain, dtype=np.float32)
    first_ifo, second_ifo = str(first["ifo"]), str(second["ifo"])
    if (
        values.ndim != 2
        or values.shape[0] != len(model_ifos)
        or first_ifo not in model_ifos
        or second_ifo not in model_ifos
        or first_ifo == second_ifo
        or sample_rate <= 0
        or crop_duration_seconds <= 0
        or clip_amplitude <= 0
    ):
        raise ValueError("candidate pair aligned strain crop inputs are invalid")
    samples = int(round(crop_duration_seconds * sample_rate))
    if samples < 16 or not np.isclose(
        samples / sample_rate, crop_duration_seconds, rtol=0, atol=1e-9
    ):
        raise ValueError("candidate pair crop duration must map to at least 16 samples")
    centers = [
        0.5 * (float(row["gps_start"]) + float(row["gps_end"]))
        for row in (first, second)
    ]
    if not np.isfinite(centers).all():
        raise ValueError("candidate pair crop centers are invalid")
    crop_start_gps = float(np.mean(centers)) - crop_duration_seconds / 2
    source_start = int(round((crop_start_gps - analysis_start_gps) * sample_rate))
    source_stop = source_start + samples
    valid_start = max(source_start, 0)
    valid_stop = min(source_stop, values.shape[1])
    output = np.zeros((2, samples), dtype=np.float32)
    if valid_stop > valid_start:
        target_start = valid_start - source_start
        target_stop = target_start + valid_stop - valid_start
        for output_index, ifo in enumerate((first_ifo, second_ifo)):
            output[output_index, target_start:target_stop] = values[
                model_ifos.index(ifo), valid_start:valid_stop
            ]
    return np.clip(output, -clip_amplitude, clip_amplitude).astype(np.float16)


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
    strain_contexts: dict[str, tuple[np.ndarray, float]] | None = None,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    sample_rate: int = 1024,
    include_strain_summary: bool = False,
    strain_crop_seconds: float | None = None,
    strain_clip_amplitude: float = 32.0,
) -> dict[str, Any]:
    if include_strain_summary and strain_crop_seconds is not None:
        raise ValueError("candidate pair examples cannot mix summary and STFT strain modes")
    if (include_strain_summary or strain_crop_seconds is not None) and strain_contexts is None:
        raise ValueError("candidate pair strain mode requires parent strain contexts")
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
    strain_crops = []
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
                feature_vector = candidate_pair_feature_vector(
                    first,
                    second,
                    physical_delay_limit_seconds,
                    width_scale_seconds,
                )
                aligned_crop = None
                if include_strain_summary:
                    if injection_id not in strain_contexts:
                        raise ValueError("candidate pair strain context is absent")
                    strain, analysis_start = strain_contexts[injection_id]
                    feature_vector = np.concatenate(
                        [
                            feature_vector,
                            candidate_pair_strain_feature_vector(
                                first,
                                second,
                                strain,
                                model_ifos,
                                analysis_start,
                                sample_rate,
                                physical_delay_limit_seconds,
                                width_scale_seconds,
                            ),
                        ]
                    ).astype(np.float32)
                elif strain_crop_seconds is not None:
                    if injection_id not in strain_contexts:
                        raise ValueError("candidate pair strain context is absent")
                    strain, analysis_start = strain_contexts[injection_id]
                    aligned_crop = candidate_pair_aligned_strain_crop(
                        first,
                        second,
                        strain,
                        model_ifos,
                        analysis_start,
                        sample_rate,
                        strain_crop_seconds,
                        strain_clip_amplitude,
                    )
                rows.append(
                    {
                        "features": feature_vector,
                        "padded": bool(support["padded"]),
                        "exact": bool(support["exact"]),
                        "peak_error": float(support["maximum_peak_error_seconds"]),
                        "pair_id": f'{first["candidate_id"]}|{second["candidate_id"]}',
                        "strain_crop": aligned_crop,
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
            if strain_crop_seconds is not None:
                strain_crops.append(row["strain_crop"])
    if not eligible or not features or not any(padded_labels) or all(padded_labels):
        raise ValueError("candidate pair training examples lack parents or class diversity")
    result = {
        "parent_ids": eligible,
        "features": np.stack(features),
        "padded_labels": np.asarray(padded_labels, dtype=bool),
        "exact_labels": np.asarray(exact_labels, dtype=bool),
        "peak_errors_seconds": np.asarray(peak_errors, dtype=np.float64),
        "example_parent_ids": example_parent_ids,
        "available_negative_pairs": available_negative_pairs,
        "retained_negative_pairs": retained_negative_pairs,
    }
    if strain_crop_seconds is not None:
        result["strain_crops"] = np.stack(strain_crops)
        if result["strain_crops"].shape != (
            len(features),
            2,
            int(round(strain_crop_seconds * sample_rate)),
        ):
            raise ValueError("candidate pair strain crops do not align with examples")
    return result


def _build_strain_contexts(
    parents: list[dict[str, Any]],
    model_ifos: tuple[str, ...],
    target_sample_rate: int,
    analysis_duration_seconds: float,
    parent_output_bins: int,
) -> dict[str, tuple[np.ndarray, float]]:
    dataset = DetectorArrivalDataset(
        parents,
        model_ifos,
        target_sample_rate,
        analysis_duration_seconds,
        parent_output_bins,
        True,
    )
    contexts = {}
    for index, row in enumerate(parents):
        strain, availability, _, offsets = dataset[index]
        present = row.get("detector_arrival_gps", {})
        starts = [
            float(present[ifo]) - float(offsets[ifo_index])
            for ifo_index, ifo in enumerate(model_ifos)
            if availability[ifo_index] and ifo in present
        ]
        if not starts or not np.allclose(starts, starts[0], rtol=0, atol=1e-6):
            raise ValueError("candidate pair parent analysis starts are inconsistent")
        contexts[str(row["injection_id"])] = (strain, starts[0])
    return contexts


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


    class CandidatePairTimeFrequencyEncoder(nn.Module):
        """Shared per-IFO STFT encoder with a symmetric pair-ranking head."""

        def __init__(
            self,
            scalar_features: int,
            hidden_features: int,
            embedding_features: int,
            stft_n_fft: int,
            stft_hop_length: int,
        ):
            super().__init__()
            if (
                scalar_features <= 0
                or hidden_features <= 0
                or embedding_features <= 0
                or stft_n_fft < 16
                or stft_hop_length <= 0
                or stft_hop_length > stft_n_fft
            ):
                raise ValueError("candidate pair STFT encoder settings are invalid")
            self.stft_n_fft = stft_n_fft
            self.stft_hop_length = stft_hop_length
            self.register_buffer("stft_window", torch.hann_window(stft_n_fft))
            self.shared_encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.GroupNorm(4, 8),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, embedding_features, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.ranker = nn.Sequential(
                nn.Linear(scalar_features + 4 * embedding_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, 1),
            )

        def forward(self, scalar: Any, strain_crops: Any) -> Any:
            if strain_crops.ndim != 3 or strain_crops.shape[1] != 2:
                raise ValueError("candidate pair STFT encoder expects [batch, 2, samples]")
            batch = strain_crops.shape[0]
            flattened = strain_crops.float().reshape(batch * 2, -1)
            spectrum = torch.stft(
                flattened,
                n_fft=self.stft_n_fft,
                hop_length=self.stft_hop_length,
                window=self.stft_window,
                center=False,
                return_complex=True,
            ).abs()
            spectrum = torch.log1p(spectrum)
            mean = spectrum.mean(dim=(-2, -1), keepdim=True)
            scale = spectrum.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            encoded = self.shared_encoder(((spectrum - mean) / scale)[:, None])
            encoded = encoded.reshape(batch, 2, -1)
            first, second = encoded[:, 0], encoded[:, 1]
            pair = torch.cat(
                [first, second, torch.abs(first - second), first * second, scalar],
                dim=1,
            )
            return self.ranker(pair)[:, 0]

else:

    class CandidatePairMLP:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            raise RuntimeError("Candidate pair training requires torch")

    class CandidatePairTimeFrequencyEncoder:  # type: ignore[no-redef]
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
            batch_features = features[start : start + batch_size].to(device)
            if "strain_crops" in examples:
                batch_crops = torch.from_numpy(
                    examples["strain_crops"][start : start + batch_size]
                ).to(device)
                logits = model(batch_features, batch_crops)
            else:
                logits = model(batch_features)
            scores.append(torch.sigmoid(logits).cpu())
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


def _stratified_parent_metrics(
    examples: dict[str, Any],
    scores: np.ndarray,
    parents: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_map = {str(row["injection_id"]): row for row in parents}
    groups: dict[str, list[str]] = defaultdict(list)
    for parent_id in examples["parent_ids"]:
        row = parent_map[parent_id]
        family = str(row["source_family"])
        snr_value = row.get(
            "training_network_optimal_snr", row.get("network_optimal_snr")
        )
        if snr_value is None:
            raise ValueError("candidate pair ranker parent lacks network SNR")
        snr = float(snr_value)
        snr_name = (
            "snr_lt_8"
            if snr < 8
            else "snr_8_15"
            if snr < 15
            else "snr_15_30"
            if snr < 30
            else "snr_ge_30"
        )
        groups[f"family:{family}"].append(parent_id)
        groups[f"snr:{snr_name}"].append(parent_id)
    output = {}
    for name, parent_ids in sorted(groups.items()):
        selected_ids = set(parent_ids)
        mask = np.asarray(
            [value in selected_ids for value in examples["example_parent_ids"]],
            dtype=bool,
        )
        output[name] = candidate_parent_top1_metrics(
            parent_ids,
            [
                value
                for value, keep in zip(examples["example_parent_ids"], mask)
                if keep
            ],
            scores[mask],
            examples["padded_labels"][mask],
            examples["exact_labels"][mask],
            examples["peak_errors_seconds"][mask],
        )
    return output


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
    use_strain_features = bool(settings.get("use_strain_pair_features", False))
    use_time_frequency_encoder = bool(
        settings.get("use_time_frequency_pair_encoder", False)
    )
    if use_strain_features and use_time_frequency_encoder:
        raise ValueError("candidate pair ranker permits only one strain representation")
    model_ifos = tuple(
        str(value)
        for value in settings.get("model_ifos", ["H1", "L1", "V1"])
    )
    target_sample_rate = int(settings.get("target_sample_rate", 1024))
    train_contexts = (
        _build_strain_contexts(
            train_parents,
            model_ifos,
            target_sample_rate,
            float(settings["analysis_duration_seconds"]),
            int(settings["parent_output_bins"]),
        )
        if use_strain_features or use_time_frequency_encoder
        else None
    )
    validation_contexts = (
        _build_strain_contexts(
            validation_parents,
            model_ifos,
            target_sample_rate,
            float(settings["analysis_duration_seconds"]),
            int(settings["parent_output_bins"]),
        )
        if use_strain_features or use_time_frequency_encoder
        else None
    )
    train_examples = _build_examples(
        train_parents,
        train_candidates,
        *common,
        int(settings["maximum_negative_pairs_per_parent"]),
        seed,
        train_contexts,
        model_ifos,
        target_sample_rate,
        use_strain_features,
        (
            float(settings["strain_crop_seconds"])
            if use_time_frequency_encoder
            else None
        ),
        float(settings.get("strain_clip_amplitude", 32.0)),
    )
    validation_examples = _build_examples(
        validation_parents,
        validation_candidates,
        *common,
        None,
        seed,
        validation_contexts,
        model_ifos,
        target_sample_rate,
        use_strain_features,
        (
            float(settings["strain_crop_seconds"])
            if use_time_frequency_encoder
            else None
        ),
        float(settings.get("strain_clip_amplitude", 32.0)),
    )
    input_features = int(train_examples["features"].shape[1])
    if validation_examples["features"].shape[1] != input_features:
        raise ValueError("candidate pair train/validation features differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = (
        "candidate_pair_trainable_stft_cnn_v3"
        if use_time_frequency_encoder
        else "candidate_pair_mlp_strain_coherence_v2"
        if use_strain_features
        else "candidate_pair_mlp_v1"
    )
    if use_time_frequency_encoder:
        model = CandidatePairTimeFrequencyEncoder(
            input_features,
            int(settings["hidden_features"]),
            int(settings["embedding_features"]),
            int(settings["stft_n_fft"]),
            int(settings["stft_hop_length"]),
        ).to(device)
    else:
        model = CandidatePairMLP(
            input_features, int(settings["hidden_features"])
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    features = torch.from_numpy(train_examples["features"])
    labels = torch.from_numpy(train_examples["padded_labels"].astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    training_tensors = (
        (features, torch.from_numpy(train_examples["strain_crops"]), labels)
        if use_time_frequency_encoder
        else (features, labels)
    )
    loader = DataLoader(
        TensorDataset(*training_tensors),
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
        for batch in loader:
            if updates >= maximum_updates:
                break
            if use_time_frequency_encoder:
                batch_features, batch_crops, batch_labels = batch
            else:
                batch_features, batch_labels = batch
            optimizer.zero_grad(set_to_none=True)
            logits = (
                model(batch_features.to(device), batch_crops.to(device))
                if use_time_frequency_encoder
                else model(batch_features.to(device))
            )
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
                    "architecture": architecture,
                    "model": model.state_dict(),
                    "input_features": input_features,
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
    selected_strata = _stratified_parent_metrics(
        validation_examples, scores, validation_parents
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
        "architecture": architecture,
        "input_features": input_features,
        "strain_pair_features": use_strain_features,
        "time_frequency_pair_encoder": use_time_frequency_encoder,
        "strain_feature_definition": (
            "absolute/signed physical-lag correlation, local RMS/peak amplitudes and ROI duration"
            if use_strain_features
            else None
        ),
        "time_frequency_feature_definition": (
            "shared-GPS aligned whitened H1/L1 crops, trainable log-STFT shared-IFO CNN, "
            "ordered detector embeddings, difference/product fusion and proposal geometry"
            if use_time_frequency_encoder
            else None
        ),
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
        "selected_validation_strata": selected_strata,
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
