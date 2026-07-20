from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .arrival_timing import DetectorArrivalDataset
from .candidate_refiner import (
    candidate_average_precision,
    candidate_interval_pair_features,
    candidate_pair_truth_support,
)
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
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


def _parse_scale_manifest_specs(specs: list[str]) -> list[tuple[int, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("candidate pair scale manifests require SIZE=PATH")
        raw_size, raw_path = spec.split("=", 1)
        try:
            size = int(raw_size)
        except ValueError as error:
            raise ValueError("candidate pair scale size must be an integer") from error
        path = Path(raw_path)
        if size <= 0 or not raw_path or not path.is_file():
            raise ValueError("candidate pair scale manifest specification is invalid")
        parsed.append((size, path))
    parsed.sort()
    if not parsed or len({size for size, _ in parsed}) != len(parsed):
        raise ValueError("candidate pair scale sizes must be nonempty and unique")
    return parsed


def run_candidate_pair_scaling_plan(
    train_injection_manifest: str | Path,
    train_candidate_manifest: str | Path,
    scale_manifest_specs: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create nested, physical-parent-counted ranker manifests without candidate pruning."""

    scale_specs = _parse_scale_manifest_specs(scale_manifest_specs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "train_injection_manifest_sha256": file_sha256(train_injection_manifest),
        "train_candidate_manifest_sha256": file_sha256(train_candidate_manifest),
        "scale_manifests": [
            {"size": size, "sha256": file_sha256(path)} for size, path in scale_specs
        ],
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_pair_scaling_plan_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_identity") != identity:
            raise ValueError("completed candidate pair scaling plan has another identity")
        return report
    if any(output.iterdir()):
        raise FileExistsError("candidate pair scaling plan output must be empty")
    parents = _read_jsonl(train_injection_manifest)
    candidates = _read_jsonl(train_candidate_manifest)
    if not parents or any(row.get("split") != "train" for row in parents):
        raise ValueError("candidate pair scaling parents must be nonempty train rows")
    parent_map = {str(row["injection_id"]): row for row in parents}
    if len(parent_map) != len(parents):
        raise ValueError("candidate pair scaling parents repeat injection IDs")
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate pair scaling candidates repeat candidate IDs")
    if any(
        row.get("split") != "train"
        or str(row["injection_id"]) not in parent_map
        or row.get("refiner_role") not in (None, "train")
        for row in candidates
    ):
        raise ValueError("candidate pair scaling candidates differ from train parents")
    candidate_parent_ids = {str(row["injection_id"]) for row in candidates}
    previous_ids: set[str] = set()
    records = []
    for size, source_path in scale_specs:
        source_rows = _read_jsonl(source_path)
        selected_ids = {str(row["injection_id"]) for row in source_rows}
        if len(source_rows) != size or len(selected_ids) != size:
            raise ValueError("candidate pair scale size differs from unique physical rows")
        if not previous_ids.issubset(selected_ids):
            raise ValueError("candidate pair scale manifests are not nested")
        if not selected_ids.issubset(parent_map):
            raise ValueError("candidate pair scale contains an unknown injection")
        for row in source_rows:
            parent = parent_map[str(row["injection_id"])]
            if (
                str(row["waveform_id"]) != str(parent["waveform_id"])
                or str(row["gps_block"]) != str(parent["gps_block"])
            ):
                raise ValueError("candidate pair scale physical identity differs")
        selected_parents = [
            row for row in parents if str(row["injection_id"]) in selected_ids
        ]
        selected_candidates = [
            {**row, "refiner_role": "train", "training_parent_scale": size}
            for row in candidates
            if str(row["injection_id"]) in selected_ids
        ]
        parent_path = output / f"candidate_pair_scale_{size}_parents.jsonl"
        candidate_path = output / f"candidate_pair_scale_{size}_candidates.jsonl"
        atomic_write_text(
            parent_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_parents),
        )
        atomic_write_text(
            candidate_path,
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in selected_candidates
            ),
        )
        parents_with_candidates = len(selected_ids & candidate_parent_ids)
        records.append(
            {
                "physical_parent_count": size,
                "parent_manifest": str(parent_path),
                "parent_manifest_sha256": file_sha256(parent_path),
                "candidate_manifest": str(candidate_path),
                "candidate_manifest_sha256": file_sha256(candidate_path),
                "candidates": len(selected_candidates),
                "parents_with_candidates": parents_with_candidates,
                "zero_candidate_parents": size - parents_with_candidates,
                "unique_waveforms": len(
                    {str(row["waveform_id"]) for row in selected_parents}
                ),
                "unique_gps_blocks": len(
                    {str(row["gps_block"]) for row in selected_parents}
                ),
                "candidate_counts_by_ifo": dict(
                    sorted(
                        Counter(str(row["ifo"]) for row in selected_candidates).items()
                    )
                ),
                "all_connected_candidates_retained": True,
                "top_k_pruning": None,
            }
        )
        previous_ids = selected_ids
    result = {
        "status": "verified_nested_candidate_pair_scaling_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "fixed-update and fixed-epoch ranker controls plus fresh calibration remain required"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "scale_records": records,
        "physical_sample_definition": "unique injection/waveform parent, never candidate rows",
        "all_connected_candidates_retained": True,
        "top_k_pruning": None,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def _parse_scale_report_specs(specs: list[str]) -> dict[int, Path]:
    parsed = _parse_scale_manifest_specs(specs)
    return {size: path for size, path in parsed}


def run_candidate_pair_scaling_evaluation(
    config_path: str | Path,
    scaling_plan_report_path: str | Path,
    fixed_update_report_specs: list[str],
    fixed_epoch_report_specs: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate predeclared 2k/5k/10k controls without authorizing larger scale."""

    config = load_yaml(config_path)
    settings = config["candidate_pair_scaling_evaluation"]
    expected_scales = tuple(int(value) for value in settings["expected_scales"])
    if expected_scales != tuple(sorted(expected_scales)) or len(expected_scales) < 2:
        raise ValueError("candidate pair scaling expected scales are invalid")
    plan = json.loads(Path(scaling_plan_report_path).read_text(encoding="utf-8"))
    if plan.get("status") != "verified_nested_candidate_pair_scaling_plan":
        raise ValueError("candidate pair scaling evaluation requires a verified plan")
    plan_records = {
        int(row["physical_parent_count"]): row for row in plan["scale_records"]
    }
    if tuple(sorted(plan_records)) != expected_scales:
        raise ValueError("candidate pair scaling plan differs from expected scales")
    modes = {
        "fixed_updates": _parse_scale_report_specs(fixed_update_report_specs),
        "fixed_epochs": _parse_scale_report_specs(fixed_epoch_report_specs),
    }
    if any(tuple(sorted(paths)) != expected_scales for paths in modes.values()):
        raise ValueError("candidate pair scaling reports differ from expected scales")
    identity = {
        "config_sha256": file_sha256(config_path),
        "scaling_plan_report_sha256": file_sha256(scaling_plan_report_path),
        "reports": {
            mode: [
                {"scale": scale, "sha256": file_sha256(paths[scale])}
                for scale in expected_scales
            ]
            for mode, paths in modes.items()
        },
        "code_commit": execution_provenance()["code_commit"],
    }
    target = Path(output_path)
    if target.is_file():
        result = json.loads(target.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed candidate pair scaling evaluation has another identity")
        return result
    reports: dict[str, dict[int, dict[str, Any]]] = {}
    common_validation_identity = None
    common_architecture = None
    common_seed = None
    curves = {}
    for mode, paths in modes.items():
        reports[mode] = {}
        curve = []
        for scale in expected_scales:
            report = json.loads(paths[scale].read_text(encoding="utf-8"))
            if report.get("status") != "validation_selection_candidate_pair_ranker":
                raise ValueError("candidate pair scaling input is not a completed ranker")
            if report.get("test_evaluation") is not None:
                raise ValueError("candidate pair scaling evaluation cannot consume test results")
            if report.get("budget_mode") != mode:
                raise ValueError("candidate pair scaling report has the wrong budget mode")
            if int(report.get("train_physical_parents", -1)) != scale:
                raise ValueError("candidate pair scaling report miscounts physical parents")
            if report["run_identity"]["train_injection_manifest_sha256"] != plan_records[
                scale
            ]["parent_manifest_sha256"]:
                raise ValueError("candidate pair scaling parent manifest differs from plan")
            if report["run_identity"]["train_candidate_manifest_sha256"] != plan_records[
                scale
            ]["candidate_manifest_sha256"]:
                raise ValueError("candidate pair scaling candidate manifest differs from plan")
            validation_identity = (
                report["run_identity"]["validation_injection_manifest_sha256"],
                report["run_identity"][
                    "validation_selection_candidate_manifest_sha256"
                ],
            )
            common_validation_identity = common_validation_identity or validation_identity
            common_architecture = common_architecture or report["architecture"]
            common_seed = common_seed if common_seed is not None else report["run_identity"]["seed"]
            if (
                validation_identity != common_validation_identity
                or report["architecture"] != common_architecture
                or report["run_identity"]["seed"] != common_seed
            ):
                raise ValueError("candidate pair scaling reports are not paired controls")
            metrics = report["selected_validation_metrics"]
            strata = report["selected_validation_strata"]
            record = {
                "physical_parents": scale,
                "optimizer_updates": int(report["optimizer_updates"]),
                "top1_padded_truth_pair_fraction": float(
                    metrics["top1_padded_truth_pair_fraction"]
                ),
                "top1_peak_p90_seconds": float(
                    metrics["top1_peak_error_seconds_quantiles"]["0.9"]
                ),
                "pair_average_precision": float(metrics["pair_average_precision"]),
                "snr_8_15_top1_padded_truth_pair_fraction": float(
                    strata["snr:snr_8_15"]["top1_padded_truth_pair_fraction"]
                ),
                "report_sha256": file_sha256(paths[scale]),
            }
            reports[mode][scale] = report
            curve.append(record)
        baseline = curve[0]
        for record in curve:
            record["top1_gain_from_smallest"] = (
                record["top1_padded_truth_pair_fraction"]
                - baseline["top1_padded_truth_pair_fraction"]
            )
            record["snr_8_15_top1_gain_from_smallest"] = (
                record["snr_8_15_top1_padded_truth_pair_fraction"]
                - baseline["snr_8_15_top1_padded_truth_pair_fraction"]
            )
            record["peak_p90_reduction_from_smallest_seconds"] = (
                baseline["top1_peak_p90_seconds"] - record["top1_peak_p90_seconds"]
            )
        curves[mode] = curve
    final_records = {mode: curves[mode][-1] for mode in modes}
    checks = {
        "top1_gain_both_controls": all(
            row["top1_gain_from_smallest"]
            >= float(settings["minimum_final_top1_gain"])
            for row in final_records.values()
        ),
        "snr_8_15_gain_both_controls": all(
            row["snr_8_15_top1_gain_from_smallest"]
            >= float(settings["minimum_final_snr_8_15_top1_gain"])
            for row in final_records.values()
        ),
        "peak_p90_reduction_both_controls": all(
            row["peak_p90_reduction_from_smallest_seconds"]
            >= float(settings["minimum_final_peak_p90_reduction_seconds"])
            for row in final_records.values()
        ),
        "top1_nearly_monotonic_both_controls": all(
            all(
                current["top1_padded_truth_pair_fraction"]
                + float(settings["maximum_intermediate_top1_regression"])
                >= previous["top1_padded_truth_pair_fraction"]
                for previous, current in zip(curve, curve[1:])
            )
            for curve in curves.values()
        ),
    }
    representation_gain = all(checks.values())
    fixed_epoch_gain = (
        final_records["fixed_epochs"]["top1_gain_from_smallest"]
        >= float(settings["minimum_final_top1_gain"])
    )
    fixed_update_gain = (
        final_records["fixed_updates"]["top1_gain_from_smallest"]
        >= float(settings["minimum_final_top1_gain"])
    )
    diagnosis = (
        "data_limited_signal"
        if representation_gain
        else "update_limited_signal"
        if fixed_epoch_gain and not fixed_update_gain
        else "representation_or_domain_limited"
    )
    result = {
        "status": "validation_only_candidate_pair_scaling_evaluation",
        "scientific_claim_allowed": False,
        "test_evaluation": None,
        "run_identity": identity,
        "validation_identity": list(common_validation_identity),
        "architecture": common_architecture,
        "seed": common_seed,
        "curves": curves,
        "predeclared_checks": checks,
        "representation_scaling_gate_passed": representation_gain,
        "scaling_diagnosis": diagnosis,
        "scale_beyond_10000_allowed": False,
        "larger_scale_blocker": (
            "fresh group-disjoint O4a calibration, continuous-background FAR/VT, and a positive "
            "fixed-update plus fixed-epoch endpoint are required before 25k/50k"
        ),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


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


def candidate_pair_optimizer_budget(
    settings: dict[str, Any], batches_per_epoch: int
) -> tuple[str, int]:
    """Resolve predeclared fixed-update or fixed-epoch controls."""

    epochs = int(settings["epochs"])
    if epochs <= 0 or batches_per_epoch <= 0:
        raise ValueError("candidate pair ranker epoch geometry is invalid")
    mode = str(settings.get("budget_mode", "fixed_updates"))
    if mode == "fixed_epochs":
        if "max_optimizer_updates" in settings:
            raise ValueError("fixed-epoch candidate ranker must not set max updates")
        updates = epochs * batches_per_epoch
    elif mode == "fixed_updates":
        updates = int(settings["max_optimizer_updates"])
    else:
        raise ValueError("candidate pair ranker budget mode is invalid")
    if updates <= 0 or updates > epochs * batches_per_epoch:
        raise ValueError("candidate pair ranker optimizer budget is invalid")
    return mode, updates


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
    resumable_names = {
        "best_candidate_pair_ranker.pt",
        "last_candidate_pair_ranker.pt",
        "history.json",
        "validation_selection_pair_scores.npz",
    }
    unexpected = sorted(
        path.name for path in output.iterdir() if path.name not in resumable_names
    )
    if unexpected:
        raise FileExistsError(
            f"candidate pair ranker output contains non-resumable files: {unexpected}"
        )
    early_resume_path = output / "last_candidate_pair_ranker.pt"
    if early_resume_path.is_file():
        early_resume = torch.load(
            early_resume_path, map_location="cpu", weights_only=False
        )
        if early_resume.get("run_identity") != identity:
            raise ValueError("candidate pair ranker resume identity differs")
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
    budget_mode, maximum_updates = candidate_pair_optimizer_budget(
        settings, len(loader)
    )
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
        "train_physical_parents": len(train_examples["parent_ids"]),
        "train_unique_waveforms": len(
            {str(row["waveform_id"]) for row in train_parents}
        ),
        "train_unique_gps_blocks": len(
            {str(row["gps_block"]) for row in train_parents}
        ),
        "train_candidate_rows": len(train_candidates),
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
        "budget_mode": budget_mode,
        "max_optimizer_updates": maximum_updates,
        "training_budget_reached": updates == maximum_updates,
        "resumable_epoch_checkpoints": True,
        "strain_crop_shape": (
            list(train_examples["strain_crops"].shape[1:])
            if use_time_frequency_encoder
            else None
        ),
        "strain_crop_storage_dtype": (
            str(train_examples["strain_crops"].dtype)
            if use_time_frequency_encoder
            else None
        ),
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
