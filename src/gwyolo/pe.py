from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, file_sha256
from .metrics import wilson_interval


PUBLICATION_PROVENANCE_FIELDS = (
    "backend_version",
    "backend_model_hash",
    "prior_hash",
    "waveform_approximant",
    "detector_set",
    "calibration_version",
    "source_event_hash",
    "hardware",
    "latency_scope",
)


def _paired_mean_bootstrap(
    values: list[float], replicates: int, seed: int
) -> dict[str, Any]:
    """Summarize a paired per-event delta with a deterministic percentile bootstrap."""
    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("paired bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[indices].mean(axis=1)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "paired_bootstrap_95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _validate_publication_provenance(row: dict[str, Any]) -> None:
    missing = [field for field in PUBLICATION_PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"PE row is missing publication provenance: {missing}")
    empty = [field for field in PUBLICATION_PROVENANCE_FIELDS if row[field] in (None, "", [])]
    if empty:
        raise ValueError(f"PE row has empty publication provenance: {empty}")


def _load_posterior(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix != ".npz":
        raise ValueError("PE adapter currently requires numeric NPZ posterior files")
    with np.load(source, allow_pickle=False) as arrays:
        posterior = {
            key: np.asarray(arrays[key], dtype=np.float64).reshape(-1) for key in arrays.files
        }
    if not posterior or any(values.size == 0 for values in posterior.values()):
        raise ValueError(f"Empty posterior in {path}")
    if any(not np.isfinite(values).all() for values in posterior.values()):
        raise ValueError(f"Non-finite posterior samples in {path}")
    return posterior


def posterior_truth_metrics(
    posterior: dict[str, np.ndarray], truth: dict[str, float], credible_level: float = 0.9
) -> dict[str, Any]:
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must be between zero and one")
    missing = sorted(set(truth) - set(posterior))
    if missing:
        raise ValueError(f"Posterior is missing truth parameters: {missing}")
    tail = (1.0 - credible_level) / 2.0
    metrics = {}
    for parameter, true_value in sorted(truth.items()):
        samples = posterior[parameter]
        lower, median, upper = np.quantile(samples, [tail, 0.5, 1.0 - tail])
        mean = float(np.mean(samples))
        standard_deviation = float(np.std(samples))
        metrics[parameter] = {
            "truth": float(true_value),
            "mean": mean,
            "median": float(median),
            "bias": mean - float(true_value),
            "absolute_bias": abs(mean - float(true_value)),
            "posterior_std": standard_deviation,
            "credible_interval": [float(lower), float(upper)],
            "credible_width": float(upper - lower),
            "covered": bool(lower <= true_value <= upper),
            "mean_absolute_distance_to_truth": float(
                np.mean(np.abs(samples - float(true_value)))
            ),
        }
    return metrics


def evaluate_pe_rows(
    rows: list[dict[str, Any]],
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260719,
    require_publication_provenance: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("PE manifest cannot be empty")
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    evaluated_rows = []
    for row in rows:
        if require_publication_provenance:
            _validate_publication_provenance(row)
        backend = str(row["backend"])
        injection_id = str(row["injection_id"])
        condition = str(row["condition"])
        if condition not in {"raw", "cleaned"}:
            raise ValueError(f"Unsupported PE condition: {condition}")
        key = (backend, injection_id)
        if condition in groups[key]:
            raise ValueError(f"Duplicate {condition} PE row for {backend}/{injection_id}")
        truth = {str(name): float(value) for name, value in row["truth"].items()}
        posterior = _load_posterior(row["posterior_path"])
        latency = float(row["latency_seconds"])
        if not np.isfinite(latency) or latency < 0:
            raise ValueError(f"Invalid PE latency for {backend}/{injection_id}: {latency}")
        evaluated = {
            **row,
            "latency_seconds": latency,
            "truth": truth,
            "posterior_sha256": file_sha256(row["posterior_path"]),
            "parameters": posterior_truth_metrics(posterior, truth, credible_level),
        }
        groups[key][condition] = evaluated
        evaluated_rows.append(evaluated)

    incomplete = [f"{backend}/{injection}" for (backend, injection), pair in groups.items() if set(pair) != {"raw", "cleaned"}]
    if incomplete:
        raise ValueError(f"Missing raw/cleaned PE pairs: {incomplete[:10]}")
    comparisons = []
    for (backend, injection_id), pair in sorted(groups.items()):
        raw = pair["raw"]
        cleaned = pair["cleaned"]
        if raw["truth"] != cleaned["truth"]:
            raise ValueError(f"Truth mismatch for {backend}/{injection_id}")
        if require_publication_provenance:
            inconsistent = [
                field
                for field in PUBLICATION_PROVENANCE_FIELDS
                if raw[field] != cleaned[field]
            ]
            if inconsistent:
                raise ValueError(
                    f"Raw/cleaned publication provenance mismatch for "
                    f"{backend}/{injection_id}: {inconsistent}"
                )
        parameter_changes = {}
        for parameter in sorted(raw["truth"]):
            raw_metrics = raw["parameters"][parameter]
            cleaned_metrics = cleaned["parameters"][parameter]
            parameter_changes[parameter] = {
                "absolute_bias_change_cleaned_minus_raw": (
                    cleaned_metrics["absolute_bias"] - raw_metrics["absolute_bias"]
                ),
                "credible_width_ratio_cleaned_over_raw": (
                    cleaned_metrics["credible_width"] / raw_metrics["credible_width"]
                    if raw_metrics["credible_width"] > 0
                    else None
                ),
                "raw_covered": raw_metrics["covered"],
                "cleaned_covered": cleaned_metrics["covered"],
                "coverage_change_cleaned_minus_raw": int(cleaned_metrics["covered"])
                - int(raw_metrics["covered"]),
            }
        comparisons.append(
            {
                "backend": backend,
                "injection_id": injection_id,
                "event_id": raw.get("event_id"),
                "raw_latency_seconds": float(raw["latency_seconds"]),
                "cleaned_latency_seconds": float(cleaned["latency_seconds"]),
                "cleaning_latency_overhead_seconds": float(cleaned["latency_seconds"])
                - float(raw["latency_seconds"]),
                "parameters": parameter_changes,
            }
        )

    coverage = {}
    for backend in sorted({str(row["backend"]) for row in evaluated_rows}):
        coverage[backend] = {}
        backend_rows = [row for row in evaluated_rows if row["backend"] == backend]
        parameters = sorted({name for row in backend_rows for name in row["parameters"]})
        for condition in ("raw", "cleaned"):
            condition_rows = [row for row in backend_rows if row["condition"] == condition]
            coverage[backend][condition] = {}
            for parameter in parameters:
                available = [row for row in condition_rows if parameter in row["parameters"]]
                successes = sum(row["parameters"][parameter]["covered"] for row in available)
                interval = wilson_interval(successes, len(available))
                coverage[backend][condition][parameter] = {
                    "covered": successes,
                    "total": len(available),
                    "rate": successes / len(available) if available else None,
                    "wilson_95": list(interval) if available else [None, None],
                }
    paired_summaries = {}
    for backend_index, backend in enumerate(
        sorted({comparison["backend"] for comparison in comparisons})
    ):
        backend_comparisons = [row for row in comparisons if row["backend"] == backend]
        parameter_names = sorted(
            {name for row in backend_comparisons for name in row["parameters"]}
        )
        parameter_summaries = {}
        for parameter_index, parameter in enumerate(parameter_names):
            parameter_rows = [
                row["parameters"][parameter]
                for row in backend_comparisons
                if parameter in row["parameters"]
            ]
            seed_offset = bootstrap_seed + 10_000 * backend_index + 100 * parameter_index
            parameter_summaries[parameter] = {
                "absolute_bias_change_cleaned_minus_raw": _paired_mean_bootstrap(
                    [row["absolute_bias_change_cleaned_minus_raw"] for row in parameter_rows],
                    bootstrap_replicates,
                    seed_offset,
                ),
                "credible_width_ratio_cleaned_over_raw": _paired_mean_bootstrap(
                    [
                        row["credible_width_ratio_cleaned_over_raw"]
                        for row in parameter_rows
                        if row["credible_width_ratio_cleaned_over_raw"] is not None
                    ],
                    bootstrap_replicates,
                    seed_offset + 1,
                ),
                "coverage_transitions": dict(
                    sorted(
                        Counter(
                            f"{int(row['raw_covered'])}->{int(row['cleaned_covered'])}"
                            for row in parameter_rows
                        ).items()
                    )
                ),
            }
        paired_summaries[backend] = {
            "paired_events": len(backend_comparisons),
            "cleaning_latency_overhead_seconds": _paired_mean_bootstrap(
                [row["cleaning_latency_overhead_seconds"] for row in backend_comparisons],
                bootstrap_replicates,
                bootstrap_seed + 10_000 * backend_index + 9_999,
            ),
            "parameters": parameter_summaries,
        }
    return {
        "protocol": "paired raw/cleaned posterior evaluation on identical injections and truth",
        "credible_level": credible_level,
        "rows": len(evaluated_rows),
        "paired_injections": len(comparisons),
        "backend_counts": dict(sorted(Counter(row["backend"] for row in evaluated_rows).items())),
        "coverage": coverage,
        "paired_summaries": paired_summaries,
        "publication_provenance_required": require_publication_provenance,
        "publication_provenance_fields": list(PUBLICATION_PROVENANCE_FIELDS),
        "comparisons": comparisons,
        "evaluated_rows": evaluated_rows,
    }


ROBUSTNESS_CONDITIONS = ("clean", "contaminated", "mask_conditioned")


def _positive_optional(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"PE {field} must be finite and positive")
    return number


def evaluate_pe_robustness_rows(
    rows: list[dict[str, Any]],
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260720,
    require_publication_provenance: bool = True,
) -> dict[str, Any]:
    """Evaluate clean/contaminated/mask-conditioned PE triplets without changing priors."""
    if not rows:
        raise ValueError("PE robustness manifest cannot be empty")
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    evaluated = []
    for row in rows:
        if require_publication_provenance:
            _validate_publication_provenance(row)
        condition = str(row["condition"])
        if condition not in ROBUSTNESS_CONDITIONS:
            raise ValueError(f"Unsupported PE robustness condition: {condition}")
        backend = str(row["backend"])
        injection_id = str(row["injection_id"])
        key = (backend, injection_id)
        if condition in groups[key]:
            raise ValueError(f"Duplicate {condition} PE row for {backend}/{injection_id}")
        posterior = _load_posterior(row["posterior_path"])
        sample_counts = {values.size for values in posterior.values()}
        if len(sample_counts) != 1:
            raise ValueError("all parameters in one posterior must have equal sample count")
        sample_count = next(iter(sample_counts))
        effective_sample_size = _positive_optional(row, "effective_sample_size")
        sky_area = _positive_optional(row, "sky_area_90_deg2")
        if require_publication_provenance and effective_sample_size is None:
            raise ValueError("publication PE robustness rows require effective_sample_size")
        if require_publication_provenance and sky_area is None:
            raise ValueError("publication PE robustness rows require sky_area_90_deg2")
        if effective_sample_size is not None and effective_sample_size > sample_count:
            raise ValueError("effective_sample_size cannot exceed posterior sample count")
        latency = float(row["latency_seconds"])
        if not np.isfinite(latency) or latency <= 0:
            raise ValueError("PE robustness latency_seconds must be finite and positive")
        truth = {str(name): float(value) for name, value in row["truth"].items()}
        record = {
            **row,
            "truth": truth,
            "latency_seconds": latency,
            "posterior_sha256": file_sha256(row["posterior_path"]),
            "posterior_sample_count": sample_count,
            "effective_sample_size": effective_sample_size,
            "effective_samples_per_second": (
                effective_sample_size / latency
                if effective_sample_size is not None
                else None
            ),
            "sky_area_90_deg2": sky_area,
            "parameters": posterior_truth_metrics(posterior, truth, credible_level),
        }
        groups[key][condition] = record
        evaluated.append(record)
    incomplete = [
        f"{backend}/{injection_id}"
        for (backend, injection_id), triplet in groups.items()
        if set(triplet) != set(ROBUSTNESS_CONDITIONS)
    ]
    if incomplete:
        raise ValueError(f"Missing clean/contaminated/mask-conditioned PE triplets: {incomplete[:10]}")

    comparisons = []
    for (backend, injection_id), triplet in sorted(groups.items()):
        clean = triplet["clean"]
        contaminated = triplet["contaminated"]
        masked = triplet["mask_conditioned"]
        if not (clean["truth"] == contaminated["truth"] == masked["truth"]):
            raise ValueError(f"PE robustness truth mismatch for {backend}/{injection_id}")
        if require_publication_provenance:
            inconsistent = [
                field
                for field in PUBLICATION_PROVENANCE_FIELDS
                if len({str(triplet[condition][field]) for condition in ROBUSTNESS_CONDITIONS})
                != 1
            ]
            if inconsistent:
                raise ValueError(
                    f"PE robustness provenance mismatch for {backend}/{injection_id}: "
                    f"{inconsistent}"
                )
        parameters = {}
        for parameter in sorted(clean["truth"]):
            clean_metric = clean["parameters"][parameter]
            contaminated_metric = contaminated["parameters"][parameter]
            masked_metric = masked["parameters"][parameter]
            parameters[parameter] = {
                "contamination_absolute_bias_change": (
                    contaminated_metric["absolute_bias"] - clean_metric["absolute_bias"]
                ),
                "mask_absolute_bias_change_vs_contaminated": (
                    masked_metric["absolute_bias"] - contaminated_metric["absolute_bias"]
                ),
                "mask_absolute_bias_change_vs_clean": (
                    masked_metric["absolute_bias"] - clean_metric["absolute_bias"]
                ),
                "contamination_width_ratio_vs_clean": (
                    contaminated_metric["credible_width"] / clean_metric["credible_width"]
                    if clean_metric["credible_width"] > 0
                    else None
                ),
                "mask_width_ratio_vs_contaminated": (
                    masked_metric["credible_width"]
                    / contaminated_metric["credible_width"]
                    if contaminated_metric["credible_width"] > 0
                    else None
                ),
                "mask_width_ratio_vs_clean": (
                    masked_metric["credible_width"] / clean_metric["credible_width"]
                    if clean_metric["credible_width"] > 0
                    else None
                ),
                "coverage_transition": (
                    f"{int(clean_metric['covered'])}->"
                    f"{int(contaminated_metric['covered'])}->"
                    f"{int(masked_metric['covered'])}"
                ),
            }
        def ratio(field: str, numerator: str, denominator: str) -> float | None:
            top = triplet[numerator][field]
            bottom = triplet[denominator][field]
            return float(top / bottom) if top is not None and bottom not in (None, 0) else None

        comparisons.append(
            {
                "backend": backend,
                "injection_id": injection_id,
                "event_id": clean.get("event_id"),
                "contamination_stratum": contaminated.get("contamination_stratum", "all"),
                "parameters": parameters,
                "latency_mask_minus_contaminated_seconds": (
                    masked["latency_seconds"] - contaminated["latency_seconds"]
                ),
                "ess_rate_mask_over_contaminated": ratio(
                    "effective_samples_per_second", "mask_conditioned", "contaminated"
                ),
                "sky_area_mask_over_contaminated": ratio(
                    "sky_area_90_deg2", "mask_conditioned", "contaminated"
                ),
                "sky_area_contaminated_over_clean": ratio(
                    "sky_area_90_deg2", "contaminated", "clean"
                ),
            }
        )

    coverage = {}
    paired_summaries = {}
    for backend_index, backend in enumerate(sorted({row["backend"] for row in evaluated})):
        backend_rows = [row for row in evaluated if row["backend"] == backend]
        coverage[backend] = {}
        for condition in ROBUSTNESS_CONDITIONS:
            selected = [row for row in backend_rows if row["condition"] == condition]
            coverage[backend][condition] = {}
            for parameter in sorted(selected[0]["parameters"]):
                successes = sum(row["parameters"][parameter]["covered"] for row in selected)
                coverage[backend][condition][parameter] = {
                    "covered": successes,
                    "total": len(selected),
                    "rate": successes / len(selected),
                    "wilson_95": list(wilson_interval(successes, len(selected))),
                }
        backend_comparisons = [row for row in comparisons if row["backend"] == backend]
        parameter_summaries = {}
        for parameter_index, parameter in enumerate(
            sorted(backend_comparisons[0]["parameters"])
        ):
            parameter_rows = [row["parameters"][parameter] for row in backend_comparisons]
            parameter_summaries[parameter] = {}
            for metric_index, metric in enumerate(
                (
                    "contamination_absolute_bias_change",
                    "mask_absolute_bias_change_vs_contaminated",
                    "mask_absolute_bias_change_vs_clean",
                    "contamination_width_ratio_vs_clean",
                    "mask_width_ratio_vs_contaminated",
                    "mask_width_ratio_vs_clean",
                )
            ):
                values = [row[metric] for row in parameter_rows if row[metric] is not None]
                parameter_summaries[parameter][metric] = (
                    _paired_mean_bootstrap(
                        values,
                        bootstrap_replicates,
                        bootstrap_seed
                        + backend_index * 100_000
                        + parameter_index * 100
                        + metric_index,
                    )
                    if values
                    else None
                )
            parameter_summaries[parameter]["coverage_transitions"] = dict(
                sorted(Counter(row["coverage_transition"] for row in parameter_rows).items())
            )
        resource_summaries = {}
        for metric_index, metric in enumerate(
            (
                "latency_mask_minus_contaminated_seconds",
                "ess_rate_mask_over_contaminated",
                "sky_area_mask_over_contaminated",
                "sky_area_contaminated_over_clean",
            )
        ):
            values = [row[metric] for row in backend_comparisons if row[metric] is not None]
            resource_summaries[metric] = (
                _paired_mean_bootstrap(
                    values,
                    bootstrap_replicates,
                    bootstrap_seed + backend_index * 100_000 + 90_000 + metric_index,
                )
                if values
                else None
            )
        paired_summaries[backend] = {
            "paired_injections": len(backend_comparisons),
            "parameters": parameter_summaries,
            "resources": resource_summaries,
        }
    backend_names = sorted(paired_summaries)
    return {
        "status": "paired_pe_contamination_mask_robustness",
        "scientific_claim_allowed": False,
        "protocol": (
            "paired clean/contaminated/mask-conditioned inference with identical backend, prior, "
            "waveform and detector assumptions"
        ),
        "conditions": list(ROBUSTNESS_CONDITIONS),
        "credible_level": credible_level,
        "backends": backend_names,
        "dingo_amplfi_joint_gate": {"DINGO", "AMPLFI"}.issubset(
            {name.upper() for name in backend_names}
        ),
        "triplets": len(comparisons),
        "coverage": coverage,
        "paired_summaries": paired_summaries,
        "comparisons": comparisons,
        "publication_provenance_required": require_publication_provenance,
        "publication_provenance_fields": list(PUBLICATION_PROVENANCE_FIELDS),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def run_pe_robustness_evaluation(
    manifest_path: str | Path,
    output_path: str | Path,
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260720,
    require_publication_provenance: bool = True,
) -> dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = evaluate_pe_robustness_rows(
        rows,
        credible_level,
        bootstrap_replicates,
        bootstrap_seed,
        require_publication_provenance,
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = file_sha256(manifest_path)
    atomic_write_json(output_path, report)
    return report


def run_pe_evaluation(
    manifest_path: str | Path,
    output_path: str | Path,
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260719,
    require_publication_provenance: bool = False,
) -> dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = evaluate_pe_rows(
        rows,
        credible_level,
        bootstrap_replicates,
        bootstrap_seed,
        require_publication_provenance,
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = file_sha256(manifest_path)
    atomic_write_json(output_path, report)
    return report
