from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .injection_bootstrap import hierarchical_injection_bootstrap
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .runtime import execution_provenance


PAIRED_PE_LATENCY_SCOPE_V1 = (
    "model-load-and-event-preprocessing-through-posterior-and-native-result-write_"
    "v1_excludes-artifact-verification-imports-and-mask-generation"
)

PAIRED_PE_LATENCY_COMPONENT_FIELDS = (
    "model_load",
    "event_preprocessing",
    "posterior_sampling",
    "posterior_postprocessing_and_write",
)

PERIODIC_POSTERIOR_PARAMETERS = {
    "ra": 2 * np.pi,
    "psi": np.pi,
}


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
    "sky_area_estimator",
)

PUBLICATION_INPUT_FIELDS = (
    "analysis_input_path",
    "analysis_input_sha256",
    "input_sample_rate_hz",
    "input_duration_seconds",
    "input_post_trigger_seconds",
    "input_ifos",
    "base_injection_manifest_path",
    "base_injection_manifest_sha256",
    "native_conditioning_path",
    "native_conditioning_sha256",
    "native_conditioning_config_path",
    "native_conditioning_config_sha256",
)

PUBLICATION_SHARED_BACKEND_FIELDS = (
    "waveform_id",
    "gps_block",
    "prior_hash",
    "waveform_approximant",
    "detector_set",
    "calibration_version",
    "source_event_hash",
    "hardware",
    "latency_scope",
    "sky_area_estimator",
)

SKY_AREA_ESTIMATOR_IDENTITY_FIELDS = (
    "method",
    "credible_level",
    "ra_bins",
    "sin_dec_bins",
    "total_pixels",
    "pixel_area_deg2",
    "coordinate_units",
    "interpretation",
)


def validate_paired_pe_latency(report: dict[str, Any]) -> dict[str, float]:
    """Validate a truthful, shared end-to-end PE inference timing contract."""

    if report.get("latency_scope") != PAIRED_PE_LATENCY_SCOPE_V1:
        raise ValueError("PE backend latency scope differs from the frozen paired contract")
    try:
        total = float(report["latency_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PE backend latency_seconds is absent or invalid") from error
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PE backend latency_seconds must be finite and positive")
    raw_components = report.get("latency_components_seconds")
    if not isinstance(raw_components, dict) or set(raw_components) != set(
        PAIRED_PE_LATENCY_COMPONENT_FIELDS
    ):
        raise ValueError("PE backend latency components differ from the frozen contract")
    components: dict[str, float] = {}
    for field in PAIRED_PE_LATENCY_COMPONENT_FIELDS:
        try:
            value = float(raw_components[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"PE backend latency component {field} is invalid") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"PE backend latency component {field} must be finite and non-negative"
            )
        components[field] = value
    tolerance = max(1e-6, total * 1e-6)
    if sum(components.values()) > total + tolerance:
        raise ValueError("PE backend latency components exceed the measured total")
    return components


def posterior_sky_area_equal_solid_angle(
    ra: np.ndarray,
    dec: np.ndarray,
    credible_level: float = 0.9,
    ra_bins: int = 360,
    sin_dec_bins: int = 180,
) -> dict[str, Any]:
    """Estimate a greedy sky credible area on a fixed equal-solid-angle grid."""

    right_ascension = np.asarray(ra, dtype=np.float64).reshape(-1)
    declination = np.asarray(dec, dtype=np.float64).reshape(-1)
    if right_ascension.shape != declination.shape or right_ascension.size == 0:
        raise ValueError("sky area requires equally sized non-empty RA/Dec samples")
    if not np.isfinite(right_ascension).all() or not np.isfinite(declination).all():
        raise ValueError("sky posterior coordinates must be finite")
    if np.any(np.abs(declination) > np.pi / 2 + 1e-12):
        raise ValueError("sky posterior declination lies outside [-pi/2, pi/2]")
    if not 0 < credible_level <= 1:
        raise ValueError("sky credible level must lie in (0, 1]")
    if ra_bins < 1 or sin_dec_bins < 1:
        raise ValueError("sky-area grid dimensions must be positive")
    wrapped_ra = np.mod(right_ascension, 2 * np.pi)
    ra_index = np.minimum(
        (wrapped_ra / (2 * np.pi) * ra_bins).astype(np.int64), ra_bins - 1
    )
    sin_dec = np.clip(np.sin(declination), -1.0, 1.0)
    dec_index = np.minimum(
        ((sin_dec + 1.0) / 2.0 * sin_dec_bins).astype(np.int64),
        sin_dec_bins - 1,
    )
    flat = dec_index * ra_bins + ra_index
    counts = np.bincount(flat, minlength=ra_bins * sin_dec_bins)
    occupied = counts[counts > 0]
    descending = np.sort(occupied)[::-1]
    required_samples = int(np.ceil(credible_level * right_ascension.size))
    credible_pixels = int(
        np.searchsorted(np.cumsum(descending), required_samples, side="left") + 1
    )
    square_degrees_per_steradian = (180.0 / np.pi) ** 2
    pixel_area = (
        4.0
        * np.pi
        / (ra_bins * sin_dec_bins)
        * square_degrees_per_steradian
    )
    return {
        "method": "fixed_equal_solid_angle_histogram_v1",
        "credible_level": credible_level,
        "sample_count": int(right_ascension.size),
        "ra_bins": ra_bins,
        "sin_dec_bins": sin_dec_bins,
        "total_pixels": ra_bins * sin_dec_bins,
        "occupied_pixels": int(occupied.size),
        "credible_pixels": credible_pixels,
        "pixel_area_deg2": pixel_area,
        "area_deg2": credible_pixels * pixel_area,
        "coordinate_units": "radians",
        "interpretation": (
            "fixed-grid posterior credible area; not a BAYESTAR or adaptive HEALPix sky map"
        ),
    }


def sky_area_estimator_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Separate frozen estimator settings from posterior-dependent diagnostics."""

    missing = [field for field in SKY_AREA_ESTIMATOR_IDENTITY_FIELDS if field not in report]
    if missing:
        raise ValueError(f"sky-area report lacks estimator identity fields: {missing}")
    return {field: report[field] for field in SKY_AREA_ESTIMATOR_IDENTITY_FIELDS}


def _paired_mean_bootstrap(
    values: list[float],
    replicates: int,
    seed: int,
    event_rows: list[dict[str, Any]] | None = None,
    minimum_physical_groups: int = 2,
) -> dict[str, Any]:
    """Summarize a paired event delta with auditable physical-noise resampling."""
    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("paired bootstrap values must be finite")
    records = event_rows if event_rows is not None else [{} for _ in values]
    if len(records) != len(values):
        raise ValueError("paired bootstrap event rows must align with values")
    bootstrap = hierarchical_injection_bootstrap(
        records,
        array,
        np.ones(array.size, dtype=np.float64),
        replicates,
        seed,
        minimum_physical_groups=minimum_physical_groups,
    )
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "paired_bootstrap_95": bootstrap["interval_95"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_independence": bootstrap["independence_audit"],
    }


def _combine_pe_bootstrap_audits(
    audits: dict[str, dict[str, Any]], minimum_physical_groups: int
) -> dict[str, Any]:
    if not audits:
        raise ValueError("PE bootstrap audit requires at least one backend")
    methods = {str(value.get("method")) for value in audits.values()}
    physical_groups = [int(value.get("physical_groups", 0)) for value in audits.values()]
    return {
        "status": "paired_pe_bootstrap_independence_audit_v1",
        "passed": all(value.get("passed") is True for value in audits.values())
        and len(methods) == 1,
        "method": next(iter(methods)) if len(methods) == 1 else "inconsistent",
        "minimum_physical_groups": minimum_physical_groups,
        "physical_groups": min(physical_groups),
        "backend_audits": dict(sorted(audits.items())),
    }


def _validate_publication_provenance(row: dict[str, Any]) -> None:
    missing = [field for field in PUBLICATION_PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"PE row is missing publication provenance: {missing}")
    empty = [field for field in PUBLICATION_PROVENANCE_FIELDS if row[field] in (None, "", [])]
    if empty:
        raise ValueError(f"PE row has empty publication provenance: {empty}")


def _verified_file_identity(
    row: dict[str, Any], path_field: str, hash_field: str
) -> dict[str, str]:
    path = Path(row[path_field]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PE provenance file does not exist: {path}")
    observed = file_sha256(path)
    if str(row[hash_field]) != observed:
        raise ValueError(f"PE {hash_field} does not match {path_field}")
    return {"path": str(path), "sha256": observed}


def _validate_publication_input_provenance(
    row: dict[str, Any], condition: str
) -> dict[str, Any]:
    missing = [field for field in PUBLICATION_INPUT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"PE row is missing publication input provenance: {missing}")
    empty = [field for field in PUBLICATION_INPUT_FIELDS if row[field] in (None, "", [])]
    if empty:
        raise ValueError(f"PE row has empty publication input provenance: {empty}")
    sample_rate = float(row["input_sample_rate_hz"])
    duration = float(row["input_duration_seconds"])
    post_trigger = float(row["input_post_trigger_seconds"])
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("PE input sample rate must be finite and positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("PE input duration must be finite and positive")
    if not np.isfinite(post_trigger) or not 0 < post_trigger < duration:
        raise ValueError("PE input post-trigger duration must lie inside the input window")
    input_ifos = tuple(str(value) for value in row["input_ifos"])
    detector_set = tuple(str(value) for value in row["detector_set"])
    if not input_ifos or len(set(input_ifos)) != len(input_ifos):
        raise ValueError("PE input IFOs must be non-empty and unique")
    if input_ifos != detector_set:
        raise ValueError("PE input IFOs differ from the declared detector set")
    identity: dict[str, Any] = {
        "condition": condition,
        "analysis_input": _verified_file_identity(
            row, "analysis_input_path", "analysis_input_sha256"
        ),
        "input_sample_rate_hz": sample_rate,
        "input_duration_seconds": duration,
        "input_post_trigger_seconds": post_trigger,
        "input_ifos": list(input_ifos),
        "base_injection_manifest": _verified_file_identity(
            row,
            "base_injection_manifest_path",
            "base_injection_manifest_sha256",
        ),
        "native_conditioning": _verified_file_identity(
            row, "native_conditioning_path", "native_conditioning_sha256"
        ),
        "native_conditioning_config": _verified_file_identity(
            row,
            "native_conditioning_config_path",
            "native_conditioning_config_sha256",
        ),
    }
    if condition in {"contaminated", "mask_conditioned"}:
        fields = (
            "glitch_id",
            "contamination_manifest_path",
            "contamination_manifest_sha256",
        )
        missing = [field for field in fields if row.get(field) in (None, "", [])]
        if missing:
            raise ValueError(f"PE contaminated input lacks lineage: {missing}")
        identity.update(
            {
                "glitch_id": str(row["glitch_id"]),
                "contamination_manifest": _verified_file_identity(
                    row,
                    "contamination_manifest_path",
                    "contamination_manifest_sha256",
                ),
            }
        )
    if condition == "mask_conditioned":
        fields = (
            "mask_conditioning_mode",
            "mask_artifact_path",
            "mask_artifact_sha256",
            "mask_model_path",
            "mask_model_sha256",
            "mask_policy_path",
            "mask_policy_sha256",
        )
        missing = [field for field in fields if row.get(field) in (None, "", [])]
        if missing:
            raise ValueError(f"PE mask-conditioned input lacks lineage: {missing}")
        mode = str(row["mask_conditioning_mode"])
        if mode not in {"cleaned_strain", "auxiliary_mask"}:
            raise ValueError("PE mask conditioning mode is unsupported")
        identity.update(
            {
                "mask_conditioning_mode": mode,
                "mask_artifact": _verified_file_identity(
                    row, "mask_artifact_path", "mask_artifact_sha256"
                ),
                "mask_model": _verified_file_identity(
                    row, "mask_model_path", "mask_model_sha256"
                ),
                "mask_policy": _verified_file_identity(
                    row, "mask_policy_path", "mask_policy_sha256"
                ),
            }
        )
    return identity


def _publication_input_semantics(identity: dict[str, Any]) -> dict[str, Any]:
    """Discard machine-local paths while retaining every content identity."""
    result = {
        "condition": identity["condition"],
        "analysis_input_sha256": identity["analysis_input"]["sha256"],
        "input_sample_rate_hz": identity["input_sample_rate_hz"],
        "input_duration_seconds": identity["input_duration_seconds"],
        "input_post_trigger_seconds": identity["input_post_trigger_seconds"],
        "input_ifos": identity["input_ifos"],
        "base_injection_manifest_sha256": identity["base_injection_manifest"]["sha256"],
        "native_conditioning_sha256": identity["native_conditioning"]["sha256"],
        "native_conditioning_config_sha256": identity["native_conditioning_config"][
            "sha256"
        ],
    }
    for field in ("glitch_id", "mask_conditioning_mode"):
        if field in identity:
            result[field] = identity[field]
    for field in ("contamination_manifest", "mask_artifact", "mask_model", "mask_policy"):
        if field in identity:
            result[f"{field}_sha256"] = identity[field]["sha256"]
    return result


def _common_source_semantics(identity: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match across backends before native conditioning diverges."""
    return {
        key: value
        for key, value in _publication_input_semantics(identity).items()
        if key
        not in {
            "native_conditioning_sha256",
            "native_conditioning_config_sha256",
        }
    }


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
        period = PERIODIC_POSTERIOR_PARAMETERS.get(parameter)
        if period is None:
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
                "periodic": False,
            }
            continue
        wrapped = np.mod(samples, period)
        phase = wrapped * (2 * np.pi / period)
        cosine = float(np.mean(np.cos(phase)))
        sine = float(np.mean(np.sin(phase)))
        mean = float(np.mod(np.arctan2(sine, cosine), 2 * np.pi) * period / (2 * np.pi))
        if np.isclose(mean, period, rtol=0, atol=1e-12):
            mean = 0.0
        residuals = np.mod(wrapped - mean + period / 2, period) - period / 2
        lower, median_residual, upper = np.quantile(
            residuals, [tail, 0.5, 1.0 - tail]
        )
        truth_wrapped = float(np.mod(true_value, period))
        truth_residual = float(
            np.mod(truth_wrapped - mean + period / 2, period) - period / 2
        )
        bias = float(np.mod(mean - truth_wrapped + period / 2, period) - period / 2)
        lower_wrapped = float(np.mod(mean + lower, period))
        upper_wrapped = float(np.mod(mean + upper, period))
        truth_distances = np.abs(
            np.mod(wrapped - truth_wrapped + period / 2, period) - period / 2
        )
        metrics[parameter] = {
            "truth": truth_wrapped,
            "mean": mean,
            "median": float(np.mod(mean + median_residual, period)),
            "bias": bias,
            "absolute_bias": abs(bias),
            "posterior_std": float(np.std(residuals)),
            "credible_interval": [lower_wrapped, upper_wrapped],
            "credible_interval_wraps": lower_wrapped > upper_wrapped,
            "credible_interval_residual_to_circular_mean": [
                float(lower),
                float(upper),
            ],
            "credible_width": float(upper - lower),
            "covered": bool(lower <= truth_residual <= upper),
            "mean_absolute_distance_to_truth": float(np.mean(truth_distances)),
            "periodic": True,
            "period": float(period),
            "circular_resultant_length": float(np.hypot(cosine, sine)),
        }
    return metrics


def evaluate_pe_rows(
    rows: list[dict[str, Any]],
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260719,
    require_publication_provenance: bool = False,
    minimum_physical_groups: int = 2,
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
        if any(
            raw.get(field) != cleaned.get(field)
            for field in ("waveform_id", "gps_block")
        ):
            raise ValueError(
                f"Physical event identity mismatch for {backend}/{injection_id}"
            )
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
                "waveform_id": raw.get("waveform_id"),
                "gps_block": raw.get("gps_block"),
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
            parameter_comparisons = [
                row
                for row in backend_comparisons
                if parameter in row["parameters"]
            ]
            parameter_rows = [
                row["parameters"][parameter] for row in parameter_comparisons
            ]
            seed_offset = bootstrap_seed + 10_000 * backend_index + 100 * parameter_index
            parameter_summaries[parameter] = {
                "absolute_bias_change_cleaned_minus_raw": _paired_mean_bootstrap(
                    [row["absolute_bias_change_cleaned_minus_raw"] for row in parameter_rows],
                    bootstrap_replicates,
                    seed_offset,
                    parameter_comparisons,
                    minimum_physical_groups,
                ),
                "credible_width_ratio_cleaned_over_raw": _paired_mean_bootstrap(
                    [
                        row["credible_width_ratio_cleaned_over_raw"]
                        for row in parameter_rows
                        if row["credible_width_ratio_cleaned_over_raw"] is not None
                    ],
                    bootstrap_replicates,
                    seed_offset + 1,
                    [
                        comparison
                        for comparison, row in zip(
                            parameter_comparisons, parameter_rows
                        )
                        if row["credible_width_ratio_cleaned_over_raw"] is not None
                    ],
                    minimum_physical_groups,
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
                backend_comparisons,
                minimum_physical_groups,
            ),
            "parameters": parameter_summaries,
        }
    bootstrap_audits = {
        backend: summary["cleaning_latency_overhead_seconds"][
            "bootstrap_independence"
        ]
        for backend, summary in paired_summaries.items()
    }
    return {
        "protocol": "paired raw/cleaned posterior evaluation on identical injections and truth",
        "credible_level": credible_level,
        "rows": len(evaluated_rows),
        "paired_injections": len(comparisons),
        "backend_counts": dict(sorted(Counter(row["backend"] for row in evaluated_rows).items())),
        "coverage": coverage,
        "paired_summaries": paired_summaries,
        "pe_bootstrap_independence": _combine_pe_bootstrap_audits(
            bootstrap_audits, minimum_physical_groups
        ),
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
    require_cross_backend_join: bool = True,
    minimum_physical_groups: int = 2,
) -> dict[str, Any]:
    """Evaluate clean/contaminated/mask-conditioned PE triplets without changing priors."""
    if not require_cross_backend_join and not require_publication_provenance:
        raise ValueError(
            "Within-backend PE robustness requires publication provenance"
        )
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
        verified_input = (
            _validate_publication_input_provenance(row, condition)
            if require_publication_provenance
            else None
        )
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
            "verified_publication_input": verified_input,
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
        if any(
            len(
                {
                    str(triplet[condition].get(field, ""))
                    for condition in ROBUSTNESS_CONDITIONS
                }
            )
            != 1
            for field in ("waveform_id", "gps_block")
        ):
            raise ValueError(
                f"PE robustness physical identity mismatch for {backend}/{injection_id}"
            )
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
            input_semantics = {
                condition: _publication_input_semantics(
                    triplet[condition]["verified_publication_input"]
                )
                for condition in ROBUSTNESS_CONDITIONS
            }
            shared_fields = (
                "input_sample_rate_hz",
                "input_duration_seconds",
                "input_post_trigger_seconds",
                "input_ifos",
                "base_injection_manifest_sha256",
                "native_conditioning_config_sha256",
            )
            inconsistent_inputs = [
                field
                for field in shared_fields
                if len(
                    {
                        json.dumps(input_semantics[condition][field], sort_keys=True)
                        for condition in ROBUSTNESS_CONDITIONS
                    }
                )
                != 1
            ]
            contaminated_input_semantics = input_semantics["contaminated"]
            masked_input_semantics = input_semantics["mask_conditioned"]
            if (
                contaminated_input_semantics["glitch_id"]
                != masked_input_semantics["glitch_id"]
                or contaminated_input_semantics["contamination_manifest_sha256"]
                != masked_input_semantics["contamination_manifest_sha256"]
            ):
                inconsistent_inputs.append("contamination_lineage")
            clean_input = input_semantics["clean"]["analysis_input_sha256"]
            contaminated_input = contaminated_input_semantics["analysis_input_sha256"]
            masked_input = masked_input_semantics["analysis_input_sha256"]
            if clean_input == contaminated_input:
                inconsistent_inputs.append("clean_vs_contaminated_input")
            if masked_input_semantics["mask_conditioning_mode"] == "cleaned_strain":
                if masked_input == contaminated_input:
                    inconsistent_inputs.append("cleaned_strain_vs_contaminated_input")
            elif masked_input != contaminated_input:
                inconsistent_inputs.append("auxiliary_mask_strain_input")
            if inconsistent_inputs:
                raise ValueError(
                    f"PE robustness input lineage mismatch for {backend}/{injection_id}: "
                    f"{sorted(set(inconsistent_inputs))}"
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
                "mask_absolute_bias_change_vs_contaminated_normalized_by_clean_width": (
                    (
                        masked_metric["absolute_bias"]
                        - contaminated_metric["absolute_bias"]
                    )
                    / clean_metric["credible_width"]
                    if clean_metric["credible_width"] > 0
                    else None
                ),
                "mask_absolute_bias_change_vs_clean_normalized_by_clean_width": (
                    (masked_metric["absolute_bias"] - clean_metric["absolute_bias"])
                    / clean_metric["credible_width"]
                    if clean_metric["credible_width"] > 0
                    else None
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
                "coverage_mask_minus_contaminated": float(
                    masked_metric["covered"] - contaminated_metric["covered"]
                ),
                "coverage_mask_minus_clean": float(
                    masked_metric["covered"] - clean_metric["covered"]
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
                "waveform_id": clean.get("waveform_id"),
                "gps_block": clean.get("gps_block"),
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

    cross_backend_input_gate = False
    common_injection_ids: list[str] = []
    if require_publication_provenance:
        normalized_backends: dict[str, str] = {}
        for backend, _ in groups:
            normalized = backend.upper()
            prior = normalized_backends.setdefault(normalized, backend)
            if prior != backend:
                raise ValueError("PE robustness backend names collide after normalization")
        if require_cross_backend_join:
            missing_backends = sorted({"DINGO", "AMPLFI"} - set(normalized_backends))
            if missing_backends:
                raise ValueError(
                    "Publication PE cross-backend robustness requires actual "
                    f"DINGO and AMPLFI: {missing_backends}"
                )
        elif len(normalized_backends) != 1:
            raise ValueError(
                "Within-backend PE robustness requires exactly one backend"
            )
        ids_by_backend = {
            normalized: {
                injection_id
                for (backend, injection_id) in groups
                if backend == original
            }
            for normalized, original in normalized_backends.items()
        }
        reference_ids = next(iter(ids_by_backend.values()))
        common_injection_ids = sorted(reference_ids)
        if require_cross_backend_join:
            if any(ids != reference_ids for ids in ids_by_backend.values()):
                raise ValueError("Publication PE backends use different injection sets")
            for injection_id in common_injection_ids:
                for condition in ROBUSTNESS_CONDITIONS:
                    backend_rows = [
                        groups[(original, injection_id)][condition]
                        for original in normalized_backends.values()
                    ]
                    if any(
                        row["truth"] != backend_rows[0]["truth"]
                        for row in backend_rows[1:]
                    ):
                        raise ValueError(
                            f"Publication PE truth differs across backends: {injection_id}"
                        )
                    inconsistent = [
                        field
                        for field in PUBLICATION_SHARED_BACKEND_FIELDS
                        if any(
                            row[field] != backend_rows[0][field]
                            for row in backend_rows[1:]
                        )
                    ]
                    if inconsistent:
                        raise ValueError(
                            f"Publication PE assumptions differ across backends for "
                            f"{injection_id}/{condition}: {inconsistent}"
                        )
                    semantic = _common_source_semantics(
                        backend_rows[0]["verified_publication_input"]
                    )
                    if any(
                        _common_source_semantics(row["verified_publication_input"])
                        != semantic
                        for row in backend_rows[1:]
                    ):
                        raise ValueError(
                            f"Publication PE inputs differ across backends: "
                            f"{injection_id}/{condition}"
                        )
            cross_backend_input_gate = True

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
                    "mask_absolute_bias_change_vs_contaminated_normalized_by_clean_width",
                    "mask_absolute_bias_change_vs_clean_normalized_by_clean_width",
                    "contamination_width_ratio_vs_clean",
                    "mask_width_ratio_vs_contaminated",
                    "mask_width_ratio_vs_clean",
                    "coverage_mask_minus_contaminated",
                    "coverage_mask_minus_clean",
                )
            ):
                selected = [
                    (comparison, row[metric])
                    for comparison, row in zip(backend_comparisons, parameter_rows)
                    if row[metric] is not None
                ]
                values = [float(value) for _, value in selected]
                parameter_summaries[parameter][metric] = (
                    _paired_mean_bootstrap(
                        values,
                        bootstrap_replicates,
                        bootstrap_seed
                        + backend_index * 100_000
                        + parameter_index * 100
                        + metric_index,
                        [comparison for comparison, _ in selected],
                        minimum_physical_groups,
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
            selected = [
                (row, row[metric])
                for row in backend_comparisons
                if row[metric] is not None
            ]
            values = [float(value) for _, value in selected]
            resource_summaries[metric] = (
                _paired_mean_bootstrap(
                    values,
                    bootstrap_replicates,
                    bootstrap_seed + backend_index * 100_000 + 90_000 + metric_index,
                    [row for row, _ in selected],
                    minimum_physical_groups,
                )
                if values
                else None
            )
        paired_summaries[backend] = {
            "paired_injections": len(backend_comparisons),
            "bootstrap_independence": resource_summaries[
                "latency_mask_minus_contaminated_seconds"
            ]["bootstrap_independence"],
            "parameters": parameter_summaries,
            "resources": resource_summaries,
        }
    backend_names = sorted(paired_summaries)
    bootstrap_audits = {
        backend: paired_summaries[backend]["bootstrap_independence"]
        for backend in backend_names
    }
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
        "cross_backend_matched_input_gate": cross_backend_input_gate,
        "within_backend_provenance_gate": require_publication_provenance,
        "cross_backend_join_required": require_cross_backend_join,
        "comparison_scope": (
            "matched_cross_backend"
            if cross_backend_input_gate
            else (
                "strict_within_backend_paired"
                if require_publication_provenance and not require_cross_backend_join
                else "engineering_only"
            )
        ),
        "common_injection_ids": common_injection_ids,
        "common_injection_count": len(common_injection_ids),
        "triplets": len(comparisons),
        "coverage": coverage,
        "paired_summaries": paired_summaries,
        "pe_bootstrap_independence": _combine_pe_bootstrap_audits(
            bootstrap_audits, minimum_physical_groups
        ),
        "comparisons": comparisons,
        "publication_provenance_required": require_publication_provenance,
        "publication_provenance_fields": list(PUBLICATION_PROVENANCE_FIELDS),
        "publication_input_fields": list(PUBLICATION_INPUT_FIELDS),
        "publication_shared_backend_fields": list(PUBLICATION_SHARED_BACKEND_FIELDS),
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
    require_cross_backend_join: bool = True,
    minimum_physical_groups: int = 2,
) -> dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report = evaluate_pe_robustness_rows(
        rows,
        credible_level,
        bootstrap_replicates,
        bootstrap_seed,
        require_publication_provenance,
        require_cross_backend_join,
        minimum_physical_groups,
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = file_sha256(manifest_path)
    report.update(execution_provenance())
    atomic_write_json(output_path, report)
    return report


def run_joint_pe_robustness_evaluation(
    dingo_batch_report_path: str | Path,
    amplfi_batch_report_path: str | Path,
    manifest_output_path: str | Path,
    output_path: str | Path,
    credible_level: float = 0.9,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260720,
    minimum_physical_groups: int = 25,
) -> dict[str, Any]:
    """Join hash-bound DINGO/AMPLFI batches and run the strict paired evaluation."""

    specifications = {
        "DINGO": (
            Path(dingo_batch_report_path).resolve(),
            "real_dingo_common_batch_complete",
        ),
        "AMPLFI": (
            Path(amplfi_batch_report_path).resolve(),
            "real_amplfi_common_batch_complete",
        ),
    }
    if specifications["DINGO"][0] == specifications["AMPLFI"][0]:
        raise ValueError("Joint PE evaluation requires distinct backend batch reports")
    backend_rows: dict[str, list[dict[str, Any]]] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    backend_keys: dict[str, set[tuple[str, str]]] = {}
    for backend, (report_path, expected_status) in specifications.items():
        if not report_path.is_file():
            raise FileNotFoundError(f"{backend} PE batch report is absent: {report_path}")
        batch_report = json.loads(report_path.read_text(encoding="utf-8"))
        if batch_report.get("status") != expected_status:
            raise ValueError(f"{backend} PE batch report has not completed successfully")
        manifest = Path(str(batch_report.get("manifest_path", ""))).resolve()
        if (
            not manifest.is_file()
            or file_sha256(manifest) != batch_report.get("manifest_sha256")
        ):
            raise ValueError(f"{backend} PE batch manifest hash mismatch")
        with manifest.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != int(batch_report.get("rows", -1)) or not rows:
            raise ValueError(f"{backend} PE batch row count differs from its report")
        if any(str(row.get("backend", "")).upper() != backend for row in rows):
            raise ValueError(f"{backend} PE batch manifest contains another backend")
        keys = {
            (str(row.get("injection_id", "")), str(row.get("condition", "")))
            for row in rows
        }
        if any(not injection_id for injection_id, _ in keys):
            raise ValueError(f"{backend} PE batch contains an empty injection ID")
        if len(keys) != len(rows):
            raise ValueError(f"{backend} PE batch repeats an injection condition")
        injection_ids = {injection_id for injection_id, _ in keys}
        expected_keys = {
            (injection_id, condition)
            for injection_id in injection_ids
            for condition in ROBUSTNESS_CONDITIONS
        }
        if keys != expected_keys:
            raise ValueError(f"{backend} PE batch lacks complete robustness triplets")
        if int(batch_report.get("paired_injections", -1)) != len(injection_ids):
            raise ValueError(f"{backend} PE paired-injection count differs from its report")
        backend_rows[backend] = rows
        backend_keys[backend] = keys
        source_reports[backend] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        }
    if backend_keys["DINGO"] != backend_keys["AMPLFI"]:
        raise ValueError("DINGO and AMPLFI PE batches use different injection conditions")

    condition_order = {condition: index for index, condition in enumerate(ROBUSTNESS_CONDITIONS)}
    rows = sorted(
        backend_rows["DINGO"] + backend_rows["AMPLFI"],
        key=lambda row: (
            str(row["injection_id"]),
            condition_order[str(row["condition"])],
            str(row["backend"]).upper(),
        ),
    )
    manifest_output = Path(manifest_output_path).resolve()
    report_output = Path(output_path).resolve()
    protected_inputs = {
        identity[field]
        for identity in source_reports.values()
        for field in ("path", "manifest_path")
    }
    if str(manifest_output) in protected_inputs or str(report_output) in protected_inputs:
        raise ValueError("Joint PE outputs may not overwrite backend inputs")
    if manifest_output == report_output:
        raise ValueError("Joint PE manifest and report outputs must be distinct")
    report = evaluate_pe_robustness_rows(
        rows,
        credible_level,
        bootstrap_replicates,
        bootstrap_seed,
        True,
        True,
        minimum_physical_groups,
    )
    if not report["dingo_amplfi_joint_gate"] or not report[
        "cross_backend_matched_input_gate"
    ]:
        raise AssertionError("Strict joint PE evaluation returned without its mandatory gates")
    atomic_write_text(
        manifest_output,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report.update(
        {
            "status": "paired_dingo_amplfi_pe_robustness_evaluation_complete",
            "rows": len(rows),
            "manifest_path": str(manifest_output),
            "manifest_sha256": file_sha256(manifest_output),
            "source_batch_reports": source_reports,
            **execution_provenance(),
        }
    )
    atomic_write_json(report_output, report)
    return report


def run_within_backend_pe_robustness_portfolio(
    dingo_batch_report_path: str | Path,
    dingo_robustness_report_path: str | Path,
    amplfi_batch_report_path: str | Path,
    amplfi_robustness_report_path: str | Path,
    manifest_output_path: str | Path,
    output_path: str | Path,
    credible_level: float = 0.9,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20260721,
    required_split: str = "val",
    minimum_physical_groups: int = 25,
) -> dict[str, Any]:
    """Join matched events while retaining only within-backend PE deltas.

    This path is intentionally distinct from an absolute DINGO/AMPLFI comparison:
    each backend keeps one fixed native prior and waveform across clean,
    contaminated and mask-conditioned conditions, while source events and truths
    must match across backends.
    """

    if required_split not in {"val", "test"}:
        raise ValueError("PE portfolio split must be val or test")
    if bootstrap_replicates < 1:
        raise ValueError("PE portfolio bootstrap count must be positive")
    specifications = {
        "DINGO": {
            "batch": Path(dingo_batch_report_path).resolve(),
            "robustness": Path(dingo_robustness_report_path).resolve(),
            "batch_statuses": {
                "real_dingo_common_batch_complete",
                "real_dingo_official_native_paired_robustness_batch_complete",
            },
        },
        "AMPLFI": {
            "batch": Path(amplfi_batch_report_path).resolve(),
            "robustness": Path(amplfi_robustness_report_path).resolve(),
            "batch_statuses": {"real_amplfi_common_batch_complete"},
        },
    }
    source_batch_reports: dict[str, Any] = {}
    source_within_reports: dict[str, Any] = {}
    backend_rows: dict[str, list[dict[str, Any]]] = {}
    backend_evaluations: dict[str, dict[str, Any]] = {}
    backend_keys: dict[str, set[tuple[str, str]]] = {}
    backend_identities: dict[str, Any] = {}
    for backend, specification in specifications.items():
        batch_path = specification["batch"]
        robustness_path = specification["robustness"]
        if not batch_path.is_file() or not robustness_path.is_file():
            raise FileNotFoundError(f"{backend} PE portfolio source report is absent")
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
        manifest = Path(str(batch.get("manifest_path", ""))).resolve()
        if (
            batch.get("status") not in specification["batch_statuses"]
            or batch.get("run_identity", {}).get("required_split") != required_split
            or not manifest.is_file()
            or batch.get("manifest_sha256") != file_sha256(manifest)
            or robustness.get("status") != "paired_pe_contamination_mask_robustness"
            or robustness.get("comparison_scope") != "strict_within_backend_paired"
            or robustness.get("within_backend_provenance_gate") is not True
            or robustness.get("cross_backend_matched_input_gate") is not False
            or robustness.get("dingo_amplfi_joint_gate") is not False
            or robustness.get("publication_provenance_required") is not True
            or Path(str(robustness.get("manifest_path", ""))).resolve() != manifest
            or robustness.get("manifest_sha256") != file_sha256(manifest)
        ):
            raise ValueError(f"{backend} strict within-backend evidence failed replay")
        with manifest.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if (
            not rows
            or len(rows) != int(batch.get("rows", -1))
            or any(str(row.get("backend", "")).upper() != backend for row in rows)
            or any(str(row.get("split")) != required_split for row in rows)
        ):
            raise ValueError(f"{backend} PE portfolio manifest identity failed")
        keys = {
            (str(row.get("injection_id", "")), str(row.get("condition", "")))
            for row in rows
        }
        injections = {injection for injection, _ in keys}
        expected_keys = {
            (injection, condition)
            for injection in injections
            for condition in ROBUSTNESS_CONDITIONS
        }
        if (
            not injections
            or keys != expected_keys
            or len(keys) != len(rows)
            or len(injections) != int(batch.get("paired_injections", -1))
        ):
            raise ValueError(f"{backend} PE portfolio lacks complete unique triplets")
        evaluation = evaluate_pe_robustness_rows(
            rows,
            credible_level,
            bootstrap_replicates,
            bootstrap_seed,
            True,
            False,
            minimum_physical_groups,
        )
        prior_hashes = {str(row["prior_hash"]) for row in rows}
        waveforms = {str(row["waveform_approximant"]) for row in rows}
        model_hashes = {str(row["backend_model_hash"]) for row in rows}
        detector_sets = {
            json.dumps(row["detector_set"], sort_keys=True) for row in rows
        }
        if any(len(values) != 1 for values in (prior_hashes, waveforms, model_hashes, detector_sets)):
            raise ValueError(f"{backend} model/prior/waveform identity changes within portfolio")
        backend_rows[backend] = rows
        backend_evaluations[backend] = evaluation
        backend_keys[backend] = keys
        backend_identities[backend] = {
            "prior_hash": next(iter(prior_hashes)),
            "waveform_approximant": next(iter(waveforms)),
            "backend_model_hash": next(iter(model_hashes)),
            "detector_set": json.loads(next(iter(detector_sets))),
        }
        source_batch_reports[backend] = {
            "path": str(batch_path),
            "sha256": file_sha256(batch_path),
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        }
        source_within_reports[backend] = {
            "path": str(robustness_path),
            "sha256": file_sha256(robustness_path),
        }

    if backend_keys["DINGO"] != backend_keys["AMPLFI"]:
        raise ValueError("PE portfolio backends use different injection conditions")
    shared_fields = (
        "waveform_id",
        "gps_block",
        "source_event_hash",
        "analysis_input_sha256",
        "base_injection_manifest_sha256",
        "input_sample_rate_hz",
        "input_duration_seconds",
        "input_post_trigger_seconds",
        "input_ifos",
        "truth",
    )
    by_backend = {
        backend: {
            (str(row["injection_id"]), str(row["condition"])): row for row in rows
        }
        for backend, rows in backend_rows.items()
    }
    for key in sorted(backend_keys["DINGO"]):
        dingo = by_backend["DINGO"][key]
        amplfi = by_backend["AMPLFI"][key]
        inconsistent = [field for field in shared_fields if dingo.get(field) != amplfi.get(field)]
        if inconsistent:
            raise ValueError(
                f"PE portfolio source event differs across backends for {key}: {inconsistent}"
            )

    condition_order = {condition: index for index, condition in enumerate(ROBUSTNESS_CONDITIONS)}
    combined_rows = sorted(
        backend_rows["DINGO"] + backend_rows["AMPLFI"],
        key=lambda row: (
            str(row["injection_id"]),
            condition_order[str(row["condition"])],
            str(row["backend"]).upper(),
        ),
    )
    manifest_output = Path(manifest_output_path).resolve()
    report_output = Path(output_path).resolve()
    protected = {
        value[field]
        for value in source_batch_reports.values()
        for field in ("path", "manifest_path")
    } | {value["path"] for value in source_within_reports.values()}
    if str(manifest_output) in protected or str(report_output) in protected:
        raise ValueError("PE portfolio outputs may not overwrite source evidence")
    if manifest_output == report_output:
        raise ValueError("PE portfolio manifest and report outputs must be distinct")
    atomic_write_text(
        manifest_output,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows),
    )
    common_ids = sorted({injection for injection, _ in backend_keys["DINGO"]})
    result = {
        "status": "paired_dingo_amplfi_within_backend_portfolio_complete",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "this matched-event portfolio measures only within-backend condition deltas; "
            "it cannot rank absolute DINGO versus AMPLFI posterior performance"
        ),
        "comparison_scope": "matched_event_within_backend_deltas_only",
        "absolute_cross_backend_comparison_allowed": False,
        "matched_event_gate": True,
        "within_backend_provenance_gate": True,
        "dingo_amplfi_joint_gate": False,
        "cross_backend_matched_input_gate": False,
        "publication_provenance_required": True,
        "required_split": required_split,
        "test_rows_read": 0 if required_split == "val" else len(combined_rows),
        "backends": ["AMPLFI", "DINGO"],
        "backend_identities": backend_identities,
        "native_prior_hashes_equal": (
            backend_identities["DINGO"]["prior_hash"]
            == backend_identities["AMPLFI"]["prior_hash"]
        ),
        "native_waveform_assumptions_equal": (
            backend_identities["DINGO"]["waveform_approximant"]
            == backend_identities["AMPLFI"]["waveform_approximant"]
        ),
        "common_injection_ids": common_ids,
        "common_injection_count": len(common_ids),
        "rows": len(combined_rows),
        "manifest_path": str(manifest_output),
        "manifest_sha256": file_sha256(manifest_output),
        "credible_level": credible_level,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "coverage": {
            backend: evaluation["coverage"][backend]
            for backend, evaluation in backend_evaluations.items()
        },
        "paired_summaries": {
            backend: evaluation["paired_summaries"][backend]
            for backend, evaluation in backend_evaluations.items()
        },
        "pe_bootstrap_independence": _combine_pe_bootstrap_audits(
            {
                backend: evaluation["pe_bootstrap_independence"]
                for backend, evaluation in backend_evaluations.items()
            },
            minimum_physical_groups,
        ),
        "source_batch_reports": source_batch_reports,
        "source_within_backend_reports": source_within_reports,
        **execution_provenance(),
    }
    atomic_write_json(report_output, result)
    return result


_LOCKED_PE_FIXED_PROVENANCE_FIELDS = tuple(
    field for field in PUBLICATION_PROVENANCE_FIELDS if field != "source_event_hash"
)


def _replay_locked_pe_validation_promotion(
    promotion_report_path: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    promotion_path = Path(promotion_report_path).resolve()
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    joint_path = Path(str(promotion.get("joint_report_path", ""))).resolve()
    manifest_path = Path(str(promotion.get("joint_manifest_path", ""))).resolve()
    config_path = Path(str(promotion.get("config_path", ""))).resolve()
    if (
        promotion.get("status") != "pe_robustness_validation_promotion_decision"
        or promotion.get("passed") is not True
        or promotion.get("promote_to_locked_test") is not True
        or promotion.get("scientific_claim_allowed") is not False
        or not joint_path.is_file()
        or promotion.get("joint_report_sha256") != file_sha256(joint_path)
        or not manifest_path.is_file()
        or promotion.get("joint_manifest_sha256") != file_sha256(manifest_path)
        or not config_path.is_file()
        or promotion.get("config_sha256") != file_sha256(config_path)
    ):
        raise ValueError("locked PE requires a passing replayable validation promotion")
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    evidence_mode = promotion.get("evidence_mode", "absolute_common_prior_joint")
    absolute_joint = (
        evidence_mode == "absolute_common_prior_joint"
        and joint.get("status")
        == "paired_dingo_amplfi_pe_robustness_evaluation_complete"
        and joint.get("dingo_amplfi_joint_gate") is True
        and joint.get("cross_backend_matched_input_gate") is True
    )
    within_backend_portfolio = (
        evidence_mode == "matched_event_within_backend_portfolio"
        and joint.get("status")
        == "paired_dingo_amplfi_within_backend_portfolio_complete"
        and joint.get("comparison_scope")
        == "matched_event_within_backend_deltas_only"
        and joint.get("matched_event_gate") is True
        and joint.get("within_backend_provenance_gate") is True
        and joint.get("absolute_cross_backend_comparison_allowed") is False
        and joint.get("dingo_amplfi_joint_gate") is False
        and joint.get("cross_backend_matched_input_gate") is False
    )
    if (
        not (absolute_joint or within_backend_portfolio)
        or joint.get("publication_provenance_required") is not True
        or Path(str(joint.get("manifest_path", ""))).resolve() != manifest_path
        or joint.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("locked PE validation joint report failed replay")
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(str(row.get("split")) != "val" for row in rows):
        raise ValueError("locked PE promotion manifest must remain validation-only")
    return promotion_path, promotion, rows


def bind_locked_pe_backend_batch(
    backend: str,
    batch_report_path: str | Path,
    validation_promotion_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind one completed locked PE backend batch to the frozen suite and validation model."""

    from .evaluation_lock import (
        validate_locked_evaluation_suite_access,
        validate_locked_evaluation_suite_input,
    )

    backend_name = str(backend).upper()
    settings = {
        "DINGO": (
            "dingo_batch",
            {
                "real_dingo_common_batch_complete",
                "real_dingo_official_native_paired_robustness_batch_complete",
            },
            "locked_dingo_paired_pe_batch_complete",
        ),
        "AMPLFI": (
            "amplfi_batch",
            {"real_amplfi_common_batch_complete"},
            "locked_amplfi_paired_pe_batch_complete",
        ),
    }
    if backend_name not in settings:
        raise ValueError("locked PE backend must be DINGO or AMPLFI")
    output_key, expected_statuses, locked_status = settings[backend_name]
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked PE backend bindings are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, output_key, output_path
    )
    suite_input = validate_locked_evaluation_suite_input(
        locked_suite_plan,
        f"{backend_name.lower()}_locked_source_batch_report",
        batch_report_path,
    )
    promotion_path, _, validation_rows = _replay_locked_pe_validation_promotion(
        validation_promotion_report
    )
    promotion_identity = suite_access["frozen_artifacts"].get(
        "validation_pe_promotion", {}
    )
    if (
        Path(str(promotion_identity.get("path", ""))).resolve() != promotion_path
        or promotion_identity.get("sha256") != file_sha256(promotion_path)
    ):
        raise ValueError("locked PE promotion differs from the access receipt")
    validation_backend_rows = [
        row for row in validation_rows if str(row.get("backend", "")).upper() == backend_name
    ]
    if not validation_backend_rows:
        raise ValueError(f"validation promotion contains no {backend_name} rows")

    batch_path = Path(batch_report_path).resolve()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(batch.get("manifest_path", ""))).resolve()
    if (
        batch.get("status") not in expected_statuses
        or not manifest_path.is_file()
        or batch.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError(f"locked {backend_name} batch report failed replay")
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if (
        not rows
        or len(rows) != int(batch.get("rows", -1))
        or any(str(row.get("backend", "")).upper() != backend_name for row in rows)
        or any(str(row.get("split")) != "test" for row in rows)
    ):
        raise ValueError(f"locked {backend_name} batch is not a complete test artifact")
    keys = {
        (str(row.get("injection_id", "")), str(row.get("condition", "")))
        for row in rows
    }
    injection_ids = {injection_id for injection_id, _ in keys}
    expected_keys = {
        (injection_id, condition)
        for injection_id in injection_ids
        for condition in ROBUSTNESS_CONDITIONS
    }
    if (
        len(keys) != len(rows)
        or keys != expected_keys
        or len(injection_ids) != int(batch.get("paired_injections", -1))
        or len(injection_ids)
        < int(suite_access["endpoints"]["minimum_paired_pe_injections"])
    ):
        raise ValueError(f"locked {backend_name} batch lacks complete paired triplets")

    def fixed_identities(selected: list[dict[str, Any]]) -> set[str]:
        identities = []
        for row in selected:
            _validate_publication_provenance(row)
            identities.append(
                canonical_hash(
                    {
                        field: row[field]
                        for field in _LOCKED_PE_FIXED_PROVENANCE_FIELDS
                    },
                    64,
                )
            )
        return set(identities)

    validation_identities = fixed_identities(validation_backend_rows)
    locked_identities = fixed_identities(rows)
    if len(validation_identities) != 1 or locked_identities != validation_identities:
        raise ValueError(
            f"locked {backend_name} model/prior/waveform/latency identity differs from validation"
        )
    result = {
        "status": locked_status,
        "endpoint_complete": True,
        "scientific_claim_allowed": False,
        "backend": backend_name,
        "rows": len(rows),
        "paired_injections": len(injection_ids),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "fixed_provenance_identity_sha256": next(iter(locked_identities)),
        "source_batch_report": {
            "path": str(batch_path),
            "sha256": file_sha256(batch_path),
        },
        "validation_promotion_report": {
            "path": str(promotion_path),
            "sha256": file_sha256(promotion_path),
        },
        "locked_suite_access": suite_access,
        "locked_suite_input": suite_input,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def run_locked_joint_pe_robustness_evaluation(
    dingo_locked_report: str | Path,
    amplfi_locked_report: str | Path,
    validation_promotion_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate the predeclared locked DINGO/AMPLFI paired PE endpoint once."""

    from .evaluation_lock import validate_locked_evaluation_suite_access

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked joint PE outputs are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, "joint_pe", output_path
    )
    promotion_path, _, _ = _replay_locked_pe_validation_promotion(
        validation_promotion_report
    )
    promotion_identity = suite_access["frozen_artifacts"].get(
        "validation_pe_promotion", {}
    )
    if (
        Path(str(promotion_identity.get("path", ""))).resolve() != promotion_path
        or promotion_identity.get("sha256") != file_sha256(promotion_path)
    ):
        raise ValueError("locked PE promotion differs from the access receipt")
    specifications = {
        "DINGO": (
            Path(dingo_locked_report).resolve(),
            "dingo_batch",
            "locked_dingo_paired_pe_batch_complete",
        ),
        "AMPLFI": (
            Path(amplfi_locked_report).resolve(),
            "amplfi_batch",
            "locked_amplfi_paired_pe_batch_complete",
        ),
    }
    rows = []
    source_reports = {}
    backend_keys = {}
    for backend, (report_path, output_key, expected_status) in specifications.items():
        binding = validate_locked_evaluation_suite_access(
            locked_suite_plan, access_log, output_key, report_path
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest_path = Path(str(report.get("manifest_path", ""))).resolve()
        if (
            report.get("status") != expected_status
            or report.get("endpoint_complete") is not True
            or report.get("locked_suite_access") != binding
            or not manifest_path.is_file()
            or report.get("manifest_sha256") != file_sha256(manifest_path)
            or Path(
                str(report.get("validation_promotion_report", {}).get("path", ""))
            ).resolve()
            != promotion_path
            or report.get("validation_promotion_report", {}).get("sha256")
            != file_sha256(promotion_path)
        ):
            raise ValueError(f"locked {backend} PE binding failed replay")
        with manifest_path.open("r", encoding="utf-8") as handle:
            backend_rows = [json.loads(line) for line in handle if line.strip()]
        keys = {
            (str(row.get("injection_id", "")), str(row.get("condition", "")))
            for row in backend_rows
        }
        if (
            not backend_rows
            or len(backend_rows) != int(report.get("rows", -1))
            or any(str(row.get("split")) != "test" for row in backend_rows)
            or any(str(row.get("backend", "")).upper() != backend for row in backend_rows)
        ):
            raise ValueError(f"locked {backend} PE manifest changed after binding")
        rows.extend(backend_rows)
        backend_keys[backend] = keys
        source_reports[backend] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        }
    if backend_keys["DINGO"] != backend_keys["AMPLFI"]:
        raise ValueError("locked DINGO and AMPLFI batches use different injection conditions")
    endpoints = suite_access["endpoints"]
    report = evaluate_pe_robustness_rows(
        rows,
        float(endpoints["pe_credible_level"]),
        int(endpoints["bootstrap_replicates"]),
        int(endpoints["bootstrap_seed"]),
        True,
        True,
        int(endpoints["minimum_injection_gps_blocks"]),
    )
    paired_injections = int(report["common_injection_count"])
    if paired_injections < int(endpoints["minimum_paired_pe_injections"]):
        raise ValueError("locked joint PE endpoint has too few common injections")
    paired_summaries = report["paired_summaries"]
    result = {
        **report,
        "status": "locked_joint_paired_pe_complete",
        "endpoint_complete": True,
        "scientific_claim_allowed": False,
        "paired_injections": paired_injections,
        "identical_priors": True,
        "identical_waveform_assumptions": True,
        "coverage": report["coverage"],
        "bias": {
            backend: summary["parameters"]
            for backend, summary in paired_summaries.items()
        },
        "posterior_width": {
            backend: summary["parameters"]
            for backend, summary in paired_summaries.items()
        },
        "latency": {
            backend: summary["resources"].get(
                "latency_mask_minus_contaminated_seconds"
            )
            for backend, summary in paired_summaries.items()
        },
        "dingo_batch": source_reports["DINGO"],
        "amplfi_batch": source_reports["AMPLFI"],
        "validation_promotion_report": {
            "path": str(promotion_path),
            "sha256": file_sha256(promotion_path),
        },
        "locked_suite_access": suite_access,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def run_locked_paired_pe_robustness_portfolio(
    dingo_locked_report: str | Path,
    amplfi_locked_report: str | Path,
    validation_promotion_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate the locked matched-event, within-backend DINGO/AMPLFI portfolio."""

    from .evaluation_lock import validate_locked_evaluation_suite_access

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked paired PE portfolio outputs are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, "paired_pe_portfolio", output_path
    )
    promotion_path, promotion, _ = _replay_locked_pe_validation_promotion(
        validation_promotion_report
    )
    if promotion.get("evidence_mode") != "matched_event_within_backend_portfolio":
        raise ValueError("locked paired PE portfolio requires portfolio validation promotion")
    promotion_identity = suite_access["frozen_artifacts"].get(
        "validation_pe_promotion", {}
    )
    if (
        Path(str(promotion_identity.get("path", ""))).resolve() != promotion_path
        or promotion_identity.get("sha256") != file_sha256(promotion_path)
    ):
        raise ValueError("locked PE portfolio promotion differs from the access receipt")

    specifications = {
        "DINGO": (
            Path(dingo_locked_report).resolve(),
            "dingo_batch",
            "locked_dingo_paired_pe_batch_complete",
        ),
        "AMPLFI": (
            Path(amplfi_locked_report).resolve(),
            "amplfi_batch",
            "locked_amplfi_paired_pe_batch_complete",
        ),
    }
    rows_by_backend: dict[str, list[dict[str, Any]]] = {}
    keys_by_backend: dict[str, set[tuple[str, str]]] = {}
    source_reports: dict[str, Any] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for backend, (report_path, output_key, expected_status) in specifications.items():
        binding = validate_locked_evaluation_suite_access(
            locked_suite_plan, access_log, output_key, report_path
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest_path = Path(str(report.get("manifest_path", ""))).resolve()
        if (
            report.get("status") != expected_status
            or report.get("endpoint_complete") is not True
            or report.get("locked_suite_access") != binding
            or not manifest_path.is_file()
            or report.get("manifest_sha256") != file_sha256(manifest_path)
            or Path(
                str(report.get("validation_promotion_report", {}).get("path", ""))
            ).resolve()
            != promotion_path
            or report.get("validation_promotion_report", {}).get("sha256")
            != file_sha256(promotion_path)
        ):
            raise ValueError(f"locked {backend} PE portfolio binding failed replay")
        with manifest_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        keys = {
            (str(row.get("injection_id", "")), str(row.get("condition", "")))
            for row in rows
        }
        injections = {injection for injection, _ in keys}
        expected_keys = {
            (injection, condition)
            for injection in injections
            for condition in ROBUSTNESS_CONDITIONS
        }
        if (
            not rows
            or len(rows) != int(report.get("rows", -1))
            or any(str(row.get("split")) != "test" for row in rows)
            or any(str(row.get("backend", "")).upper() != backend for row in rows)
            or keys != expected_keys
            or len(keys) != len(rows)
            or len(injections) != int(report.get("paired_injections", -1))
        ):
            raise ValueError(f"locked {backend} PE portfolio manifest is incomplete")
        evaluation = evaluate_pe_robustness_rows(
            rows,
            float(suite_access["endpoints"]["pe_credible_level"]),
            int(suite_access["endpoints"]["bootstrap_replicates"]),
            int(suite_access["endpoints"]["bootstrap_seed"]),
            True,
            False,
            int(suite_access["endpoints"]["minimum_injection_gps_blocks"]),
        )
        rows_by_backend[backend] = rows
        keys_by_backend[backend] = keys
        evaluations[backend] = evaluation
        source_reports[backend] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        }

    if keys_by_backend["DINGO"] != keys_by_backend["AMPLFI"]:
        raise ValueError("locked PE portfolio backends use different injection conditions")
    indexed = {
        backend: {
            (str(row["injection_id"]), str(row["condition"])): row for row in rows
        }
        for backend, rows in rows_by_backend.items()
    }
    shared_fields = (
        "waveform_id",
        "gps_block",
        "source_event_hash",
        "analysis_input_sha256",
        "base_injection_manifest_sha256",
        "input_sample_rate_hz",
        "input_duration_seconds",
        "input_post_trigger_seconds",
        "input_ifos",
        "truth",
    )
    for key in sorted(keys_by_backend["DINGO"]):
        dingo = indexed["DINGO"][key]
        amplfi = indexed["AMPLFI"][key]
        inconsistent = [field for field in shared_fields if dingo.get(field) != amplfi.get(field)]
        if inconsistent:
            raise ValueError(
                f"locked PE portfolio source event differs for {key}: {inconsistent}"
            )
    paired_injections = len({injection for injection, _ in keys_by_backend["DINGO"]})
    if paired_injections < int(suite_access["endpoints"]["minimum_paired_pe_injections"]):
        raise ValueError("locked paired PE portfolio has too few common injections")
    backend_identities = {}
    for backend, rows in rows_by_backend.items():
        identities = {
            field: sorted(
                {
                    json.dumps(row[field], sort_keys=True)
                    if isinstance(row[field], (dict, list))
                    else str(row[field])
                    for row in rows
                }
            )
            for field in (
                "prior_hash",
                "waveform_approximant",
                "backend_model_hash",
                "detector_set",
            )
        }
        if any(len(values) != 1 for values in identities.values()):
            raise ValueError(f"locked {backend} PE identity changes within the portfolio")
        backend_identities[backend] = {
            field: values[0] for field, values in identities.items()
        }
    result = {
        "status": "locked_paired_pe_robustness_portfolio_complete",
        "endpoint_complete": True,
        "scientific_claim_allowed": False,
        "comparison_scope": "matched_event_within_backend_deltas_only",
        "absolute_cross_backend_comparison_allowed": False,
        "matched_event_gate": True,
        "within_backend_provenance_gate": True,
        "paired_injections": paired_injections,
        "backend_identities": backend_identities,
        "coverage": {
            backend: evaluation["coverage"][backend]
            for backend, evaluation in evaluations.items()
        },
        "bias": {
            backend: evaluation["paired_summaries"][backend]["parameters"]
            for backend, evaluation in evaluations.items()
        },
        "posterior_width": {
            backend: evaluation["paired_summaries"][backend]["parameters"]
            for backend, evaluation in evaluations.items()
        },
        "latency": {
            backend: evaluation["paired_summaries"][backend]["resources"].get(
                "latency_mask_minus_contaminated_seconds"
            )
            for backend, evaluation in evaluations.items()
        },
        "pe_bootstrap_independence": _combine_pe_bootstrap_audits(
            {
                backend: evaluation["pe_bootstrap_independence"]
                for backend, evaluation in evaluations.items()
            },
            int(suite_access["endpoints"]["minimum_injection_gps_blocks"]),
        ),
        "dingo_batch": source_reports["DINGO"],
        "amplfi_batch": source_reports["AMPLFI"],
        "validation_promotion_report": {
            "path": str(promotion_path),
            "sha256": file_sha256(promotion_path),
        },
        "locked_suite_access": suite_access,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def promote_pe_robustness_validation(
    joint_report_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Apply a frozen validation-only PE gate before any locked-test access."""

    joint_path = Path(joint_report_path).resolve()
    report = json.loads(joint_path.read_text(encoding="utf-8"))
    status = report.get("status")
    absolute_joint = (
        status == "paired_dingo_amplfi_pe_robustness_evaluation_complete"
        and report.get("dingo_amplfi_joint_gate") is True
        and report.get("cross_backend_matched_input_gate") is True
    )
    within_backend_portfolio = (
        status == "paired_dingo_amplfi_within_backend_portfolio_complete"
        and report.get("comparison_scope")
        == "matched_event_within_backend_deltas_only"
        and report.get("matched_event_gate") is True
        and report.get("within_backend_provenance_gate") is True
        and report.get("absolute_cross_backend_comparison_allowed") is False
        and report.get("dingo_amplfi_joint_gate") is False
        and report.get("cross_backend_matched_input_gate") is False
    )
    if (
        not (absolute_joint or within_backend_portfolio)
        or report.get("publication_provenance_required") is not True
    ):
        raise ValueError(
            "PE promotion requires a strict absolute joint or matched-event "
            "within-backend validation portfolio"
        )
    evidence_mode = (
        "absolute_common_prior_joint"
        if absolute_joint
        else "matched_event_within_backend_portfolio"
    )
    manifest = Path(str(report.get("manifest_path", ""))).resolve()
    if not manifest.is_file() or file_sha256(manifest) != report.get("manifest_sha256"):
        raise ValueError("PE promotion joint manifest hash mismatch")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != int(report.get("rows", -1)) or not rows:
        raise ValueError("PE promotion joint manifest row count mismatch")
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("PE promotion may only inspect validation posterior rows")
    for backend, identity in report.get("source_batch_reports", {}).items():
        source_report = Path(str(identity.get("path", ""))).resolve()
        source_manifest = Path(str(identity.get("manifest_path", ""))).resolve()
        if (
            backend not in {"DINGO", "AMPLFI"}
            or not source_report.is_file()
            or not source_manifest.is_file()
            or file_sha256(source_report) != identity.get("sha256")
            or file_sha256(source_manifest) != identity.get("manifest_sha256")
        ):
            raise ValueError("PE promotion source batch identity mismatch")
    if set(report.get("source_batch_reports", {})) != {"DINGO", "AMPLFI"}:
        raise ValueError("PE promotion requires both hash-bound source batch reports")

    config = load_yaml(config_path)
    settings = config.get("pe_robustness_promotion")
    if not isinstance(settings, dict):
        raise ValueError("PE robustness promotion configuration is missing")
    required_backends = [str(value).upper() for value in settings["required_backends"]]
    required_parameters = [str(value) for value in settings["required_parameters"]]
    if set(required_backends) != {"DINGO", "AMPLFI"} or not required_parameters:
        raise ValueError("PE promotion must predeclare DINGO, AMPLFI and paper parameters")
    minimum_injections = int(settings["minimum_paired_injections"])
    minimum_bootstraps = int(settings["minimum_bootstrap_replicates"])
    minimum_physical_groups = int(settings.get("minimum_injection_gps_blocks", 25))
    clean_coverage_margin = float(settings["coverage_noninferiority_margin_vs_clean"])
    contaminated_coverage_margin = float(
        settings["coverage_noninferiority_margin_vs_contaminated"]
    )
    maximum_bias_regression = float(settings["maximum_normalized_bias_regression_upper"])
    significant_bias_upper = float(
        settings["significant_normalized_bias_improvement_upper"]
    )
    minimum_bias_improvements = int(
        settings["minimum_significant_bias_improvements_per_backend"]
    )
    minimum_width_ratio = float(settings["minimum_width_ratio_vs_clean_lower"])
    maximum_width_ratio = float(settings["maximum_width_ratio_vs_clean_upper"])
    maximum_sky_ratio = float(settings["maximum_sky_area_ratio_upper"])
    minimum_ess_ratio = float(settings["minimum_ess_rate_ratio_lower"])
    maximum_latency_overhead = float(settings["maximum_latency_overhead_upper_seconds"])
    if (
        minimum_injections <= 0
        or minimum_bootstraps <= 0
        or minimum_physical_groups < 1
        or not 0 <= clean_coverage_margin < 1
        or not 0 <= contaminated_coverage_margin < 1
        or maximum_bias_regression < 0
        or minimum_bias_improvements < 0
        or minimum_width_ratio <= 0
        or maximum_width_ratio < minimum_width_ratio
        or maximum_sky_ratio <= 0
        or minimum_ess_ratio <= 0
        or maximum_latency_overhead < 0
    ):
        raise ValueError("PE robustness promotion thresholds are invalid")

    def interval(summary: Any, label: str) -> tuple[float, float]:
        if not isinstance(summary, dict):
            raise ValueError(f"PE promotion metric is absent: {label}")
        bounds = summary.get("paired_bootstrap_95")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"PE promotion metric lacks a paired interval: {label}")
        low, high = map(float, bounds)
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError(f"PE promotion metric has an invalid interval: {label}")
        return low, high

    global_checks = {
        "minimum_bootstrap_replicates": {
            "observed": int(report.get("bootstrap_replicates", -1)),
            "required": minimum_bootstraps,
            "passed": int(report.get("bootstrap_replicates", -1)) >= minimum_bootstraps,
        },
        "validation_only": {"passed": True},
        "physical_injection_bootstrap": {
            "observed": report.get("pe_bootstrap_independence"),
            "minimum_injection_gps_blocks": minimum_physical_groups,
            "passed": bool(
                report.get("pe_bootstrap_independence", {}).get("status")
                == "paired_pe_bootstrap_independence_audit_v1"
                and report.get("pe_bootstrap_independence", {}).get("passed")
                is True
                and report.get("pe_bootstrap_independence", {}).get("method")
                == "gps_block_then_paired_injection_hierarchical_bootstrap_v1"
                and int(
                    report.get("pe_bootstrap_independence", {}).get(
                        "physical_groups", 0
                    )
                )
                >= minimum_physical_groups
            ),
        },
        "input_scope_gate": {
            "mode": evidence_mode,
            "passed": True,
        },
    }
    backend_checks: dict[str, Any] = {}
    for backend in required_backends:
        summary = report.get("paired_summaries", {}).get(backend)
        if not isinstance(summary, dict):
            raise ValueError(f"PE promotion lacks backend summary: {backend}")
        paired_injections = int(summary.get("paired_injections", -1))
        parameter_checks = {}
        significant_improvements = 0
        for parameter in required_parameters:
            metrics = summary.get("parameters", {}).get(parameter)
            if not isinstance(metrics, dict):
                raise ValueError(f"PE promotion lacks {backend}/{parameter} metrics")
            coverage_clean = interval(
                metrics.get("coverage_mask_minus_clean"),
                f"{backend}/{parameter}/coverage_vs_clean",
            )
            coverage_contaminated = interval(
                metrics.get("coverage_mask_minus_contaminated"),
                f"{backend}/{parameter}/coverage_vs_contaminated",
            )
            normalized_bias = interval(
                metrics.get(
                    "mask_absolute_bias_change_vs_contaminated_normalized_by_clean_width"
                ),
                f"{backend}/{parameter}/normalized_bias",
            )
            width_ratio = interval(
                metrics.get("mask_width_ratio_vs_clean"),
                f"{backend}/{parameter}/width_ratio_vs_clean",
            )
            significant = normalized_bias[1] < significant_bias_upper
            significant_improvements += int(significant)
            checks = {
                "coverage_vs_clean": {
                    "paired_bootstrap_95": list(coverage_clean),
                    "minimum_lower": -clean_coverage_margin,
                    "passed": coverage_clean[0] >= -clean_coverage_margin,
                },
                "coverage_vs_contaminated": {
                    "paired_bootstrap_95": list(coverage_contaminated),
                    "minimum_lower": -contaminated_coverage_margin,
                    "passed": coverage_contaminated[0] >= -contaminated_coverage_margin,
                },
                "normalized_bias_nonregression": {
                    "paired_bootstrap_95": list(normalized_bias),
                    "maximum_upper": maximum_bias_regression,
                    "passed": normalized_bias[1] <= maximum_bias_regression,
                },
                "significant_normalized_bias_improvement": {
                    "paired_bootstrap_95": list(normalized_bias),
                    "required_upper_below": significant_bias_upper,
                    "passed": significant,
                },
                "width_ratio_vs_clean": {
                    "paired_bootstrap_95": list(width_ratio),
                    "required_interval": [minimum_width_ratio, maximum_width_ratio],
                    "passed": (
                        width_ratio[0] >= minimum_width_ratio
                        and width_ratio[1] <= maximum_width_ratio
                    ),
                },
            }
            parameter_checks[parameter] = {
                "passed_safety": all(
                    value["passed"]
                    for key, value in checks.items()
                    if key != "significant_normalized_bias_improvement"
                ),
                "checks": checks,
            }
        resources = summary.get("resources", {})
        sky = interval(
            resources.get("sky_area_mask_over_contaminated"),
            f"{backend}/sky_area_mask_over_contaminated",
        )
        ess = interval(
            resources.get("ess_rate_mask_over_contaminated"),
            f"{backend}/ess_rate_mask_over_contaminated",
        )
        latency = interval(
            resources.get("latency_mask_minus_contaminated_seconds"),
            f"{backend}/latency_mask_minus_contaminated_seconds",
        )
        resource_checks = {
            "sky_area_nonregression": {
                "paired_bootstrap_95": list(sky),
                "maximum_upper": maximum_sky_ratio,
                "passed": sky[1] <= maximum_sky_ratio,
            },
            "effective_sample_rate_noninferiority": {
                "paired_bootstrap_95": list(ess),
                "minimum_lower": minimum_ess_ratio,
                "passed": ess[0] >= minimum_ess_ratio,
            },
            "latency_overhead": {
                "paired_bootstrap_95": list(latency),
                "maximum_upper_seconds": maximum_latency_overhead,
                "passed": latency[1] <= maximum_latency_overhead,
            },
        }
        backend_checks[backend] = {
            "paired_injections": paired_injections,
            "minimum_paired_injections": minimum_injections,
            "sample_size_passed": paired_injections >= minimum_injections,
            "significant_bias_improvements": significant_improvements,
            "minimum_significant_bias_improvements": minimum_bias_improvements,
            "bias_improvement_passed": significant_improvements >= minimum_bias_improvements,
            "parameters": parameter_checks,
            "resources": resource_checks,
            "passed": (
                paired_injections >= minimum_injections
                and significant_improvements >= minimum_bias_improvements
                and all(value["passed_safety"] for value in parameter_checks.values())
                and all(value["passed"] for value in resource_checks.values())
            ),
        }
    passed = all(value["passed"] for value in global_checks.values()) and all(
        value["passed"] for value in backend_checks.values()
    )
    result = {
        "status": "pe_robustness_validation_promotion_decision",
        "passed": passed,
        "promote_to_locked_test": passed,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "this is a validation-only promotion decision; a frozen one-time locked-test "
            "evaluation is still required"
        ),
        "evidence_mode": evidence_mode,
        "absolute_cross_backend_comparison_allowed": absolute_joint,
        "joint_report_path": str(joint_path),
        "joint_report_sha256": file_sha256(joint_path),
        "joint_manifest_path": str(manifest),
        "joint_manifest_sha256": file_sha256(manifest),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "required_backends": required_backends,
        "required_parameters": required_parameters,
        "pe_bootstrap_independence": report["pe_bootstrap_independence"],
        "global_checks": global_checks,
        "backend_checks": backend_checks,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


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
    report.update(execution_provenance())
    atomic_write_json(output_path, report)
    return report
