from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import atomic_write_json, atomic_write_text, file_sha256, load_yaml
from .pe import (
    PAIRED_PE_LATENCY_SCOPE_V1,
    posterior_sky_area_equal_solid_angle,
    sky_area_estimator_identity,
    validate_paired_pe_latency,
)
from .runtime import execution_provenance


CONDITIONS = ("clean", "contaminated", "mask_conditioned")


_DEFAULT_DINGO_PRIORS = {
    "phase": ("uniform", 0.0, 2 * np.pi),
    "tilt_1": ("sine", 0.0, np.pi),
    "tilt_2": ("sine", 0.0, np.pi),
    "phi_12": ("uniform", 0.0, 2 * np.pi),
    "phi_jl": ("uniform", 0.0, 2 * np.pi),
    "theta_jn": ("sine", 0.0, np.pi),
    "ra": ("uniform", 0.0, 2 * np.pi),
    "dec": ("cosine", -np.pi / 2, np.pi / 2),
    "psi": ("uniform", 0.0, np.pi),
}


def _load_dingo_settings(path: str | Path) -> dict[str, Any]:
    """Load YAML or the text-prefixed output emitted by DINGO model inspection."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        value = None
    if isinstance(value, dict):
        return value
    marker = "dataset_settings:"
    offset = text.find(marker)
    if offset < 0:
        raise ValueError(f"Expected DINGO settings mapping in {source}")
    try:
        value = yaml.safe_load(text[offset:])
    except yaml.YAMLError as error:
        raise ValueError(f"Cannot parse DINGO model settings in {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected DINGO settings mapping in {source}")
    return value


def _parse_dingo_prior(value: Any, parameter: str) -> dict[str, Any]:
    if value == "default":
        default = _DEFAULT_DINGO_PRIORS.get(parameter)
        if default is None:
            return {"family": "unsupported_default", "raw": value}
        family, minimum, maximum = default
        return {
            "family": family,
            "minimum": float(minimum),
            "maximum": float(maximum),
            "class": "bilby_default",
            "raw": value,
        }
    if not isinstance(value, str):
        return {"family": "fixed", "value": value, "raw": value}
    class_match = re.match(r"\s*([A-Za-z0-9_.]+)\s*\(", value)
    if class_match is None:
        return {"family": "unparsed", "raw": value}
    class_name = class_match.group(1)
    suffix = class_name.rsplit(".", 1)[-1]
    families = {
        "Uniform": "uniform",
        "Sine": "sine",
        "Cosine": "cosine",
        "Constraint": "constraint",
        "UniformInComponentsChirpMass": "uniform_in_components_chirp_mass",
        "UniformInComponentsMassRatio": "uniform_in_components_mass_ratio",
    }
    result: dict[str, Any] = {
        "family": families.get(suffix, "unsupported"),
        "class": class_name,
        "raw": value,
    }
    for field in ("minimum", "maximum"):
        match = re.search(rf"\b{field}\s*=\s*([-+0-9.eE]+)", value)
        if match is not None:
            result[field] = float(match.group(1))
    if result["family"] in {"sine", "cosine"}:
        default = _DEFAULT_DINGO_PRIORS.get(parameter)
        if default is not None:
            result.setdefault("minimum", float(default[1]))
            result.setdefault("maximum", float(default[2]))
    return result


def audit_dingo_common_prior_projection(
    canonical_prior_path: str | Path,
    dingo_prior_config_path: str | Path,
    dingo_training_config_path: str | Path,
) -> dict[str, Any]:
    """Prove that DINGO's native training prior equals the common PE prior."""

    canonical = load_yaml(canonical_prior_path)
    prior_config = _load_dingo_settings(dingo_prior_config_path)
    training_config = _load_dingo_settings(dingo_training_config_path)
    failures: list[str] = []
    if canonical.get("schema_version") != 1 or canonical.get("population") != "BBH":
        failures.append("canonical prior must be schema v1 BBH")
    distributions = canonical.get("distributions")
    nuisance = canonical.get("nuisance_distributions")
    if not isinstance(distributions, dict) or not isinstance(nuisance, dict):
        failures.append("canonical prior distributions are malformed")
        distributions = distributions if isinstance(distributions, dict) else {}
        nuisance = nuisance if isinstance(nuisance, dict) else {}
    dataset = prior_config.get("dataset_settings", prior_config)
    train = training_config.get("train_settings", training_config)
    intrinsic = dataset.get("intrinsic_prior", {}) if isinstance(dataset, dict) else {}
    data = train.get("data", {}) if isinstance(train, dict) else {}
    extrinsic = data.get("extrinsic_prior", {}) if isinstance(data, dict) else {}
    if not isinstance(intrinsic, dict) or not isinstance(extrinsic, dict):
        failures.append("DINGO intrinsic or extrinsic prior configuration is malformed")
        intrinsic = intrinsic if isinstance(intrinsic, dict) else {}
        extrinsic = extrinsic if isinstance(extrinsic, dict) else {}
    mappings = {
        "chirp_mass": (distributions.get("chirp_mass"), intrinsic.get("chirp_mass")),
        "mass_ratio": (distributions.get("mass_ratio"), intrinsic.get("mass_ratio")),
        "luminosity_distance": (
            distributions.get("luminosity_distance"),
            extrinsic.get("luminosity_distance"),
        ),
        "theta_jn": (distributions.get("theta_jn"), intrinsic.get("theta_jn")),
        "phase": (nuisance.get("phase"), intrinsic.get("phase")),
        "a_1": (nuisance.get("a_1"), intrinsic.get("a_1")),
        "a_2": (nuisance.get("a_2"), intrinsic.get("a_2")),
        "tilt_1": (nuisance.get("tilt_1"), intrinsic.get("tilt_1")),
        "tilt_2": (nuisance.get("tilt_2"), intrinsic.get("tilt_2")),
        "phi_jl": (nuisance.get("phi_jl"), intrinsic.get("phi_jl")),
        "phi_12": (nuisance.get("phi_12"), intrinsic.get("phi_12")),
        "ra": (distributions.get("ra"), extrinsic.get("ra")),
        "dec": (distributions.get("dec"), extrinsic.get("dec")),
        "psi": (distributions.get("psi"), extrinsic.get("psi")),
    }
    expected_native_families = {
        "uniform": "uniform",
        "uniform_periodic": "uniform",
        "sine": "sine",
        "cosine": "cosine",
    }
    checks = {}
    for parameter, (expected, raw_native) in mappings.items():
        if not isinstance(expected, dict) or raw_native is None:
            failures.append(f"prior projection is missing {parameter}")
            continue
        native = _parse_dingo_prior(raw_native, parameter)
        expected_family = str(expected.get("family"))
        expected_native = expected_native_families.get(expected_family)
        family_match = expected_native is not None and native.get("family") == expected_native
        try:
            bounds_match = np.isclose(
                float(native.get("minimum")), float(expected.get("minimum"))
            ) and np.isclose(float(native.get("maximum")), float(expected.get("maximum")))
        except (TypeError, ValueError):
            bounds_match = False
        if not family_match:
            failures.append(f"prior family mismatch for {parameter}")
        if not bounds_match:
            failures.append(f"prior bounds mismatch for {parameter}")
        checks[parameter] = {
            "canonical_family": expected_family,
            "native_family": native.get("family"),
            "native_class": native.get("class"),
            "canonical_bounds": [expected.get("minimum"), expected.get("maximum")],
            "native_bounds": [native.get("minimum"), native.get("maximum")],
            "family_match": family_match,
            "bounds_match": bool(bounds_match),
        }
    extra_constraints = sorted(
        parameter
        for parameter, value in intrinsic.items()
        if _parse_dingo_prior(value, parameter).get("family") == "constraint"
        and parameter not in mappings
    )
    if extra_constraints:
        failures.append(
            "DINGO native prior has constraints absent from the canonical prior: "
            + ", ".join(extra_constraints)
        )
    extra_stochastic_parameters = sorted(
        parameter
        for parameter, value in {**intrinsic, **extrinsic}.items()
        if parameter not in mappings
        and _parse_dingo_prior(value, parameter).get("family")
        not in {"fixed", "constraint"}
    )
    if extra_stochastic_parameters:
        failures.append(
            "DINGO native prior has stochastic parameters absent from the canonical prior: "
            + ", ".join(extra_stochastic_parameters)
        )
    if data.get("detectors") != ["H1", "L1"]:
        failures.append("DINGO common training detector set must be H1/L1")
    inference_parameters = data.get("inference_parameters")
    required_inference_parameters = {
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "theta_jn",
        "ra",
        "dec",
        "psi",
    }
    missing_inference = sorted(
        required_inference_parameters
        - set(inference_parameters if isinstance(inference_parameters, list) else [])
    )
    if missing_inference:
        failures.append(f"DINGO inference parameters omit common fields: {missing_inference}")
    window = data.get("window", {})
    if not isinstance(window, dict) or (
        float(window.get("f_s", 0)) != 4096 or float(window.get("T", 0)) != 16
    ):
        failures.append("DINGO native sample rate/window differs from frozen contract")
    domain = data.get("domain_update", {})
    if not isinstance(domain, dict) or (
        float(domain.get("f_min", 0)) != 20 or float(domain.get("f_max", 0)) != 1024
    ):
        failures.append("DINGO analysis frequency band differs from frozen contract")
    return {
        "status": "passed" if not failures else "failed",
        "publication_ready": not failures,
        "canonical_prior_path": str(Path(canonical_prior_path).resolve()),
        "canonical_prior_sha256": file_sha256(canonical_prior_path),
        "dingo_prior_config_path": str(Path(dingo_prior_config_path).resolve()),
        "dingo_prior_config_sha256": file_sha256(dingo_prior_config_path),
        "dingo_training_config_path": str(Path(dingo_training_config_path).resolve()),
        "dingo_training_config_sha256": file_sha256(dingo_training_config_path),
        "checks": checks,
        "extra_native_constraints": extra_constraints,
        "extra_native_stochastic_parameters": extra_stochastic_parameters,
        "failures": failures,
        **execution_provenance(),
    }


def run_dingo_common_prior_audit(
    canonical_prior_path: str | Path,
    dingo_prior_config_path: str | Path,
    dingo_training_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = audit_dingo_common_prior_projection(
        canonical_prior_path,
        dingo_prior_config_path,
        dingo_training_config_path,
    )
    atomic_write_json(output_path, report)
    if not report["publication_ready"]:
        raise RuntimeError(f"DINGO common prior projection failed; inspect {output_path}")
    return report


def _load_native_rows(path: str | Path, required_split: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("DINGO native conditioning manifest is empty")
    if any(row.get("backend") != "DINGO" for row in rows):
        raise ValueError("DINGO batch received a non-DINGO conditioning row")
    if any(str(row.get("split")) != required_split for row in rows):
        raise ValueError("DINGO batch native manifest contains another split")
    by_injection: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_injection[str(row["injection_id"])].add(str(row["condition"]))
    if any(values != set(CONDITIONS) for values in by_injection.values()):
        raise ValueError("DINGO batch requires three conditions for every injection")
    if len(rows) != 3 * len(by_injection):
        raise ValueError("DINGO batch native manifest repeats an injection condition")
    return rows


def _validated_completed_report(
    path: Path,
    event_sha256: str,
    model_sha256: str,
    model_init_sha256: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "real_dingo_gnpe_posterior_complete",
        "backend": "DINGO",
        "event_sha256": event_sha256,
        "model_sha256": model_sha256,
        "model_init_sha256": model_init_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing DINGO posterior report belongs to another run")
    for prefix in ("posterior", "native_result"):
        artifact = Path(report[f"{prefix}_path"])
        if not artifact.is_file() or file_sha256(artifact) != report[f"{prefix}_sha256"]:
            raise ValueError(f"Existing DINGO {prefix} artifact hash mismatch")
    validate_paired_pe_latency(report)
    return report


def run_dingo_common_batch(
    native_manifest: str | Path,
    model_metadata_path: str | Path,
    native_prior_path: str | Path,
    model_init_path: str | Path,
    python_executable: str | Path,
    runner_script: str | Path,
    output_dir: str | Path,
    required_split: str,
    num_samples: int = 10000,
    batch_size: int = 1000,
    num_gnpe_iterations: int = 30,
    device: str = "cuda",
    seed: int = 20260721,
) -> dict[str, Any]:
    if required_split not in {"val", "test"}:
        raise ValueError("DINGO batch is restricted to val or test")
    if num_samples <= 0 or batch_size <= 0 or num_gnpe_iterations <= 0:
        raise ValueError("DINGO batch sampling settings must be positive")
    metadata_path = Path(model_metadata_path).resolve()
    metadata = load_yaml(metadata_path)
    if metadata.get("backend") != "DINGO" or metadata.get("selection_split") != "validation":
        raise ValueError("DINGO batch requires validation-selected standardized model metadata")
    model = Path(metadata["model_path"]).resolve()
    if file_sha256(model) != str(metadata["model_sha256"]):
        raise ValueError("DINGO model hash differs from standardized metadata")
    model_init = Path(model_init_path).resolve()
    model_init_sha = file_sha256(model_init)
    artifacts = metadata.get("artifacts", {})
    required_artifacts = (
        "training_config",
        "training_data_manifest",
        "analysis_prior",
        "native_prior",
        "prior_projection_report",
        "selection_report",
        "native_conditioning_config",
        "initialization_model",
    )
    verified_artifacts = {}
    for label in required_artifacts:
        identity = artifacts.get(label, {})
        artifact = Path(str(identity.get("path", ""))).resolve()
        if (
            not artifact.is_file()
            or file_sha256(artifact) != identity.get("sha256")
        ):
            raise ValueError(f"DINGO {label} hash differs from model metadata")
        verified_artifacts[label] = identity
    native_prior = Path(native_prior_path).resolve()
    native_prior_sha = file_sha256(native_prior)
    if native_prior_sha != verified_artifacts["native_prior"]["sha256"]:
        raise ValueError("DINGO runtime native prior differs from model metadata")
    projection = load_yaml(verified_artifacts["prior_projection_report"]["path"])
    if (
        projection.get("status") != "passed"
        or projection.get("publication_ready") is not True
        or projection.get("failures") not in (None, [])
        or projection.get("canonical_prior_sha256")
        != verified_artifacts["analysis_prior"]["sha256"]
        or projection.get("dingo_prior_config_sha256") != native_prior_sha
        or projection.get("dingo_training_config_sha256")
        != verified_artifacts["training_config"]["sha256"]
    ):
        raise ValueError("DINGO prior projection differs from model metadata")
    if model_init_sha != verified_artifacts["initialization_model"]["sha256"]:
        raise ValueError("DINGO runtime initialization model differs from metadata")
    selection = load_yaml(verified_artifacts["selection_report"]["path"])
    if (
        selection.get("status") != "validation_selected_checkpoint"
        or selection.get("publication_eligible") is not True
        or selection.get("selection_split") != "validation"
        or selection.get("selected_checkpoint_sha256") != metadata["model_sha256"]
        or selection.get("selection_metric") != metadata.get("selection_metric")
    ):
        raise ValueError("DINGO validation selection report differs from metadata")
    python = Path(python_executable).resolve()
    runner = Path(runner_script).resolve()
    if not python.is_file() or not runner.is_file():
        raise FileNotFoundError("DINGO pinned interpreter or runner script is absent")
    source_input = metadata.get("source_input", {})
    if (
        source_input.get("ifos") != ["H1", "L1"]
        or not source_input.get("common_asd_required")
        or float(source_input.get("sample_rate_hz", 0)) <= 0
        or float(source_input.get("duration_seconds", 0)) <= 0
        or float(source_input.get("post_trigger_seconds", 0)) <= 0
    ):
        raise ValueError("DINGO model metadata lacks the common H1/L1 ASD contract")
    rows = _load_native_rows(native_manifest, required_split)
    conditioning_sha = verified_artifacts["native_conditioning_config"]["sha256"]
    if any(
        row.get("input_ifos") != source_input["ifos"]
        or not np.isclose(
            float(row.get("input_sample_rate_hz", 0)),
            float(source_input["sample_rate_hz"]),
        )
        or not np.isclose(
            float(row.get("input_duration_seconds", 0)),
            float(source_input["duration_seconds"]),
        )
        or not np.isclose(
            float(row.get("input_post_trigger_seconds", 0)),
            float(source_input["post_trigger_seconds"]),
        )
        for row in rows
    ):
        raise ValueError("DINGO native rows differ from the model common-source contract")
    if any(
        row.get("native_conditioning_config_sha256") != conditioning_sha
        for row in rows
    ):
        raise ValueError("DINGO native rows use conditioning outside model metadata")
    if any(
        row.get("common_prior_sha256")
        != verified_artifacts["analysis_prior"]["sha256"]
        for row in rows
    ):
        raise ValueError("DINGO native rows use a common prior outside model metadata")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": "dingo_common_batch_v1",
        "native_manifest_sha256": file_sha256(native_manifest),
        "model_metadata_sha256": file_sha256(metadata_path),
        "model_sha256": metadata["model_sha256"],
        "model_init_sha256": model_init_sha,
        "training_config_sha256": verified_artifacts["training_config"]["sha256"],
        "training_data_manifest_sha256": verified_artifacts[
            "training_data_manifest"
        ]["sha256"],
        "analysis_prior_sha256": verified_artifacts["analysis_prior"]["sha256"],
        "native_prior_sha256": native_prior_sha,
        "prior_projection_report_sha256": verified_artifacts[
            "prior_projection_report"
        ]["sha256"],
        "selection_report_sha256": verified_artifacts["selection_report"]["sha256"],
        "native_conditioning_config_sha256": conditioning_sha,
        "python_executable": str(python),
        "runner_sha256": file_sha256(runner),
        "required_split": required_split,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_gnpe_iterations": num_gnpe_iterations,
        "device": device,
        "seed": seed,
    }
    state_path = output / "dingo_batch_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing DINGO batch output belongs to another run")
    else:
        atomic_write_json(
            state_path, {"status": "in_progress", "run_identity": run_identity, "completed": 0}
        )
    result_rows = []
    for index, row in enumerate(rows, start=1):
        event = Path(row["native_conditioning_path"]).resolve()
        if file_sha256(event) != str(row["native_conditioning_sha256"]):
            raise ValueError("DINGO native conditioning artifact hash mismatch")
        event_output = output / "events" / str(row["injection_id"]) / str(row["condition"])
        posterior = event_output / "posterior.npz"
        native_result = event_output / "dingo_result.hdf5"
        report_path = event_output / "dingo_inference_report.json"
        log_path = event_output / "dingo_inference.log"
        if report_path.is_file():
            report = _validated_completed_report(
                report_path,
                row["native_conditioning_sha256"],
                metadata["model_sha256"],
                model_init_sha,
            )
        else:
            event_output.mkdir(parents=True, exist_ok=True)
            command = [
                str(python),
                str(runner),
                "--event",
                str(event),
                "--model",
                str(model),
                "--model-init",
                str(model_init),
                "--posterior-output",
                str(posterior),
                "--result-output",
                str(native_result),
                "--report-output",
                str(report_path),
                "--expected-event-sha256",
                row["native_conditioning_sha256"],
                "--expected-model-sha256",
                metadata["model_sha256"],
                "--expected-model-init-sha256",
                model_init_sha,
                "--num-samples",
                str(num_samples),
                "--batch-size",
                str(batch_size),
                "--num-gnpe-iterations",
                str(num_gnpe_iterations),
                "--device",
                device,
                "--seed",
                str(seed + index - 1),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            atomic_write_text(
                log_path,
                "command: " + json.dumps(command) + "\nstdout:\n" + completed.stdout
                + "\nstderr:\n" + completed.stderr,
            )
            if completed.returncode != 0:
                atomic_write_json(
                    event_output / "dingo_inference_failure.json",
                    {
                        "status": "failed",
                        "returncode": completed.returncode,
                        "log_path": str(log_path),
                        "log_sha256": file_sha256(log_path),
                        "event_sha256": row["native_conditioning_sha256"],
                        "model_sha256": metadata["model_sha256"],
                        "model_init_sha256": model_init_sha,
                    },
                )
                raise RuntimeError(f"DINGO inference failed; inspect {log_path}")
            report = _validated_completed_report(
                report_path,
                row["native_conditioning_sha256"],
                metadata["model_sha256"],
                model_init_sha,
            )
        with np.load(report["posterior_path"], allow_pickle=False) as posterior:
            if "ra" not in posterior.files or "dec" not in posterior.files:
                raise ValueError("DINGO posterior lacks RA/Dec sky samples")
            sky_area = posterior_sky_area_equal_solid_angle(
                posterior["ra"], posterior["dec"]
            )
        latency_components = validate_paired_pe_latency(report)
        result_rows.append(
            {
                **row,
                "backend": "DINGO",
                "posterior_path": report["posterior_path"],
                "posterior_sha256": report["posterior_sha256"],
                "latency_seconds": report["latency_seconds"],
                "effective_sample_size": report["effective_sample_size"],
                "sky_area_90_deg2": sky_area["area_deg2"],
                "sky_area_estimator": sky_area_estimator_identity(sky_area),
                "sky_area_diagnostics": {
                    field: sky_area[field]
                    for field in ("sample_count", "occupied_pixels", "credible_pixels")
                },
                "backend_version": report["backend_version"],
                "backend_model_hash": report["model_sha256"],
                "prior_hash": row["common_prior_sha256"],
                "waveform_approximant": metadata["analysis_waveform_approximant"],
                "detector_set": row["input_ifos"],
                "calibration_version": "none_software_injection_o4a_strain",
                "source_event_hash": row["source_event_hash"],
                "hardware": {
                    "hostname": report["environment"]["hostname"],
                    "gpu": report["environment"]["gpu"],
                },
                "latency_scope": PAIRED_PE_LATENCY_SCOPE_V1,
                "backend_native_latency_scope": report["latency_scope"],
                "backend_native_latency_components_seconds": latency_components,
            }
        )
        atomic_write_json(
            state_path,
            {"status": "in_progress", "run_identity": run_identity, "completed": index},
        )
    manifest = output / "dingo_posterior_manifest.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    report = {
        "status": "real_dingo_common_batch_complete",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "matched AMPLFI results and paired robustness evaluation are still required"
        ),
        "rows": len(result_rows),
        "paired_injections": len({row["injection_id"] for row in result_rows}),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(output / "dingo_batch_report.json", report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(result_rows),
            "manifest_sha256": report["manifest_sha256"],
        },
    )
    return report
