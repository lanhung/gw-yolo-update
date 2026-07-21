from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256, load_yaml
from .runtime import execution_provenance


REQUIRED_BACKENDS = ("DINGO", "AMPLFI")
REQUIRED_CONDITIONS = ("clean", "contaminated", "mask_conditioned")
UNRESOLVED_VALUES = {"", "UNRESOLVED", "TO_BE_FROZEN"}
MODEL_METADATA_ARTIFACTS = (
    "training_config",
    "training_data_manifest",
    "analysis_prior",
    "selection_report",
    "native_conditioning_config",
)
AMPLFI_MODEL_METADATA_ARTIFACTS = (
    "native_prior",
    "prior_projection_report",
)
DINGO_MODEL_METADATA_ARTIFACTS = (
    "native_prior",
    "prior_projection_report",
    "initialization_model",
)


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _env_path(variable: str, failures: list[str]) -> Path | None:
    value = os.environ.get(variable)
    if not value:
        failures.append(f"environment variable {variable} is not set")
        return None
    return Path(value).expanduser().resolve()


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        raise ValueError(f"Unsupported version string: {value}")
    return tuple(int(part or 0) for part in match.groups())


def _python_satisfies(version: str, specification: str) -> bool:
    observed = _version_tuple(version)
    for term in (part.strip() for part in specification.split(",")):
        match = re.fullmatch(r"(>=|<=|==|>|<)(\d+(?:\.\d+){0,2})", term)
        if not match:
            raise ValueError(f"Unsupported Python constraint: {term}")
        expected = _version_tuple(match.group(2))
        width = max(len(observed), len(expected))
        left = observed + (0,) * (width - len(observed))
        right = expected + (0,) * (width - len(expected))
        operation = match.group(1)
        accepted = {
            ">=": left >= right,
            "<=": left <= right,
            "==": left == right,
            ">": left > right,
            "<": left < right,
        }[operation]
        if not accepted:
            return False
    return True


def _audit_source(name: str, settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected_commit = str(settings.get("expected_git_commit", ""))
    expected_tag = str(settings.get("expected_git_tag", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        failures.append("expected_git_commit must be a full 40-character lowercase SHA")
    if not expected_tag:
        failures.append("expected_git_tag is required")
    source_env = str(settings.get("source_path_env", ""))
    if not source_env:
        return {}, [f"{name}: source_path_env is required"]
    source = _env_path(source_env, failures)
    result: dict[str, Any] = {"path_env": source_env, "path": str(source) if source else None}
    if source is None:
        return result, [f"{name}: {failure}" for failure in failures]
    if not (source / ".git").is_dir():
        return result, [f"{name}: source is not a Git repository: {source}"]
    try:
        commit = _run(["git", "rev-parse", "HEAD"], source)
        tag = _run(["git", "describe", "--tags", "--exact-match", "HEAD"], source)
        dirty = bool(_run(["git", "status", "--porcelain"], source))
    except (OSError, subprocess.CalledProcessError) as error:
        return result, [f"{name}: cannot inspect source repository: {error}"]
    result.update({"commit": commit, "tag": tag, "dirty": dirty})
    if commit != expected_commit:
        failures.append(f"source commit {commit} does not match lock")
    if tag != expected_tag:
        failures.append(f"source tag {tag} does not match lock")
    if dirty:
        failures.append("source repository is dirty")
    return result, [f"{name}: {failure}" for failure in failures]


def _audit_interpreter(
    name: str, settings: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    executable_env = str(settings.get("python_executable_env", ""))
    if not executable_env:
        return {}, [f"{name}: python_executable_env is required"]
    executable = _env_path(executable_env, failures)
    result: dict[str, Any] = {
        "executable_env": executable_env,
        "executable": str(executable) if executable else None,
    }
    if executable is None:
        return result, [f"{name}: {failure}" for failure in failures]
    if not executable.is_file():
        return result, [f"{name}: Python executable does not exist: {executable}"]
    probe = (
        "import hashlib,importlib.metadata as m,json,platform,sys; "
        "p=sorted((x.metadata['Name'].lower(),x.version) for x in m.distributions() "
        "if x.metadata.get('Name')); b=json.dumps(p,separators=(',',':')).encode(); "
        "d={'python':platform.python_version(),'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'packages':p,"
        "'environment_packages_sha256':hashlib.sha256(b).hexdigest(),"
        "'distribution':m.version("
        + repr(str(settings.get("distribution", "")))
        + ")}; "
        "\ntry:\n import torch; d.update(torch=torch.__version__,"
        "cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,"
        "gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))"
        "\nexcept Exception as e: d['torch_probe_error']=repr(e)"
        "\nprint(json.dumps(d,sort_keys=True))"
    )
    try:
        observed = json.loads(_run([str(executable), "-c", probe]))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return result, [f"{name}: environment probe failed: {error}"]
    result.update(observed)
    required_python = str(settings.get("python_requires", ""))
    if not required_python:
        failures.append("python_requires is required")
    elif not _python_satisfies(str(observed["python"]), required_python):
        failures.append(
            f"Python {observed['python']} does not satisfy {required_python}"
        )
    expected_version = str(settings.get("expected_distribution_version", ""))
    if observed.get("distribution") != expected_version:
        failures.append(
            f"distribution version {observed.get('distribution')} does not match "
            f"{expected_version}"
        )
    expected_environment = str(settings.get("environment_packages_sha256", ""))
    if expected_environment in UNRESOLVED_VALUES:
        failures.append("environment package-set SHA256 is unresolved")
    elif observed.get("environment_packages_sha256") != expected_environment:
        failures.append("environment package-set SHA256 does not match lock")
    if not observed.get("cuda_available"):
        failures.append("CUDA is unavailable in backend environment")
    return result, [f"{name}: {failure}" for failure in failures]


def _audit_file_lock(
    name: str,
    label: str,
    path_env: Any,
    expected_sha256: Any,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    variable = str(path_env or "")
    expected = str(expected_sha256 or "")
    result: dict[str, Any] = {"path_env": variable, "expected_sha256": expected}
    if not variable:
        return result, [f"{name}: {label}_path_env is required"]
    path = _env_path(variable, failures)
    result["path"] = str(path) if path else None
    if expected in UNRESOLVED_VALUES:
        failures.append(f"{label} SHA256 is unresolved")
    if path is not None:
        if not path.is_file():
            failures.append(f"{label} file does not exist: {path}")
        else:
            observed = file_sha256(path)
            result["observed_sha256"] = observed
            if expected not in UNRESOLVED_VALUES and observed != expected:
                failures.append(f"{label} SHA256 does not match lock")
    return result, [f"{name}: {failure}" for failure in failures]


def _valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _verified_metadata_artifact(
    name: str, label: str, value: Any, failures: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(f"{label} metadata artifact is missing")
        return {}
    path_value = value.get("path")
    expected = value.get("sha256")
    result = {"path": path_value, "sha256": expected}
    if not _valid_sha256(expected):
        failures.append(f"{label} metadata SHA256 is invalid")
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        failures.append(f"{label} metadata artifact does not exist: {path}")
    else:
        observed = file_sha256(path)
        result["observed_sha256"] = observed
        if observed != expected:
            failures.append(f"{label} metadata artifact SHA256 does not match")
    return result


def _audit_amplfi_prior_projection_metadata(
    artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    projection_artifact = artifacts.get("prior_projection_report", {})
    projection_path = Path(
        str(projection_artifact.get("path") or "")
    ).expanduser().resolve()
    if not projection_path.is_file():
        return {}, failures
    try:
        projection = load_yaml(projection_path)
    except (OSError, ValueError) as error:
        return {}, [f"cannot load AMPLFI prior projection report: {error}"]
    if projection.get("status") != "passed" or projection.get("publication_ready") is not True:
        failures.append("AMPLFI prior projection report did not pass")
    if projection.get("failures") not in (None, []):
        failures.append("AMPLFI prior projection report contains failures")
    expected_hashes = {
        "canonical_prior_sha256": artifacts.get("analysis_prior", {}).get(
            "observed_sha256"
        ),
        "amplfi_prior_sha256": artifacts.get("native_prior", {}).get(
            "observed_sha256"
        ),
        "amplfi_training_config_sha256": artifacts.get("training_config", {}).get(
            "observed_sha256"
        ),
    }
    for field, expected in expected_hashes.items():
        if not _valid_sha256(expected) or projection.get(field) != expected:
            failures.append(f"AMPLFI prior projection {field} does not match model metadata")
    return projection, failures


def _audit_dingo_prior_projection_metadata(
    artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    projection_artifact = artifacts.get("prior_projection_report", {})
    projection_path = Path(
        str(projection_artifact.get("path") or "")
    ).expanduser().resolve()
    if not projection_path.is_file():
        return {}, failures
    try:
        projection = load_yaml(projection_path)
    except (OSError, ValueError) as error:
        return {}, [f"cannot load DINGO prior projection report: {error}"]
    if projection.get("status") != "passed" or projection.get("publication_ready") is not True:
        failures.append("DINGO prior projection report did not pass")
    if projection.get("failures") not in (None, []):
        failures.append("DINGO prior projection report contains failures")
    expected_hashes = {
        "canonical_prior_sha256": artifacts.get("analysis_prior", {}).get(
            "observed_sha256"
        ),
        "dingo_prior_config_sha256": artifacts.get("native_prior", {}).get(
            "observed_sha256"
        ),
        "dingo_training_config_sha256": artifacts.get("training_config", {}).get(
            "observed_sha256"
        ),
    }
    for field, expected in expected_hashes.items():
        if not _valid_sha256(expected) or projection.get(field) != expected:
            failures.append(f"DINGO prior projection {field} does not match model metadata")
    return projection, failures


def _audit_model_metadata_semantics(
    name: str,
    metadata_path: str | None,
    model_sha256: str | None,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if not metadata_path or not Path(metadata_path).is_file():
        return {}, []
    try:
        metadata = load_yaml(metadata_path)
    except (OSError, ValueError) as error:
        return {}, [f"{name}: cannot load standardized model metadata: {error}"]
    if metadata.get("schema_version") != 1:
        failures.append("model metadata schema_version must be 1")
    if metadata.get("backend") != name:
        failures.append("model metadata backend does not match")
    if metadata.get("model_sha256") != model_sha256:
        failures.append("model metadata model_sha256 does not match checkpoint")
    if metadata.get("population") != contract.get("population"):
        failures.append("model metadata population does not match comparison contract")
    source_input = metadata.get("source_input")
    expected_source = {
        "ifos": contract.get("source_ifos"),
        "sample_rate_hz": contract.get("source_sample_rate_hz"),
        "duration_seconds": contract.get("source_duration_seconds"),
        "post_trigger_seconds": contract.get("source_post_trigger_seconds"),
        "common_asd_required": contract.get("common_asd_required"),
    }
    if source_input != expected_source:
        failures.append("model metadata source_input does not match comparison contract")
    for field in (
        "analysis_waveform_approximant",
        "native_model_waveform_approximant",
        "model_training_backend_version",
        "selection_metric",
    ):
        if metadata.get(field) in (None, ""):
            failures.append(f"model metadata {field} is required")
    parameters = metadata.get("native_inference_parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(not isinstance(value, str) or not value for value in parameters)
        or len(set(parameters)) != len(parameters)
    ):
        failures.append("model metadata native_inference_parameters must be non-empty and unique")
    mapping = metadata.get("reported_parameter_mapping")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(
            not isinstance(canonical, str)
            or not canonical
            or not isinstance(native, str)
            or not native
            for canonical, native in mapping.items()
        )
    ):
        failures.append("model metadata reported_parameter_mapping must be non-empty")
    elif isinstance(parameters, list) and any(native not in parameters for native in mapping.values()):
        failures.append("reported parameter mapping references a non-native parameter")
    if metadata.get("selection_split") != "validation":
        failures.append("model checkpoint must be selected only on validation")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        failures.append("model metadata artifacts must be a mapping")
    required_artifacts = MODEL_METADATA_ARTIFACTS
    if name == "AMPLFI":
        required_artifacts += AMPLFI_MODEL_METADATA_ARTIFACTS
    elif name == "DINGO":
        required_artifacts += DINGO_MODEL_METADATA_ARTIFACTS
    verified_artifacts = {
        label: _verified_metadata_artifact(name, label, artifacts.get(label), failures)
        for label in required_artifacts
    }
    prior_projection: dict[str, Any] = {}
    if name == "AMPLFI":
        prior_projection, projection_failures = _audit_amplfi_prior_projection_metadata(
            verified_artifacts
        )
        failures.extend(projection_failures)
    elif name == "DINGO":
        prior_projection, projection_failures = _audit_dingo_prior_projection_metadata(
            verified_artifacts
        )
        failures.extend(projection_failures)
    selection = verified_artifacts.get("selection_report", {})
    selection_path = selection.get("path")
    if selection_path and Path(selection_path).is_file():
        try:
            selection_report = load_yaml(selection_path)
        except (OSError, ValueError) as error:
            failures.append(f"cannot load selection report: {error}")
        else:
            if selection_report.get("status") != "validation_selected_checkpoint":
                failures.append("selection report status is not validation-selected")
            if selection_report.get("publication_eligible") is not True:
                failures.append("selection report is not publication eligible")
            if selection_report.get("selection_split") != "validation":
                failures.append("selection report is not validation-only")
            if selection_report.get("selected_checkpoint_sha256") != model_sha256:
                failures.append("selection report selected checkpoint does not match model")
            if selection_report.get("selection_metric") != metadata.get("selection_metric"):
                failures.append("selection metric differs between report and model metadata")
    normalized = dict(metadata)
    normalized["verified_artifacts"] = verified_artifacts
    if name in {"AMPLFI", "DINGO"}:
        normalized["verified_prior_projection"] = prior_projection
    return normalized, [f"{name}: {failure}" for failure in failures]


def audit_pe_backend_lock(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    failures: list[str] = []
    if config.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    contract = config.get("comparison_contract")
    if not isinstance(contract, dict):
        contract = {}
        failures.append("comparison_contract must be a mapping")
    if contract.get("population") != "BBH":
        failures.append("the initial shared PE comparison population must be BBH")
    if contract.get("source_ifos") != ["H1", "L1"]:
        failures.append("the initial shared PE comparison must freeze source_ifos as H1/L1")
    if not contract.get("identical_source_bytes_across_backends"):
        failures.append("identical source bytes across backends must be required")
    if contract.get("conditions") != list(REQUIRED_CONDITIONS):
        failures.append(f"conditions must be {list(REQUIRED_CONDITIONS)}")
    for numeric_field in (
        "source_sample_rate_hz",
        "source_duration_seconds",
        "source_post_trigger_seconds",
    ):
        try:
            value = float(contract[numeric_field])
            if value <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            failures.append(f"comparison_contract.{numeric_field} must be positive")
    try:
        if float(contract["source_post_trigger_seconds"]) >= float(
            contract["source_duration_seconds"]
        ):
            failures.append("source post-trigger duration must be shorter than source duration")
    except (KeyError, TypeError, ValueError):
        pass
    if not contract.get("common_asd_required"):
        failures.append("a common source ASD must be required")
    if not contract.get("common_asd_condition_invariant"):
        failures.append("the common source ASD must be invariant across conditions")

    backend_config = config.get("backends")
    if not isinstance(backend_config, dict):
        backend_config = {}
        failures.append("backends must be a mapping")
    if set(backend_config) != set(REQUIRED_BACKENDS):
        failures.append(f"backends must be exactly {list(REQUIRED_BACKENDS)}")

    backends: dict[str, Any] = {}
    interpreter_paths: list[str] = []
    for name in REQUIRED_BACKENDS:
        settings = backend_config.get(name)
        if not isinstance(settings, dict):
            failures.append(f"{name}: backend settings are missing")
            continue
        source, source_failures = _audit_source(name, settings)
        environment, environment_failures = _audit_interpreter(name, settings)
        model, model_failures = _audit_file_lock(
            name, "model", settings.get("model_path_env"), settings.get("model_sha256")
        )
        metadata, metadata_failures = _audit_file_lock(
            name,
            "model_metadata",
            settings.get("model_metadata_path_env"),
            settings.get("model_metadata_sha256"),
        )
        model_metadata, semantic_failures = _audit_model_metadata_semantics(
            name,
            metadata.get("path"),
            model.get("observed_sha256"),
            contract,
        )
        failures.extend(
            source_failures
            + environment_failures
            + model_failures
            + metadata_failures
            + semantic_failures
        )
        if environment.get("executable"):
            interpreter_paths.append(environment["executable"])
        backends[name] = {
            "source": source,
            "environment": environment,
            "model": model,
            "model_metadata": metadata,
            "model_metadata_semantics": model_metadata,
        }
    if len(interpreter_paths) == len(REQUIRED_BACKENDS) and len(set(interpreter_paths)) != 2:
        failures.append("DINGO and AMPLFI must use separate Python interpreters")
    if all(name in backends for name in REQUIRED_BACKENDS):
        semantics = [backends[name]["model_metadata_semantics"] for name in REQUIRED_BACKENDS]
        if all(semantics):
            if semantics[0].get("analysis_waveform_approximant") != semantics[1].get(
                "analysis_waveform_approximant"
            ):
                failures.append("DINGO/AMPLFI model metadata differ in analysis waveform")
            common_parameters = [
                sorted(value.get("reported_parameter_mapping", {})) for value in semantics
            ]
            if common_parameters[0] != common_parameters[1]:
                failures.append("DINGO/AMPLFI reported common parameter sets differ")
            prior_hashes = [
                value.get("verified_artifacts", {})
                .get("analysis_prior", {})
                .get("observed_sha256")
                for value in semantics
            ]
            if prior_hashes[0] != prior_hashes[1]:
                failures.append("DINGO/AMPLFI analysis prior artifacts differ")

    return {
        "status": "ready" if not failures else "incomplete",
        "publication_ready": not failures,
        "scientific_claim_allowed": False,
        "comparison_contract": contract,
        "backends": backends,
        "failures": failures,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        **execution_provenance(),
    }


def run_pe_backend_lock_audit(
    config_path: str | Path,
    output_path: str | Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    report = audit_pe_backend_lock(config_path)
    atomic_write_json(output_path, report)
    if not report["publication_ready"] and not allow_incomplete:
        raise RuntimeError(
            f"PE backend lock is incomplete; inspect the atomic report at {output_path}"
        )
    return report


def freeze_pe_backend_model_metadata(
    *,
    backend: str,
    model_path: str | Path,
    training_config_path: str | Path,
    training_data_manifest_path: str | Path,
    analysis_prior_path: str | Path,
    selection_report_path: str | Path,
    native_conditioning_config_path: str | Path,
    output_path: str | Path,
    population: str,
    source_ifos: list[str],
    source_sample_rate_hz: float,
    source_duration_seconds: float,
    source_post_trigger_seconds: float,
    analysis_waveform_approximant: str,
    native_model_waveform_approximant: str,
    model_training_backend_version: str,
    native_inference_parameters: list[str],
    reported_parameter_mapping: list[str],
    native_prior_path: str | Path | None = None,
    prior_projection_report_path: str | Path | None = None,
    initialization_model_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_backend = backend.upper()
    if normalized_backend not in REQUIRED_BACKENDS:
        raise ValueError(f"backend must be one of {list(REQUIRED_BACKENDS)}")
    if population != "BBH":
        raise ValueError("initial shared PE model metadata supports BBH only")
    if source_ifos != ["H1", "L1"]:
        raise ValueError("initial shared PE source IFOs must be H1/L1")
    if (
        source_sample_rate_hz <= 0
        or source_duration_seconds <= 0
        or not 0 < source_post_trigger_seconds < source_duration_seconds
    ):
        raise ValueError("source rate, duration and post-trigger duration are invalid")
    if not native_inference_parameters or len(set(native_inference_parameters)) != len(
        native_inference_parameters
    ):
        raise ValueError("native inference parameters must be non-empty and unique")
    mapping: dict[str, str] = {}
    for value in reported_parameter_mapping:
        if value.count("=") != 1:
            raise ValueError("reported parameter mappings must use canonical=native")
        canonical, native = value.split("=", 1)
        if not canonical or not native or canonical in mapping:
            raise ValueError("reported parameter mapping is empty or duplicated")
        if native not in native_inference_parameters:
            raise ValueError("reported parameter mapping references a non-native parameter")
        mapping[canonical] = native
    if not mapping:
        raise ValueError("at least one reported common parameter is required")
    if not model_training_backend_version:
        raise ValueError("model training backend version is required")
    paths = {
        "training_config": Path(training_config_path).resolve(),
        "training_data_manifest": Path(training_data_manifest_path).resolve(),
        "analysis_prior": Path(analysis_prior_path).resolve(),
        "selection_report": Path(selection_report_path).resolve(),
        "native_conditioning_config": Path(native_conditioning_config_path).resolve(),
    }
    prior_specific = (native_prior_path, prior_projection_report_path)
    if any(value is None for value in prior_specific):
        raise ValueError(
            f"{normalized_backend} model metadata requires native prior and prior projection report"
        )
    paths.update(
        {
            "native_prior": Path(str(native_prior_path)).resolve(),
            "prior_projection_report": Path(
                str(prior_projection_report_path)
            ).resolve(),
        }
    )
    if normalized_backend == "DINGO":
        if initialization_model_path is None:
            raise ValueError("DINGO model metadata requires an initialization model")
        paths["initialization_model"] = Path(initialization_model_path).resolve()
    elif initialization_model_path is not None:
        raise ValueError("initialization model artifact is DINGO-specific")
    model = Path(model_path).resolve()
    for label, path in {"model": model, **paths}.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact does not exist: {path}")
    model_hash = file_sha256(model)
    selection_report = load_yaml(paths["selection_report"])
    if selection_report.get("status") != "validation_selected_checkpoint":
        raise ValueError("selection report status must be validation_selected_checkpoint")
    if selection_report.get("publication_eligible") is not True:
        raise ValueError("selection report must be publication eligible")
    if selection_report.get("selection_split") != "validation":
        raise ValueError("selection report must use validation split")
    if selection_report.get("selected_checkpoint_sha256") != model_hash:
        raise ValueError("selection report checkpoint hash does not match model")
    selection_metric = selection_report.get("selection_metric")
    if not isinstance(selection_metric, str) or not selection_metric:
        raise ValueError("selection report requires selection_metric")
    verified_artifacts = {
        label: {
            "path": str(path),
            "sha256": file_sha256(path),
            "observed_sha256": file_sha256(path),
        }
        for label, path in paths.items()
    }
    if normalized_backend == "AMPLFI":
        _, projection_failures = _audit_amplfi_prior_projection_metadata(
            verified_artifacts
        )
        if projection_failures:
            raise ValueError("; ".join(projection_failures))
    else:
        _, projection_failures = _audit_dingo_prior_projection_metadata(
            verified_artifacts
        )
        if projection_failures:
            raise ValueError("; ".join(projection_failures))
    metadata = {
        "schema_version": 1,
        "backend": normalized_backend,
        "model_path": str(model),
        "model_sha256": model_hash,
        "population": population,
        "source_input": {
            "ifos": source_ifos,
            "sample_rate_hz": source_sample_rate_hz,
            "duration_seconds": source_duration_seconds,
            "post_trigger_seconds": source_post_trigger_seconds,
            "common_asd_required": True,
        },
        "analysis_waveform_approximant": analysis_waveform_approximant,
        "native_model_waveform_approximant": native_model_waveform_approximant,
        "model_training_backend_version": model_training_backend_version,
        "native_inference_parameters": native_inference_parameters,
        "reported_parameter_mapping": mapping,
        "selection_split": "validation",
        "selection_metric": selection_metric,
        "artifacts": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in paths.items()
        },
        **execution_provenance(),
    }
    atomic_write_json(output_path, metadata)
    return metadata


def select_lightning_validation_checkpoint(
    *,
    training_config_path: str | Path,
    training_data_manifest_path: str | Path,
    metrics_csv_path: str | Path,
    checkpoint_index_path: str | Path,
    output_path: str | Path,
    selection_metric: str = "valid_loss",
    selection_metric_mode: str = "min",
    minimum_publication_epochs: int = 100,
    minimum_validation_points: int = 50,
) -> dict[str, Any]:
    if selection_metric_mode not in {"min", "max"}:
        raise ValueError("selection metric mode must be min or max")
    if minimum_publication_epochs <= 0 or minimum_validation_points <= 0:
        raise ValueError("publication epoch and validation-point minima must be positive")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("validation checkpoint selection reports are immutable")
    training_config = load_yaml(training_config_path)
    try:
        configured_max_epochs = int(training_config["trainer"]["max_epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training config requires trainer.max_epochs") from error
    if configured_max_epochs <= 0:
        raise ValueError("configured max epochs must be positive")

    with Path(metrics_csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if selection_metric not in fieldnames or "epoch" not in fieldnames:
            raise ValueError("metrics CSV lacks epoch or the configured validation metric")
        rows = list(reader)
    test_fields = [field for field in fieldnames if "test" in field.lower()]
    if any(str(row.get(field, "")).strip() for row in rows for field in test_fields):
        raise ValueError("checkpoint selection metrics include test-set values")
    validation_rows = []
    for row_number, row in enumerate(rows, start=2):
        metric_value = str(row.get(selection_metric, "")).strip()
        if not metric_value:
            continue
        try:
            epoch = int(float(row["epoch"]))
            step = int(float(row.get("step", epoch)))
            value = float(metric_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid validation metric row {row_number}") from error
        if epoch < 0 or step < 0 or not math.isfinite(value):
            raise ValueError(f"invalid validation metric row {row_number}")
        validation_rows.append(
            {"epoch": epoch, "global_step": step, "value": value, "row": row_number}
        )
    if not validation_rows:
        raise ValueError("metrics CSV contains no finite validation measurements")
    keys = [(row["epoch"], row["global_step"]) for row in validation_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("metrics CSV repeats a validation epoch/global-step identity")
    best = min(
        validation_rows,
        key=lambda row: (
            row["value"] if selection_metric_mode == "min" else -row["value"],
            row["epoch"],
            row["global_step"],
        ),
    )

    index = load_yaml(checkpoint_index_path)
    if index.get("status") != "indexed_lightning_checkpoints":
        raise ValueError("checkpoint index has the wrong status")
    checkpoints = index.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("checkpoint index is empty")
    verified = []
    for entry in checkpoints:
        path = Path(str(entry.get("path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise ValueError("checkpoint index artifact hash mismatch")
        try:
            epoch = int(entry["epoch"])
            global_step = int(entry["global_step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint index lacks epoch/global_step") from error
        verified.append({**entry, "path": str(path), "epoch": epoch, "global_step": global_step})
    selectable = [entry for entry in verified if Path(entry["path"]).name != "last.ckpt"]
    exact = [
        entry
        for entry in selectable
        if entry["epoch"] == best["epoch"]
        and entry["global_step"] == best["global_step"]
    ]
    if len(exact) == 1:
        selected = exact[0]
        match_rule = "exact_epoch_and_global_step"
    else:
        same_epoch = [entry for entry in selectable if entry["epoch"] == best["epoch"]]
        if len(same_epoch) != 1:
            raise ValueError("validation-selected checkpoint is absent or ambiguous")
        selected = same_epoch[0]
        match_rule = "unique_checkpoint_for_validation_epoch"

    observed_epochs = sorted({row["epoch"] for row in validation_rows})
    training_complete = observed_epochs[-1] + 1 >= configured_max_epochs
    blockers = []
    if not training_complete:
        blockers.append("configured training budget is incomplete")
    if configured_max_epochs < minimum_publication_epochs:
        blockers.append("configured training budget is below the publication minimum")
    if len(observed_epochs) < minimum_validation_points:
        blockers.append("validation trajectory is below the publication minimum")
    result = {
        "status": "validation_selected_checkpoint",
        "publication_eligible": not blockers,
        "scientific_claim_allowed": False,
        "selection_split": "validation",
        "selection_metric": selection_metric,
        "selection_metric_mode": selection_metric_mode,
        "selected_metric_value": best["value"],
        "selected_epoch": best["epoch"],
        "selected_global_step": best["global_step"],
        "checkpoint_match_rule": match_rule,
        "selected_checkpoint_path": selected["path"],
        "selected_checkpoint_sha256": selected["sha256"],
        "configured_max_epochs": configured_max_epochs,
        "observed_validation_epochs": observed_epochs,
        "validation_points": len(validation_rows),
        "training_complete": training_complete,
        "minimum_publication_epochs": minimum_publication_epochs,
        "minimum_validation_points": minimum_validation_points,
        "selection_inputs_include_test_metrics": False,
        "blockers": blockers,
        "training_config_path": str(Path(training_config_path).resolve()),
        "training_config_sha256": file_sha256(training_config_path),
        "training_data_manifest_path": str(
            Path(training_data_manifest_path).resolve()
        ),
        "training_data_manifest_sha256": file_sha256(training_data_manifest_path),
        "metrics_csv_path": str(Path(metrics_csv_path).resolve()),
        "metrics_csv_sha256": file_sha256(metrics_csv_path),
        "checkpoint_index_path": str(Path(checkpoint_index_path).resolve()),
        "checkpoint_index_sha256": file_sha256(checkpoint_index_path),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result
