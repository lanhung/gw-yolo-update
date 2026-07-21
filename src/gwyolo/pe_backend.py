from __future__ import annotations

import json
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
    if commit != str(settings.get("expected_git_commit", "")):
        failures.append(f"source commit {commit} does not match lock")
    if tag != str(settings.get("expected_git_tag", "")):
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
    for numeric_field in ("source_sample_rate_hz", "source_duration_seconds"):
        try:
            value = float(contract[numeric_field])
            if value <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            failures.append(f"comparison_contract.{numeric_field} must be positive")

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
        failures.extend(source_failures + environment_failures + model_failures + metadata_failures)
        if environment.get("executable"):
            interpreter_paths.append(environment["executable"])
        backends[name] = {
            "source": source,
            "environment": environment,
            "model": model,
            "model_metadata": metadata,
        }
    if len(interpreter_paths) == len(REQUIRED_BACKENDS) and len(set(interpreter_paths)) != 2:
        failures.append("DINGO and AMPLFI must use separate Python interpreters")

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
