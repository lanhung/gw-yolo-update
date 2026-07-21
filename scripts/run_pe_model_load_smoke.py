#!/usr/bin/env python
"""Fail-closed model-load smoke for pinned DINGO or AMPLFI interpreters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--backend", choices=("DINGO", "AMPLFI"), required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--expected-model-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--model-init")
    result.add_argument("--expected-model-init-sha256")
    result.add_argument("--model-config")
    result.add_argument("--expected-model-config-sha256")
    return result


def _verified(path_value: str | None, expected: str | None, label: str) -> tuple[Path, str]:
    if not path_value or not expected:
        raise ValueError(f"{label} path and expected SHA256 are required")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, observed


def _parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> int:
    args = parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("PE model-load smoke refuses to overwrite its report")
    model_path, model_sha = _verified(
        args.model, args.expected_model_sha256, f"{args.backend} model"
    )

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA model-load smoke requested but CUDA is unavailable")
    started = time.perf_counter()
    artifacts = {
        "model": {"path": str(model_path), "sha256": model_sha},
    }
    if args.backend == "DINGO":
        init_path, init_sha = _verified(
            args.model_init,
            args.expected_model_init_sha256,
            "DINGO time-initialization model",
        )
        from dingo.core.posterior_models.build_model import build_model_from_kwargs

        model = build_model_from_kwargs(
            filename=str(model_path), device=args.device, load_training_info=False
        )
        init_model = build_model_from_kwargs(
            filename=str(init_path), device=args.device, load_training_info=False
        )
        observations = {
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "model_parameter_count": _parameter_count(model.network),
            "initialization_model_class": (
                f"{type(init_model).__module__}.{type(init_model).__name__}"
            ),
            "initialization_model_parameter_count": _parameter_count(
                init_model.network
            ),
        }
        artifacts["model_init"] = {"path": str(init_path), "sha256": init_sha}
        distribution = "dingo-gw"
    else:
        config_path, config_sha = _verified(
            args.model_config,
            args.expected_model_config_sha256,
            "AMPLFI model config",
        )
        import run_amplfi_common_event as amplfi_runner

        training = amplfi_runner.load_yaml(config_path)
        model = amplfi_runner.build_model(training, model_path, args.device)
        observations = {
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "architecture_class": (
                f"{type(model.model).__module__}.{type(model.model).__name__}"
            ),
            "model_parameter_count": _parameter_count(model),
            "scaler_built": bool(model.scaler.built),
            "inference_parameters": list(model.hparams.inference_params),
        }
        artifacts["model_config"] = {"path": str(config_path), "sha256": config_sha}
        distribution = "amplfi"
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "status": "real_pe_backend_model_load_smoke_complete",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "model loading alone is not a posterior, calibration, search or sensitivity result"
        ),
        "backend": args.backend,
        "backend_version": importlib.metadata.version(distribution),
        "artifacts": artifacts,
        "device": args.device,
        "elapsed_seconds": elapsed,
        "observations": observations,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
