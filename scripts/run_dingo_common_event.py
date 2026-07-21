#!/usr/bin/env python
"""Run one real DINGO posterior from a frozen GW-YOLO EventDataset artifact.

This script is intentionally executed by the pinned DINGO interpreter rather than the
GW-YOLO environment. It never fabricates a posterior when DINGO cannot load the model.
"""

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

import numpy as np


LATENCY_SCOPE = (
    "model-load-and-event-preprocessing-through-posterior-and-native-result-write_"
    "v1_excludes-artifact-verification-imports-and-mask-generation"
)


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


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--event", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--model-init", required=True)
    result.add_argument("--posterior-output", required=True)
    result.add_argument("--result-output", required=True)
    result.add_argument("--report-output", required=True)
    result.add_argument("--expected-event-sha256", required=True)
    result.add_argument("--expected-model-sha256", required=True)
    result.add_argument("--expected-model-init-sha256", required=True)
    result.add_argument("--num-samples", type=int, default=10000)
    result.add_argument("--batch-size", type=int, default=1000)
    result.add_argument("--num-gnpe-iterations", type=int, default=30)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260721)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_gnpe_iterations <= 0:
        raise ValueError("DINGO sampling sizes and iteration count must be positive")
    event_path = Path(args.event).resolve()
    model_path = Path(args.model).resolve()
    init_path = Path(args.model_init).resolve()
    identities = {
        "event": (event_path, args.expected_event_sha256),
        "model": (model_path, args.expected_model_sha256),
        "model_init": (init_path, args.expected_model_init_sha256),
    }
    observed = {}
    for label, (path, expected) in identities.items():
        if not path.is_file():
            raise FileNotFoundError(f"DINGO {label} artifact is absent: {path}")
        observed[label] = file_sha256(path)
        if observed[label] != expected:
            raise ValueError(f"DINGO {label} artifact hash mismatch")
    posterior_output = Path(args.posterior_output).resolve()
    result_output = Path(args.result_output).resolve()
    report_output = Path(args.report_output).resolve()
    if any(path.exists() for path in (posterior_output, result_output, report_output)):
        raise FileExistsError("DINGO inference refuses to overwrite an existing output")

    import torch
    from dingo.core.posterior_models.build_model import build_model_from_kwargs
    from dingo.gw.data.event_dataset import EventDataset
    from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    total_started = time.perf_counter()
    event_started = time.perf_counter()
    event = EventDataset(file_name=str(event_path))
    event_preprocessing_seconds = time.perf_counter() - event_started

    model_load_started = time.perf_counter()
    model = build_model_from_kwargs(
        filename=str(model_path), device=args.device, load_training_info=False
    )
    init_model = build_model_from_kwargs(
        filename=str(init_path), device=args.device, load_training_info=False
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started

    sampler = GWSamplerGNPE(
        model=model,
        init_sampler=GWSampler(model=init_model),
        num_iterations=args.num_gnpe_iterations,
    )
    sampler.context = event.data
    sampler.event_metadata = event.settings
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    sampling_started = time.perf_counter()
    sampler.run_sampler(args.num_samples, batch_size=args.batch_size)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    sampling_seconds = time.perf_counter() - sampling_started

    postprocessing_started = time.perf_counter()
    samples = sampler.samples
    if samples is None or len(samples) != args.num_samples:
        raise RuntimeError("DINGO returned an unexpected posterior sample count")
    arrays = {}
    for column in samples.columns:
        values = np.asarray(samples[column])
        if np.issubdtype(values.dtype, np.number):
            values = values.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"DINGO posterior column {column} is non-finite")
            arrays[str(column)] = values
    if not arrays:
        raise RuntimeError("DINGO returned no numeric posterior parameters")
    atomic_npz(posterior_output, arrays)

    result_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_output.name}.", suffix=".hdf5", dir=result_output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        sampler.to_result().to_file(file_name=temporary)
        os.replace(temporary, result_output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    posterior_postprocessing_and_write_seconds = (
        time.perf_counter() - postprocessing_started
    )
    total_seconds = time.perf_counter() - total_started

    report = {
        "status": "real_dingo_gnpe_posterior_complete",
        "backend": "DINGO",
        "backend_version": importlib.metadata.version("dingo-gw"),
        "event_path": str(event_path),
        "event_sha256": observed["event"],
        "model_path": str(model_path),
        "model_sha256": observed["model"],
        "model_init_path": str(init_path),
        "model_init_sha256": observed["model_init"],
        "posterior_path": str(posterior_output),
        "posterior_sha256": file_sha256(posterior_output),
        "native_result_path": str(result_output),
        "native_result_sha256": file_sha256(result_output),
        "parameters": sorted(arrays),
        "posterior_samples": args.num_samples,
        "effective_sample_size": float(args.num_samples),
        "importance_sampled": False,
        "num_gnpe_iterations": args.num_gnpe_iterations,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "latency_seconds": total_seconds,
        "latency_scope": LATENCY_SCOPE,
        "latency_components_seconds": {
            "model_load": model_load_seconds,
            "event_preprocessing": event_preprocessing_seconds,
            "posterior_sampling": sampling_seconds,
            "posterior_postprocessing_and_write": (
                posterior_postprocessing_and_write_seconds
            ),
        },
        "device": args.device,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    atomic_json(report_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
