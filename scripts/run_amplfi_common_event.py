#!/usr/bin/env python
"""Run one real AMPLFI posterior from a frozen common-source artifact.

The script is executed by the pinned AMPLFI interpreter.  It reconstructs the
exact architecture from the hashed training configuration, loads a real
validation-selected Lightning checkpoint, whitens with the condition-invariant
ASD stored beside the event, and fails rather than fabricating samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np


INFERENCE_PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "distance",
    "phic",
    "inclination",
    "dec",
    "psi",
    "phi",
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


def atomic_hdf5(path: Path, arrays: dict[str, np.ndarray], attributes: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".hdf5", dir=path.parent
    )
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(attributes)
            posterior = handle.create_group("posterior")
            for name, values in arrays.items():
                posterior.create_dataset(name, data=values, compression="gzip")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"AMPLFI configuration is not a mapping: {path}")
    return value


def _architecture_settings(training: dict) -> tuple[dict, dict, list[str]]:
    model = training.get("model", {})
    if model.get("class_path") != "amplfi.train.models.flow.FlowModel":
        raise ValueError("AMPLFI training config does not declare FlowModel")
    model_args = model.get("init_args", {})
    inference = list(model_args.get("inference_params", []))
    if tuple(inference) != INFERENCE_PARAMETERS:
        raise ValueError("AMPLFI checkpoint inference parameter order differs from contract")
    arch = model_args.get("arch", {})
    if arch.get("class_path") != "amplfi.train.architectures.NSF":
        raise ValueError("AMPLFI training config does not declare the frozen NSF")
    arch_args = arch.get("init_args", {})
    embedding = arch_args.get("embedding_net", {})
    if embedding.get("class_path") != (
        "amplfi.train.architectures.embeddings.MultiModalPsd"
    ):
        raise ValueError("AMPLFI training config does not declare MultiModalPsd")
    embedding_args = embedding.get("init_args", {})
    norm = embedding_args.get("norm_layer", {})
    if norm.get("class_path") != "ml4gw.nn.norm.GroupNorm1DGetter":
        raise ValueError("AMPLFI training config does not declare GroupNorm1DGetter")
    return arch_args, embedding_args, inference


def build_model(training: dict, checkpoint: Path, device: str):
    from amplfi.train.architectures import NSF
    from amplfi.train.architectures.embeddings import MultiModalPsd
    from amplfi.train.models.flow import FlowModel
    from ml4gw.nn.norm import GroupNorm1DGetter

    arch_args, embedding_args, inference = _architecture_settings(training)
    norm_args = dict(embedding_args["norm_layer"].get("init_args", {}))
    embedding_kwargs = {
        key: value
        for key, value in embedding_args.items()
        if key != "norm_layer"
    }
    embedding = MultiModalPsd(
        num_ifos=2,
        norm_layer=GroupNorm1DGetter(**norm_args),
        **embedding_kwargs,
    )
    arch_kwargs = {
        key: value
        for key, value in arch_args.items()
        if key != "embedding_net"
    }
    architecture = NSF(
        num_params=len(inference), embedding_net=embedding, **arch_kwargs
    )
    model = FlowModel.load_from_checkpoint(
        str(checkpoint), arch=architecture, map_location=device, strict=True
    )
    if tuple(model.hparams.inference_params) != tuple(inference):
        raise ValueError("Loaded AMPLFI checkpoint parameter order differs from training config")
    if not model.scaler.built:
        raise ValueError("Loaded AMPLFI checkpoint does not contain a fitted parameter scaler")
    model.to(device)
    model.eval()
    return model


def native_bounds(native_prior: dict, training: dict) -> dict[str, tuple[float, float]]:
    priors = native_prior.get("init_args", {}).get("priors", {})
    bounds: dict[str, tuple[float, float]] = {}
    for name in ("chirp_mass", "mass_ratio", "distance", "phic", "inclination"):
        specification = priors.get(name, {})
        if specification.get("class_path") not in {
            "torch.distributions.Uniform",
            "ml4gw.distributions.Sine",
        }:
            raise ValueError(f"AMPLFI native prior is unsupported for {name}")
        arguments = specification.get("init_args", {})
        bounds[name] = (float(arguments["low"]), float(arguments["high"]))
    data = training.get("data", {}).get("init_args", {})
    if data.get("dec", {}).get("class_path") != "ml4gw.distributions.Cosine":
        raise ValueError("AMPLFI declination prior differs from the frozen cosine prior")
    bounds["dec"] = (-math.pi / 2, math.pi / 2)
    for name, high in (("psi", math.pi), ("phi", 2 * math.pi)):
        specification = data.get(name, {})
        if specification.get("class_path") != "torch.distributions.Uniform":
            raise ValueError(f"AMPLFI extrinsic prior is unsupported for {name}")
        arguments = specification.get("init_args", {})
        observed = (float(arguments["low"]), float(arguments["high"]))
        if not np.allclose(observed, (0.0, high), rtol=0.0, atol=1e-12):
            raise ValueError(f"AMPLFI extrinsic prior bounds differ for {name}")
        bounds[name] = observed
    if set(bounds) != set(INFERENCE_PARAMETERS):
        raise ValueError("AMPLFI native prior does not bound every inference parameter")
    return bounds


def load_and_preprocess_event(event_path: Path, training: dict, device: str):
    import h5py
    import torch
    from ml4gw.transforms import Whiten

    data = training.get("data", {}).get("init_args", {})
    ifos = tuple(str(value) for value in data.get("ifos", []))
    if ifos != ("H1", "L1"):
        raise ValueError("AMPLFI common-event inference requires H1/L1")
    sample_rate = int(data["sample_rate"])
    kernel = float(data["kernel_length"])
    fduration = float(data["fduration"])
    highpass = float(data["highpass"])
    right_pad = float(data["waveform_sampler"]["init_args"]["right_pad"])
    with h5py.File(event_path, "r") as handle:
        if handle.attrs.get("schema") != "gwyolo-amplfi-common-source-v1":
            raise ValueError("AMPLFI event does not use the common-source native schema")
        strain = np.asarray(handle["strain"], dtype=np.float64)
        asd = np.asarray(handle["asd"], dtype=np.float64)
        asd_frequencies = np.asarray(handle["asd_frequencies"], dtype=np.float64)
        stored_ifos = tuple(value.decode() for value in handle["ifos"][:])
        gps_start = float(handle.attrs["gps_start"])
        geocent_time = float(handle.attrs["geocent_time"])
        attributes = {
            "sample_rate_hz": float(handle.attrs["sample_rate_hz"]),
            "kernel_seconds": float(handle.attrs["kernel_seconds"]),
            "whitening_duration_seconds": float(
                handle.attrs["whitening_duration_seconds"]
            ),
            "highpass_hz": float(handle.attrs["highpass_hz"]),
            "right_pad_seconds": float(handle.attrs["right_pad_seconds"]),
            "condition": str(handle.attrs["condition"]),
            "injection_id": str(handle.attrs["injection_id"]),
            "common_asd_sha256": str(handle.attrs["common_asd_sha256"]),
            "source_sha256": str(handle.attrs["source_sha256"]),
        }
    expected_attributes = {
        "sample_rate_hz": float(sample_rate),
        "kernel_seconds": kernel,
        "whitening_duration_seconds": fduration,
        "highpass_hz": highpass,
        "right_pad_seconds": right_pad,
    }
    if any(
        not np.isclose(attributes[name], value, rtol=0.0, atol=1e-12)
        for name, value in expected_attributes.items()
    ):
        raise ValueError("AMPLFI event preprocessing attributes differ from training config")
    expected_source_samples = int(round(16 * sample_rate))
    if stored_ifos != ifos or strain.shape != (2, expected_source_samples):
        raise ValueError("AMPLFI event strain detector or duration contract mismatch")
    if (
        asd.shape != (2, asd_frequencies.size)
        or asd_frequencies.ndim != 1
        or not np.isfinite(asd).all()
        or np.any(asd <= 0)
        or not np.isfinite(strain).all()
    ):
        raise ValueError("AMPLFI event strain/ASD arrays are invalid")
    expected_asd_frequencies = np.fft.rfftfreq(
        expected_source_samples, d=1.0 / sample_rate
    )
    if not np.array_equal(asd_frequencies, expected_asd_frequencies):
        raise ValueError("AMPLFI event common ASD frequency grid is invalid")

    raw_duration = kernel + fduration
    event_offset = geocent_time - gps_start
    event_position_raw = kernel - right_pad + fduration / 2
    start_float = (event_offset - event_position_raw) * sample_rate
    start = int(round(start_float))
    stop = start + int(round(raw_duration * sample_rate))
    if not np.isclose(start_float, start, rtol=0.0, atol=1e-6):
        raise ValueError("AMPLFI event crop does not align to the native sample grid")
    if start < 0 or stop > strain.shape[-1]:
        raise ValueError("AMPLFI model window falls outside the common source")

    X = torch.as_tensor(strain[:, start:stop], dtype=torch.float32, device=device)[None]
    psd = torch.as_tensor(asd**2, dtype=torch.float64, device=device)[None]
    whitener = Whiten(fduration, sample_rate, highpass).to(device)
    whitened = whitener(X, psd)
    expected_model_samples = int(round(kernel * sample_rate))
    if whitened.shape != (1, 2, expected_model_samples):
        raise ValueError("AMPLFI whitened model input has an unexpected shape")
    output_frequencies = torch.fft.rfftfreq(
        expected_model_samples, d=1.0 / sample_rate, device=whitened.device
    )
    interpolated_psd = torch.nn.functional.interpolate(
        psd, size=(output_frequencies.numel(),), mode="linear"
    )
    frequency_mask = output_frequencies > highpass
    model_asd = torch.sqrt(interpolated_psd[:, :, frequency_mask]).float()
    if not torch.isfinite(whitened).all() or not torch.isfinite(model_asd).all():
        raise ValueError("AMPLFI preprocessed model context is non-finite")
    timing = {
        "gps_start": gps_start,
        "geocent_time": geocent_time,
        "source_event_offset_seconds": event_offset,
        "raw_crop_start_seconds": start / sample_rate,
        "raw_crop_duration_seconds": raw_duration,
        "model_event_offset_seconds": kernel - right_pad,
        "model_input_samples": expected_model_samples,
        "model_asd_bins": int(model_asd.shape[-1]),
    }
    return (whitened, model_asd), attributes, timing


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--event", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--model-config", required=True)
    result.add_argument("--native-prior", required=True)
    result.add_argument("--posterior-output", required=True)
    result.add_argument("--result-output", required=True)
    result.add_argument("--report-output", required=True)
    result.add_argument("--expected-event-sha256", required=True)
    result.add_argument("--expected-model-sha256", required=True)
    result.add_argument("--expected-model-config-sha256", required=True)
    result.add_argument("--expected-native-prior-sha256", required=True)
    result.add_argument("--num-samples", type=int, default=10000)
    result.add_argument("--sample-batch-size", type=int, default=1000)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260721)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.num_samples <= 0 or args.sample_batch_size <= 0:
        raise ValueError("AMPLFI sampling sizes must be positive")
    paths = {
        "event": Path(args.event).resolve(),
        "model": Path(args.model).resolve(),
        "model_config": Path(args.model_config).resolve(),
        "native_prior": Path(args.native_prior).resolve(),
    }
    expected = {
        "event": args.expected_event_sha256,
        "model": args.expected_model_sha256,
        "model_config": args.expected_model_config_sha256,
        "native_prior": args.expected_native_prior_sha256,
    }
    observed = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"AMPLFI {label} artifact is absent: {path}")
        observed[label] = file_sha256(path)
        if observed[label] != expected[label]:
            raise ValueError(f"AMPLFI {label} artifact hash mismatch")
    outputs = {
        "posterior": Path(args.posterior_output).resolve(),
        "native_result": Path(args.result_output).resolve(),
        "report": Path(args.report_output).resolve(),
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("AMPLFI inference refuses to overwrite an existing output")

    import torch
    from amplfi.train.data.datasets.testing import ra_from_phi

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("AMPLFI CUDA inference was requested but CUDA is unavailable")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    training = load_yaml(paths["model_config"])
    native_prior = load_yaml(paths["native_prior"])
    bounds = native_bounds(native_prior, training)

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    model = build_model(training, paths["model"], args.device)
    model_load_seconds = time.perf_counter() - load_started
    preprocessing_started = time.perf_counter()
    context, event_attributes, timing = load_and_preprocess_event(
        paths["event"], training, args.device
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    sampling_started = time.perf_counter()
    scaled_chunks = []
    log_prob_chunks = []
    with torch.inference_mode():
        remaining = args.num_samples
        while remaining:
            count = min(args.sample_batch_size, remaining)
            # MultiModalPsd v0.6 scales its ASD argument in place.  Give every
            # embedding call a fresh ASD tensor so sampling, log-probability
            # evaluation and later chunks see identical physical context.
            sample_context = (context[0], context[1].clone())
            scaled = model.model.sample(count, context=sample_context)
            probability_context = (context[0], context[1].clone())
            log_prob = model.model.log_prob(scaled, context=probability_context)
            if scaled.ndim != 3 or scaled.shape[1:] != (1, len(INFERENCE_PARAMETERS)):
                raise RuntimeError("AMPLFI flow returned an unexpected sample shape")
            scaled_chunks.append(scaled.squeeze(1))
            log_prob_chunks.append(log_prob.squeeze(1))
            remaining -= count
        scaled = torch.cat(scaled_chunks, dim=0)
        log_prob = torch.cat(log_prob_chunks, dim=0)
        samples = model.scale(scaled, reverse=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    sampling_seconds = time.perf_counter() - sampling_started
    finite = torch.isfinite(samples).all(dim=1) & torch.isfinite(log_prob)
    for index, name in enumerate(INFERENCE_PARAMETERS):
        low, high = bounds[name]
        finite &= (samples[:, index] >= low) & (samples[:, index] <= high)
    retained = samples[finite]
    retained_log_prob = log_prob[finite]
    if retained.shape[0] == 0:
        raise RuntimeError("AMPLFI flow returned no finite samples inside its training support")

    native_arrays = {
        name: retained[:, index].detach().cpu().numpy().astype(np.float64)
        for index, name in enumerate(INFERENCE_PARAMETERS)
    }
    native_arrays["log_prob"] = retained_log_prob.detach().cpu().numpy().astype(np.float64)
    phi = retained[:, INFERENCE_PARAMETERS.index("phi")]
    ra = ra_from_phi(phi, timing["geocent_time"])
    posterior_arrays = dict(native_arrays)
    posterior_arrays.update(
        {
            "luminosity_distance": native_arrays["distance"],
            "theta_jn": native_arrays["inclination"],
            "ra": ra.detach().cpu().numpy().astype(np.float64),
        }
    )
    if any(not np.isfinite(values).all() for values in posterior_arrays.values()):
        raise ValueError("AMPLFI posterior contains non-finite values")
    atomic_npz(outputs["posterior"], posterior_arrays)
    atomic_hdf5(
        outputs["native_result"],
        native_arrays,
        {
            "schema": "gwyolo-real-amplfi-flow-posterior-v1",
            "event_sha256": observed["event"],
            "model_sha256": observed["model"],
            "model_config_sha256": observed["model_config"],
            "native_prior_sha256": observed["native_prior"],
            "geocent_time": timing["geocent_time"],
            "requested_samples": args.num_samples,
            "retained_samples": int(retained.shape[0]),
        },
    )
    total_seconds = time.perf_counter() - total_started
    report = {
        "status": "real_amplfi_flow_posterior_complete",
        "backend": "AMPLFI",
        "backend_version": importlib.metadata.version("amplfi"),
        "event_path": str(paths["event"]),
        "event_sha256": observed["event"],
        "model_path": str(paths["model"]),
        "model_sha256": observed["model"],
        "model_config_path": str(paths["model_config"]),
        "model_config_sha256": observed["model_config"],
        "native_prior_path": str(paths["native_prior"]),
        "native_prior_sha256": observed["native_prior"],
        "posterior_path": str(outputs["posterior"]),
        "posterior_sha256": file_sha256(outputs["posterior"]),
        "native_result_path": str(outputs["native_result"]),
        "native_result_sha256": file_sha256(outputs["native_result"]),
        "native_parameters": list(INFERENCE_PARAMETERS),
        "reported_parameters": sorted(posterior_arrays),
        "requested_samples": args.num_samples,
        "posterior_samples": int(retained.shape[0]),
        "retained_fraction": float(retained.shape[0] / args.num_samples),
        "effective_sample_size": float(retained.shape[0]),
        "importance_sampled": False,
        "sample_batch_size": args.sample_batch_size,
        "seed": args.seed,
        "event_attributes": event_attributes,
        "timing": timing,
        "asd_context_cloned_per_embedding_call": True,
        "model_load_seconds": model_load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "sampling_seconds": sampling_seconds,
        "latency_seconds": total_seconds,
        "latency_scope": "verified-source-and-model-load-through-posterior-write",
        "device": args.device,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(outputs["report"], report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
