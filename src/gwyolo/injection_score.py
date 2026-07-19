from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, file_sha256, load_yaml
from .trigger import probability_summaries


def score_materialized_injections(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Injection scoring requires torch") from exc
    from .numeric import MultiIFOQNet

    config = load_yaml(config_path)
    tensor_config = config["numeric_training"]["tensor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != expected_channels:
        raise ValueError(
            f"Checkpoint has {checkpoint['input_channels']} channels; scorer requires {expected_channels}"
        )
    model = MultiIFOQNet(expected_channels, int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    if not manifest_rows:
        raise ValueError("Materialized injection manifest cannot be empty")
    output_rows = []
    failures = []
    verified_background_hashes: dict[str, str] = {}
    started = time.time()
    for row in manifest_rows:
        try:
            artifact = Path(row["materialized_path"])
            if file_sha256(artifact) != str(row["materialized_sha256"]):
                raise ValueError("materialized array hash mismatch")
            with np.load(artifact, allow_pickle=False) as arrays:
                source_rate = int(arrays["sample_rate"])
                ifos = [str(value) for value in arrays["ifos"].tolist()]
                context_start = float(arrays["context_gps_start"])
                analysis_start = float(arrays["analysis_gps_start"])
                source_start = int(arrays["analysis_start_index"])
                source_stop = int(arrays["analysis_stop_index"])
                source_duration = (source_stop - source_start) / source_rate
                if "strain" in arrays:
                    mixture = np.asarray(arrays["strain"], dtype=np.float64)
                else:
                    signal = np.asarray(arrays["signal"], dtype=np.float64)
                    context_duration = signal.shape[1] / source_rate
                    context_center = context_start + context_duration / 2.0
                    detector_noise = []
                    for ifo in ifos:
                        source = row["background_source_files"][ifo]
                        source_path = str(source["path"])
                        expected_hash = str(source["sha256"])
                        actual_hash = verified_background_hashes.get(source_path)
                        if actual_hash is None:
                            actual_hash = file_sha256(source_path)
                            verified_background_hashes[source_path] = actual_hash
                        if actual_hash != expected_hash:
                            raise ValueError(f"background source hash mismatch for {ifo}")
                        segment = read_hdf5_segment(
                            source_path, context_center, context_duration
                        )
                        detector_noise.append(
                            _fft_downsample(
                                np.asarray(segment["strain"], dtype=np.float64),
                                int(segment["sample_rate"]),
                                source_rate,
                            )
                        )
                    noise = np.stack(detector_noise)
                    if noise.shape != signal.shape:
                        raise ValueError("reconstructed background shape differs from signal")
                    mixture = noise + signal
            if source_rate < target_sample_rate or source_rate % target_sample_rate:
                raise ValueError("materialized sample rate must be an integer multiple of target")
            transformed = []
            output_samples = int(round(source_duration * target_sample_rate))
            target_start = int(round((analysis_start - context_start) * target_sample_rate))
            for ifo in model_ifos:
                if ifo not in ifos:
                    transformed.append(np.zeros(output_samples, dtype=np.float32))
                    continue
                values = mixture[ifos.index(ifo)]
                values = _fft_downsample(values, source_rate, target_sample_rate)
                whitened = _whiten(values)
                transformed.append(whitened[target_start : target_start + output_samples])
            strain = np.stack(transformed)
            if strain.shape[1] != output_samples:
                raise ValueError("analysis crop is incomplete after downsampling")
            power = multiresolution_power(
                strain,
                target_sample_rate,
                q_values,
                int(tensor_config["frequency_bins"]),
                int(tensor_config["time_bins"]),
                float(tensor_config["fmin"]),
                float(tensor_config["fmax"]),
            )
            features = _normalize_power(power).reshape(
                1, expected_channels, power.shape[-2], power.shape[-1]
            )
            with torch.no_grad():
                logits = model(torch.from_numpy(features).to(device))
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            probabilities = probabilities.reshape(
                2, len(model_ifos), len(q_values), power.shape[-2], power.shape[-1]
            )
            summary = probability_summaries(
                probabilities,
                model_ifos,
                ifos,
                analysis_start,
                source_duration,
            )
            output_rows.append(
                {
                    "injection_id": row["injection_id"],
                    "waveform_id": row["waveform_id"],
                    "split": row["split"],
                    "source_family": row["source_family"],
                    "stratum": row["source_family"],
                    "gps_block": row["gps_block"],
                    "gps_time": row["gps_time"],
                    "redshift": row["redshift"],
                    "luminosity_distance_mpc": row["luminosity_distance_mpc"],
                    "vt_weight": row["vt_weight"],
                    "vt_weight_unit": row["vt_weight_unit"],
                    "materialized_sha256": row["materialized_sha256"],
                    "padded_ifos": [ifo for ifo in model_ifos if ifo not in ifos],
                    **summary,
                }
            )
        except (ValueError, OSError, KeyError) as exc:
            failures.append({"injection_id": row.get("injection_id"), "error": str(exc)})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    triggers_path = output / "injection_triggers.jsonl"
    atomic_write_text(
        triggers_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
    )
    report = {
        "status": "physical_waveform_real_noise_domain_transfer_diagnostic",
        "scientific_claim_allowed": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "input_injections": len(manifest_rows),
        "scored_injections": len(output_rows),
        "failed_injections": len(failures),
        "failures": failures,
        "family_counts": dict(
            sorted(Counter(row["source_family"] for row in output_rows).items())
        ),
        "triggers_path": str(triggers_path),
        "triggers_sha256": file_sha256(triggers_path),
        "elapsed_seconds": time.time() - started,
        "preprocessing": "full-context PSD whitening then central analysis-window crop",
    }
    atomic_write_json(output / "injection_score_report.json", report)
    return report
