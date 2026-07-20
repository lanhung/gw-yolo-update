from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .waveforms import _atomic_save_npz


def _load_resumable_trigger_rows(
    output: Path,
    run_identity: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_path = output / "trigger_score_state.json"
    partial_path = output / "background_triggers.partial.jsonl"
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing trigger-score state belongs to a different run")
    elif partial_path.is_file():
        raise ValueError("Partial background triggers exist without a run-identity state")
    else:
        atomic_write_json(
            state_path,
            {
                "status": "in_progress",
                "run_identity": run_identity,
                "completed": 0,
                "requested": len(manifest_rows),
            },
        )
        return []
    if not partial_path.is_file():
        return []
    with partial_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    requested = {str(row["window_id"]) for row in manifest_rows}
    completed = set()
    for row in rows:
        window_id = str(row["window_id"])
        if window_id not in requested or window_id in completed:
            raise ValueError("Partial background scores contain unexpected or duplicate IDs")
        if run_identity.get("save_probabilities"):
            path = row.get("probability_path")
            expected_sha = row.get("probability_sha256")
            if not path or not expected_sha or file_sha256(path) != expected_sha:
                raise ValueError(f"Partial probability hash mismatch for {window_id}")
        completed.add(window_id)
    return rows


def _save_trigger_progress(
    output: Path,
    rows: list[dict[str, Any]],
    run_identity: dict[str, Any],
    requested: int,
    status: str = "in_progress",
) -> None:
    atomic_write_text(
        output / "background_triggers.partial.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    atomic_write_json(
        output / "trigger_score_state.json",
        {
            "status": status,
            "run_identity": run_identity,
            "completed": len(rows),
            "requested": requested,
        },
    )


def network_ranking(
    chirp_scores: dict[str, float],
    glitch_scores: dict[str, float],
    valid_ifos: list[str],
) -> dict[str, Any]:
    if not valid_ifos:
        raise ValueError("at least one valid IFO is required")
    missing = [ifo for ifo in valid_ifos if ifo not in chirp_scores or ifo not in glitch_scores]
    if missing:
        raise ValueError(f"Missing scores for valid IFOs: {missing}")
    ordered_chirp = sorted((float(chirp_scores[ifo]) for ifo in valid_ifos), reverse=True)
    coherent_score = ordered_chirp[1] if len(ordered_chirp) >= 2 else ordered_chirp[0]
    maximum_glitch = max(float(glitch_scores[ifo]) for ifo in valid_ifos)
    return {
        "ranking_score": coherent_score,
        "ranking_definition": "second-highest valid-IFO chirp maximum; single-IFO maximum if needed",
        "network_mode": "coincident" if len(valid_ifos) >= 2 else "single_ifo_diagnostic",
        "valid_ifos": valid_ifos,
        "maximum_chirp_score": ordered_chirp[0],
        "coherent_chirp_score": coherent_score,
        "maximum_glitch_score": maximum_glitch,
        "chirp_glitch_margin": coherent_score - maximum_glitch,
    }


def probability_summaries(
    probabilities: np.ndarray,
    model_ifos: tuple[str, ...],
    valid_ifos: list[str],
    gps_start: float,
    duration: float,
) -> dict[str, Any]:
    if probabilities.ndim != 5 or probabilities.shape[0] != 2:
        raise ValueError("probabilities must have shape [2, IFO, Q, frequency, time]")
    if probabilities.shape[1] != len(model_ifos):
        raise ValueError("probability IFO axis does not match model_ifos")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    chirp_scores = {
        ifo: float(np.max(probabilities[0, index])) for index, ifo in enumerate(model_ifos)
    }
    glitch_scores = {
        ifo: float(np.max(probabilities[1, index])) for index, ifo in enumerate(model_ifos)
    }
    peak_times = {"chirp": {}, "glitch": {}}
    for class_index, class_name in enumerate(("chirp", "glitch")):
        for ifo_index, ifo in enumerate(model_ifos):
            profile = np.max(probabilities[class_index, ifo_index], axis=(0, 1))
            peak_index = int(np.argmax(profile))
            peak_offset = (peak_index + 0.5) / profile.size * duration
            peak_times[class_name][ifo] = {
                "gps": gps_start + peak_offset,
                "offset_seconds": peak_offset,
                "time_bin": peak_index,
                "time_bins": int(profile.size),
                "score": float(profile[peak_index]),
            }
    return {
        "chirp_scores": chirp_scores,
        "glitch_scores": glitch_scores,
        "peak_times": peak_times,
        **network_ranking(chirp_scores, glitch_scores, valid_ifos),
    }


def _window_strain(
    row: dict[str, Any],
    model_ifos: tuple[str, ...],
    target_sample_rate: int,
    context_duration: float,
) -> tuple[np.ndarray, list[str]]:
    center = (float(row["gps_start"]) + float(row["gps_end"])) / 2.0
    window_duration = float(row["duration"])
    valid_ifos = [str(item) for item in row["ifos"]]
    context_by_ifo = {}
    for ifo in valid_ifos:
        source = row["source_files"][ifo]["path"]
        segment = read_hdf5_segment(source, center, context_duration)
        context_by_ifo[ifo] = _fft_downsample(
            segment["strain"], segment["sample_rate"], target_sample_rate
        )
    output_samples = int(round(window_duration * target_sample_rate))
    strains = []
    for ifo in model_ifos:
        if ifo not in context_by_ifo:
            strains.append(np.zeros(output_samples, dtype=np.float32))
            continue
        whitened = _whiten(context_by_ifo[ifo])
        center_index = whitened.size // 2
        start = center_index - output_samples // 2
        strains.append(whitened[start : start + output_samples])
    return np.stack(strains), valid_ifos


def score_background_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
    save_probabilities: bool = False,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Trigger scoring requires torch") from exc
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
        raise ValueError("Background manifest cannot be empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": file_sha256(config_path),
        "window_ids_hash": canonical_hash(
            [str(row["window_id"]) for row in manifest_rows], 64
        ),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "save_probabilities": save_probabilities,
    }
    resumed_rows = _load_resumable_trigger_rows(output, run_identity, manifest_rows)
    resumed_by_id = {str(row["window_id"]): row for row in resumed_rows}
    rows = []
    started = time.time()
    failures = []
    newly_scored = 0
    for row in manifest_rows:
        window_id = str(row["window_id"])
        if window_id in resumed_by_id:
            rows.append(resumed_by_id[window_id])
            continue
        try:
            strain, valid_ifos = _window_strain(
                row, model_ifos, target_sample_rate, context_duration
            )
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
                valid_ifos,
                float(row["gps_start"]),
                float(row["duration"]),
            )
            probability_record = {}
            if save_probabilities:
                probability_path = output / "probabilities" / f"{row['window_id']}.npz"
                _atomic_save_npz(
                    probability_path,
                    chirp_probability=probabilities[0].astype(np.float16),
                    glitch_probability=probabilities[1].astype(np.float16),
                    ifos=np.asarray(model_ifos),
                    q_values=np.asarray(q_values, dtype=np.float32),
                )
                probability_record = {
                    "probability_path": str(probability_path),
                    "probability_sha256": file_sha256(probability_path),
                }
            rows.append(
                {
                    "window_id": row["window_id"],
                    "split": row["split"],
                    "gps_start": row["gps_start"],
                    "gps_end": row["gps_end"],
                    "gps_block": row["gps_block"],
                    "padded_ifos": [ifo for ifo in model_ifos if ifo not in valid_ifos],
                    **probability_record,
                    **summary,
                }
            )
            newly_scored += 1
            if newly_scored % 5 == 0:
                _save_trigger_progress(output, rows, run_identity, len(manifest_rows))
        except (ValueError, OSError, KeyError) as exc:
            failures.append({"window_id": row.get("window_id"), "error": str(exc)})
    _save_trigger_progress(output, rows, run_identity, len(manifest_rows))
    triggers_path = output / "background_triggers.jsonl"
    atomic_write_text(
        triggers_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report = {
        "status": "real_o4a_domain_transfer_diagnostic",
        "scientific_claim_allowed": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "probabilities_saved": save_probabilities,
        "run_identity_hash": canonical_hash(run_identity, 64),
        "input_windows": len(manifest_rows),
        "resumed_windows": len(resumed_rows),
        "newly_scored_windows": newly_scored,
        "scored_windows": len(rows),
        "failed_windows": len(failures),
        "failures": failures,
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "network_mode_counts": dict(
            sorted(Counter(row["network_mode"] for row in rows).items())
        ),
        "triggers_path": str(triggers_path),
        "triggers_sha256": file_sha256(triggers_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "trigger_score_report.json", report)
    _save_trigger_progress(
        output,
        rows,
        run_identity,
        len(manifest_rows),
        "failed" if failures else "complete",
    )
    if failures:
        raise RuntimeError(
            f"Trigger scoring failed for {len(failures)} windows; inspect "
            f"{output / 'trigger_score_report.json'}"
        )
    return report
