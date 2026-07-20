from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten, _whiten_with_reference
from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    load_yaml,
    training_tensor_config,
)
from .trigger import (
    _coherence_settings,
    coherence_assisted_summary,
    probability_summaries,
)
from .runtime import execution_provenance
from .waveforms import _atomic_save_npz, load_materialized_context


def _load_resumable_rows(
    output: Path,
    run_identity: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_path = output / "injection_score_state.json"
    partial_path = output / "injection_triggers.partial.jsonl"
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("run_identity") != run_identity:
            raise ValueError("Existing injection-score state belongs to a different run")
    elif partial_path.is_file():
        raise ValueError("Partial injection triggers exist without a run-identity state")
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
    requested = {str(row["injection_id"]) for row in manifest_rows}
    completed = set()
    for row in rows:
        injection_id = str(row["injection_id"])
        if injection_id not in requested or injection_id in completed:
            raise ValueError("Partial injection scores contain unexpected or duplicate IDs")
        if run_identity["save_probabilities"]:
            path = row.get("probability_path")
            expected_sha = row.get("probability_sha256")
            if not path or not expected_sha or file_sha256(path) != expected_sha:
                raise ValueError(f"Partial probability hash mismatch for {injection_id}")
        completed.add(injection_id)
    return rows


def _save_progress(
    output: Path,
    rows: list[dict[str, Any]],
    run_identity: dict[str, Any],
    requested: int,
    status: str = "in_progress",
) -> None:
    atomic_write_text(
        output / "injection_triggers.partial.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    atomic_write_json(
        output / "injection_score_state.json",
        {
            "status": status,
            "run_identity": run_identity,
            "completed": len(rows),
            "requested": requested,
        },
    )


def score_materialized_injections(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    save_probabilities: bool = False,
    required_split: str | None = None,
    enabled_ifos: tuple[str, ...] | None = None,
    coherence_config_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Injection scoring requires torch") from exc
    from .numeric import model_from_checkpoint

    config = load_yaml(config_path)
    tensor_config = training_tensor_config(config)
    coherence = _coherence_settings(coherence_config_path, target_sample_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    enabled_ifos = model_ifos if enabled_ifos is None else enabled_ifos
    if (
        not enabled_ifos
        or len(set(enabled_ifos)) != len(enabled_ifos)
        or not set(enabled_ifos).issubset(model_ifos)
    ):
        raise ValueError("enabled_ifos must be a non-empty unique subset of model_ifos")
    expected_channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != expected_channels:
        raise ValueError(
            f"Checkpoint has {checkpoint['input_channels']} channels; scorer requires {expected_channels}"
        )
    model, architecture = model_from_checkpoint(checkpoint, model_ifos, q_values)
    model = model.to(device)
    model.eval()
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    if not manifest_rows:
        raise ValueError("Materialized injection manifest cannot be empty")
    observed_splits = sorted({str(row.get("split")) for row in manifest_rows})
    if required_split is not None and observed_splits != [required_split]:
        raise ValueError(
            f"Injection scorer required split {required_split!r}, observed {observed_splits}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": file_sha256(config_path),
        "injection_ids_hash": canonical_hash(
            [str(row["injection_id"]) for row in manifest_rows], 64
        ),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "save_probabilities": save_probabilities,
        "architecture": architecture,
        "enabled_ifos": list(enabled_ifos),
        "coherence_config_sha256": (
            coherence["config_sha256"] if coherence is not None else None
        ),
        "whitening": str(tensor_config.get("whitening", "self")),
        "required_split": required_split,
        "code_commit": execution_provenance()["code_commit"],
    }
    resumed_rows = _load_resumable_rows(output, run_identity, manifest_rows)
    resumed_by_id = {str(row["injection_id"]): row for row in resumed_rows}
    output_rows = []
    failures = []
    newly_scored = 0
    verified_background_hashes: dict[str, str] = {}
    started = time.time()
    for row in manifest_rows:
        injection_id = str(row["injection_id"])
        if injection_id in resumed_by_id:
            output_rows.append(resumed_by_id[injection_id])
            continue
        try:
            context = load_materialized_context(row, verified_background_hashes)
            source_rate = int(context["sample_rate"])
            ifos = list(context["ifos"])
            context_start = float(context["context_gps_start"])
            analysis_start = float(context["analysis_gps_start"])
            source_start = int(context["analysis_start_index"])
            source_stop = int(context["analysis_stop_index"])
            source_duration = (source_stop - source_start) / source_rate
            mixture = np.asarray(context["mixture"], dtype=np.float64)
            noise = np.asarray(context["noise"], dtype=np.float64)
            if source_rate < target_sample_rate or source_rate % target_sample_rate:
                raise ValueError("materialized sample rate must be an integer multiple of target")
            transformed = []
            valid_ifos = [ifo for ifo in model_ifos if ifo in ifos and ifo in enabled_ifos]
            if not valid_ifos:
                raise ValueError("materialized context has no enabled detector")
            output_samples = int(round(source_duration * target_sample_rate))
            target_start = int(round((analysis_start - context_start) * target_sample_rate))
            for ifo in model_ifos:
                if ifo not in valid_ifos:
                    transformed.append(np.zeros(output_samples, dtype=np.float32))
                    continue
                ifo_index = ifos.index(ifo)
                values = mixture[ifo_index]
                values = _fft_downsample(values, source_rate, target_sample_rate)
                whitening = str(tensor_config.get("whitening", "self"))
                if whitening == "self":
                    whitened = _whiten(values)
                elif whitening == "noise_reference":
                    reference = _fft_downsample(
                        noise[ifo_index], source_rate, target_sample_rate
                    )
                    whitened = _whiten_with_reference(reference, values)
                else:
                    raise ValueError("injection whitening must be self or noise_reference")
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
                feature_tensor = torch.from_numpy(features).to(device)
                if architecture == "detector_set":
                    availability = torch.as_tensor(
                        [[ifo in valid_ifos for ifo in model_ifos]],
                        dtype=feature_tensor.dtype,
                        device=device,
                    )
                    logits = model(feature_tensor, availability)
                else:
                    logits = model(feature_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            probabilities = probabilities.reshape(
                2, len(model_ifos), len(q_values), power.shape[-2], power.shape[-1]
            )
            summary = probability_summaries(
                probabilities,
                model_ifos,
                valid_ifos,
                analysis_start,
                source_duration,
            )
            if coherence is not None:
                summary.update(
                    coherence_assisted_summary(
                        strain,
                        model_ifos,
                        valid_ifos,
                        target_sample_rate,
                        summary["peak_times"]["chirp"],
                        summary["ranking_score"],
                        coherence["limits_seconds"],
                        coherence["timing_uncertainty_seconds"],
                        coherence["roi_duration_seconds"],
                    )
                )
            probability_record = {}
            if save_probabilities:
                probability_path = output / "probabilities" / f"{row['injection_id']}.npz"
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
            output_row = {
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
                    "source_ifos": ifos,
                    "enabled_ifos": list(enabled_ifos),
                    "padded_ifos": [ifo for ifo in model_ifos if ifo not in valid_ifos],
                    **probability_record,
                    **summary,
                }
            output_rows.append(output_row)
            newly_scored += 1
            if newly_scored % 5 == 0:
                _save_progress(output, output_rows, run_identity, len(manifest_rows))
        except (ValueError, OSError, KeyError) as exc:
            failures.append({"injection_id": row.get("injection_id"), "error": str(exc)})
    _save_progress(output, output_rows, run_identity, len(manifest_rows))
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
        "config_sha256": file_sha256(config_path),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "architecture": architecture,
        "enabled_ifos": list(enabled_ifos),
        "coherence": coherence,
        "target_sample_rate": target_sample_rate,
        "probabilities_saved": save_probabilities,
        "required_split": required_split,
        "observed_splits": observed_splits,
        "run_identity_hash": canonical_hash(run_identity, 64),
        "input_injections": len(manifest_rows),
        "resumed_injections": len(resumed_rows),
        "newly_scored_injections": newly_scored,
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
        **execution_provenance(torch),
    }
    atomic_write_json(output / "injection_score_report.json", report)
    _save_progress(
        output,
        output_rows,
        run_identity,
        len(manifest_rows),
        "failed" if failures else "complete",
    )
    if failures:
        raise RuntimeError(
            f"Injection scoring failed for {len(failures)} rows; inspect "
            f"{output / 'injection_score_report.json'}"
        )
    return report
