from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, file_sha256, load_yaml
from .pe import (
    PAIRED_PE_LATENCY_SCOPE_V1,
    posterior_sky_area_equal_solid_angle,
    sky_area_estimator_identity,
    validate_paired_pe_latency,
)
from .runtime import execution_provenance


CONDITIONS = ("clean", "contaminated", "mask_conditioned")


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
