from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .factory import multiresolution_power
from .gwosc import _whiten
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .physical_training import relative_component_mask, scale_component_for_transform
from .runtime import execution_provenance


AUTOMATIC_MASK_SOURCE = "isolated_real_glitch_component_power_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"automatic mask manifest is empty or invalid: {path}")
    return rows


def audit_automatic_mask_policy(
    overlap_manifest_path: str | Path,
    overlap_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Recompute every validation mask from isolated physical components.

    Chirp support is derived from the isolated injected waveform. Glitch support
    is a deterministic pseudo-mask derived from the isolated real-glitch strain.
    Neither target is a human annotation, and no pixel-accuracy claim is made.
    """

    manifest = Path(overlap_manifest_path).resolve()
    config_file = Path(overlap_config_path).resolve()
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("automatic mask policy audits are immutable")
    if not manifest.is_file() or not config_file.is_file():
        raise FileNotFoundError("automatic mask audit input is absent")
    rows = _jsonl(manifest)
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("automatic mask publication audit is validation-only")
    config = load_yaml(config_file)
    settings = config.get("overlap_factory")
    tensor = settings.get("tensor") if isinstance(settings, dict) else None
    if not isinstance(tensor, dict):
        raise ValueError("automatic mask audit lacks overlap tensor settings")
    model_ifos = [str(value) for value in settings["model_ifos"]]
    q_values = tuple(float(value) for value in settings["q_values"])
    sample_rate = int(settings["target_sample_rate"])
    frequency_bins = int(tensor["frequency_bins"])
    time_bins = int(tensor["time_bins"])
    fmin = float(tensor["fmin"])
    fmax = float(tensor["fmax"])
    chirp_fraction = float(tensor["mask_fraction"])
    glitch_fraction = float(tensor["glitch_mask_fraction"])
    if (
        tensor.get("glitch_mask_source") != AUTOMATIC_MASK_SOURCE
        or tensor.get("manual_annotation_required") is not False
        or not 0 < min(chirp_fraction, glitch_fraction)
        or max(chirp_fraction, glitch_fraction) >= 1
    ):
        raise ValueError("automatic mask audit policy differs from the frozen contract")

    seen: dict[str, set[str]] = {
        field: set()
        for field in (
            "mixture_id",
            "injection_id",
            "waveform_id",
            "glitch_id",
        )
    }
    support = []
    labels: Counter[str] = Counter()
    detector_sets: Counter[str] = Counter()
    gps_blocks = set()
    for index, row in enumerate(rows):
        for field, values in seen.items():
            value = str(row.get(field, ""))
            if not value or value in values:
                raise ValueError(
                    f"automatic mask audit repeats {field} at row {index}"
                )
            values.add(value)
        artifact = Path(str(row.get("path", ""))).resolve()
        if (
            not artifact.is_file()
            or row.get("sha256") != file_sha256(artifact)
            or row.get("mask_provenance") != AUTOMATIC_MASK_SOURCE
            or row.get("automatic_pseudo_mask") is not True
            or row.get("human_pixel_mask") is not False
            or not np.isclose(
                float(row.get("mask_fraction", -1)),
                glitch_fraction,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"automatic mask artifact identity failed at row {index}"
            )
        with np.load(artifact, allow_pickle=False) as arrays:
            required = {
                "chirp_mask",
                "glitch_mask",
                "raw_glitch_strain",
                "target_signal_strain",
                "detector_availability",
                "ifos",
                "q_values",
                "sample_rate",
            }
            if required - set(arrays.files):
                raise ValueError(
                    f"automatic mask artifact is incomplete at row {index}"
                )
            ifos = [str(value) for value in arrays["ifos"].tolist()]
            stored_q = tuple(float(value) for value in arrays["q_values"].tolist())
            stored_rate = int(arrays["sample_rate"])
            availability = np.asarray(
                arrays["detector_availability"], dtype=np.uint8
            )
            glitch_strain = np.asarray(
                arrays["raw_glitch_strain"], dtype=np.float64
            )
            signal_strain = np.asarray(
                arrays["target_signal_strain"], dtype=np.float64
            )
            observed_chirp = np.asarray(arrays["chirp_mask"], dtype=np.uint8)
            observed_glitch = np.asarray(arrays["glitch_mask"], dtype=np.uint8)
        expected_shape = (
            len(model_ifos),
            len(q_values),
            frequency_bins,
            time_bins,
        )
        if (
            ifos != model_ifos
            or stored_q != q_values
            or stored_rate != sample_rate
            or availability.shape != (len(model_ifos),)
            or glitch_strain.shape != signal_strain.shape
            or glitch_strain.ndim != 2
            or glitch_strain.shape[0] != len(model_ifos)
            or observed_chirp.shape != expected_shape
            or observed_glitch.shape != expected_shape
            or not np.isfinite(glitch_strain).all()
            or not np.isfinite(signal_strain).all()
        ):
            raise ValueError(
                f"automatic mask tensor contract failed at row {index}"
            )
        whitened_glitch = np.zeros_like(glitch_strain)
        for detector_index in np.flatnonzero(availability):
            whitened_glitch[detector_index] = _whiten(
                glitch_strain[detector_index]
            )
        glitch_power = multiresolution_power(
            whitened_glitch,
            sample_rate,
            q_values,
            frequency_bins,
            time_bins,
            fmin,
            fmax,
        )
        chirp_power = multiresolution_power(
            scale_component_for_transform(signal_strain),
            sample_rate,
            q_values,
            frequency_bins,
            time_bins,
            fmin,
            fmax,
        )
        expected_glitch = relative_component_mask(
            glitch_power, glitch_fraction
        ).astype(np.uint8)
        expected_chirp = relative_component_mask(
            chirp_power, chirp_fraction
        ).astype(np.uint8)
        if not np.array_equal(observed_glitch, expected_glitch) or not np.array_equal(
            observed_chirp, expected_chirp
        ):
            raise ValueError(
                f"automatic mask replay differs at row {index}"
            )
        support.append(
            {
                "mixture_id": row["mixture_id"],
                "chirp_pixels": int(np.count_nonzero(observed_chirp)),
                "glitch_pixels": int(np.count_nonzero(observed_glitch)),
                "available_ifos": int(np.count_nonzero(availability)),
            }
        )
        labels[str(row.get("ml_label", "unknown"))] += 1
        detector_sets["+".join(row.get("available_ifos", []))] += 1
        gps_blocks.add(str(row["network_gps_block"]))

    report = {
        "status": "verified_validation_automatic_mask_policy",
        "passed": True,
        "validation_only": True,
        "test_rows_read": 0,
        "scientific_claim_allowed": False,
        "human_annotation_used": False,
        "human_ground_truth_claimed": False,
        "pixel_accuracy_claim_allowed": False,
        "automatic_glitch_masks_are_pseudo_labels": True,
        "soft_model_probabilities_required_downstream": True,
        "unknown_glitch_abstention_required": True,
        "rows": len(rows),
        "unique_injections": len(seen["injection_id"]),
        "unique_waveforms": len(seen["waveform_id"]),
        "unique_glitches": len(seen["glitch_id"]),
        "unique_gps_blocks": len(gps_blocks),
        "labels": dict(sorted(labels.items())),
        "detector_sets": dict(sorted(detector_sets.items())),
        "zero_chirp_masks": sum(row["chirp_pixels"] == 0 for row in support),
        "zero_glitch_masks": sum(row["glitch_pixels"] == 0 for row in support),
        "mask_policy": {
            "chirp_source": "isolated_injected_waveform_component_power_v1",
            "glitch_source": AUTOMATIC_MASK_SOURCE,
            "chirp_fraction": chirp_fraction,
            "glitch_fraction": glitch_fraction,
            "whitening": "per_available_ifo_self_whitening",
            "transform": "fresh_multi_q_numeric_power",
            "selection": "fixed_fraction_of_per_plane_peak",
            "manual_annotation_required": False,
        },
        "support_summary": {
            "chirp_pixels_min": min(row["chirp_pixels"] for row in support),
            "chirp_pixels_median": float(
                np.median([row["chirp_pixels"] for row in support])
            ),
            "glitch_pixels_min": min(row["glitch_pixels"] for row in support),
            "glitch_pixels_median": float(
                np.median([row["glitch_pixels"] for row in support])
            ),
        },
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "config_hash": canonical_hash(config),
        **execution_provenance(),
    }
    atomic_write_json(target, report)
    return report


def bind_raw_mask_automatic_publication_evidence(
    raw_mask_endpoint_path: str | Path,
    automatic_mask_audit_path: str | Path,
    gate_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind functional mask gains to a reproducible non-human mask policy."""

    raw_path = Path(raw_mask_endpoint_path).resolve()
    audit_path = Path(automatic_mask_audit_path).resolve()
    config_path = Path(gate_config_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("automatic raw/mask endpoint bindings are immutable")
    if not raw_path.is_file() or not audit_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("automatic raw/mask binding input is absent")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config = load_yaml(config_path)
    gate = config.get("automatic_mask_publication_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("schema") != "automatic_mask_publication_gate_v1"
        or gate.get("manual_annotation_required") is not False
        or gate.get("human_ground_truth_claimed") is not False
        or gate.get("require_soft_model_probabilities") is not True
        or gate.get("require_unknown_glitch_abstention") is not True
    ):
        raise ValueError("automatic mask publication gate is invalid")
    dependence = raw.get("background_dependence_audits", {})
    injection_bootstrap = raw.get("injection_bootstrap_independence", {})
    if (
        raw.get("status")
        != "bound_validation_raw_mask_continuous_background_evidence"
        or raw.get("passed") is not True
        or raw.get("mask_locked_test_arm_eligible") is not True
        or raw.get("scientific_claim_allowed") is not False
        or int(raw.get("test_rows_read", -1)) != 0
        or any(
            dependence.get(arm, {}).get("passed") is not True
            for arm in ("raw", "mask")
        )
        or injection_bootstrap.get("passed") is not True
    ):
        raise ValueError("functional raw/mask endpoint failed automatic binding")
    if (
        audit.get("status") != "verified_validation_automatic_mask_policy"
        or audit.get("passed") is not True
        or audit.get("validation_only") is not True
        or audit.get("human_annotation_used") is not False
        or audit.get("human_ground_truth_claimed") is not False
        or audit.get("automatic_glitch_masks_are_pseudo_labels") is not True
        or audit.get("soft_model_probabilities_required_downstream") is not True
        or audit.get("unknown_glitch_abstention_required") is not True
        or int(audit.get("test_rows_read", -1)) != 0
        or not Path(str(audit.get("manifest_path", ""))).is_file()
        or audit.get("manifest_sha256")
        != file_sha256(audit["manifest_path"])
        or not Path(str(audit.get("config_path", ""))).is_file()
        or audit.get("config_sha256") != file_sha256(audit["config_path"])
    ):
        raise ValueError("automatic mask policy audit failed replay")
    checks = {
        "minimum_rows": int(audit["rows"]) >= int(gate["minimum_rows"]),
        "minimum_unique_glitches": int(audit["unique_glitches"])
        >= int(gate["minimum_unique_glitches"]),
        "minimum_gps_blocks": int(audit["unique_gps_blocks"])
        >= int(gate["minimum_gps_blocks"]),
        "minimum_labels": len(audit["labels"]) >= int(gate["minimum_labels"]),
        # A visibility-gated injection may legitimately have an empty target
        # mask. Preserve those hard/null rows rather than filtering them.
        "chirp_masks_replayed": True,
        "nonempty_glitch_masks": int(audit["zero_glitch_masks"]) == 0,
        "automatic_replay": True,
        "functional_raw_mask_endpoint": True,
    }
    passed = all(checks.values())
    result = {
        "status": "bound_validation_raw_mask_automatic_evidence",
        "passed": passed,
        "mask_locked_test_arm_eligible": passed,
        "functional_raw_mask_endpoint_passed": True,
        "automatic_mask_policy_passed": passed,
        "human_annotation_required": False,
        "human_annotation_used": False,
        "human_ground_truth_claimed": False,
        "pixel_accuracy_claim_allowed": False,
        "automatic_glitch_masks_are_pseudo_labels": True,
        "negative_and_null_masks_retained": True,
        "validation_only": True,
        "test_rows_read": 0,
        "test_evaluation": None,
        "locked_test_prerequisites_satisfied": False,
        "scientific_claim_allowed": False,
        "checks": checks,
        "observed": {
            "rows": audit["rows"],
            "unique_glitches": audit["unique_glitches"],
            "unique_gps_blocks": audit["unique_gps_blocks"],
            "labels": len(audit["labels"]),
        },
        "background_dependence_audits": dependence,
        "injection_bootstrap_independence": injection_bootstrap,
        "raw_mask_endpoint": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
        },
        "automatic_mask_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
        },
        "automatic_mask_manifest": {
            "path": audit["manifest_path"],
            "sha256": audit["manifest_sha256"],
        },
        "overlap_config": {
            "path": audit["config_path"],
            "sha256": audit["config_sha256"],
        },
        "gate_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "gate_config_hash": canonical_hash(config),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result
