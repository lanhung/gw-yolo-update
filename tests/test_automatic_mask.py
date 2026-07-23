from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.automatic_mask import (
    AUTOMATIC_MASK_SOURCE,
    audit_automatic_mask_policy,
    bind_raw_mask_automatic_publication_evidence,
)
from gwyolo.factory import multiresolution_power
from gwyolo.gwosc import _whiten
from gwyolo.io import file_sha256
from gwyolo.overlaps import array_sha256
from gwyolo.physical_training import (
    relative_component_mask,
    scale_component_for_transform,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_automatic_mask_audit_replays_isolated_components_and_binds_endpoint(
    tmp_path: Path,
) -> None:
    config = tmp_path / "overlap.yaml"
    config.write_text(
        "overlap_factory:\n"
        "  model_ifos: [H1, L1]\n"
        "  q_values: [4]\n"
        "  target_sample_rate: 64\n"
        "  tensor:\n"
        "    whitening: self\n"
        "    target_whitening: morphology\n"
        "    mask_fraction: 0.08\n"
        "    glitch_mask_source: isolated_real_glitch_component_power_v1\n"
        "    glitch_mask_fraction: 0.08\n"
        "    manual_annotation_required: false\n"
        "    frequency_bins: 8\n"
        "    time_bins: 8\n"
        "    fmin: 4\n"
        "    fmax: 30\n",
        encoding="utf-8",
    )
    samples = 128
    glitch = np.zeros((2, samples), dtype=np.float64)
    signal = np.zeros_like(glitch)
    glitch[0] = np.sin(np.linspace(0, 20, samples)) * 1e-21
    signal[0] = np.sin(np.linspace(0, 40, samples)) * 2e-22
    stored_glitch = glitch.astype(np.float32)
    whitened = np.zeros_like(glitch)
    whitened[0] = _whiten(stored_glitch[0].astype(np.float64))
    kwargs = {
        "sample_rate": 64,
        "q_values": (4.0,),
        "frequency_bins": 8,
        "time_bins": 8,
        "fmin": 4.0,
        "fmax": 30.0,
    }
    glitch_mask = relative_component_mask(
        multiresolution_power(whitened, **kwargs), 0.08
    ).astype(np.uint8)
    chirp_mask = relative_component_mask(
        multiresolution_power(scale_component_for_transform(signal), **kwargs),
        0.08,
    ).astype(np.uint8)
    source_glitch = tmp_path / "gravityspy.npz"
    np.savez_compressed(
        source_glitch,
        raw_strain=stored_glitch,
        glitch_mask=np.zeros_like(glitch_mask),
        ifos=np.asarray(["H1", "L1"]),
        q_values=np.asarray([4.0]),
        sample_rate=np.asarray(64),
    )
    source_injection = tmp_path / "injection.npz"
    np.savez_compressed(
        source_injection,
        signal=signal,
        noise=np.zeros_like(signal),
        strain=signal,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(64),
        context_gps_start=np.asarray(0.0),
        analysis_gps_start=np.asarray(0.0),
        analysis_start_index=np.asarray(0),
        analysis_stop_index=np.asarray(samples),
    )
    mixture = (stored_glitch.astype(np.float64) + signal).astype(np.float32)
    artifact = tmp_path / "overlap.npz"
    np.savez_compressed(
        artifact,
        chirp_mask=chirp_mask,
        glitch_mask=glitch_mask,
        raw_glitch_strain=stored_glitch,
        signal_strain=signal,
        target_signal_strain=signal,
        mixture_strain=mixture,
        detector_availability=np.asarray([1, 0], dtype=np.uint8),
        ifos=np.asarray(["H1", "L1"]),
        q_values=np.asarray([4.0]),
        sample_rate=np.asarray(64),
    )
    manifest = tmp_path / "validation.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "mixture_id": "m0",
                "injection_id": "i0",
                "waveform_id": "w0",
                "glitch_id": "g0",
                "split": "val",
                "path": str(artifact),
                "sha256": file_sha256(artifact),
                "mask_provenance": AUTOMATIC_MASK_SOURCE,
                "mask_fraction": 0.08,
                "automatic_pseudo_mask": True,
                "human_pixel_mask": False,
                "glitch_ifo": "H1",
                "glitch_artifact_path": str(source_glitch),
                "glitch_artifact_sha256": file_sha256(source_glitch),
                "injection_materialized_path": str(source_injection),
                "injection_materialized_sha256": file_sha256(source_injection),
                "raw_glitch_component_sha256": array_sha256(stored_glitch),
                "signal_component_sha256": array_sha256(signal),
                "target_signal_component_sha256": array_sha256(signal),
                "mixture_component_sha256": array_sha256(mixture),
                "network_gps_block": "O3a:0:8",
                "ml_label": "Blip",
                "available_ifos": ["H1"],
            }
        ],
    )
    audit_path = tmp_path / "audit.json"
    audit = audit_automatic_mask_policy(manifest, config, audit_path)
    assert audit["passed"] is True
    assert audit["human_annotation_used"] is False
    assert audit["zero_glitch_masks"] == 0
    assert audit["zero_chirp_masks"] == 0
    assert audit["source_components_replayed"] is True
    assert audit["mixture_identity_verified"] is True

    raw = tmp_path / "raw-mask.json"
    raw.write_text(
        json.dumps(
            {
                "status": "bound_validation_raw_mask_continuous_background_evidence",
                "passed": True,
                "mask_locked_test_arm_eligible": True,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "background_dependence_audits": {
                    "raw": {"passed": True},
                    "mask": {"passed": True},
                },
                "injection_bootstrap_independence": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.yaml"
    gate.write_text(
        "automatic_mask_publication_gate:\n"
        "  schema: automatic_mask_publication_gate_v1\n"
        "  minimum_rows: 1\n"
        "  minimum_unique_glitches: 1\n"
        "  minimum_gps_blocks: 1\n"
        "  minimum_labels: 1\n"
        "  manual_annotation_required: false\n"
        "  human_ground_truth_claimed: false\n"
        "  require_soft_model_probabilities: true\n"
        "  require_unknown_glitch_abstention: true\n",
        encoding="utf-8",
    )
    endpoint = bind_raw_mask_automatic_publication_evidence(
        raw, audit_path, gate, tmp_path / "endpoint.json"
    )
    assert endpoint["passed"] is True
    assert endpoint["human_annotation_required"] is False
    assert endpoint["pixel_accuracy_claim_allowed"] is False
    assert endpoint["negative_and_null_masks_retained"] is True
    assert endpoint["checks"]["chirp_masks_replayed"] is True

    with np.load(source_glitch, allow_pickle=False) as arrays:
        altered = {key: arrays[key] for key in arrays.files}
    altered["raw_strain"] = altered["raw_strain"].copy()
    altered["raw_strain"][0, 0] += np.float32(1e-20)
    np.savez_compressed(source_glitch, **altered)
    with pytest.raises(ValueError, match="artifact identity"):
        audit_automatic_mask_policy(
            manifest,
            config,
            tmp_path / "tampered-audit.json",
        )
