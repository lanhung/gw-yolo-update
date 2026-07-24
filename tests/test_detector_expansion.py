from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.detector_expansion import (
    audit_detector_set_expansion_readiness,
    expand_materialized_injection_detector_set,
)
from gwyolo.io import file_sha256
from gwyolo.waveforms import _atomic_save_npz


class FakeBackend:
    metadata = {"backend": "fake-physical-projection", "version": "1"}

    def __init__(self, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    def generate(self, _row, ifos, _sample_rate):
        values = {
            "H1": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "L1": np.asarray([2.0, 1.0, 2.0, 1.0]),
            "V1": np.asarray([0.5, 1.0, 0.5, 1.0]),
        }
        if self.mismatch:
            values["H1"] = -values["H1"]
        return (
            {ifo: (100.0, values[ifo]) for ifo in ifos},
            {
                ifo: {
                    "detector_arrival_gps": 100.0,
                    "geocenter_to_detector_delay_seconds": 0.0,
                }
                for ifo in ifos
            },
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "source.npz"
    _atomic_save_npz(
        artifact,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(4, dtype=np.int64),
        context_gps_start=np.asarray(100.0),
        analysis_gps_start=np.asarray(100.0),
        analysis_start_index=np.asarray(0, dtype=np.int64),
        analysis_stop_index=np.asarray(4, dtype=np.int64),
        signal=np.asarray(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0]], dtype=np.float32
        ),
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "injection_id": "injection-1",
                "waveform_id": "waveform-1",
                "gps_block": "gps-1",
                "gps_time": 100.0,
                "split": "train",
                "ifos": ["H1", "L1"],
                "materialized_path": str(artifact),
                "materialized_sha256": file_sha256(artifact),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
detector_set_expansion:
  target_ifos: [H1, L1, V1]
  reference_psd_models_by_ifo:
    H1: h-psd
    L1: l-psd
    V1: v-psd
  low_frequency_hz: 0.1
  high_frequency_hz: 1.5
  minimum_common_ifo_normalized_overlap: 0.999
  maximum_common_ifo_relative_l2_error: 0.01
""".lstrip(),
        encoding="utf-8",
    )
    validation = tmp_path / "waveform.json"
    validation.write_text(
        json.dumps(
            {
                "passed": True,
                "validation_scope": "external_reference_waveform_equivalence",
            }
        ),
        encoding="utf-8",
    )
    return manifest, config, validation


def _snr(signal, _sample_rate, _ifo, _model, _low, _high):
    return float(np.linalg.norm(signal))


def test_detector_expansion_projects_v1_and_reports_reference_psd_snr(
    tmp_path: Path,
) -> None:
    manifest, config, validation = _inputs(tmp_path)
    report = expand_materialized_injection_detector_set(
        manifest,
        config,
        validation,
        tmp_path / "output",
        backend=FakeBackend(),
        snr_calculator=_snr,
    )
    assert report["passed"] is True
    assert report["same_distribution_data_scaling_claim_allowed"] is False
    assert report["detector_set_signal_bank_ready"] is True
    assert report["detector_set_robustness_ablation_ready"] is False
    assert report["target_detector_sets"] == {"H1+L1+V1": 1}
    assert report["test_rows_read"] == 0
    row = json.loads(
        Path(report["manifest_path"]).read_text(encoding="utf-8").strip()
    )
    assert row["ifos"] == ["H1", "L1", "V1"]
    assert set(row["optimal_snr_by_ifo"]) == {"H1", "L1", "V1"}
    assert row["detector_set_expansion_role"] == (
        "robustness_ablation_not_sample_scaling"
    )
    with np.load(row["materialized_path"], allow_pickle=False) as arrays:
        assert arrays["signal_scaled"].shape == (3, 4)
        assert arrays["ifos"].tolist() == ["H1", "L1", "V1"]


def test_detector_expansion_rejects_changed_common_ifo_projection(
    tmp_path: Path,
) -> None:
    manifest, config, validation = _inputs(tmp_path)
    with pytest.raises(ValueError, match="regenerated common-IFO projection differs"):
        expand_materialized_injection_detector_set(
            manifest,
            config,
            validation,
            tmp_path / "output",
            backend=FakeBackend(mismatch=True),
            snr_calculator=_snr,
        )


def test_readiness_audit_rejects_signal_bank_as_complete_clean_data(
    tmp_path: Path,
) -> None:
    reports = []
    for split in ("train", "val"):
        root = tmp_path / split
        manifest, config, validation = _inputs(root)
        row = json.loads(manifest.read_text(encoding="utf-8"))
        row["split"] = split
        row["injection_id"] = f"injection-{split}"
        row["waveform_id"] = f"waveform-{split}"
        row["gps_block"] = f"gps-{split}"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = expand_materialized_injection_detector_set(
            manifest,
            config,
            validation,
            root / "output",
            split=split,
            backend=FakeBackend(),
            snr_calculator=_snr,
        )
        reports.append(root / "output" / "detector_set_expansion_report.json")
        assert report["detector_set_signal_bank_ready"] is True
    readiness = audit_detector_set_expansion_readiness(
        reports, tmp_path / "readiness.json"
    )
    assert readiness["signal_overlap_materialization_authorized"] is True
    assert readiness["detector_complete_clean_training_authorized"] is False
    assert readiness["detector_set_robustness_ablation_ready"] is False
    assert readiness["detector_complete_clean_background_rows"] == 0
    assert readiness["test_rows_read"] == 0
