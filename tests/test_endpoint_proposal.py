import json

import numpy as np
import pytest

import gwyolo.endpoint_proposal as endpoint_proposal_module
from gwyolo.endpoint_proposal import (
    _proposal_epoch,
    application_shard_ranges,
    dense_endpoint_targets,
    extract_dense_endpoint_candidates,
    proposal_gate_record,
    run_detector_endpoint_proposal_application,
    select_dense_proposal_record,
)


def test_application_shards_are_deterministic_exhaustive_and_non_overlapping() -> None:
    assert application_shard_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError):
        application_shard_ranges(0, 4)
    with pytest.raises(ValueError):
        application_shard_ranges(10, 0)


def test_endpoint_application_resumes_verified_parts_without_rescoring(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "detector_endpoint_proposal": {
                    "model_ifos": ["H1", "L1"],
                    "target_sample_rate": 8,
                    "analysis_duration": 4.0,
                    "output_bins": 4,
                    "base_channels": 1,
                    "batch_size": 2,
                    "cache_in_memory": True,
                    "minimum_bins": 1,
                }
            }
        )
    )
    rows = [
        {
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "split": "train",
            "source_family": "BBH",
            "gps_block": f"b{index}",
            "detector_arrival_gps": {"H1": 101.0, "L1": 102.0},
        }
        for index in range(5)
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint identity")
    checkpoint = {
        "architecture": "detector_endpoint_spectrogram_dense_v1",
        "model_ifos": ["H1", "L1"],
        "target_sample_rate": 8,
        "analysis_duration": 4.0,
        "output_bins": 4,
        "base_channels": 1,
        "model": {},
    }

    class FakeDataset:
        def __init__(self, shard_rows, *_args):
            self.rows = shard_rows

    class FakeLoader:
        def __init__(self, dataset, **_kwargs):
            self.dataset = dataset

    class FakeModel:
        def __init__(self, *_args):
            pass

        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            return None

    scoring_calls = []

    def fake_predict(_model, loader, _device):
        count = len(loader.dataset.rows)
        scoring_calls.append(count)
        probabilities = np.zeros((count, 2, 4), dtype=np.float32)
        probabilities[:, :, 1] = 0.9
        availability = np.ones((count, 2), dtype=bool)
        offsets = np.tile(np.asarray([[1.0, 2.0]]), (count, 1))
        return probabilities, availability, offsets

    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(endpoint_proposal_module, "DetectorArrivalDataset", FakeDataset)
    monkeypatch.setattr(endpoint_proposal_module, "DataLoader", FakeLoader)
    monkeypatch.setattr(
        endpoint_proposal_module, "DetectorArrivalSpectrogramNet", FakeModel
    )
    monkeypatch.setattr(endpoint_proposal_module, "_predict_proposals", fake_predict)
    output = tmp_path / "output"
    result = run_detector_endpoint_proposal_application(
        config_path,
        manifest_path,
        checkpoint_path,
        0.5,
        "train",
        output,
        shard_size=2,
    )
    assert scoring_calls == [2, 2, 1]
    assert result["shards"] == 3
    assert result["candidates"] == 10
    assert sum(1 for _ in open(result["candidate_manifest"])) == 10

    (output / "detector_endpoint_proposal_application.json").unlink()
    (output / "endpoint_train_candidates.jsonl").unlink()
    scoring_calls.clear()
    resumed = run_detector_endpoint_proposal_application(
        config_path,
        manifest_path,
        checkpoint_path,
        0.5,
        "train",
        output,
        shard_size=2,
    )
    assert scoring_calls == []
    assert resumed["candidate_manifest_sha256"] == result["candidate_manifest_sha256"]


def test_dense_endpoint_targets_preserve_multiple_instances_by_hand() -> None:
    target, availability = dense_endpoint_targets(
        {"H1": [1.0, 3.0], "L1": 2.0},
        ("H1", "L1", "V1"),
        duration_seconds=4.0,
        output_bins=8,
        half_width_seconds=0.0,
    )
    assert availability.tolist() == [True, True, False]
    assert np.flatnonzero(target[0]).tolist() == [2, 6]
    assert np.flatnonzero(target[1]).tolist() == [4]
    assert not target[2].any()


def test_dense_endpoint_extraction_retains_every_connected_peak() -> None:
    probabilities = np.zeros((1, 2, 8), dtype=np.float32)
    probabilities[0, 0, 1:3] = [0.7, 0.9]
    probabilities[0, 0, 6] = 0.8
    probabilities[0, 1, 4:6] = [0.95, 0.7]
    row = {
        "injection_id": "i1",
        "waveform_id": "w1",
        "split": "val",
        "source_family": "BBH",
        "gps_block": "b1",
        "detector_arrival_gps": {"H1": 101.0, "L1": 102.0},
    }
    rows = extract_dense_endpoint_candidates(
        probabilities,
        np.asarray([[True, True]]),
        np.asarray([[1.0, 2.0]]),
        [row],
        ("H1", "L1"),
        duration_seconds=4.0,
        threshold=0.6,
    )
    assert [(item["ifo"], item["start_bin"], item["stop_bin_exclusive"]) for item in rows] == [
        ("H1", 1, 3),
        ("H1", 6, 7),
        ("L1", 4, 6),
    ]
    assert rows[0]["gps_start"] == 100.5
    assert rows[0]["gps_end"] == 101.5
    assert rows[0]["gps_peak"] == 101.25


def _coverage(padded: float, median_union: float, p90_union: float, width: float):
    group = {
        "padded_coverage_fraction": padded,
        "proposal_union_fraction_of_analysis_quantiles": {
            "0.5": median_union,
            "0.9": p90_union,
        },
        "minimum_containing_proposal_width_seconds_quantiles": {"0.5": width},
    }
    return {
        "candidates": 10,
        "groups": {"all": group, "family:BBH": group, "snr:snr_4_8": group},
    }


def test_dense_proposal_gate_requires_coverage_and_compactness() -> None:
    settings = {
        "required_groups": ["family:BBH", "snr:snr_4_8"],
        "minimum_all_padded_coverage": 0.95,
        "minimum_required_group_padded_coverage": 0.90,
        "maximum_median_union_fraction": 0.50,
        "maximum_p90_union_fraction": 0.80,
        "maximum_median_containing_width_seconds": 2.0,
    }
    passing = proposal_gate_record(_coverage(0.96, 0.4, 0.7, 1.0), 0.5, settings)
    broad = proposal_gate_record(_coverage(0.99, 0.7, 0.9, 3.0), 0.3, settings)
    assert passing["qualified"] is True
    assert broad["qualified"] is False
    assert select_dense_proposal_record([broad, passing]) == passing


def test_dense_proposal_loss_excludes_negative_infinity_missing_ifo() -> None:
    torch = pytest.importorskip("torch")

    class MissingDetectorModel(torch.nn.Module):
        def forward(self, strain, availability):
            logits = torch.zeros((strain.shape[0], 3, 8), device=strain.device)
            return logits.masked_fill(~availability[:, :, None], -torch.inf)

    loader = [
        (
            torch.zeros((1, 3, 64)),
            torch.tensor([[True, True, False]]),
            torch.tensor([[2, 4, -1]]),
            torch.tensor([[1.0, 2.0, float("nan")]]),
        )
    ]
    metrics = _proposal_epoch(
        MissingDetectorModel(),
        loader,
        torch.device("cpu"),
        None,
        output_bins=8,
        half_width_bins=0,
        positive_weight=2.0,
        focal_gamma=2.0,
    )
    assert np.isfinite(metrics["loss"])
