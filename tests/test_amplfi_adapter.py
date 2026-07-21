from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gwyolo.amplfi_adapter import (
    audit_amplfi_common_prior_projection,
    export_amplfi_group_safe_background,
)
from gwyolo.io import file_sha256


def _source(path: Path, ifo: str, start: int = 1000, rate: int = 4) -> None:
    values = np.arange(64 * rate, dtype=np.float64) + (1 if ifo == "L1" else 0)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("strain/Strain", data=values)
        dataset.attrs["Xspacing"] = 1 / rate
        dataset.attrs["Xstart"] = start


def _rows(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    sources = {ifo: tmp_path / f"{ifo}.hdf5" for ifo in ("H1", "L1")}
    for ifo, path in sources.items():
        _source(path, ifo)
    identities = {
        ifo: {"path": str(path), "sha256": file_sha256(path)}
        for ifo, path in sources.items()
    }
    rows = []
    for split, block_start in (("train", 1000), ("val", 1032)):
        for offset in (0, 8, 16, 24):
            rows.append(
                {
                    "split": split,
                    "ifos": ["H1", "L1"],
                    "gps_block": f"gps:{block_start}:32",
                    "pair_id": f"pair-{block_start}",
                    "gps_start": block_start + offset,
                    "gps_end": block_start + offset + 8,
                    "source_files": identities,
                }
            )
    manifest = tmp_path / "background.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest, sources


def test_amplfi_background_export_preserves_group_safe_splits(tmp_path: Path) -> None:
    manifest, _ = _rows(tmp_path)
    report = export_amplfi_group_safe_background(
        manifest,
        tmp_path / "amplfi",
        target_sample_rate=4,
        minimum_segment_seconds=16,
    )
    assert report["split_file_counts"] == {"train": 1, "val": 1, "test": 0}
    assert report["split_duration_seconds"] == {"train": 32.0, "val": 32.0, "test": 0.0}
    assert report["cross_split_gps_block_overlap"] == 0
    validation = Path(report["files"][1]["path"])
    assert validation.parts[-3:-1] == ("validation", "background")
    with h5py.File(validation) as handle:
        assert handle["H1"].shape == (128,)
        assert handle["H1"].attrs["dx"] == 0.25
        assert handle.attrs["gps_block"] == "gps:1032:32"


def test_amplfi_background_export_rejects_cross_split_gps_block(
    tmp_path: Path,
) -> None:
    manifest, _ = _rows(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[-1]["gps_block"] = "gps:1000:32"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="cross AMPLFI export splits"):
        export_amplfi_group_safe_background(manifest, tmp_path / "amplfi", target_sample_rate=4)


def test_amplfi_common_prior_projection_matches_every_native_distribution() -> None:
    root = Path(__file__).parents[1]
    report = audit_amplfi_common_prior_projection(
        root / "configs/pe_common_bbh_analysis_prior.yaml",
        root / "configs/amplfi_common_bbh_training_prior.yaml",
        root / "configs/amplfi_common_bbh_publication.yaml",
    )
    assert report["publication_ready"] is True
    assert len(report["checks"]) == 14
    assert report["checks"]["luminosity_distance"]["native_bounds"] == [100.0, 3100.0]
