from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from gwyolo.background import plan_background_windows


def _write_quality_file(path: Path, gps_start: int, duration: int, bad_second: int | None = None) -> None:
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=gps_start)
        meta.create_dataset("Duration", data=duration)
        quality = handle.create_group("quality")
        simple = quality.create_group("simple")
        dq = np.full(duration, 7, dtype=np.int32)
        if bad_second is not None:
            dq[bad_second] = 0
        simple.create_dataset("DQmask", data=dq)
        injections = quality.create_group("injections")
        injections.create_dataset("Injmask", data=np.full(duration, 3, dtype=np.int32))


def test_background_windows_use_common_dq_and_disjoint_blocks(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    l1 = tmp_path / "l1.hdf5"
    _write_quality_file(h1, 1000, 64)
    _write_quality_file(l1, 1000, 64, bad_second=10)
    rows, report = plan_background_windows(
        {"H1": h1, "L1": l1},
        window_duration=4,
        stride=4,
        block_duration=16,
        required_dq_bits=1,
        excluded_intervals=[(1032, 1036)],
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )
    starts = {row["gps_start"] for row in rows}
    assert 1008 not in starts  # L1 DQ failure at GPS 1010 removes the whole window.
    assert 1032 not in starts  # Explicit event exclusion.
    assert report["windows"] == 14
    assert report["unique_gps_blocks"] == 4
    assert report["passed"]
    assert all(not values for values in report["cross_split_block_overlaps"].values())
    assert sum(item["live_time_seconds"] for item in report["splits"].values()) == 56


def test_background_live_time_uses_interval_union(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    _write_quality_file(h1, 2000, 16)
    _, report = plan_background_windows(
        {"H1": h1},
        window_duration=8,
        stride=4,
        block_duration=16,
        validation_fraction=0,
        test_fraction=0,
    )
    assert report["windows"] == 3
    assert report["splits"]["train"]["live_time_seconds"] == 16
