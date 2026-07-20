from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from gwyolo.gwosc import (
    _fft_downsample,
    _whiten,
    download_resumable,
    event_strain_files,
    plan_run_strain_pairs,
    read_hdf5_segment,
    run_gwosc_event_exclusions,
    run_gwosc_pilot,
    verify_hdf5_against_detail,
)


def test_event_strain_file_filtering() -> None:
    response = {
        "results": [
            {
                "detector": "H1",
                "sample_rate_kHz": 4,
                "gps_start": 10,
                "hdf5_url": "h1",
                "detail_url": "h1-detail",
            },
            {
                "detector": "L1",
                "sample_rate_kHz": 4,
                "gps_start": 10,
                "hdf5_url": "l1",
                "detail_url": "l1-detail",
            },
            {
                "detector": "H1",
                "sample_rate_kHz": 16,
                "gps_start": 10,
                "hdf5_url": "h1-16",
                "detail_url": "h1-16-detail",
            },
        ]
    }
    with patch("gwyolo.gwosc._api_json", return_value=response):
        records = event_strain_files("event", ["L1"], 4)
    assert records == [
        {
            "detector": "L1",
            "sample_rate": 4096,
            "gps_start": 10,
            "hdf5_url": "l1",
            "detail_url": "l1-detail",
        }
    ]


def test_full_hdf5_verification_matches_official_statistics(tmp_path: Path) -> None:
    path = tmp_path / "verified.hdf5"
    values = np.array([1.0, 2.0, 3.0, 4.0])
    with h5py.File(path, "w") as handle:
        handle.create_group("strain").create_dataset(
            "Strain", data=values, chunks=(2,), fletcher32=True
        )
        quality = handle.create_group("quality")
        quality.create_group("simple").create_dataset("DQmask", data=[3, 1, 0, 2])
        quality.create_group("injections").create_dataset("Injmask", data=[1, 0, 1, 0])
    detail = {
        "filesize_bytes": path.stat().st_size,
        "mean_strain": 2.5,
        "stdev_strain": float(np.std(values)),
        "min_strain": 1.0,
        "max_strain": 4.0,
        "nans_fraction": 0.0,
        "bitsums": [
            {"bit": 0, "sum": 2},
            {"bit": 1, "sum": 2},
            {"bit": 32, "sum": 2},
        ],
    }
    report = verify_hdf5_against_detail(path, detail, chunk_samples=2)
    assert report["passed"]
    assert report["strain_samples"] == 4
    assert report["observed_bitsums"] == {"0": 2, "1": 2, "32": 2}


def test_run_plan_keeps_only_aligned_pairs_and_stratifies() -> None:
    first = {
        "results_count": 7,
        "results": [
            {
                "detector": ifo,
                "gps_start": gps,
                "sample_rate_kHz": 4,
                "hdf5_url": f"https://example/{ifo}-{gps}.hdf5",
                "detail_url": f"https://example/{ifo}-{gps}",
            }
            for gps in (100, 200, 300)
            for ifo in ("H1", "L1")
        ]
        + [
            {
                "detector": "H1",
                "gps_start": 400,
                "sample_rate_kHz": 4,
                "hdf5_url": "https://example/H1-400.hdf5",
                "detail_url": "https://example/H1-400",
            }
        ],
        "next": "next-page",
    }
    second = {"results_count": 7, "results": [], "next": None}
    with patch("gwyolo.gwosc._api_json", side_effect=[first, second]):
        plan = plan_run_strain_pairs("O4a", ["H1", "L1"], maximum_pairs=2, seed=3)
    assert plan["api_pages"] == 2
    assert plan["aligned_pairs_available"] == 3
    assert plan["selected_pairs"] == 2
    assert all(set(row["detectors"]) == {"H1", "L1"} for row in plan["pairs"])


def test_run_plan_rejects_locked_o4b_without_api_access() -> None:
    with patch("gwyolo.gwosc._api_json") as api, pytest.raises(
        ValueError, match="locked evaluation"
    ):
        plan_run_strain_pairs("O4b")
    api.assert_not_called()


def test_event_exclusions_use_latest_version_and_padding(tmp_path: Path) -> None:
    events = [
        {
            "name": "event",
            "versions": [
                {"version": 1, "detail_url": "v1"},
                {"version": 2, "detail_url": "v2"},
            ],
        }
    ]
    with patch(
        "gwyolo.gwosc._api_results",
        return_value=(events, {"api_results_count": 1, "api_pages": 1}),
    ), patch(
        "gwyolo.gwosc._api_json",
        return_value={"run": "O4a", "gps": 1234.5, "catalog": "catalog", "version": 2},
    ) as detail:
        report = run_gwosc_event_exclusions(
            "O4a", tmp_path / "exclusions.json", padding_seconds=8, workers=1
        )
    detail.assert_called_once_with("v2")
    assert report["events"] == 1
    assert report["intervals"][0]["exclusion_start"] == 1226.5
    assert report["intervals"][0]["exclusion_end"] == 1242.5


def test_read_hdf5_segment_and_downsample(tmp_path: Path) -> None:
    path = tmp_path / "strain.hdf5"
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=100)
        strain = handle.create_group("strain").create_dataset("Strain", data=np.arange(80, dtype=float))
        strain.attrs["Xspacing"] = 0.25
        simple = handle.create_group("quality").create_group("simple")
        simple.create_dataset("DQmask", data=np.arange(20))
        injections = handle["quality"].create_group("injections")
        injections.create_dataset("Injmask", data=np.arange(20) + 100)
    segment = read_hdf5_segment(path, gps_center=105.0, duration=4.0)
    assert segment["sample_rate"] == 4
    assert segment["strain"].tolist() == list(np.arange(12, 28, dtype=float))
    assert len(segment["quality"]["DQmask"]) == 4
    assert segment["quality"]["Injmask"].tolist() == [103, 104, 105, 106]
    downsampled = _fft_downsample(segment["strain"], 4, 2)
    assert downsampled.shape == (8,)


def test_whitening_rejects_nonfinite_and_zero_variance() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _whiten(np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="zero-variance"):
        _whiten(np.zeros(64))


def test_o4b_is_locked_before_any_download(tmp_path: Path) -> None:
    with patch(
        "gwyolo.gwosc.resolve_event",
        return_value={"event": "locked", "gps": 1.0, "run": "O4b", "detectors": ["H1"]},
    ), pytest.raises(ValueError, match="locked evaluation"):
        run_gwosc_pilot("locked", tmp_path / "cache", tmp_path / "output")


def test_report_shape_is_json_serializable() -> None:
    value = {"shape": list(np.zeros((2, 3)).shape)}
    assert json.loads(json.dumps(value)) == {"shape": [2, 3]}


def test_parallel_download_resumes_exact_prefix(tmp_path: Path) -> None:
    payload = bytes(range(251)) * 100

    class RangeHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_GET(self) -> None:
            units, requested = self.headers["Range"].split("=")
            assert units == "bytes"
            start_text, stop_text = requested.split("-")
            start, stop = int(start_text), int(stop_text)
            body = payload[start : stop + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{stop}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "payload.bin"
    target.write_bytes(payload[:137])
    try:
        report = download_resumable(
            f"http://127.0.0.1:{server.server_port}/payload", target, workers=3
        )
    finally:
        server.shutdown()
        thread.join()
    assert target.read_bytes() == payload
    assert report["bytes"] == len(payload)
    assert report["downloaded"]
