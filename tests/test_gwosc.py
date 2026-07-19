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
    download_resumable,
    event_strain_files,
    read_hdf5_segment,
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
