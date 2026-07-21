from __future__ import annotations

import json
import threading
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.gwosc import (
    _fft_downsample,
    _whiten,
    _whiten_with_reference,
    _remote_size,
    _urlopen_metadata,
    download_resumable,
    extend_gwosc_run_plan,
    event_strain_files,
    plan_run_strain_pairs,
    run_disjoint_gwosc_run_plan,
    run_gwosc_batch_download,
    read_hdf5_segment,
    run_gwosc_event_exclusions,
    run_gwosc_plan_shard,
    run_gwosc_pilot,
    verify_hdf5_against_detail,
)


class _MetadataResponse:
    def __init__(self, payload: bytes = b"{}", content_length: str = "123") -> None:
        self.payload = payload
        self.headers = Message()
        self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_public_metadata_retry_is_bounded_and_rejects_permanent_http_errors() -> None:
    request = urllib.request.Request("https://example.test/metadata")
    response = _MetadataResponse()
    with (
        patch(
            "gwyolo.gwosc.urllib.request.urlopen",
            side_effect=[URLError("TLS timeout"), response],
        ) as opener,
        patch("gwyolo.gwosc.time.sleep") as sleep,
    ):
        assert _urlopen_metadata(request, timeout=1, max_attempts=2) is response
    assert opener.call_count == 2
    sleep.assert_called_once_with(0.5)

    permanent = HTTPError(request.full_url, 404, "not found", {}, None)
    with (
        patch("gwyolo.gwosc.urllib.request.urlopen", side_effect=permanent) as opener,
        pytest.raises(HTTPError),
    ):
        _urlopen_metadata(request, timeout=1, max_attempts=5)
    assert opener.call_count == 1


def test_remote_size_retries_transient_head_failure() -> None:
    response = _MetadataResponse(content_length="456")
    with (
        patch(
            "gwyolo.gwosc.urllib.request.urlopen",
            side_effect=[TimeoutError("handshake"), response],
        ),
        patch("gwyolo.gwosc.time.sleep"),
    ):
        assert _remote_size("https://example.test/file.hdf5") == 456


def test_gwosc_plan_shards_are_disjoint_and_parent_bound(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    pairs = [
        {
            "pair_id": f"O4a-{gps}-H1-L1",
            "run": "O4a",
            "gps_start": gps,
            "detectors": {
                ifo: {
                    "detector": ifo,
                    "gps_start": gps,
                    "sample_rate": 4096,
                    "hdf5_url": f"https://example/{ifo}-{gps}.hdf5",
                    "detail_url": f"https://example/{ifo}-{gps}.json",
                }
                for ifo in ("H1", "L1")
            },
        }
        for gps in (100, 200, 300, 400, 500)
    ]
    plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "locked_evaluation_data": False,
                "run": "O4a",
                "detectors": ["H1", "L1"],
                "sample_rate_khz": 4,
                "seed": 7,
                "source_endpoint": "https://gwosc.org/api/v2/runs/O4a/strain-files",
                "aligned_pairs_available": 5,
                "selected_pairs": 5,
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )

    first = run_gwosc_plan_shard(plan, tmp_path / "shard-0.json", 0, 2)
    second = run_gwosc_plan_shard(plan, tmp_path / "shard-1.json", 1, 2)
    last = run_gwosc_plan_shard(plan, tmp_path / "shard-2.json", 2, 2)

    assert first["parent_plan_sha256"] == second["parent_plan_sha256"]
    assert [row["gps_start"] for row in first["pairs"]] == [100, 200]
    assert [row["gps_start"] for row in second["pairs"]] == [300, 400]
    assert [row["gps_start"] for row in last["pairs"]] == [500]
    selected = {
        row["pair_id"] for report in (first, second, last) for row in report["pairs"]
    }
    assert selected == {row["pair_id"] for row in pairs}
    with pytest.raises(ValueError, match="beyond"):
        run_gwosc_plan_shard(plan, tmp_path / "shard-3.json", 3, 2)


def test_gwosc_plan_extension_preserves_parent_prefix_and_is_score_blind(
    tmp_path: Path,
) -> None:
    def pair(gps: int) -> dict:
        return {
            "pair_id": f"O4a-{gps}-H1-L1",
            "run": "O4a",
            "gps_start": gps,
            "detectors": {
                ifo: {
                    "detector": ifo,
                    "gps_start": gps,
                    "sample_rate": 4096,
                    "hdf5_url": f"https://example/{ifo}-{gps}.hdf5",
                    "detail_url": f"https://example/{ifo}-{gps}.json",
                }
                for ifo in ("H1", "L1")
            },
        }

    parent = tmp_path / "parent.json"
    base_pairs = [pair(100), pair(400)]
    base = {
        "status": "development_acquisition_plan",
        "locked_evaluation_data": False,
        "run": "O4a",
        "detectors": ["H1", "L1"],
        "sample_rate_khz": 4,
        "seed": 7,
        "source_endpoint": "https://gwosc.org/api/v2/runs/O4a/strain-files",
        "aligned_pairs_available": 5,
        "selected_pairs": 2,
        "pairs": base_pairs,
    }
    parent.write_text(json.dumps(base), encoding="utf-8")
    full = {
        **base,
        "api_results_count": 10,
        "api_pages": 1,
        "aligned_pairs_available": 5,
        "selected_pairs": 5,
        "selected_gps_span": [100, 500],
        "pairs": [pair(gps) for gps in (100, 200, 300, 400, 500)],
    }
    output = tmp_path / "extended.json"
    with patch("gwyolo.gwosc.plan_run_strain_pairs", return_value=full):
        extended = extend_gwosc_run_plan(parent, output, 4, extension_seed=9)

    assert extended["pairs"][:2] == base_pairs
    assert extended["selected_pairs"] == 4
    assert len({row["pair_id"] for row in extended["pairs"]}) == 4
    assert extended["base_parent_plan_sha256"] == file_sha256(parent)
    assert extended["base_selected_pairs"] == 2
    assert extended["extension_pairs"] == 2
    assert extended["candidate_scores_inspected"] is False
    assert extended["selection_data"] == "GWOSC strain-file metadata only"
    with pytest.raises(FileExistsError, match="immutable"):
        extend_gwosc_run_plan(parent, output, 5)


def test_disjoint_gwosc_plan_excludes_frozen_source_pairs(tmp_path: Path) -> None:
    def pair(gps: int) -> dict:
        return {
            "pair_id": f"O4a-{gps}-H1-L1",
            "run": "O4a",
            "gps_start": gps,
            "detectors": {
                ifo: {
                    "detector": ifo,
                    "gps_start": gps,
                    "sample_rate": 4096,
                    "hdf5_url": f"https://example/{ifo}-{gps}.hdf5",
                    "detail_url": f"https://example/{ifo}-{gps}.json",
                }
                for ifo in ("H1", "L1")
            },
        }

    full_pairs = [pair(gps) for gps in (100, 200, 300, 400, 500, 600)]
    full = {
        "status": "development_acquisition_plan",
        "locked_evaluation_data": False,
        "run": "O4a",
        "detectors": ["H1", "L1"],
        "sample_rate_khz": 4,
        "seed": 11,
        "source_endpoint": "https://gwosc.org/api/v2/runs/O4a/strain-files",
        "aligned_pairs_available": len(full_pairs),
        "selected_pairs": len(full_pairs),
        "selected_gps_span": [100, 600],
        "pairs": full_pairs,
    }
    exclusion = tmp_path / "reserved.json"
    exclusion.write_text(
        json.dumps({**full, "selected_pairs": 2, "pairs": [full_pairs[1], full_pairs[4]]}),
        encoding="utf-8",
    )
    output = tmp_path / "training-plan.json"
    with patch("gwyolo.gwosc.plan_run_strain_pairs", return_value=full):
        report = run_disjoint_gwosc_run_plan(
            "O4a",
            ["H1", "L1"],
            [exclusion],
            output,
            target_pairs=3,
            seed=17,
        )
    selected = {row["gps_start"] for row in report["pairs"]}
    assert len(selected) == 3
    assert not selected & {200, 500}
    assert report["excluded_unique_pair_ids"] == 2
    assert report["eligible_pairs_after_exclusion"] == 4
    assert report["candidate_scores_inspected"] is False
    assert report["test_data_opened"] is False
    assert report["exclusion_plans"][0]["sha256"] == file_sha256(exclusion)


def test_reference_whitening_is_linear_for_signal_component() -> None:
    rng = np.random.default_rng(5)
    noise = rng.normal(size=2048)
    signal = 0.02 * np.sin(np.linspace(0, 30, 2048))
    whitened_noise = _whiten_with_reference(noise, noise)
    whitened_signal = _whiten_with_reference(noise, signal, component=True)
    whitened_sum = _whiten_with_reference(noise, noise + signal)
    assert whitened_noise == pytest.approx(_whiten(noise))
    assert whitened_sum == pytest.approx(whitened_noise + whitened_signal, abs=1e-6)


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


def test_batch_download_replays_exact_verified_inventory_without_network(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    run_cache = cache / "O4a"
    run_cache.mkdir(parents=True)
    pair = {
        "pair_id": "O4a-100-H1-L1",
        "run": "O4a",
        "gps_start": 100,
        "detectors": {},
    }
    files = []
    for index, ifo in enumerate(("H1", "L1")):
        filename = f"{ifo}-100-4096.hdf5"
        path = run_cache / filename
        values = np.arange(16, dtype=float) + index
        with h5py.File(path, "w") as handle:
            handle.create_group("strain").create_dataset("Strain", data=values)
            quality = handle.create_group("quality")
            quality.create_group("simple").create_dataset("DQmask", data=[3, 1])
            quality.create_group("injections").create_dataset("Injmask", data=[1, 0])
        detail = {
            "filesize_bytes": path.stat().st_size,
            "mean_strain": float(np.mean(values)),
            "stdev_strain": float(np.std(values)),
            "min_strain": float(np.min(values)),
            "max_strain": float(np.max(values)),
            "nans_fraction": 0.0,
            "bitsums": [
                {"bit": 0, "sum": 2},
                {"bit": 1, "sum": 1},
                {"bit": 32, "sum": 1},
            ],
        }
        verification = verify_hdf5_against_detail(path, detail, chunk_samples=4)
        pair["detectors"][ifo] = {
            "detector": ifo,
            "gps_start": 100,
            "sample_rate": 4096,
            "hdf5_url": f"https://example.test/{filename}",
            "detail_url": f"https://example.test/{ifo}-detail",
        }
        files.append(
            {
                "pair_id": pair["pair_id"],
                "run": "O4a",
                "gps_start": 100,
                "detector": ifo,
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "downloaded": True,
                "detail_url": pair["detectors"][ifo]["detail_url"],
                "verification": verification,
            }
        )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "locked_evaluation_data": False,
                "run": "O4a",
                "detectors": ["H1", "L1"],
                "sample_rate_khz": 4,
                "seed": 7,
                "selected_pairs": 1,
                "pairs": [pair],
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "status": "verified_development_strain_batch",
                "passed": True,
                "run": "O4a",
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    with patch("gwyolo.gwosc.download_resumable") as download, patch(
        "gwyolo.gwosc._api_json"
    ) as metadata:
        result = run_gwosc_batch_download(
            plan,
            cache,
            tmp_path / "output",
            download_workers=2,
            chunk_samples=4,
            verified_source_inventories=[inventory],
        )

    download.assert_not_called()
    metadata.assert_not_called()
    assert result["verified_files"] == 2
    assert result["imported_verified_files"] == 2
    assert all(not row["downloaded"] for row in result["files"])
    assert result["verified_source_inventory_sha256s"] == [file_sha256(inventory)]

    with Path(files[0]["path"]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="byte count changed"):
        run_gwosc_batch_download(
            plan,
            cache,
            tmp_path / "tampered-output",
            download_workers=2,
            chunk_samples=4,
            verified_source_inventories=[inventory],
        )

    inventory.unlink()
    with pytest.raises(ValueError, match="different run"):
        run_gwosc_batch_download(
            plan,
            cache,
            tmp_path / "output",
            download_workers=2,
            chunk_samples=4,
        )


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
