import json

import gwyolo.catalog as catalog


def test_catalog_join_and_warning(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"event": "GW000001_000001", "has_chirp": True},
        {"event": "GW000002_000002", "has_chirp": False},
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(
        catalog,
        "load_gwosc_events",
        lambda _: {
            "GW000001_000001": {"network_matched_filter_snr": 9.0, "far": 0.1, "p_astro": 0.9},
            "GW000002_000002": {"network_matched_filter_snr": 21.0, "far": 0.01, "p_astro": 0.99},
        },
    )
    output = tmp_path / "summary.json"
    result = catalog.evaluate_catalog_predictions(predictions, "unused", output)
    assert result["catalog_image_hit_rate"] == 0.5
    assert result["high_snr_misses"][0]["event"] == "GW000002_000002"
    assert "not a search recall" in result["warning"]
    assert output.is_file()
