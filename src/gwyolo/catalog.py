from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .metrics import snr_binned_hit_rate, wilson_interval


def load_gwosc_events(api_url: str) -> dict[str, dict[str, Any]]:
    with urllib.request.urlopen(api_url, timeout=30) as response:  # nosec B310: configured official API
        payload = json.load(response)
    events = payload.get("events", {})
    return {event["commonName"]: event for event in events.values()}


def evaluate_catalog_predictions(
    predictions_path: str | Path, api_url: str, output_path: str | Path
) -> dict[str, Any]:
    metadata = load_gwosc_events(api_url)
    rows: list[dict[str, Any]] = []
    with Path(predictions_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            prediction = json.loads(line)
            name = prediction.get("event")
            event = metadata.get(name, {})
            rows.append(
                {
                    "event": name,
                    "matched": bool(event),
                    "hit": bool(prediction.get("has_chirp")),
                    "snr": event.get("network_matched_filter_snr"),
                    "far_per_year": event.get("far"),
                    "p_astro": event.get("p_astro"),
                    "total_mass_source": event.get("total_mass_source"),
                }
            )
    matched = [row for row in rows if row["matched"]]
    hits = sum(row["hit"] for row in matched)
    lower, upper = wilson_interval(hits, len(matched))
    high_snr_misses = sorted(
        [row for row in matched if not row["hit"] and (row["snr"] or 0.0) >= 20.0],
        key=lambda row: row["snr"],
        reverse=True,
    )
    summary = {
        "images": len(rows),
        "metadata_matches": len(matched),
        "chirp_hits": hits,
        "catalog_image_hit_rate": hits / len(matched) if matched else None,
        "catalog_image_hit_rate_wilson_95": [lower, upper] if matched else [None, None],
        "snr_bins": snr_binned_hit_rate(matched),
        "high_snr_misses": high_snr_misses,
        "warning": (
            "Catalog-image hit rate is not a search recall. It has no continuous-background denominator "
            "and may use only one detector view."
        ),
    }
    atomic_write_json(output_path, summary)
    return summary
