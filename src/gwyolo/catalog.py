from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256
from .metrics import snr_binned_hit_rate, wilson_interval
from .runtime import execution_provenance


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


def _load_locked_catalog_metadata(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        values = payload.get("events", payload)
        if isinstance(values, dict):
            rows = []
            for name, value in values.items():
                if not isinstance(value, dict):
                    raise ValueError("locked catalog metadata entries must be objects")
                rows.append({"event": value.get("commonName", name), **value})
        elif isinstance(values, list):
            rows = values
        else:
            raise ValueError("locked catalog metadata has an unsupported schema")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("locked catalog metadata has an unsupported schema")
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("locked catalog metadata rows must be objects")
        event = row.get("event") or row.get("commonName")
        if not event:
            raise ValueError("locked catalog metadata rows require an event name")
        normalized.append({**row, "event": str(event)})
    if not normalized or len({row["event"] for row in normalized}) != len(normalized):
        raise ValueError("locked catalog metadata event names must be unique and non-empty")
    return normalized


def run_locked_gwtc5_catalog_diagnostic(
    prediction_manifest: str | Path,
    candidate_search_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Describe GWTC-5 recovery at one own-search validation-frozen threshold."""

    from .evaluation_lock import (
        validate_locked_evaluation_suite_access,
        validate_locked_evaluation_suite_input,
    )

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked catalog diagnostics are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, "catalog_diagnostic", output_path
    )
    suite_input = validate_locked_evaluation_suite_input(
        locked_suite_plan,
        "catalog_prediction_manifest",
        prediction_manifest,
    )
    search_arm = str(suite_access["endpoints"]["catalog_search_arm"])
    search_path = Path(candidate_search_report).resolve()
    search_binding = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, search_arm, search_path
    )
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if (
        search.get("status") != "locked_candidate_search_evaluation"
        or search.get("candidate_endpoint_gates_passed") is not True
        or search.get("threshold_source")
        != "frozen_validation_candidate_search_calibration"
        or search.get("locked_suite_access", {}).get("output_key") != search_arm
        or search_binding.get("output_key") != search_arm
    ):
        raise ValueError("locked catalog diagnostic requires the predeclared search arm")
    threshold = float(search["test_evaluation"]["threshold"])
    own_search_background = search["test_evaluation"]["background"]

    metadata_identity = suite_access["frozen_artifacts"].get("catalog_metadata", {})
    metadata_path = Path(str(metadata_identity.get("path", ""))).resolve()
    if (
        not metadata_path.is_file()
        or metadata_identity.get("sha256") != file_sha256(metadata_path)
    ):
        raise ValueError("locked catalog metadata differs from the access receipt")
    metadata_rows = _load_locked_catalog_metadata(metadata_path)
    metadata_by_event = {row["event"]: row for row in metadata_rows}

    prediction_path = Path(prediction_manifest).resolve()
    with prediction_path.open("r", encoding="utf-8") as handle:
        prediction_rows = [json.loads(line) for line in handle if line.strip()]
    if not prediction_rows or any(
        not isinstance(row, dict) or str(row.get("split")) != "test"
        for row in prediction_rows
    ):
        raise ValueError("locked catalog predictions must be non-empty test rows")
    prediction_events = [str(row.get("event", "")) for row in prediction_rows]
    if (
        len(set(prediction_events)) != len(prediction_events)
        or set(prediction_events) != set(metadata_by_event)
    ):
        raise ValueError("locked catalog predictions must cover every metadata event exactly once")

    evaluated = []
    instance_count = 0
    mask_count = 0
    candidate_count = 0
    for row in prediction_rows:
        event = str(row["event"])
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"locked catalog event lacks a candidate list: {event}")
        candidate_ids = []
        scores = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"locked catalog candidate is not an object: {event}")
            candidate_id = str(candidate.get("candidate_id", ""))
            score = float(candidate.get("ranking_score", float("nan")))
            instances = candidate.get("instances")
            if not candidate_id or not math.isfinite(score) or not isinstance(instances, list):
                raise ValueError(f"locked catalog candidate is incomplete: {event}")
            candidate_ids.append(candidate_id)
            scores.append(score)
            for instance in instances:
                if not isinstance(instance, dict):
                    raise ValueError(f"locked catalog instance is not an object: {event}")
                mask_path = Path(str(instance.get("mask_path", ""))).resolve()
                confidence = float(instance.get("confidence", float("nan")))
                if (
                    not str(instance.get("instance_id", ""))
                    or not str(instance.get("class_name", ""))
                    or not math.isfinite(confidence)
                    or not mask_path.is_file()
                    or instance.get("mask_sha256") != file_sha256(mask_path)
                ):
                    raise ValueError(f"locked catalog instance/mask failed replay: {event}")
                instance_count += 1
                mask_count += 1
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"locked catalog candidate IDs repeat: {event}")
        candidate_count += len(candidates)
        hit = bool(scores and max(scores) >= threshold)
        metadata = metadata_by_event[event]
        evaluated.append(
            {
                "event": event,
                "hit": hit,
                "maximum_ranking_score": max(scores) if scores else None,
                "candidate_count": len(candidates),
                "snr": metadata.get("network_matched_filter_snr", metadata.get("snr")),
                "published_far": metadata.get("far"),
                "p_astro": metadata.get("p_astro"),
                "total_mass_source": metadata.get("total_mass_source"),
            }
        )
    hits = sum(row["hit"] for row in evaluated)
    interval = wilson_interval(hits, len(evaluated))
    result = {
        "status": "locked_gwtc5_catalog_diagnostic",
        "endpoint_complete": True,
        "descriptive_only": True,
        "scientific_claim_allowed": False,
        "threshold_refits_on_catalog": 0,
        "threshold": threshold,
        "threshold_source": search["threshold_source"],
        "search_arm": search_arm,
        "own_search_far_per_year": own_search_background["far_per_year"],
        "own_search_ifar_years": own_search_background["ifar_years"],
        "events": len(evaluated),
        "hits": hits,
        "catalog_hit_rate": hits / len(evaluated),
        "catalog_hit_rate_wilson_95": list(interval),
        "snr_bins": snr_binned_hit_rate(evaluated),
        "candidate_count": candidate_count,
        "instance_count": instance_count,
        "mask_count": mask_count,
        "predictions_retain_all_instances_and_masks": instance_count == mask_count,
        "event_results": evaluated,
        "warning": (
            "GWTC-5 catalog recovery is descriptive and is not search recall; the primary "
            "endpoint remains continuous-background FAR and paired injection VT"
        ),
        "candidate_search_report": {
            "path": str(search_path),
            "sha256": file_sha256(search_path),
        },
        "catalog_metadata": {
            "path": str(metadata_path),
            "sha256": file_sha256(metadata_path),
        },
        "prediction_manifest": {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        },
        "locked_suite_access": suite_access,
        "locked_suite_input": suite_input,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result
