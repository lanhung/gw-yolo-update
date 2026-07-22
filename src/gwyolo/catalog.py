from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, file_sha256
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


def _catalog_suite_bindings(
    locked_suite_plan: str | Path,
    access_log: str | Path,
    *,
    include_predictions: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Replay the opened suite and every catalog intermediate path."""

    from .evaluation_lock import (
        validate_locked_evaluation_suite_access,
        validate_locked_evaluation_suite_input,
    )

    plan_path = Path(locked_suite_plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    catalog_output = plan.get("outputs", {}).get("catalog_diagnostic")
    if not catalog_output:
        raise ValueError("locked suite plan lacks the catalog diagnostic output")
    access = validate_locked_evaluation_suite_access(
        plan_path, access_log, "catalog_diagnostic", catalog_output
    )
    keys = [
        "catalog_source_manifest",
        "catalog_candidate_manifest",
        "catalog_candidate_report",
    ]
    if include_predictions:
        keys.extend(["catalog_prediction_manifest", "catalog_prediction_report"])
    bindings = {
        key: validate_locked_evaluation_suite_input(
            plan_path, key, plan.get("inputs", {}).get(key, "")
        )
        for key in keys
    }
    return access, bindings


def _frozen_artifact(
    access: dict[str, Any], label: str
) -> tuple[Path, dict[str, str]]:
    identity = access.get("frozen_artifacts", {}).get(label, {})
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or identity.get("sha256") != file_sha256(path):
        raise ValueError(f"locked catalog frozen artifact failed replay: {label}")
    return path, {"path": str(path), "sha256": file_sha256(path)}


def run_locked_catalog_prediction_manifest(
    candidate_manifest: str | Path,
    candidate_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    prediction_manifest: str | Path,
    prediction_report: str | Path,
) -> dict[str, Any]:
    """Group every frozen catalog candidate without thresholding or pruning."""

    access, bindings = _catalog_suite_bindings(
        locked_suite_plan, access_log, include_predictions=True
    )
    candidate_path = Path(candidate_manifest).resolve()
    candidate_report_path = Path(candidate_report).resolve()
    prediction_path = Path(prediction_manifest).resolve()
    report_path = Path(prediction_report).resolve()
    expected_paths = {
        "catalog_source_manifest": Path(
            bindings["catalog_source_manifest"]["input_path"]
        ).resolve(),
        "catalog_candidate_manifest": candidate_path,
        "catalog_candidate_report": candidate_report_path,
        "catalog_prediction_manifest": prediction_path,
        "catalog_prediction_report": report_path,
    }
    for key, path in expected_paths.items():
        if Path(bindings[key]["input_path"]).resolve() != path:
            raise ValueError(f"locked catalog path differs from its suite plan: {key}")
    if report_path.is_file():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "locked_catalog_prediction_manifest_complete"
            or not prediction_path.is_file()
            or completed.get("prediction_manifest", {}).get("sha256")
            != file_sha256(prediction_path)
            or completed.get("candidate_manifest", {}).get("sha256")
            != file_sha256(candidate_path)
            or completed.get("candidate_scoring_report", {}).get("sha256")
            != file_sha256(candidate_report_path)
            or completed.get("locked_suite_access") != access
            or completed.get("locked_suite_inputs") != bindings
        ):
            raise ValueError("completed locked catalog prediction output failed replay")
        return completed
    if not candidate_path.is_file() or not candidate_report_path.is_file():
        raise FileNotFoundError("locked catalog candidate manifest/report is absent")

    metadata_path, metadata_identity = _frozen_artifact(access, "catalog_metadata")
    model_path, model_identity = _frozen_artifact(access, "model")
    config_path, config_identity = _frozen_artifact(access, "config")
    metadata_rows = _load_locked_catalog_metadata(metadata_path)
    metadata_events = [row["event"] for row in metadata_rows]
    metadata_set = set(metadata_events)

    source_path = expected_paths["catalog_source_manifest"]
    if not source_path.is_file():
        raise FileNotFoundError("locked catalog source manifest is absent")
    with source_path.open("r", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    source_by_event: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError("locked catalog source manifest must contain JSON objects")
        event = str(row.get("event", ""))
        sample_path = Path(str(row.get("path", ""))).resolve()
        available_ifos = tuple(str(value) for value in row.get("available_ifos", ()))
        availability = row.get("detector_availability")
        try:
            gps_start = float(row.get("gps_start", float("nan")))
            gps_end = float(row.get("gps_end", float("nan")))
        except (TypeError, ValueError) as error:
            raise ValueError("locked catalog source GPS interval is invalid") from error
        if (
            str(row.get("split")) != "test"
            or str(row.get("observing_run")) != "O4b"
            or event not in metadata_set
            or event in source_by_event
            or not sample_path.is_file()
            or row.get("sha256") != file_sha256(sample_path)
            or not available_ifos
            or not set(available_ifos).issubset({"H1", "L1", "V1"})
            or not isinstance(availability, list)
            or len(availability) != 3
            or any(value not in {0, 1} for value in availability)
            or [ifo for ifo, valid in zip(("H1", "L1", "V1"), availability) if valid]
            != list(available_ifos)
            or not math.isfinite(gps_start)
            or not math.isfinite(gps_end)
            or gps_end <= gps_start
        ):
            raise ValueError("locked catalog source row is incomplete or inconsistent")
        try:
            with np.load(sample_path, allow_pickle=False) as arrays:
                features = np.asarray(arrays["features"], dtype=np.float32)
                stored_ifos = tuple(str(value) for value in arrays["ifos"].tolist())
                stored_q = tuple(float(value) for value in arrays["q_values"].tolist())
                stored_availability = np.asarray(
                    arrays["detector_availability"], dtype=np.int8
                )
        except (OSError, KeyError, ValueError) as error:
            raise ValueError("locked catalog source is not a numeric multi-Q tensor") from error
        if (
            features.ndim != 4
            or features.shape[:2] != (3, 3)
            or stored_ifos != ("H1", "L1", "V1")
            or not np.allclose(stored_q, (4.0, 8.0, 16.0), rtol=0.0, atol=1e-6)
            or stored_availability.shape != (3,)
            or not np.array_equal(stored_availability, np.asarray(availability))
            or not np.isfinite(features).all()
            or np.any(features[stored_availability == 0] != 0)
        ):
            raise ValueError("locked catalog numeric source detector/Q contract differs")
        source_by_event[event] = row
    if set(source_by_event) != metadata_set:
        raise ValueError("locked catalog source must cover every metadata event exactly once")

    with candidate_path.open("r", encoding="utf-8") as handle:
        candidate_rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in candidate_rows):
        raise ValueError("locked catalog candidate manifest must contain JSON objects")
    scoring = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    scoring_identity = scoring.get("candidate_manifest", {})
    scoring_inputs = scoring.get("locked_suite_inputs")
    required_scoring_inputs = {
        key: bindings[key]
        for key in (
            "catalog_source_manifest",
            "catalog_candidate_manifest",
            "catalog_candidate_report",
        )
    }
    if (
        scoring.get("status") != "locked_catalog_candidate_scoring_complete"
        or scoring.get("split") != "test"
        or scoring.get("threshold_applied") is not False
        or scoring.get("candidate_pruning_applied") is not False
        or scoring.get("all_instances_retained") is not True
        or int(scoring.get("test_rows_scored", -1)) != len(metadata_events)
        or int(scoring.get("candidate_rows", -1)) != len(candidate_rows)
        or Path(str(scoring_identity.get("path", ""))).resolve() != candidate_path
        or scoring_identity.get("sha256") != file_sha256(candidate_path)
        or scoring.get("model") != model_identity
        or scoring.get("config") != config_identity
        or scoring.get("catalog_metadata") != metadata_identity
        or scoring.get("source_manifest")
        != {"path": str(source_path), "sha256": file_sha256(source_path)}
        or scoring.get("locked_suite_access") != access
        or scoring_inputs != required_scoring_inputs
    ):
        raise ValueError("locked catalog candidate scoring report is not publication eligible")

    candidates_by_event: dict[str, list[dict[str, Any]]] = {
        event: [] for event in metadata_events
    }
    candidate_ids: set[str] = set()
    instance_references = 0
    unique_instance_ids: set[str] = set()
    mask_references = 0
    for row in candidate_rows:
        event = str(row.get("event", ""))
        candidate_id = str(row.get("candidate_id", ""))
        try:
            ranking_score = float(row.get("ranking_score", float("nan")))
        except (TypeError, ValueError) as error:
            raise ValueError("locked catalog candidate ranking score is invalid") from error
        instances = row.get("instances")
        source_sha256 = str(source_by_event.get(event, {}).get("sha256", ""))
        if (
            str(row.get("split")) != "test"
            or event not in metadata_set
            or not candidate_id
            or candidate_id in candidate_ids
            or not math.isfinite(ranking_score)
            or not isinstance(instances, list)
            or row.get("source_sha256") != source_sha256
        ):
            raise ValueError("locked catalog candidate row is incomplete or duplicated")
        candidate_ids.add(candidate_id)
        local_instances: set[str] = set()
        normalized_instances = []
        for instance in instances:
            if not isinstance(instance, dict):
                raise ValueError("locked catalog instance must be an object")
            instance_id = str(instance.get("instance_id", ""))
            class_name = str(instance.get("class_name", ""))
            mask_path = Path(str(instance.get("mask_path", ""))).resolve()
            try:
                confidence = float(instance.get("confidence", float("nan")))
            except (TypeError, ValueError) as error:
                raise ValueError("locked catalog instance confidence is invalid") from error
            if (
                not instance_id
                or instance_id in local_instances
                or not class_name
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
                or not mask_path.is_file()
                or instance.get("mask_sha256") != file_sha256(mask_path)
            ):
                raise ValueError("locked catalog instance/mask failed producer replay")
            local_instances.add(instance_id)
            unique_instance_ids.add(instance_id)
            instance_references += 1
            mask_references += 1
            normalized_instances.append(
                {
                    **instance,
                    "instance_id": instance_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "mask_path": str(mask_path),
                    "mask_sha256": file_sha256(mask_path),
                }
            )
        normalized = {
            key: value
            for key, value in row.items()
            if key not in {"event", "split", "candidate_id", "ranking_score", "instances"}
        }
        normalized.update(
            {
                "candidate_id": candidate_id,
                "ranking_score": ranking_score,
                "instances": normalized_instances,
            }
        )
        candidates_by_event[event].append(normalized)

    prediction_rows = [
        {
            "split": "test",
            "event": event,
            "candidates": sorted(
                candidates_by_event[event], key=lambda row: str(row["candidate_id"])
            ),
        }
        for event in metadata_events
    ]
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in prediction_rows
    )
    if prediction_path.is_file():
        if prediction_path.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing locked catalog prediction manifest is not deterministic")
    else:
        atomic_write_text(prediction_path, serialized)
    result = {
        "status": "locked_catalog_prediction_manifest_complete",
        "producer_complete": True,
        "scientific_claim_allowed": False,
        "split": "test",
        "threshold_applied": False,
        "candidate_pruning_applied": False,
        "events": len(prediction_rows),
        "candidate_count": len(candidate_rows),
        "instance_reference_count": instance_references,
        "unique_instance_count": len(unique_instance_ids),
        "mask_reference_count": mask_references,
        "all_candidates_retained": sum(
            len(row["candidates"]) for row in prediction_rows
        )
        == len(candidate_rows),
        "all_instances_and_masks_retained": instance_references == mask_references,
        "model": model_identity,
        "config": config_identity,
        "catalog_metadata": metadata_identity,
        "source_manifest": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "candidate_scoring_report": {
            "path": str(candidate_report_path),
            "sha256": file_sha256(candidate_report_path),
        },
        "prediction_manifest": {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        },
        "locked_suite_access": access,
        "locked_suite_inputs": bindings,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


def run_locked_gwtc5_catalog_diagnostic(
    prediction_manifest: str | Path,
    prediction_report: str | Path,
    candidate_search_report: str | Path,
    locked_suite_plan: str | Path,
    access_log: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Describe GWTC-5 recovery at one own-search validation-frozen threshold."""

    from .evaluation_lock import validate_locked_evaluation_suite_access

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("locked catalog diagnostics are immutable")
    suite_access = validate_locked_evaluation_suite_access(
        locked_suite_plan, access_log, "catalog_diagnostic", output_path
    )
    producer_access, suite_inputs = _catalog_suite_bindings(
        locked_suite_plan, access_log, include_predictions=True
    )
    if producer_access != suite_access:
        raise ValueError("locked catalog producer/evaluator access bindings differ")
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

    producer_path = Path(prediction_report).resolve()
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    prediction_path = Path(prediction_manifest).resolve()
    if (
        producer.get("status") != "locked_catalog_prediction_manifest_complete"
        or producer.get("producer_complete") is not True
        or producer.get("threshold_applied") is not False
        or producer.get("candidate_pruning_applied") is not False
        or producer.get("all_candidates_retained") is not True
        or producer.get("all_instances_and_masks_retained") is not True
        or producer.get("locked_suite_access") != suite_access
        or producer.get("locked_suite_inputs") != suite_inputs
        or Path(str(producer.get("prediction_manifest", {}).get("path", ""))).resolve()
        != prediction_path
        or producer.get("prediction_manifest", {}).get("sha256")
        != file_sha256(prediction_path)
        or producer.get("catalog_metadata")
        != {"path": str(metadata_path), "sha256": file_sha256(metadata_path)}
        or producer.get("model", {}).get("sha256")
        != search.get("identity", {}).get("candidate_checkpoint_sha256")
        or producer.get("config", {}).get("sha256")
        != search.get("identity", {}).get("candidate_config_sha256")
    ):
        raise ValueError("locked catalog prediction producer report failed replay")
    for key in ("source_manifest", "candidate_manifest", "candidate_scoring_report"):
        identity = producer.get(key, {})
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or identity.get("sha256") != file_sha256(path):
            raise ValueError(f"locked catalog producer input failed replay: {key}")

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
        "prediction_report": {
            "path": str(producer_path),
            "sha256": file_sha256(producer_path),
        },
        "locked_suite_access": suite_access,
        "locked_suite_inputs": suite_inputs,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result
