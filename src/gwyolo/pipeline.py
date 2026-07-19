from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import evaluate_catalog_predictions
from .data import audit_and_split
from .io import atomic_write_json
from .prediction import predict_catalog
from .training import evaluate_checkpoint, train_candidate


def _metric_value(summary: dict[str, Any], key: str) -> float | None:
    for location in (summary.get("metrics", {}), summary.get("training_metrics", {})):
        if key in location:
            return float(location[key])
    return None


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["project"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "status": "running",
    }
    atomic_write_json(output / "pipeline_state.json", state)

    data_summary = audit_and_split(config, output)
    state["data"] = data_summary
    atomic_write_json(output / "pipeline_state.json", state)

    quality = config["quality"]
    metric_key = str(quality["metric"])
    minimum = float(quality["minimum"])
    selected: dict[str, Any] | None = None
    experiments: list[dict[str, Any]] = []
    common = config["training"].get("common", {})
    dataset_yaml = data_summary["dataset"]["yaml"]
    for candidate in config["training"]["candidates"]:
        training = train_candidate(candidate, common, dataset_yaml, output, selection_metric=metric_key)
        checkpoint = training.get("selection_checkpoint") or training.get("best_checkpoint")
        if not checkpoint:
            raise RuntimeError(f"Candidate {candidate['name']} did not produce best.pt")
        validation = evaluate_checkpoint(
            checkpoint,
            dataset_yaml,
            "val",
            output,
            f"{candidate['name']}_validation",
            imgsz=int(common.get("imgsz", 640)),
            batch=int(common.get("batch", 16)),
        )
        score = _metric_value(validation, metric_key)
        experiment = {"candidate": candidate, "training": training, "validation": validation, "score": score}
        experiments.append(experiment)
        state["experiments"] = experiments
        atomic_write_json(output / "pipeline_state.json", state)
        if score is not None and score >= minimum:
            selected = experiment
            if bool(quality.get("stop_when_met", True)):
                break

    if selected is None and experiments:
        valid = [experiment for experiment in experiments if experiment["score"] is not None]
        selected = max(valid, key=lambda experiment: experiment["score"]) if valid else experiments[-1]
    if selected is None:
        raise RuntimeError("No experiment completed")

    checkpoint = selected["training"].get("selection_checkpoint") or selected["training"]["best_checkpoint"]
    test_summary = evaluate_checkpoint(
        checkpoint,
        dataset_yaml,
        "test",
        output,
        f"{selected['candidate']['name']}_locked_test",
        imgsz=int(common.get("imgsz", 640)),
        batch=int(common.get("batch", 16)),
    )
    state["selected"] = selected["candidate"]["name"]
    state["test"] = test_summary

    catalog_config = config.get("catalog", {})
    if catalog_config and Path(catalog_config.get("source", "")).is_dir():
        prediction = predict_catalog(
            checkpoint,
            catalog_config["source"],
            output,
            confidence=float(catalog_config.get("confidence", 0.25)),
            imgsz=int(common.get("imgsz", 640)),
        )
        catalog_eval = evaluate_catalog_predictions(
            prediction["predictions"],
            str(catalog_config["api_url"]),
            output / "catalog" / "catalog_evaluation.json",
        )
        state["catalog_prediction"] = prediction
        state["catalog_evaluation"] = catalog_eval

    selected_score = selected.get("score")
    state["quality_gate"] = {
        "metric": metric_key,
        "minimum": minimum,
        "validation_value": selected_score,
        "passed": selected_score is not None and selected_score >= minimum,
    }
    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output / "pipeline_state.json", state)
    return state


def load_existing_state(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / "pipeline_state.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
