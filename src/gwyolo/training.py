from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256


def _metrics_dict(result: Any) -> dict[str, float]:
    raw = getattr(result, "results_dict", {}) or {}
    output: dict[str, float] = {}
    for key, value in raw.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _target_checkpoint_callback(metric_key: str, destination: Path):
    """Build a callback that saves the best epoch for one explicit metric."""
    state: dict[str, Any] = {"value": None, "epoch": None}

    def callback(trainer: Any) -> None:
        raw_value = (getattr(trainer, "metrics", {}) or {}).get(metric_key)
        if raw_value is None:
            return
        value = float(raw_value)
        if state["value"] is not None and value <= float(state["value"]):
            return
        source = Path(trainer.last)
        if not source.is_file():
            raise RuntimeError(f"Target-metric checkpoint source does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        state.update({"value": value, "epoch": int(trainer.epoch) + 1})

    return callback, state


def train_candidate(
    candidate: dict[str, Any],
    common: dict[str, Any],
    dataset_yaml: str | Path,
    output_dir: str | Path,
    selection_metric: str | None = None,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Training requires the 'train' optional dependencies") from exc

    name = str(candidate["name"])
    project = Path(output_dir).resolve() / "training"
    project.mkdir(parents=True, exist_ok=True)
    summary_path = project / name / "training_summary.json"
    if summary_path.is_file() and bool(candidate.get("reuse", True)):
        import json

        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    settings = dict(common)
    settings.update(candidate.get("overrides", {}))
    settings["epochs"] = int(candidate["epochs"])
    settings.update(
        {
            "data": str(Path(dataset_yaml).resolve()),
            "project": str(project),
            "name": name,
            "exist_ok": True,
            "save": True,
            "plots": True,
            "val": True,
        }
    )
    start = time.perf_counter()
    model = YOLO(str(candidate["model"]))
    target_checkpoint = project / name / "weights" / "best_target.pt"
    target_state: dict[str, Any] = {"value": None, "epoch": None}
    if selection_metric:
        target_callback, target_state = _target_checkpoint_callback(selection_metric, target_checkpoint)
        model.add_callback("on_model_save", target_callback)
    result = model.train(**settings)
    elapsed = time.perf_counter() - start
    save_dir = Path(getattr(result, "save_dir", project / name)).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    selected = target_checkpoint if target_checkpoint.is_file() else best
    summary = {
        "candidate": candidate,
        "settings": settings,
        "elapsed_seconds": elapsed,
        "save_dir": str(save_dir),
        "best_checkpoint": str(best) if best.is_file() else None,
        "best_sha256": file_sha256(best) if best.is_file() else None,
        "selection_metric": selection_metric,
        "selection_metric_value": target_state["value"],
        "selection_epoch": target_state["epoch"],
        "selection_checkpoint": str(selected) if selected.is_file() else None,
        "selection_sha256": file_sha256(selected) if selected.is_file() else None,
        "last_checkpoint": str(last) if last.is_file() else None,
        "last_sha256": file_sha256(last) if last.is_file() else None,
        "training_metrics": _metrics_dict(result),
    }
    atomic_write_json(summary_path, summary)
    return summary


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_yaml: str | Path,
    split: str,
    output_dir: str | Path,
    name: str,
    imgsz: int = 640,
    batch: int = 16,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Evaluation requires the 'train' optional dependencies") from exc

    project = Path(output_dir).resolve() / "evaluation"
    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(Path(dataset_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "dataset": str(Path(dataset_yaml).resolve()),
        "split": split,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": _metrics_dict(result),
        "speed_ms_per_image": {
            str(key): float(value) for key, value in (getattr(result, "speed", {}) or {}).items()
        },
    }
    atomic_write_json(project / name / f"{split}_summary.json", summary)
    return summary
