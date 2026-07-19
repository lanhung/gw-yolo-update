from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256


EVENT_NAME = re.compile(r"(GW\d{6}_\d{6})")


def event_name_from_path(path: str | Path) -> str | None:
    match = EVENT_NAME.search(Path(path).name)
    return match.group(1) if match else None


def predict_catalog(
    checkpoint: str | Path,
    source: str | Path,
    output_dir: str | Path,
    confidence: float,
    imgsz: int = 640,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Prediction requires the 'train' optional dependencies") from exc

    output = Path(output_dir).resolve() / "catalog"
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "predictions.jsonl"
    temporary = jsonl_path.with_suffix(".jsonl.tmp")
    model = YOLO(str(checkpoint))
    start = time.perf_counter()
    images = 0
    detections = 0
    chirp_images = 0
    mask_instances = 0
    with temporary.open("w", encoding="utf-8") as handle:
        results = model.predict(source=str(source), conf=confidence, imgsz=imgsz, stream=True, verbose=False)
        for result in results:
            images += 1
            boxes = result.boxes
            names = result.names
            polygons: list[list[list[float]]] = []
            if result.masks is not None and result.masks.xyn is not None:
                polygons = [polygon.tolist() for polygon in result.masks.xyn]
            instances: list[dict[str, Any]] = []
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                scores = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                for index, (box, score, class_value) in enumerate(zip(xyxy, scores, classes)):
                    class_id = int(class_value)
                    polygon = polygons[index] if index < len(polygons) else None
                    if polygon:
                        mask_instances += 1
                    instances.append(
                        {
                            "class_id": class_id,
                            "class_name": str(names[class_id]),
                            "confidence": float(score),
                            "xyxy": [float(value) for value in box],
                            "polygon_xyn": polygon,
                        }
                    )
            detections += len(instances)
            has_chirp = any(item["class_name"].lower() == "chirp" for item in instances)
            chirp_images += int(has_chirp)
            record = {
                "image": str(Path(result.path).resolve()),
                "event": event_name_from_path(result.path),
                "orig_shape": list(result.orig_shape),
                "instances": instances,
                "has_chirp": has_chirp,
                "speed_ms": {str(key): float(value) for key, value in (result.speed or {}).items()},
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(jsonl_path)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source": str(Path(source).resolve()),
        "confidence": confidence,
        "images": images,
        "detections": detections,
        "chirp_images": chirp_images,
        "chirp_image_rate": chirp_images / images if images else None,
        "mask_instances": mask_instances,
        "elapsed_seconds": time.perf_counter() - start,
        "predictions": str(jsonl_path),
    }
    atomic_write_json(output / "prediction_summary.json", summary)
    return summary
