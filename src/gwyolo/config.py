from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import canonical_hash, load_yaml


REQUIRED_SECTIONS = ("project", "data", "training", "quality")


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    sources = config["data"].get("sources", [])
    if not sources:
        raise ValueError("data.sources must contain at least one image/label pair")
    for source in sources:
        for key in ("name", "images", "labels"):
            if key not in source:
                raise ValueError(f"Data source is missing {key}: {source}")

    fractions = config["data"].get("split", {})
    total = sum(float(fractions.get(key, 0.0)) for key in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    if not config["training"].get("candidates"):
        raise ValueError("training.candidates cannot be empty")

    config["_meta"] = {
        "path": str(Path(path).resolve()),
        "hash": canonical_hash(config),
    }
    return config
