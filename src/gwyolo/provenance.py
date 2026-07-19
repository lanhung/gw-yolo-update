from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash


SPLITS = ("train", "val", "test")
SCENE_TYPES = ("chirp_only", "noise_only", "overlap", "quiet")
LEAKAGE_AXES = ("waveform_id", "injection_id", "glitch_id", "gps_block")


@dataclass(frozen=True)
class SceneRecipe:
    """A deterministic physical scene description, independent of its renderings."""

    split: str
    scene_type: str
    observing_run: str
    gps_start: int
    duration: float
    sample_rate: int
    ifos: tuple[str, ...]
    q_values: tuple[float, ...]
    seed: int
    waveform_id: str | None = None
    injection_id: str | None = None
    glitch_id: str | None = None
    glitch_ifo: str | None = None
    source_family: str | None = None
    target_snr: float | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.scene_type not in SCENE_TYPES:
            raise ValueError(f"Unsupported scene type: {self.scene_type}")
        if self.duration <= 0 or self.sample_rate <= 0:
            raise ValueError("duration and sample_rate must be positive")
        if not self.ifos or not self.q_values:
            raise ValueError("ifos and q_values cannot be empty")
        has_chirp = self.scene_type in {"chirp_only", "overlap"}
        has_glitch = self.scene_type in {"noise_only", "overlap"}
        if has_chirp and not (self.waveform_id and self.injection_id and self.source_family):
            raise ValueError("chirp scenes require waveform, injection, and source-family IDs")
        if not has_chirp and (self.waveform_id or self.injection_id):
            raise ValueError("non-chirp scenes cannot carry waveform or injection IDs")
        if has_glitch and not (self.glitch_id and self.glitch_ifo):
            raise ValueError("glitch scenes require glitch_id and glitch_ifo")
        if self.glitch_ifo and self.glitch_ifo not in self.ifos:
            raise ValueError("glitch_ifo must be present in ifos")

    @property
    def gps_block(self) -> str:
        return f"{self.observing_run}:{self.gps_start}:{self.duration:g}"

    @property
    def scene_id(self) -> str:
        return f"scene-{canonical_hash(self.to_dict(), length=20)}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ifos"] = list(self.ifos)
        value["q_values"] = list(self.q_values)
        value["gps_block"] = self.gps_block
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneRecipe":
        fields = dict(value)
        fields.pop("gps_block", None)
        fields["ifos"] = tuple(fields["ifos"])
        fields["q_values"] = tuple(float(item) for item in fields["q_values"])
        return cls(**fields)


def write_recipe_manifest(path: str | Path, recipes: Iterable[SceneRecipe]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for recipe in recipes:
                handle.write(json.dumps(recipe.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_recipe_manifest(path: str | Path) -> list[SceneRecipe]:
    recipes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                recipes.append(SceneRecipe.from_dict(json.loads(line)))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid recipe at {path}:{line_number}: {exc}") from exc
    return recipes


def audit_provenance(recipes: Iterable[SceneRecipe]) -> dict[str, Any]:
    rows = list(recipes)
    seen_scene_ids: set[str] = set()
    duplicate_scene_ids: list[str] = []
    values: dict[str, dict[str, set[str]]] = {
        axis: {split: set() for split in SPLITS} for axis in LEAKAGE_AXES
    }
    split_counts = {split: 0 for split in SPLITS}
    type_counts = {split: {kind: 0 for kind in SCENE_TYPES} for split in SPLITS}

    for recipe in rows:
        if recipe.scene_id in seen_scene_ids:
            duplicate_scene_ids.append(recipe.scene_id)
        seen_scene_ids.add(recipe.scene_id)
        split_counts[recipe.split] += 1
        type_counts[recipe.split][recipe.scene_type] += 1
        for axis in LEAKAGE_AXES:
            value = recipe.gps_block if axis == "gps_block" else getattr(recipe, axis)
            if value:
                values[axis][recipe.split].add(str(value))

    overlaps: dict[str, dict[str, list[str]]] = {}
    for axis, split_values in values.items():
        axis_overlaps: dict[str, list[str]] = {}
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                shared = sorted(split_values[left] & split_values[right])
                if shared:
                    axis_overlaps[f"{left}:{right}"] = shared
        overlaps[axis] = axis_overlaps

    overlap_count = sum(len(items) for axis in overlaps.values() for items in axis.values())
    passed = not duplicate_scene_ids and overlap_count == 0
    return {
        "passed": passed,
        "scene_count": len(rows),
        "unique_scene_count": len(seen_scene_ids),
        "duplicate_scene_ids": sorted(set(duplicate_scene_ids)),
        "split_counts": split_counts,
        "scene_type_counts": type_counts,
        "cross_split_overlaps": overlaps,
        "cross_split_overlap_count": overlap_count,
        "unique_physical_ids": {
            axis: {split: len(split_values[split]) for split in SPLITS}
            for axis, split_values in values.items()
        },
    }
