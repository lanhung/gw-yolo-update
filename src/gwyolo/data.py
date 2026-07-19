from __future__ import annotations

import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SOURCE_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    source: str
    image: str
    label: str
    image_sha256: str
    label_sha256: str
    image_bytes: int
    objects: int
    class_0: int
    class_1: int
    empty: bool
    split: str = ""


def derive_group_id(filename: str) -> str:
    """Derive a provenance group while excluding the Roboflow export hash."""
    stem = Path(filename).stem
    physical_prefix = stem.split(".rf.", 1)[0]
    tokens = SOURCE_TOKEN.findall(physical_prefix)
    if tokens:
        return tokens[-1].lower()
    return physical_prefix.lower()


def parse_label(path: Path) -> tuple[int, Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    objects = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            objects += 1
            try:
                class_id = int(fields[0])
            except (ValueError, IndexError):
                errors.append(f"{path}:{line_number}: invalid class id")
                continue
            counts[class_id] += 1
            if len(fields) < 7 or (len(fields) - 1) % 2:
                errors.append(f"{path}:{line_number}: invalid segmentation polygon")
                continue
            try:
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(f"{path}:{line_number}: non-numeric coordinate")
                continue
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                errors.append(f"{path}:{line_number}: coordinate outside [0,1]")
    return objects, counts, errors


def scan_sources(sources: list[dict[str, Any]]) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for source in sources:
        image_dir = Path(source["images"])
        label_dir = Path(source["labels"])
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source directories: {image_dir}, {label_dir}")
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"Missing label for {image}")
                continue
            sample_id = image.name
            if sample_id in seen_ids:
                errors.append(f"Duplicate image name across sources: {sample_id}")
                continue
            seen_ids.add(sample_id)
            objects, counts, label_errors = parse_label(label)
            errors.extend(label_errors)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    group_id=derive_group_id(image.name),
                    source=str(source["name"]),
                    image=str(image.resolve()),
                    label=str(label.resolve()),
                    image_sha256=file_sha256(image),
                    label_sha256=file_sha256(label),
                    image_bytes=image.stat().st_size,
                    objects=objects,
                    class_0=counts[0],
                    class_1=counts[1],
                    empty=objects == 0,
                )
            )

    exact_image_duplicates = len(samples) - len({sample.image_sha256 for sample in samples})
    source_groups: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        source_groups[sample.source].add(sample.group_id)
    overlap: dict[str, list[str]] = {}
    source_names = sorted(source_groups)
    for index, left in enumerate(source_names):
        for right in source_names[index + 1 :]:
            shared = sorted(source_groups[left] & source_groups[right])
            if shared:
                overlap[f"{left}__{right}"] = shared

    report = {
        "samples": len(samples),
        "groups": len({sample.group_id for sample in samples}),
        "objects": sum(sample.objects for sample in samples),
        "class_instances": {
            "0": sum(sample.class_0 for sample in samples),
            "1": sum(sample.class_1 for sample in samples),
        },
        "empty_labels": sum(sample.empty for sample in samples),
        "exact_image_duplicates": exact_image_duplicates,
        "source_group_overlap": overlap,
        "errors": errors,
    }
    return samples, report


def _group_rows(samples: list[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    return dict(groups)


def _split_score(
    counts: dict[str, Counter[str]], targets: dict[str, dict[str, float]], split_names: tuple[str, ...]
) -> float:
    score = 0.0
    for split in split_names:
        for key in ("samples", "class_0", "class_1", "empty"):
            target = targets[split][key]
            score += ((counts[split][key] - target) / max(1.0, target)) ** 2
    return score


def assign_group_splits(
    samples: list[Sample], fractions: dict[str, float], seed: int, trials: int = 2000
) -> tuple[list[Sample], dict[str, Any]]:
    split_names = tuple(fractions)
    groups = _group_rows(samples)
    totals = {
        "samples": len(samples),
        "class_0": sum(sample.class_0 for sample in samples),
        "class_1": sum(sample.class_1 for sample in samples),
        "empty": sum(sample.empty for sample in samples),
    }
    targets = {
        split: {key: value * fractions[split] for key, value in totals.items()} for split in split_names
    }
    group_stats: dict[str, Counter[str]] = {}
    for group_id, rows in groups.items():
        group_stats[group_id] = Counter(
            samples=len(rows),
            class_0=sum(row.class_0 for row in rows),
            class_1=sum(row.class_1 for row in rows),
            empty=sum(row.empty for row in rows),
        )

    best_score = float("inf")
    best_assignment: dict[str, str] | None = None
    for trial in range(max(1, trials)):
        rng = random.Random(seed + trial)
        group_ids = list(groups)
        rng.shuffle(group_ids)
        group_ids.sort(
            key=lambda group_id: (
                group_stats[group_id]["samples"]
                + group_stats[group_id]["class_0"]
                + group_stats[group_id]["class_1"]
            ),
            reverse=True,
        )
        counts = {split: Counter() for split in split_names}
        assignment: dict[str, str] = {}
        for group_id in group_ids:
            candidate_scores: list[tuple[float, float, str]] = []
            for split in split_names:
                proposed = {name: Counter(counter) for name, counter in counts.items()}
                proposed[split].update(group_stats[group_id])
                capacity = proposed[split]["samples"] / max(1.0, targets[split]["samples"])
                candidate_scores.append((_split_score(proposed, targets, split_names), capacity, split))
            _, _, selected = min(candidate_scores)
            assignment[group_id] = selected
            counts[selected].update(group_stats[group_id])
        score = _split_score(counts, targets, split_names)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Unable to create a group split")
    assigned = [replace(sample, split=best_assignment[sample.group_id]) for sample in samples]
    group_sets = {
        split: {sample.group_id for sample in assigned if sample.split == split} for split in split_names
    }
    overlap = {
        f"{left}__{right}": sorted(group_sets[left] & group_sets[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
        if group_sets[left] & group_sets[right]
    }
    report = {
        "seed": seed,
        "trials": trials,
        "score": best_score,
        "fractions": fractions,
        "splits": {
            split: {
                "samples": sum(sample.split == split for sample in assigned),
                "groups": len(group_sets[split]),
                "class_0": sum(sample.class_0 for sample in assigned if sample.split == split),
                "class_1": sum(sample.class_1 for sample in assigned if sample.split == split),
                "empty": sum(sample.empty for sample in assigned if sample.split == split),
            }
            for split in split_names
        },
        "cross_split_group_overlap": overlap,
    }
    return assigned, report


def write_manifest(samples: list[Sample], path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(samples[0]).keys()) if samples else [field.name for field in Sample.__dataclass_fields__.values()]
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in sorted(samples, key=lambda row: row.sample_id):
            writer.writerow(asdict(sample))
    os.replace(temporary, target)
    return file_sha256(target)


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown link_mode: {mode}")


def materialize_dataset(
    samples: list[Sample], root: str | Path, class_names: list[str], link_mode: str = "symlink"
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    for split in sorted({sample.split for sample in samples}):
        (root_path / split / "images").mkdir(parents=True, exist_ok=True)
        (root_path / split / "labels").mkdir(parents=True, exist_ok=True)
    for sample in samples:
        _materialize(Path(sample.image), root_path / sample.split / "images" / sample.sample_id, link_mode)
        _materialize(Path(sample.label), root_path / sample.split / "labels" / f"{Path(sample.sample_id).stem}.txt", link_mode)

    dataset_yaml = {
        "path": str(root_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path = root_path / "dataset.yaml"
    atomic_write_text(yaml_path, yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True))
    return {"root": str(root_path), "yaml": str(yaml_path), "link_mode": link_mode}


def audit_and_split(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    data_config = config["data"]
    samples, audit = scan_sources(data_config["sources"])
    audit_path = output / "data" / "source_audit.json"
    atomic_write_json(audit_path, audit)
    if audit["errors"]:
        raise ValueError(f"Data audit failed with {len(audit['errors'])} errors; see {audit_path}")

    split_config = data_config["split"]
    fractions = {key: float(split_config[key]) for key in ("train", "val", "test")}
    assigned, split_report = assign_group_splits(
        samples,
        fractions,
        seed=int(config["project"]["seed"]),
        trials=int(split_config.get("trials", 2000)),
    )
    if split_report["cross_split_group_overlap"]:
        raise RuntimeError("Group leakage detected after splitting")

    manifest_path = output / "data" / "manifest.csv"
    manifest_hash = write_manifest(assigned, manifest_path)
    dataset = materialize_dataset(
        assigned,
        output / "data" / "dataset",
        list(data_config["names"]),
        link_mode=str(split_config.get("link_mode", "symlink")),
    )
    result = {
        "audit": audit,
        "split": split_report,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "dataset": dataset,
        "data_signature": canonical_hash([asdict(sample) for sample in assigned]),
    }
    atomic_write_json(output / "data" / "summary.json", result)
    return result
