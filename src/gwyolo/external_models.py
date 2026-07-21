from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256, load_yaml
from .runtime import execution_provenance


def _digest(path: Path, algorithm: str) -> str:
    try:
        digest = hashlib.new(algorithm)
    except ValueError as error:
        raise ValueError(f"unsupported model checksum algorithm: {algorithm}") from error
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model source entry must be a mapping")
    required = ("backend", "role", "filename", "url", "size_bytes", "checksum")
    missing = [field for field in required if value.get(field) in (None, "")]
    if missing:
        raise ValueError(f"model source entry is missing fields: {missing}")
    filename = str(value["filename"])
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"unsafe model source filename: {filename}")
    try:
        size = int(value["size_bytes"])
    except (TypeError, ValueError) as error:
        raise ValueError("model source size_bytes must be an integer") from error
    if size <= 0:
        raise ValueError("model source size_bytes must be positive")
    checksum = value["checksum"]
    if not isinstance(checksum, dict):
        raise ValueError("model source checksum must be a mapping")
    algorithm = str(checksum.get("algorithm", ""))
    expected = str(checksum.get("value", ""))
    expected_lengths = {"md5": 32, "sha256": 64}
    if algorithm not in expected_lengths or len(expected) != expected_lengths[algorithm]:
        raise ValueError("model source checksum is malformed")
    return {**value, "filename": filename, "size_bytes": size}


def _verify(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    observed_size = path.stat().st_size
    algorithm = str(entry["checksum"]["algorithm"])
    observed_digest = _digest(path, algorithm)
    expected_digest = str(entry["checksum"]["value"])
    valid = observed_size == entry["size_bytes"] and observed_digest == expected_digest
    return {
        "path": str(path),
        "size_bytes": observed_size,
        "expected_size_bytes": entry["size_bytes"],
        "checksum_algorithm": algorithm,
        "checksum": observed_digest,
        "expected_checksum": expected_digest,
        "sha256": file_sha256(path),
        "valid": valid,
    }


def acquire_external_model_sources(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    download: bool = False,
    minimum_free_bytes: int = 0,
) -> dict[str, Any]:
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    config = load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("external model source schema_version must be 1")
    raw_entries = config.get("sources")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("external model source config requires a non-empty sources list")
    entries = [_validate_entry(value) for value in raw_entries]
    names = [entry["filename"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("external model source filenames must be unique")
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        target = target_dir / entry["filename"]
        if target.exists():
            verification = _verify(target, entry)
            if not verification["valid"]:
                raise ValueError(f"existing model source fails integrity lock: {target}")
        else:
            if not download:
                raise FileNotFoundError(f"model source is absent in verify-only mode: {target}")
            partial = target.with_suffix(target.suffix + ".part")
            partial_size = partial.stat().st_size if partial.exists() else 0
            remaining = max(0, entry["size_bytes"] - partial_size)
            free = shutil.disk_usage(target_dir).free
            if free - remaining < minimum_free_bytes:
                raise OSError(
                    f"insufficient free space for {target.name}: free={free}, "
                    f"remaining={remaining}, required_reserve={minimum_free_bytes}"
                )
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "8",
                    "--retry-delay",
                    "5",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    str(entry["url"]),
                ],
                check=True,
            )
            verification = _verify(partial, entry)
            if not verification["valid"]:
                raise ValueError(f"downloaded model source fails integrity lock: {partial}")
            os.replace(partial, target)
            verification["path"] = str(target)
        results.append(
            {
                "backend": entry["backend"],
                "role": entry["role"],
                "record_url": entry.get("record_url"),
                **verification,
            }
        )
    return {
        "status": "verified",
        "download_enabled": download,
        "output_dir": str(target_dir),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "minimum_free_bytes": minimum_free_bytes,
        "files": results,
        **execution_provenance(),
    }


def run_external_model_source_acquisition(
    config_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    download: bool = False,
    minimum_free_bytes: int = 0,
) -> dict[str, Any]:
    try:
        report = acquire_external_model_sources(
            config_path,
            output_dir,
            download=download,
            minimum_free_bytes=minimum_free_bytes,
        )
    except Exception as error:
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "download_enabled": download,
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": file_sha256(config_path),
            "minimum_free_bytes": minimum_free_bytes,
            **execution_provenance(),
        }
        atomic_write_json(report_path, report)
        raise
    atomic_write_json(report_path, report)
    return report
