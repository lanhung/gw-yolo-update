from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .metrics import wilson_interval
from .physical_training import physical_split_audit
from .runtime import execution_provenance


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _validation_role(injection_id: str, selection_fraction: float, seed: int) -> str:
    draw = int(
        canonical_hash(
            {
                "injection_id": injection_id,
                "seed": seed,
                "purpose": "candidate_refiner_validation_selection_v1",
            },
            16,
        ),
        16,
    ) / float(16**16 - 1)
    return "selection" if draw < selection_fraction else "calibration"


def label_candidate_refiner_rows(
    injection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    split: str,
    positive_padding_seconds: float,
    validation_selection_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label every candidate without top-k pruning and preserve parent-level roles."""

    if split not in {"train", "val"} or positive_padding_seconds < 0:
        raise ValueError("candidate refiner split or positive padding is invalid")
    if not 0 < validation_selection_fraction < 1:
        raise ValueError("candidate refiner validation selection fraction is invalid")
    parents = {}
    roles = {}
    for row in injection_rows:
        if row.get("split") != split:
            raise ValueError("candidate refiner injection manifest has the wrong split")
        injection_id = str(row["injection_id"])
        if injection_id in parents:
            raise ValueError(f"candidate refiner repeats injection: {injection_id}")
        parents[injection_id] = row
        roles[injection_id] = (
            "train"
            if split == "train"
            else _validation_role(injection_id, validation_selection_fraction, seed)
        )
    if not parents:
        raise ValueError("candidate refiner requires injection parents")
    candidate_ids = set()
    output = []
    by_arrival: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"candidate refiner repeats candidate: {candidate_id}")
        candidate_ids.add(candidate_id)
        injection_id = str(row["injection_id"])
        if injection_id not in parents or row.get("split") != split:
            raise ValueError("candidate refiner candidate has the wrong parent or split")
        ifo = str(row["ifo"])
        arrivals = parents[injection_id].get("detector_arrival_gps", {})
        if ifo not in arrivals:
            raise ValueError("candidate refiner candidate uses an unavailable detector")
        start = float(row["gps_start"])
        stop = float(row["gps_end"])
        peak = float(row["gps_peak"])
        arrival = float(arrivals[ifo])
        if not np.isfinite([start, stop, peak, arrival]).all() or stop <= start:
            raise ValueError("candidate refiner candidate geometry is invalid")
        distance = max(start - arrival, 0.0, arrival - stop)
        positive = distance <= positive_padding_seconds
        enriched = {
            **row,
            "refiner_role": roles[injection_id],
            "refiner_positive": bool(positive),
            "target_detector_arrival_gps": arrival,
            "interval_distance_to_arrival_seconds": distance,
            "peak_error_seconds": abs(peak - arrival),
            "positive_padding_seconds": positive_padding_seconds,
            "top_k_pruned": False,
        }
        output.append(enriched)
        by_arrival[(injection_id, ifo)].append(bool(positive))
    expected_arrivals = {
        (injection_id, str(ifo))
        for injection_id, parent in parents.items()
        for ifo in parent.get("detector_arrival_gps", {})
    }
    missing_candidate_arrivals = sorted(expected_arrivals - set(by_arrival))
    if missing_candidate_arrivals:
        raise ValueError(
            f"candidate refiner input lacks candidates for arrivals: {missing_candidate_arrivals[:10]}"
        )
    covered = sum(any(by_arrival[key]) for key in expected_arrivals)
    positive_count = sum(bool(row["refiner_positive"]) for row in output)
    role_counts = Counter(str(row["refiner_role"]) for row in output)
    parent_role_counts = Counter(roles.values())
    report = {
        "split": split,
        "injections": len(parents),
        "waveforms": len({str(row["waveform_id"]) for row in parents.values()}),
        "gps_blocks": len({str(row["gps_block"]) for row in parents.values()}),
        "candidates": len(output),
        "positive_candidates": positive_count,
        "negative_candidates": len(output) - positive_count,
        "positive_candidate_fraction": positive_count / max(len(output), 1),
        "expected_detector_arrivals": len(expected_arrivals),
        "arrivals_with_positive_candidate": covered,
        "positive_candidate_coverage_fraction": covered / len(expected_arrivals),
        "positive_candidate_coverage_wilson_95": list(
            wilson_interval(covered, len(expected_arrivals))
        ),
        "candidate_counts_by_role": dict(sorted(role_counts.items())),
        "parent_counts_by_role": dict(sorted(parent_role_counts.items())),
        "candidate_counts_by_ifo": dict(
            sorted(Counter(str(row["ifo"]) for row in output).items())
        ),
        "positive_counts_by_ifo": dict(
            sorted(
                Counter(
                    str(row["ifo"]) for row in output if row["refiner_positive"]
                ).items()
            )
        ),
        "all_connected_candidates_retained": len(output) == len(candidate_rows),
        "top_k_pruning": None,
    }
    return output, report


def run_candidate_refiner_plan(
    train_injection_manifest: str | Path,
    train_candidate_manifest: str | Path,
    validation_injection_manifest: str | Path,
    validation_candidate_manifest: str | Path,
    output_dir: str | Path,
    positive_padding_seconds: float = 0.5,
    validation_selection_fraction: float = 0.2,
    seed: int = 20260720,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "train_injection_manifest_sha256": file_sha256(train_injection_manifest),
        "train_candidate_manifest_sha256": file_sha256(train_candidate_manifest),
        "validation_injection_manifest_sha256": file_sha256(
            validation_injection_manifest
        ),
        "validation_candidate_manifest_sha256": file_sha256(
            validation_candidate_manifest
        ),
        "positive_padding_seconds": float(positive_padding_seconds),
        "validation_selection_fraction": float(validation_selection_fraction),
        "seed": int(seed),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_refiner_plan_report.json"
    if report_path.is_file():
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != identity:
            raise ValueError("completed candidate refiner plan has another identity")
        return result
    if any(output.iterdir()):
        raise FileExistsError("candidate refiner plan output must be empty")
    train_injections = _read_jsonl(train_injection_manifest)
    validation_injections = _read_jsonl(validation_injection_manifest)
    split_audit = physical_split_audit(train_injections, validation_injections)
    train_rows, train_summary = label_candidate_refiner_rows(
        train_injections,
        _read_jsonl(train_candidate_manifest),
        "train",
        positive_padding_seconds,
        validation_selection_fraction,
        seed,
    )
    validation_rows, validation_summary = label_candidate_refiner_rows(
        validation_injections,
        _read_jsonl(validation_candidate_manifest),
        "val",
        positive_padding_seconds,
        validation_selection_fraction,
        seed,
    )
    destinations = {
        "train": output / "candidate_refiner_train.jsonl",
        "selection": output / "candidate_refiner_validation_selection.jsonl",
        "calibration": output / "candidate_refiner_validation_calibration.jsonl",
    }
    atomic_write_text(
        destinations["train"],
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
    )
    for role in ("selection", "calibration"):
        rows = [row for row in validation_rows if row["refiner_role"] == role]
        atomic_write_text(
            destinations[role],
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    result = {
        "status": "candidate_local_refiner_group_safe_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "labeled candidate plans are train/validation supervision, not search recall; "
            "continuous background and locked-test VT remain required"
        ),
        "test_evaluation": None,
        "run_identity": identity,
        "split_audit": split_audit,
        "train": train_summary,
        "validation": validation_summary,
        "manifests": {role: str(path) for role, path in destinations.items()},
        "manifest_sha256": {
            role: file_sha256(path) for role, path in destinations.items()
        },
        "validation_parent_roles_are_group_safe": True,
        "all_connected_candidates_retained": True,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result
