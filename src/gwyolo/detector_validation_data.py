from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration
from .injections import plan_injection_recipes
from .io import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
)
from .runtime import execution_provenance
from .waveforms import _atomic_save_npz


DEFAULT_DETECTOR_SUBSETS = (
    "H1+L1",
    "H1+V1",
    "L1+V1",
    "H1+L1+V1",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def export_network_numeric_validation_background(
    network_manifest: str | Path,
    corpus_audit_path: str | Path,
    output_dir: str | Path,
    analysis_duration_seconds: float = 4.0,
    required_detector_subsets: Iterable[str] = DEFAULT_DETECTOR_SUBSETS,
    minimum_per_detector_subset: int = 25,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Export one independent real-noise bank per validation GPS block.

    This adapter preserves the already materialized aligned H1/L1/V1 numeric
    strain. It does not create injections: a later waveform materialization
    must project a fresh physical signal into every available detector.
    """

    source_path = Path(network_manifest).resolve()
    audit_path = Path(corpus_audit_path).resolve()
    audit = _read_json(audit_path)
    required = tuple(str(value) for value in required_detector_subsets)
    if (
        not required
        or len(required) != len(set(required))
        or any(not value for value in required)
        or minimum_per_detector_subset < 1
        or not np.isfinite(analysis_duration_seconds)
        or analysis_duration_seconds <= 0
    ):
        raise ValueError("Detector-validation background policy is invalid")
    if (
        audit.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or audit.get("passed") is not True
        or audit.get("scientific_claim_allowed") is not False
        or audit.get("validation_manifest_sha256") != file_sha256(source_path)
        or any(audit.get("split_audit", {}).get("cross_split_overlaps", {}).values())
    ):
        raise ValueError("Network background requires a source-safe validation corpus")

    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Network validation manifest is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_glitches: set[str] = set()
    for row in rows:
        glitch_id = str(row.get("glitch_id", ""))
        block = str(row.get("network_gps_block", ""))
        if (
            row.get("split") != "val"
            or row.get("aligned_network_context") is not True
            or not glitch_id
            or glitch_id in seen_glitches
            or not block
        ):
            raise ValueError("Network validation row violates its physical identity")
        seen_glitches.add(glitch_id)
        grouped[block].append(row)

    # One row per physical network GPS block prevents repeated glitches in the
    # same source context from masquerading as independent noise realizations.
    selected = [
        min(
            block_rows,
            key=lambda row: canonical_hash(
                {
                    "glitch_id": row["glitch_id"],
                    "numeric_sha256": row["sha256"],
                },
                64,
            ),
        )
        for _, block_rows in sorted(grouped.items())
    ]
    output = Path(output_dir).resolve()
    report_path = output / "detector_validation_background_report.json"
    manifest_path = output / "background_windows.jsonl"
    run_identity = {
        "network_manifest_sha256": file_sha256(source_path),
        "corpus_audit_sha256": file_sha256(audit_path),
        "analysis_duration_seconds": analysis_duration_seconds,
        "required_detector_subsets": list(required),
        "minimum_per_detector_subset": minimum_per_detector_subset,
        "selection": "one_canonical_row_per_network_gps_block_v1",
    }
    if report_path.is_file():
        prior = _read_json(report_path)
        if prior.get("run_identity") != run_identity:
            raise ValueError("Existing detector-validation bank has another identity")
        if file_sha256(manifest_path) != prior.get("manifest_sha256"):
            raise ValueError("Existing detector-validation manifest changed")
        if require_ready and prior.get("passed") is not True:
            raise RuntimeError(
                "Detector-validation background is below frozen subset floors: "
                f"{prior.get('detector_subset_deficits', {})}"
            )
        return prior

    output.mkdir(parents=True, exist_ok=True)
    bank_dir = output / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    counts: Counter[str] = Counter()
    run_counts: Counter[str] = Counter()
    for row in selected:
        source = Path(str(row["path"])).resolve()
        if file_sha256(source) != str(row["sha256"]):
            raise ValueError(f"Network numeric source hash changed: {source}")
        with np.load(source, allow_pickle=False) as arrays:
            required_arrays = {
                "raw_strain",
                "ifos",
                "sample_rate",
                "event_gps",
                "detector_availability",
            }
            missing = required_arrays - set(arrays.files)
            if missing:
                raise ValueError(f"Network numeric source lacks arrays: {sorted(missing)}")
            raw = np.asarray(arrays["raw_strain"], dtype=np.float64)
            model_ifos = [str(value) for value in arrays["ifos"].tolist()]
            availability = np.asarray(
                arrays["detector_availability"], dtype=np.uint8
            )
            sample_rate = int(arrays["sample_rate"])
            event_gps = float(arrays["event_gps"])
        if (
            raw.ndim != 2
            or raw.shape[0] != len(model_ifos)
            or availability.shape != (len(model_ifos),)
            or not np.isfinite(raw).all()
            or sample_rate <= 0
            or not np.isfinite(event_gps)
        ):
            raise ValueError("Network numeric strain has an invalid tensor contract")
        available = [
            ifo for ifo, valid in zip(model_ifos, availability) if int(valid) == 1
        ]
        if available != [str(value) for value in row.get("available_ifos", [])]:
            raise ValueError("Numeric and manifest detector availability differ")
        indices = [model_ifos.index(ifo) for ifo in available]
        noise = raw[indices]
        if len(available) < 2 or any(not np.any(noise[index]) for index in range(len(available))):
            raise ValueError("Available detector strain must be physically nonzero")
        context_duration = raw.shape[1] / sample_rate
        if analysis_duration_seconds >= context_duration:
            raise ValueError("Analysis duration must be shorter than numeric context")
        context_start = event_gps - context_duration / 2.0
        analysis_start = event_gps - analysis_duration_seconds / 2.0
        analysis_start_index = int(
            round((analysis_start - context_start) * sample_rate)
        )
        analysis_stop_index = analysis_start_index + int(
            round(analysis_duration_seconds * sample_rate)
        )
        if analysis_start_index < 0 or analysis_stop_index > raw.shape[1]:
            raise ValueError("Analysis crop falls outside numeric context")
        subset = "+".join(available)
        window_id = f"detector-validation:{canonical_hash({'gps_block': row['network_gps_block'], 'source_sha256': row['sha256']}, 24)}"
        bank_path = bank_dir / f"{canonical_hash(window_id, 32)}.npz"
        _atomic_save_npz(
            bank_path,
            noise=noise.astype(np.float32),
            ifos=np.asarray(available),
            sample_rate=np.asarray(sample_rate, dtype=np.int64),
            context_gps_start=np.asarray(context_start, dtype=np.float64),
            analysis_gps_start=np.asarray(analysis_start, dtype=np.float64),
            analysis_start_index=np.asarray(analysis_start_index, dtype=np.int64),
            analysis_stop_index=np.asarray(analysis_stop_index, dtype=np.int64),
            window_id=np.asarray(window_id),
        )
        bank_sha = file_sha256(bank_path)
        sources = {
            ifo: {
                "path": str(source),
                "sha256": str(row["sha256"]),
                "kind": "aligned_network_numeric_source",
            }
            for ifo in available
        }
        exported.append(
            {
                "window_id": window_id,
                "split": "val",
                "observing_run": str(row["observing_run"]),
                "gps_block": str(row["network_gps_block"]),
                "network_gps_block": str(row["network_gps_block"]),
                "gps_start": analysis_start,
                "gps_end": analysis_start + analysis_duration_seconds,
                "duration": analysis_duration_seconds,
                "ifos": available,
                "detector_subset": subset,
                "source_glitch_id": str(row["glitch_id"]),
                "source_numeric_path": str(source),
                "source_numeric_sha256": str(row["sha256"]),
                "source_files": sources,
                "background_bank": {
                    "path": str(bank_path),
                    "sha256": bank_sha,
                },
                "aligned_network_context": True,
                "candidate_scores_inspected": False,
                "physical_signal_present": False,
                "physical_signal_projection_required": True,
            }
        )
        counts[subset] += 1
        run_counts[str(row["observing_run"])] += 1

    manifest_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in exported
    )
    atomic_write_text(manifest_path, manifest_text)
    observed_counts = {subset: int(counts.get(subset, 0)) for subset in required}
    deficits = {
        subset: max(0, minimum_per_detector_subset - count)
        for subset, count in observed_counts.items()
    }
    ready = all(value == 0 for value in deficits.values())
    result = {
        "status": "exported_source_safe_detector_validation_background_bank",
        "passed": ready,
        "publication_calibration_eligible": ready,
        "scientific_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "candidate_scores_inspected": False,
        "physical_signal_present": False,
        "physical_signal_projection_required": True,
        "source_rows": len(rows),
        "selected_rows": len(exported),
        "selection": "one_canonical_row_per_network_gps_block_v1",
        "unique_network_gps_blocks": len(exported),
        "detector_subset_counts": observed_counts,
        "detector_subset_deficits": deficits,
        "minimum_per_detector_subset": minimum_per_detector_subset,
        "required_detector_subsets": list(required),
        "observing_run_counts": dict(sorted(run_counts.items())),
        "splits": {
            "val": {
                "windows": len(exported),
                "unique_gps_blocks": len(exported),
                "live_time_seconds": _union_duration(
                    (row["gps_start"], row["gps_end"]) for row in exported
                ),
                "live_time_years": _union_duration(
                    (row["gps_start"], row["gps_end"]) for row in exported
                )
                / SECONDS_PER_YEAR,
            },
            "test": {
                "windows": 0,
                "unique_gps_blocks": 0,
                "live_time_seconds": 0.0,
                "live_time_years": 0.0,
            },
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "corpus_audit_path": str(audit_path),
        "corpus_audit_sha256": file_sha256(audit_path),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    if require_ready and not ready:
        raise RuntimeError(
            "Detector-validation background is below frozen subset floors: "
            f"{deficits}"
        )
    return result


def plan_detector_stratified_validation_injections(
    background_manifest: str | Path,
    background_report: str | Path,
    output_dir: str | Path,
    injections_per_detector_subset: int = 100,
    required_detector_subsets: Iterable[str] = DEFAULT_DETECTOR_SUBSETS,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Freeze equal-count physical recipes for every detector subset."""

    manifest_path = Path(background_manifest).resolve()
    report_path = Path(background_report).resolve()
    report = _read_json(report_path)
    required = tuple(str(value) for value in required_detector_subsets)
    if (
        injections_per_detector_subset < 1
        or seed < 1
        or not required
        or len(required) != len(set(required))
    ):
        raise ValueError("Detector-stratified injection plan policy is invalid")
    if (
        report.get("status")
        != "exported_source_safe_detector_validation_background_bank"
        or report.get("passed") is not True
        or report.get("publication_calibration_eligible") is not True
        or report.get("physical_signal_projection_required") is not True
        or report.get("candidate_scores_inspected") is not False
        or report.get("manifest_sha256") != file_sha256(manifest_path)
        or tuple(report.get("required_detector_subsets", [])) != required
    ):
        raise ValueError("Detector-stratified injections require a ready background bank")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recipes = []
    component_reports = {}
    for subset_index, subset in enumerate(required):
        subset_rows = [row for row in rows if row["detector_subset"] == subset]
        if len({row["gps_block"] for row in subset_rows}) < int(
            report["minimum_per_detector_subset"]
        ):
            raise ValueError(
                f"Detector subset {subset} lacks independent background blocks"
            )
        live_time_years = (
            _union_duration(
                (float(row["gps_start"]), float(row["gps_end"]))
                for row in subset_rows
            )
            / SECONDS_PER_YEAR
        )
        subset_seed = seed + subset_index * 1_000_000
        subset_recipes, subset_report = plan_injection_recipes(
            subset_rows,
            {"val": live_time_years},
            {"val": injections_per_detector_subset},
            seed=subset_seed,
        )
        recipes.extend(subset_recipes)
        component_reports[subset] = {
            "seed": subset_seed,
            "background_windows": len(subset_rows),
            "unique_gps_blocks": len(
                {row["gps_block"] for row in subset_rows}
            ),
            "recipes": len(subset_recipes),
            "plan_hash": canonical_hash(subset_report, 64),
        }
    injection_ids = [str(row["injection_id"]) for row in recipes]
    waveform_ids = [str(row["waveform_id"]) for row in recipes]
    counts = Counter("+".join(row["ifos"]) for row in recipes)
    if (
        len(injection_ids) != len(set(injection_ids))
        or len(waveform_ids) != len(set(waveform_ids))
        or any(counts[subset] != injections_per_detector_subset for subset in required)
    ):
        raise ValueError("Detector-stratified recipe identities or quotas differ")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe_path = output / "detector_stratified_injection_recipes.jsonl"
    plan_path = output / "detector_stratified_injection_plan.json"
    run_identity = {
        "background_manifest_sha256": file_sha256(manifest_path),
        "background_report_sha256": file_sha256(report_path),
        "injections_per_detector_subset": injections_per_detector_subset,
        "required_detector_subsets": list(required),
        "seed": seed,
    }
    if plan_path.is_file():
        prior = _read_json(plan_path)
        if prior.get("run_identity") != run_identity:
            raise ValueError("Existing detector-stratified plan has another identity")
        if file_sha256(recipe_path) != prior.get("manifest_sha256"):
            raise ValueError("Existing detector-stratified recipes changed")
        return prior
    atomic_write_text(
        recipe_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in recipes),
    )
    result = {
        "status": "frozen_detector_stratified_validation_injection_plan",
        "passed": True,
        "scientific_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "candidate_scores_inspected": False,
        "physical_signal_materialized": False,
        "physical_signal_projection_required": True,
        "split": "val",
        "rows": len(recipes),
        "unique_injection_ids": len(set(injection_ids)),
        "unique_waveform_ids": len(set(waveform_ids)),
        "detector_subset_counts": {
            subset: int(counts[subset]) for subset in required
        },
        "injections_per_detector_subset": injections_per_detector_subset,
        "required_detector_subsets": list(required),
        "component_plans": component_reports,
        "background_manifest_path": str(manifest_path),
        "background_manifest_sha256": file_sha256(manifest_path),
        "background_report_path": str(report_path),
        "background_report_sha256": file_sha256(report_path),
        "manifest_path": str(recipe_path),
        "manifest_sha256": file_sha256(recipe_path),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(plan_path, result)
    return result
