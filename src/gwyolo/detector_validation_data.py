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
from .gwosc import _stratified_records


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


def freeze_source_disjoint_detector_acquisition_plan(
    inventory_plan_path: str | Path,
    frozen_network_manifests: Iterable[str | Path],
    output_path: str | Path,
    target_pairs: int,
    seed: int = 20260723,
    exclusion_plan_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Select unlocked GWOSC pairs disjoint from every frozen source component."""

    inventory_path = Path(inventory_plan_path).resolve()
    inventory = _read_json(inventory_path)
    frozen_paths = [Path(path).resolve() for path in frozen_network_manifests]
    exclusion_paths = [Path(path).resolve() for path in exclusion_plan_paths]
    target = Path(output_path).resolve()
    pairs = list(inventory.get("pairs", []))
    detectors = tuple(str(value) for value in inventory.get("detectors", []))
    if (
        inventory.get("status") != "development_acquisition_plan"
        or inventory.get("locked_evaluation_data") is not False
        or str(inventory.get("run", "")).lower().startswith("o4b")
        or len(detectors) != 2
        or not set(detectors) <= {"H1", "L1", "V1"}
        or not pairs
        or target_pairs < 1
        or seed < 1
        or not frozen_paths
    ):
        raise ValueError("Detector acquisition inventory or policy is invalid")
    pair_ids = [str(row.get("pair_id", "")) for row in pairs]
    if (
        any(not value for value in pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or int(inventory.get("selected_pairs", -1)) != len(pairs)
    ):
        raise ValueError("Detector acquisition inventory has invalid pair identities")

    excluded_urls: set[str] = set()
    excluded_gps_starts: set[int] = set()
    frozen_records = []
    for path in frozen_paths:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("Frozen network exclusion manifest is empty")
        for row in rows:
            sources = row.get("network_strain_sources")
            if not isinstance(sources, dict) or len(sources) < 2:
                raise ValueError("Frozen network row lacks source components")
            for source in sources.values():
                excluded_urls.add(str(source["hdf5_url"]))
                excluded_gps_starts.add(int(source["gps_start"]))
        frozen_records.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
        )
    plan_records = []
    for path in exclusion_paths:
        plan = _read_json(path)
        if (
            plan.get("status") != "development_acquisition_plan"
            or plan.get("locked_evaluation_data") is not False
            or str(plan.get("run")) != str(inventory["run"])
            or not isinstance(plan.get("pairs"), list)
        ):
            raise ValueError("Detector acquisition exclusion plan is invalid")
        for pair in plan["pairs"]:
            excluded_gps_starts.add(int(pair["gps_start"]))
            excluded_urls.update(
                str(source["hdf5_url"])
                for source in pair["detectors"].values()
            )
        plan_records.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "pairs": len(plan["pairs"]),
            }
        )
    eligible = [
        pair
        for pair in pairs
        if int(pair["gps_start"]) not in excluded_gps_starts
        and not (
            {
                str(source["hdf5_url"])
                for source in pair["detectors"].values()
            }
            & excluded_urls
        )
    ]
    if len(eligible) < target_pairs:
        raise ValueError(
            "Source-disjoint detector acquisition inventory is below target"
        )
    selected = _stratified_records(eligible, target_pairs, seed)
    selected_gps = [int(row["gps_start"]) for row in selected]
    selected_urls = {
        str(source["hdf5_url"])
        for pair in selected
        for source in pair["detectors"].values()
    }
    if (
        len(selected_gps) != len(set(selected_gps))
        or set(selected_gps) & excluded_gps_starts
        or selected_urls & excluded_urls
    ):
        raise RuntimeError("Detector acquisition selection is not source disjoint")
    result = {
        **{key: value for key, value in inventory.items() if key != "pairs"},
        "status": "development_acquisition_plan",
        "selected_pairs": len(selected),
        "selected_gps_span": [min(selected_gps), max(selected_gps)],
        "pairs": selected,
        "seed": seed,
        "selection_rule": "source_file_and_gps_disjoint_stratified_v1",
        "selection_data": "GWOSC strain-file metadata only",
        "candidate_scores_inspected": False,
        "test_data_opened": False,
        "locked_evaluation_data": False,
        "inventory_plan_path": str(inventory_path),
        "inventory_plan_sha256": file_sha256(inventory_path),
        "frozen_network_exclusions": frozen_records,
        "acquisition_plan_exclusions": plan_records,
        "excluded_source_urls": len(excluded_urls),
        "excluded_gps_starts": len(excluded_gps_starts),
        "eligible_pairs_after_exclusion": len(eligible),
        "selected_pair_ids_hash": canonical_hash(
            [str(row["pair_id"]) for row in selected], 64
        ),
        "selected_gps_starts_hash": canonical_hash(selected_gps, 64),
        **execution_provenance(),
    }
    if target.exists():
        prior = _read_json(target)
        identity_fields = (
            "inventory_plan_sha256",
            "frozen_network_exclusions",
            "acquisition_plan_exclusions",
            "selected_pair_ids_hash",
            "seed",
        )
        if any(prior.get(field) != result.get(field) for field in identity_fields):
            raise ValueError("Existing detector acquisition plan has another identity")
        return prior
    atomic_write_json(target, result)
    return result


def seal_streamed_detector_validation_shard(
    parent_plan_path: str | Path,
    shard_plan_path: str | Path,
    batch_report_path: str | Path,
    background_report_path: str | Path,
    background_bank_report_path: str | Path,
    eviction_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one score-blind, hash-thresholded validation acquisition shard."""

    parent_path = Path(parent_plan_path).resolve()
    shard_path = Path(shard_plan_path).resolve()
    batch_path = Path(batch_report_path).resolve()
    background_path = Path(background_report_path).resolve()
    bank_path = Path(background_bank_report_path).resolve()
    eviction_path = Path(eviction_report_path).resolve()
    target = Path(output_path).resolve()
    parent = _read_json(parent_path)
    shard = _read_json(shard_path)
    batch = _read_json(batch_path)
    background = _read_json(background_path)
    bank = _read_json(bank_path)
    eviction = _read_json(eviction_path)
    bank_manifest = Path(str(bank.get("manifest_path", ""))).resolve()
    background_manifest = Path(
        str(background.get("manifest_path", ""))
    ).resolve()
    detectors = tuple(str(value) for value in parent.get("detectors", []))
    if (
        parent.get("status") != "development_acquisition_plan"
        or parent.get("selection_rule")
        != "source_file_and_gps_disjoint_stratified_v1"
        or parent.get("locked_evaluation_data") is not False
        or parent.get("candidate_scores_inspected") is not False
        or parent.get("test_data_opened") is not False
        or len(detectors) != 2
        or str(parent.get("run", "")).lower().startswith("o4b")
    ):
        raise ValueError("Detector-validation parent plan is not score-blind development data")
    if (
        shard.get("status") != "development_acquisition_plan"
        or shard.get("locked_evaluation_data") is not False
        or shard.get("parent_plan_sha256") != file_sha256(parent_path)
        or tuple(str(value) for value in shard.get("detectors", [])) != detectors
        or int(shard.get("selected_pairs", 0)) < 1
    ):
        raise ValueError("Detector-validation shard plan breaks its parent identity")
    if (
        batch.get("status") != "verified_development_strain_batch"
        or batch.get("passed") is not True
        or batch.get("plan_sha256") != file_sha256(shard_path)
        or int(batch.get("selected_pairs", -1))
        != int(shard["selected_pairs"])
        or int(batch.get("verified_files", -1))
        != int(shard["selected_pairs"]) * len(detectors)
    ):
        raise ValueError("Detector-validation batch download is incomplete")
    if (
        background.get("status")
        != "verified_multi_segment_development_background"
        or background.get("passed") is not True
        or tuple(str(value) for value in background.get("ifos", []))
        != tuple(sorted(detectors))
        or background.get("split_strategy") != "hash_threshold_v1"
        or int(background.get("splits", {}).get("test", {}).get("windows", -1))
        != 0
        or background.get("source_batch_report_sha256s")
        != [file_sha256(batch_path)]
        or not background_manifest.is_file()
        or background.get("manifest_sha256")
        != file_sha256(background_manifest)
    ):
        raise ValueError("Detector-validation background planning is not stable and test-free")
    if (
        bank.get("status") != "verified_numeric_background_bank"
        or bank.get("selected_split") != "val"
        or int(bank.get("selected_windows", 0)) < 1
        or int(bank.get("unique_gps_blocks", -1))
        != int(bank.get("selected_windows", -2))
        or bank.get("background_manifest_sha256")
        != background["manifest_sha256"]
        or bank.get("manifest_sha256") != file_sha256(bank_manifest)
    ):
        raise ValueError("Detector-validation numeric bank is incomplete or not one row per block")
    if (
        eviction.get("status") != "verified_background_source_eviction"
        or eviction.get("recoverable") is not True
        or eviction.get("background_bank_report_sha256")
        != file_sha256(bank_path)
        or eviction.get("background_bank_manifest_sha256")
        != bank["manifest_sha256"]
        or int(eviction.get("removed_files", -1))
        != int(batch["verified_files"])
    ):
        raise ValueError("Detector-validation source eviction failed replay")

    rows = [
        json.loads(line)
        for line in bank_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blocks = [str(row.get("gps_block", "")) for row in rows]
    pair_ids = {str(row.get("pair_id", "")) for row in rows}
    expected_pair_ids = {str(row["pair_id"]) for row in shard["pairs"]}
    bank_hashes: dict[str, str] = {}
    for row in rows:
        artifact = Path(str(row.get("background_bank", {}).get("path", "")))
        expected = str(row.get("background_bank", {}).get("sha256", ""))
        actual = bank_hashes.setdefault(str(artifact.resolve()), file_sha256(artifact))
        if (
            row.get("split") != "val"
            or tuple(str(value) for value in row.get("ifos", []))
            != tuple(sorted(detectors))
            or actual != expected
        ):
            raise ValueError("Detector-validation numeric bank row failed replay")
    if (
        not rows
        or any(not value for value in blocks)
        or len(blocks) != len(set(blocks))
        or not pair_ids
        or not pair_ids <= expected_pair_ids
    ):
        raise ValueError("Detector-validation shard repeats blocks or has foreign pairs")

    result = {
        "status": "verified_streamed_detector_validation_shard",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_scores_inspected": False,
        "test_rows_read": 0,
        "test_strain_rows_read": 0,
        "detector_subset": "+".join(sorted(detectors)),
        "observing_run": str(parent["run"]),
        "selected_pairs": int(shard["selected_pairs"]),
        "validation_windows": len(rows),
        "unique_validation_gps_blocks": len(set(blocks)),
        "parent_plan": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
        },
        "shard_plan": {
            "path": str(shard_path),
            "sha256": file_sha256(shard_path),
        },
        "batch_report": {
            "path": str(batch_path),
            "sha256": file_sha256(batch_path),
        },
        "background_report": {
            "path": str(background_path),
            "sha256": file_sha256(background_path),
        },
        "background_bank_report": {
            "path": str(bank_path),
            "sha256": file_sha256(bank_path),
        },
        "source_eviction_report": {
            "path": str(eviction_path),
            "sha256": file_sha256(eviction_path),
        },
        "background_bank_manifest": {
            "path": str(bank_manifest),
            "sha256": file_sha256(bank_manifest),
        },
        "background_bank_artifacts_hash": canonical_hash(
            dict(sorted(bank_hashes.items())), 64
        ),
        **execution_provenance(),
    }
    if target.is_file():
        prior = _read_json(target)
        identity_fields = (
            "parent_plan",
            "shard_plan",
            "batch_report",
            "background_report",
            "background_bank_report",
            "source_eviction_report",
            "background_bank_manifest",
            "background_bank_artifacts_hash",
        )
        if any(prior.get(field) != result.get(field) for field in identity_fields):
            raise ValueError("Existing detector-validation shard receipt changed")
        return prior
    atomic_write_json(target, result)
    return result


def merge_streamed_detector_validation_backgrounds(
    base_manifest_path: str | Path,
    base_report_path: str | Path,
    shard_receipt_paths: Iterable[str | Path],
    output_dir: str | Path,
    required_detector_subsets: Iterable[str] = DEFAULT_DETECTOR_SUBSETS,
    minimum_per_detector_subset: int = 25,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Merge frozen numeric validation backgrounds without reopening source HDF."""

    base_manifest = Path(base_manifest_path).resolve()
    base_report_file = Path(base_report_path).resolve()
    base_report = _read_json(base_report_file)
    receipt_paths = [Path(path).resolve() for path in shard_receipt_paths]
    required = tuple(str(value) for value in required_detector_subsets)
    if (
        base_report.get("status")
        != "exported_source_safe_detector_validation_background_bank"
        or base_report.get("manifest_sha256") != file_sha256(base_manifest)
        or base_report.get("candidate_scores_inspected") is not False
        or int(base_report.get("test_rows_read", -1)) != 0
        or not receipt_paths
        or minimum_per_detector_subset < 1
        or not required
        or len(required) != len(set(required))
    ):
        raise ValueError("Detector-validation merge inputs or policy are invalid")
    rows = [
        json.loads(line)
        for line in base_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Base detector-validation manifest is empty")
    sources = [
        {
            "kind": "source_safe_network_validation_bank",
            "path": str(base_manifest),
            "sha256": file_sha256(base_manifest),
            "report_path": str(base_report_file),
            "report_sha256": file_sha256(base_report_file),
            "rows": len(rows),
        }
    ]
    receipt_records = []
    seen_receipts: set[str] = set()
    for receipt_path in receipt_paths:
        receipt_hash = file_sha256(receipt_path)
        if receipt_hash in seen_receipts:
            raise ValueError("Detector-validation merge repeats a shard receipt")
        seen_receipts.add(receipt_hash)
        receipt = _read_json(receipt_path)
        if (
            receipt.get("status")
            != "verified_streamed_detector_validation_shard"
            or receipt.get("passed") is not True
            or receipt.get("candidate_scores_inspected") is not False
            or int(receipt.get("test_rows_read", -1)) != 0
            or int(receipt.get("test_strain_rows_read", -1)) != 0
            or receipt.get("detector_subset") not in required
        ):
            raise ValueError("Detector-validation shard receipt is not publication-safe")
        manifest_record = receipt.get("background_bank_manifest", {})
        shard_manifest = Path(str(manifest_record.get("path", ""))).resolve()
        if manifest_record.get("sha256") != file_sha256(shard_manifest):
            raise ValueError("Detector-validation shard manifest changed after sealing")
        shard_rows = [
            json.loads(line)
            for line in shard_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(shard_rows) != int(receipt["validation_windows"]):
            raise ValueError("Detector-validation shard receipt row count changed")
        subset = str(receipt["detector_subset"])
        for row in shard_rows:
            artifact = Path(str(row.get("background_bank", {}).get("path", "")))
            if (
                row.get("split") != "val"
                or "+".join(str(value) for value in row.get("ifos", [])) != subset
                or file_sha256(artifact)
                != str(row.get("background_bank", {}).get("sha256", ""))
            ):
                raise ValueError("Detector-validation shard artifact failed merge replay")
            rows.append(
                {
                    **row,
                    "detector_subset": subset,
                    "aligned_network_context": True,
                    "candidate_scores_inspected": False,
                    "physical_signal_present": False,
                    "physical_signal_projection_required": True,
                    "source_kind": "streamed_source_disjoint_gwosc_background",
                }
            )
        receipt_records.append(
            {
                "path": str(receipt_path),
                "sha256": receipt_hash,
                "detector_subset": subset,
                "rows": len(shard_rows),
            }
        )

    window_ids = [str(row.get("window_id", "")) for row in rows]
    gps_blocks = [str(row.get("gps_block", "")) for row in rows]
    if (
        any(not value for value in window_ids + gps_blocks)
        or len(window_ids) != len(set(window_ids))
        or len(gps_blocks) != len(set(gps_blocks))
    ):
        raise ValueError("Merged detector-validation backgrounds repeat physical groups")
    rows.sort(
        key=lambda row: (
            str(row["detector_subset"]),
            float(row["gps_start"]),
            str(row["window_id"]),
        )
    )
    counts = Counter(str(row["detector_subset"]) for row in rows)
    observed = {subset: int(counts.get(subset, 0)) for subset in required}
    deficits = {
        subset: max(0, minimum_per_detector_subset - count)
        for subset, count in observed.items()
    }
    ready = all(value == 0 for value in deficits.values())
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "background_windows.jsonl"
    report_path = output / "detector_validation_background_report.json"
    run_identity = {
        "base_manifest_sha256": file_sha256(base_manifest),
        "base_report_sha256": file_sha256(base_report_file),
        "shard_receipt_sha256s": [file_sha256(path) for path in receipt_paths],
        "required_detector_subsets": list(required),
        "minimum_per_detector_subset": minimum_per_detector_subset,
        "merge_policy": "unique_gps_block_source_disjoint_stream_v1",
    }
    if report_path.is_file():
        prior = _read_json(report_path)
        if prior.get("run_identity") != run_identity:
            raise ValueError("Existing merged detector-validation bank has another identity")
        if prior.get("manifest_sha256") != file_sha256(manifest):
            raise ValueError("Existing merged detector-validation manifest changed")
        if require_ready and prior.get("passed") is not True:
            raise RuntimeError(
                "Merged detector-validation background remains below floors: "
                f"{prior.get('detector_subset_deficits', {})}"
            )
        return prior
    atomic_write_text(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    live_seconds = _union_duration(
        (float(row["gps_start"]), float(row["gps_end"])) for row in rows
    )
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
        "selected_rows": len(rows),
        "selection": "unique_gps_block_source_disjoint_stream_v1",
        "unique_network_gps_blocks": len(gps_blocks),
        "detector_subset_counts": observed,
        "detector_subset_deficits": deficits,
        "minimum_per_detector_subset": minimum_per_detector_subset,
        "required_detector_subsets": list(required),
        "sources": sources,
        "streamed_shard_receipts": receipt_records,
        "splits": {
            "val": {
                "windows": len(rows),
                "unique_gps_blocks": len(gps_blocks),
                "live_time_seconds": live_seconds,
                "live_time_years": live_seconds / SECONDS_PER_YEAR,
            },
            "test": {
                "windows": 0,
                "unique_gps_blocks": 0,
                "live_time_seconds": 0.0,
                "live_time_years": 0.0,
            },
        },
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    if require_ready and not ready:
        raise RuntimeError(
            f"Merged detector-validation background remains below floors: {deficits}"
        )
    return result
