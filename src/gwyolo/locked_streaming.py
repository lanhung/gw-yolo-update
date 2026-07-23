from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, file_sha256
from .runtime import execution_provenance


SHARD_ARTIFACT_KEYS = {
    "raw_candidate_rows",
    "mask_candidate_rows",
    "ood_source_rows",
    "pe_input_rows",
}


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not allow_empty and not rows:
        raise ValueError(f"locked streaming manifest is empty: {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"locked streaming manifest is invalid: {path}")
    return rows


def _write_immutable_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"existing locked streaming artifact changed: {path}")
        return
    if path.exists():
        raise FileExistsError(f"locked streaming artifact path is not a file: {path}")
    atomic_write_text(path, payload)


def _bound_shard(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    shard_index: int,
    code_commit: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    if (
        shard_index < 0
        or not code_commit.strip()
        or not plan_file.is_file()
        or not access_file.is_file()
    ):
        raise ValueError("locked shard publication inputs are invalid")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    frozen = access.get("frozen_artifacts", {}).get("locked_execution_plan", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    if (
        plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or plan.get("code_commit") != code_commit
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("code_commit") != code_commit
        or frozen.get("path") != str(plan_file)
        or frozen.get("sha256") != file_sha256(plan_file)
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
    ):
        raise ValueError("locked shard publication plan/access binding failed replay")
    shards = _load_jsonl(shard_manifest)
    if shard_index >= len(shards):
        raise ValueError("locked shard publication index is outside the frozen plan")
    shard = shards[shard_index]
    if int(shard.get("shard_index", -1)) != shard_index:
        raise ValueError("locked shard publication order changed")
    return plan_file, access_file, plan, access, shard


def publish_locked_o4b_streaming_shard_artifacts(
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    shard_index: int,
    raw_background_candidates_path: str | Path,
    raw_injection_candidates_path: str | Path,
    mask_background_candidates_path: str | Path,
    mask_injection_candidates_path: str | Path,
    ood_source_manifest_path: str | Path,
    pe_input_manifest_path: str | Path,
    code_commit: str,
) -> dict[str, Any]:
    """Publish one shard's complete score products into its frozen paths.

    The five source manifests may be empty because null detections are valid.
    Candidate rows are never threshold-selected here: every extracted instance
    is retained and tagged as background or injection.  OOD rows must already
    point to aligned numeric detector-set tensors and are likewise copied
    without score-based filtering.
    """

    plan_file, access_file, plan, _, shard = _bound_shard(
        execution_plan_path, access_log_path, shard_index, code_commit
    )
    work_dir = Path(str(shard["work_dir"])).resolve()
    preparation_path = Path(str(shard["manifest_preparation_report_path"])).resolve()
    report_path = Path(str(shard["artifact_publication_report_path"])).resolve()
    if not preparation_path.is_file():
        raise FileNotFoundError("locked shard preparation report is absent")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("status")
        != "prepared_locked_o4b_streaming_shard_manifests"
        or preparation.get("passed") is not True
        or preparation.get("shard_index") != shard_index
        or preparation.get("run_identity", {}).get("execution_plan_sha256")
        != file_sha256(plan_file)
        or preparation.get("run_identity", {}).get("access_log_sha256")
        != file_sha256(access_file)
    ):
        raise ValueError("locked shard preparation failed artifact publication replay")

    sources = {
        "raw_background": Path(raw_background_candidates_path).resolve(),
        "raw_injection": Path(raw_injection_candidates_path).resolve(),
        "mask_background": Path(mask_background_candidates_path).resolve(),
        "mask_injection": Path(mask_injection_candidates_path).resolve(),
        "ood_source": Path(ood_source_manifest_path).resolve(),
        "pe_input": Path(pe_input_manifest_path).resolve(),
    }
    if any(not path.is_file() or work_dir not in path.parents for path in sources.values()):
        raise ValueError("locked shard score products are absent or escaped the work dir")

    background_path = Path(str(shard["background_manifest_path"])).resolve()
    recipe_path = Path(str(shard["injection_recipe_manifest_path"])).resolve()
    background = {
        str(row["window_id"]): row
        for row in _load_jsonl(background_path, allow_empty=True)
    }
    recipes = {
        str(row["injection_id"]): row
        for row in _load_jsonl(recipe_path, allow_empty=True)
    }
    if len(background) != len(_load_jsonl(background_path, allow_empty=True)):
        raise ValueError("locked shard background windows repeat identities")
    if len(recipes) != len(_load_jsonl(recipe_path, allow_empty=True)):
        raise ValueError("locked shard eligible recipes repeat identities")

    def candidates(
        path: Path,
        row_kind: str,
        arm: str,
    ) -> list[dict[str, Any]]:
        rows = _load_jsonl(path, allow_empty=True)
        output = []
        seen = set()
        for index, source in enumerate(rows):
            candidate_id = str(source.get("candidate_id", ""))
            ifo = str(source.get("ifo", ""))
            split = str(source.get("split", ""))
            score = source.get("chirp_score")
            gps_peak = source.get("gps_peak")
            if (
                not candidate_id
                or candidate_id in seen
                or ifo not in {"H1", "L1", "V1"}
                or split != "test"
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or isinstance(gps_peak, bool)
                or not isinstance(gps_peak, (int, float))
                or not math.isfinite(float(gps_peak))
            ):
                raise ValueError(
                    f"locked {arm}/{row_kind} candidate is invalid at row {index}"
                )
            seen.add(candidate_id)
            if row_kind == "background_candidate":
                identity = str(source.get("window_id", ""))
                parent = background.get(identity)
            else:
                identity = str(source.get("injection_id", ""))
                parent = recipes.get(identity)
            if parent is None:
                raise ValueError(
                    f"locked {arm}/{row_kind} candidate has no frozen parent: {identity}"
                )
            availability_id = str(parent["availability_id"])
            if (
                availability_id not in shard["availability_ids"]
                or str(source.get("gps_block")) != str(parent["gps_block"])
                or (
                    row_kind == "injection_candidate"
                    and str(source.get("waveform_id")) != str(parent["waveform_id"])
                )
                or source.get("shard_index", shard_index) != shard_index
                or source.get("availability_id", availability_id) != availability_id
            ):
                raise ValueError(
                    f"locked {arm}/{row_kind} candidate identity differs from its parent"
                )
            output.append(
                {
                    **source,
                    "shard_index": shard_index,
                    "availability_id": availability_id,
                    "locked_row_kind": row_kind,
                    "locked_arm": arm,
                }
            )
        return output

    raw_rows = candidates(
        sources["raw_background"], "background_candidate", "raw"
    ) + candidates(sources["raw_injection"], "injection_candidate", "raw")
    mask_rows = candidates(
        sources["mask_background"], "background_candidate", "mask"
    ) + candidates(sources["mask_injection"], "injection_candidate", "mask")
    for arm, rows in (("raw", raw_rows), ("mask", mask_rows)):
        identities = [str(row["candidate_id"]) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"locked {arm} candidate IDs collide across row kinds")

    ood_rows = []
    seen_glitches = set()
    required_ood = {
        "glitch_id",
        "gps_block",
        "glitch_family",
        "observing_run",
        "is_unknown",
        "available_ifos",
        "split",
        "aligned_network_context",
        "path",
        "sha256",
        "detector_availability",
        "ifo",
        "availability_id",
    }
    for index, source in enumerate(
        _load_jsonl(sources["ood_source"], allow_empty=True)
    ):
        glitch_id = str(source.get("glitch_id", ""))
        availability_id = str(source.get("availability_id", ""))
        tensor_path = Path(str(source.get("path", ""))).resolve()
        if (
            required_ood - set(source)
            or not glitch_id
            or glitch_id in seen_glitches
            or availability_id not in shard["availability_ids"]
            or str(source.get("gps_block")) not in shard["gps_blocks"]
            or source.get("split") != "test"
            or source.get("observing_run") != "O4b"
            or source.get("aligned_network_context") is not True
            or source.get("shard_index", shard_index) != shard_index
            or not tensor_path.is_file()
            or work_dir not in tensor_path.parents
            or source.get("sha256") != file_sha256(tensor_path)
        ):
            raise ValueError(f"locked OOD source is invalid at row {index}")
        seen_glitches.add(glitch_id)
        ood_rows.append(
            {
                **source,
                "shard_index": shard_index,
                "locked_row_kind": "ood_source",
            }
        )

    outcomes = {
        str(row["injection_id"]): row
        for row in _load_jsonl(
            Path(str(shard["availability_outcome_path"])), allow_empty=True
        )
    }
    selected_pe_ids = {
        str(value) for value in shard.get("pe_retention_injection_ids", [])
    }
    eligible_pe_ids = {
        injection_id
        for injection_id in selected_pe_ids
        if outcomes.get(injection_id, {}).get("eligible") is True
    }
    pe_conditions = tuple(plan.get("pe_retention", {}).get("conditions", []))
    required_pe_ifos = [
        str(value)
        for value in plan.get("pe_retention", {}).get("required_ifos", [])
    ]
    if pe_conditions != ("clean", "contaminated", "mask_conditioned"):
        raise ValueError("locked PE retention conditions changed after freezing")
    pe_rows = []
    pe_keys = set()
    for index, source in enumerate(
        _load_jsonl(sources["pe_input"], allow_empty=True)
    ):
        injection_id = str(source.get("injection_id", ""))
        condition = str(source.get("condition", ""))
        key = (injection_id, condition)
        analysis_path = Path(str(source.get("analysis_input_path", ""))).resolve()
        if (
            injection_id not in eligible_pe_ids
            or condition not in pe_conditions
            or key in pe_keys
            or source.get("split") != "test"
            or str(source.get("waveform_id", ""))
            not in {str(value) for value in shard["waveform_ids"]}
            or str(source.get("gps_block", "")) not in shard["gps_blocks"]
            or source.get("input_ifos") != required_pe_ifos
            or not analysis_path.is_file()
            or work_dir not in analysis_path.parents
            or source.get("analysis_input_sha256") != file_sha256(analysis_path)
        ):
            raise ValueError(f"locked PE retained input is invalid at row {index}")
        with np.load(analysis_path, allow_pickle=False) as payload:
            required_arrays = {
                "strain",
                "asd",
                "asd_frequencies",
                "ifos",
                "sample_rate",
                "condition",
                "injection_id",
            }
            if required_arrays - set(payload.files):
                raise ValueError(
                    f"locked PE retained array is incomplete at row {index}"
                )
            strain = np.asarray(payload["strain"])
            if (
                strain.ndim != 2
                or strain.shape[0] != len(required_pe_ifos)
                or strain.shape[1] < 1
                or not np.isfinite(strain).all()
                or list(map(str, payload["ifos"].tolist())) != required_pe_ifos
                or str(payload["condition"].item()) != condition
                or str(payload["injection_id"].item()) != injection_id
            ):
                raise ValueError(
                    f"locked PE retained array failed replay at row {index}"
                )
        pe_keys.add(key)
        pe_rows.append(
            {
                **source,
                "analysis_input_path": str(analysis_path),
                "shard_index": shard_index,
                "availability_id": outcomes[injection_id]["availability_id"],
                "locked_row_kind": "pe_input",
            }
        )
    expected_pe_keys = {
        (injection_id, condition)
        for injection_id in eligible_pe_ids
        for condition in pe_conditions
    }
    if pe_keys != expected_pe_keys:
        raise ValueError(
            "locked PE retention is incomplete; source eviction is forbidden"
        )

    output_paths = {
        key: Path(str(value)).resolve()
        for key, value in shard.get("artifact_paths", {}).items()
    }
    if set(output_paths) != SHARD_ARTIFACT_KEYS or any(
        work_dir not in path.parents for path in output_paths.values()
    ):
        raise ValueError("locked shard artifact paths were not frozen")
    outputs = {
        "raw_candidate_rows": raw_rows,
        "mask_candidate_rows": mask_rows,
        "ood_source_rows": ood_rows,
        "pe_input_rows": pe_rows,
    }
    for label, rows in outputs.items():
        _write_immutable_jsonl(output_paths[label], rows)

    run_identity = {
        "execution_plan_sha256": file_sha256(plan_file),
        "access_log_sha256": file_sha256(access_file),
        "manifest_preparation_report_sha256": file_sha256(preparation_path),
        "shard_index": shard_index,
        "source_artifacts": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in sorted(sources.items())
        },
        "code_commit": code_commit,
    }
    result = {
        "status": "published_locked_o4b_streaming_shard_artifacts",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_rows_filtered_by_score": False,
        "all_candidate_instances_retained": True,
        "negative_and_null_results_retained": True,
        "shard_index": shard_index,
        "run_identity": run_identity,
        "artifacts": {
            label: {
                "path": str(output_paths[label]),
                "sha256": file_sha256(output_paths[label]),
                "rows": len(rows),
            }
            for label, rows in sorted(outputs.items())
        },
        "row_counts": {
            "raw_background_candidates": sum(
                row["locked_row_kind"] == "background_candidate"
                for row in raw_rows
            ),
            "raw_injection_candidates": sum(
                row["locked_row_kind"] == "injection_candidate"
                for row in raw_rows
            ),
            "mask_background_candidates": sum(
                row["locked_row_kind"] == "background_candidate"
                for row in mask_rows
            ),
            "mask_injection_candidates": sum(
                row["locked_row_kind"] == "injection_candidate"
                for row in mask_rows
            ),
            "ood_sources": len(ood_rows),
            "pe_input_rows": len(pe_rows),
            "pe_retained_injections": len(eligible_pe_ids),
            "pe_selected_but_dq_unavailable": len(
                selected_pe_ids - eligible_pe_ids
            ),
        },
        "code_commit": plan["code_commit"],
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = plan["code_commit"]
    if report_path.is_file():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != result["status"]
            or completed.get("run_identity") != run_identity
            or completed.get("artifacts") != result["artifacts"]
        ):
            raise ValueError("existing locked shard artifact publication changed")
        return completed
    atomic_write_json(report_path, result)
    return result


def merge_locked_o4b_streaming_suite_input_sources(
    suite_plan_path: str | Path,
    execution_plan_path: str | Path,
    access_log_path: str | Path,
    streaming_completion_audit_path: str | Path,
    post_dq_weight_report_path: str | Path,
    code_commit: str,
) -> dict[str, Any]:
    """Merge every frozen shard into score-unselected locked suite sources."""

    suite_file = Path(suite_plan_path).resolve()
    plan_file = Path(execution_plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    completion_file = Path(streaming_completion_audit_path).resolve()
    weight_report_file = Path(post_dq_weight_report_path).resolve()
    for path in (
        suite_file,
        plan_file,
        access_file,
        completion_file,
        weight_report_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"locked suite source merge input is absent: {path}")
    suite = json.loads(suite_file.read_text(encoding="utf-8"))
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    completion = json.loads(completion_file.read_text(encoding="utf-8"))
    weights = json.loads(weight_report_file.read_text(encoding="utf-8"))
    frozen = access.get("frozen_artifacts", {})
    shard_manifest = Path(str(plan.get("shard_manifest_path", ""))).resolve()
    report_path = Path(str(plan.get("suite_input_merge_report_path", ""))).resolve()
    work_root = Path(str(plan.get("freeze_identity", {}).get("work_root", ""))).resolve()
    if (
        suite.get("status") != "frozen_locked_evaluation_suite_plan"
        or suite.get("passed") is not True
        or suite.get("code_commit") != code_commit
        or plan.get("status") != "frozen_locked_o4b_streaming_execution_plan"
        or plan.get("passed") is not True
        or plan.get("code_commit") != code_commit
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("code_commit") != code_commit
        or frozen.get("locked_suite_plan", {}).get("path") != str(suite_file)
        or frozen.get("locked_suite_plan", {}).get("sha256") != file_sha256(suite_file)
        or frozen.get("locked_execution_plan", {}).get("path") != str(plan_file)
        or frozen.get("locked_execution_plan", {}).get("sha256")
        != file_sha256(plan_file)
        or completion.get("status")
        != "completed_locked_o4b_streaming_execution_audit"
        or completion.get("passed") is not True
        or completion.get("completed_shards") != plan.get("shards")
        or completion.get("execution_plan", {}).get("sha256")
        != file_sha256(plan_file)
        or weights.get("status") != "reduced_locked_o4b_post_dq_injection_weights"
        or weights.get("passed") is not True
        or weights.get("raw_mask_shared_physical_denominator") is not True
        or weights.get("streaming_completion_audit", {}).get("sha256")
        != file_sha256(completion_file)
        or report_path.parent != work_root
        or not shard_manifest.is_file()
        or plan.get("shard_manifest_sha256") != file_sha256(shard_manifest)
    ):
        raise ValueError("locked suite source merge binding failed replay")

    weight_manifest = Path(str(weights.get("weight_manifest_path", ""))).resolve()
    if (
        weight_manifest
        != Path(str(plan.get("post_dq_weight_manifest_path", ""))).resolve()
        or not weight_manifest.is_file()
        or weights.get("weight_manifest_sha256") != file_sha256(weight_manifest)
    ):
        raise ValueError("locked suite source merge weight manifest failed replay")
    weight_rows = _load_jsonl(weight_manifest)
    weight_by_id = {str(row["injection_id"]): row for row in weight_rows}
    if len(weight_by_id) != len(weight_rows):
        raise ValueError("locked suite source merge weights repeat injections")

    shards = _load_jsonl(shard_manifest)
    background_rows = []
    raw_rows = []
    mask_rows = []
    ood_rows = []
    pe_rows = []
    publication_reports = []
    for expected_index, shard in enumerate(shards):
        publication_path = Path(
            str(shard["artifact_publication_report_path"])
        ).resolve()
        if not publication_path.is_file():
            raise FileNotFoundError(
                f"locked shard publication report is absent: {expected_index}"
            )
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        if (
            publication.get("status")
            != "published_locked_o4b_streaming_shard_artifacts"
            or publication.get("passed") is not True
            or publication.get("shard_index") != expected_index
            or publication.get("candidate_rows_filtered_by_score") is not False
            or publication.get("all_candidate_instances_retained") is not True
            or publication.get("run_identity", {}).get("execution_plan_sha256")
            != file_sha256(plan_file)
            or publication.get("run_identity", {}).get("access_log_sha256")
            != file_sha256(access_file)
        ):
            raise ValueError(
                f"locked shard publication failed suite merge: {expected_index}"
            )
        publication_reports.append(
            {"path": str(publication_path), "sha256": file_sha256(publication_path)}
        )
        background_rows.extend(
            _load_jsonl(
                Path(str(shard["background_manifest_path"])), allow_empty=True
            )
        )
        for label, target in (
            ("raw_candidate_rows", raw_rows),
            ("mask_candidate_rows", mask_rows),
            ("ood_source_rows", ood_rows),
            ("pe_input_rows", pe_rows),
        ):
            path = Path(str(shard["artifact_paths"][label])).resolve()
            identity = publication["artifacts"][label]
            rows = _load_jsonl(path, allow_empty=True)
            if (
                identity.get("path") != str(path)
                or identity.get("sha256") != file_sha256(path)
                or identity.get("rows") != len(rows)
            ):
                raise ValueError(
                    f"locked shard artifact failed suite merge: {expected_index}/{label}"
                )
            target.extend(rows)

    background_ids = [str(row["window_id"]) for row in background_rows]
    if len(background_ids) != len(set(background_ids)):
        raise ValueError("locked suite background windows repeat across shards")
    for arm, rows in (("raw", raw_rows), ("mask", mask_rows)):
        candidate_ids = [str(row["candidate_id"]) for row in rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"locked suite {arm} candidates repeat across shards")
        for row in rows:
            if row["locked_row_kind"] != "injection_candidate":
                continue
            weight = weight_by_id.get(str(row["injection_id"]))
            if weight is None or weight.get("eligible") is not True:
                raise ValueError(
                    f"locked suite {arm} candidate has no eligible post-DQ weight"
                )
            row.update(
                {
                    "vt_weight": weight["vt_weight"],
                    "vt_weight_unit": weight["vt_weight_unit"],
                    "vt_measure": weight["vt_measure"],
                    "post_dq_weight_manifest_sha256": file_sha256(weight_manifest),
                }
            )
    glitch_ids = [str(row["glitch_id"]) for row in ood_rows]
    if len(glitch_ids) != len(set(glitch_ids)):
        raise ValueError("locked suite OOD sources repeat across shards")
    pe_keys = [
        (str(row["injection_id"]), str(row["condition"])) for row in pe_rows
    ]
    if len(pe_keys) != len(set(pe_keys)):
        raise ValueError("locked suite PE inputs repeat across shards")
    retained_pe_ids = {injection_id for injection_id, _ in pe_keys}
    expected_pe_conditions = set(plan["pe_retention"]["conditions"])
    if any(
        {
            condition
            for selected_id, condition in pe_keys
            if selected_id == injection_id
        }
        != expected_pe_conditions
        for injection_id in retained_pe_ids
    ):
        raise ValueError("locked suite PE retained triplets are incomplete")
    selected_pe_ids = set(plan["pe_retention"]["selected_injection_ids"])
    eligible_ids = {
        str(row["injection_id"])
        for row in weight_rows
        if row.get("eligible") is True
    }
    if retained_pe_ids != selected_pe_ids & eligible_ids:
        raise ValueError("locked suite PE retained pool differs from frozen DQ intersection")

    def split_candidates(
        rows: list[dict[str, Any]], kind: str
    ) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("locked_row_kind") == kind]

    raw_background_candidates = split_candidates(raw_rows, "background_candidate")
    raw_injection_candidates = split_candidates(raw_rows, "injection_candidate")
    mask_background_candidates = split_candidates(mask_rows, "background_candidate")
    mask_injection_candidates = split_candidates(mask_rows, "injection_candidate")
    if len(raw_background_candidates) + len(raw_injection_candidates) != len(raw_rows):
        raise ValueError("locked suite raw candidate kinds are incomplete")
    if len(mask_background_candidates) + len(mask_injection_candidates) != len(mask_rows):
        raise ValueError("locked suite mask candidate kinds are incomplete")

    output_paths = {
        "raw_background_candidates": Path(
            str(plan["merged_raw_background_candidates_path"])
        ).resolve(),
        "raw_injection_candidates": Path(
            str(plan["merged_raw_injection_candidates_path"])
        ).resolve(),
        "mask_background_candidates": Path(
            str(plan["merged_mask_background_candidates_path"])
        ).resolve(),
        "mask_injection_candidates": Path(
            str(plan["merged_mask_injection_candidates_path"])
        ).resolve(),
        "null_outcomes": Path(str(plan["injection_null_outcomes_path"])).resolve(),
        "raw_background_manifest": Path(
            str(suite["inputs"]["raw_test_background_manifest"])
        ).resolve(),
        "mask_background_manifest": Path(
            str(suite["inputs"]["mask_test_background_manifest"])
        ).resolve(),
        "ood_source_manifest": Path(
            str(suite["inputs"]["locked_ood_source_manifest"])
        ).resolve(),
        "pe_input_manifest": Path(
            str(plan["merged_pe_input_manifest_path"])
        ).resolve(),
    }
    if any(
        path.parent != work_root
        for label, path in output_paths.items()
        if label
        in {
            "raw_background_candidates",
            "raw_injection_candidates",
            "mask_background_candidates",
            "mask_injection_candidates",
            "null_outcomes",
            "pe_input_manifest",
        }
    ):
        raise ValueError("locked suite staging output escaped the frozen work root")
    payloads = {
        "raw_background_candidates": raw_background_candidates,
        "raw_injection_candidates": raw_injection_candidates,
        "mask_background_candidates": mask_background_candidates,
        "mask_injection_candidates": mask_injection_candidates,
        "null_outcomes": [
            row for row in weight_rows if row.get("eligible") is not True
        ],
        "raw_background_manifest": background_rows,
        "mask_background_manifest": background_rows,
        "ood_source_manifest": ood_rows,
        "pe_input_manifest": pe_rows,
    }
    for label, rows in payloads.items():
        _write_immutable_jsonl(output_paths[label], rows)

    endpoints = suite["endpoints"]
    result = {
        "status": "merged_locked_o4b_streaming_suite_input_sources",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_rows_filtered_by_score": False,
        "all_candidate_instances_retained": True,
        "negative_and_null_results_retained": True,
        "post_access_dq_replacement_used": False,
        "raw_mask_shared_physical_denominator": True,
        "completed_shards": len(shards),
        "background_windows": len(background_rows),
        "background_gps_blocks": len(
            {str(row["gps_block"]) for row in background_rows}
        ),
        "planned_injections": len(weight_rows),
        "eligible_injections": sum(
            row.get("eligible") is True for row in weight_rows
        ),
        "unavailable_injections": sum(
            row.get("eligible") is not True for row in weight_rows
        ),
        "raw_background_candidates": len(raw_background_candidates),
        "raw_injection_candidates": len(raw_injection_candidates),
        "mask_background_candidates": len(mask_background_candidates),
        "mask_injection_candidates": len(mask_injection_candidates),
        "ood_sources": len(ood_rows),
        "pe_retained_injections": len(retained_pe_ids),
        "pe_input_rows": len(pe_rows),
        "endpoint_source_readiness": {
            "minimum_background_gps_blocks": (
                len({str(row["gps_block"]) for row in background_rows})
                >= int(endpoints["minimum_background_gps_blocks"])
            ),
            "minimum_test_injections": (
                sum(row.get("eligible") is True for row in weight_rows)
                >= int(endpoints["minimum_test_injections"])
            ),
            "minimum_locked_ood_rows": (
                len(ood_rows) >= int(endpoints["minimum_locked_ood_rows"])
            ),
            "minimum_paired_pe_injections": (
                len(retained_pe_ids)
                >= int(endpoints["minimum_paired_pe_injections"])
            ),
        },
        "artifacts": {
            label: {
                "path": str(output_paths[label]),
                "sha256": file_sha256(output_paths[label]),
                "rows": len(payloads[label]),
            }
            for label in sorted(output_paths)
        },
        "streaming_completion_audit": {
            "path": str(completion_file),
            "sha256": file_sha256(completion_file),
        },
        "post_dq_weight_report": {
            "path": str(weight_report_file),
            "sha256": file_sha256(weight_report_file),
        },
        "publication_reports": publication_reports,
        "code_commit": code_commit,
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = code_commit
    if report_path.is_file():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != result["status"]
            or completed.get("artifacts") != result["artifacts"]
            or completed.get("streaming_completion_audit")
            != result["streaming_completion_audit"]
            or completed.get("post_dq_weight_report")
            != result["post_dq_weight_report"]
        ):
            raise ValueError("existing locked suite source merge changed")
        return completed
    atomic_write_json(report_path, result)
    return result
