from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_human_mask_audit_freeze_is_blinded_and_commit_bound() -> None:
    script = (ROOT / "scripts" / "freeze_human_mask_publication_audit.sh").read_text()
    assert "GWYOLO_CODE_COMMIT" in script
    assert "gravityspy-mask-audit-plan" in script
    assert "mask_targets_exposed_to_annotators" in script
    assert "annotation_task_manifest_sha256" in script
    assert "int(report.get(\"tasks\", 0)) < 90" in script


def test_human_mask_publication_chain_orders_all_evidence_before_binding() -> None:
    script = (ROOT / "scripts" / "run_human_mask_publication_evidence.sh").read_text()
    assert "LEGACY OPTIONAL DIAGNOSTIC ONLY" in script
    commands = (
        "gravityspy-mask-audit-evaluate",
        "gravityspy-mask-consensus-materialize",
        "gravityspy-mask-segmentation-predict",
        "gravityspy-mask-segmentation-evaluate",
        "candidate-search-raw-mask-human-endpoint-bind",
    )
    positions = [script.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "COMPLETED_ANNOTATION_MANIFEST" in script
    assert "MASK_BOOTSTRAP_REPLICATES:-10000" in script
    assert "report.get(\"passed\") is not True" in script


def test_human_mask_publication_queue_waits_for_annotations_and_gpu() -> None:
    script = (ROOT / "scripts" / "run_human_mask_publication_queue.sh").read_text()
    assert "LEGACY OPTIONAL DIAGNOSTIC ONLY" in script
    annotation = script.index('"$COMPLETED_ANNOTATION_MANIFEST"')
    model = script.index('"$MODEL_SELECTION_REPORT"', annotation)
    raw_endpoint = script.index('"$RAW_MASK_ENDPOINT"', model)
    execute = script.index("run_human_mask_publication_evidence.sh")
    assert annotation < model < raw_endpoint < execute
    assert "nvidia-smi --query-compute-apps=pid" in script
    assert 'exec bash "$TASK_CODE_DIR/scripts/run_human_mask_publication_evidence.sh"' in script


def test_human_mask_merge_queue_requires_three_finalized_reviewers() -> None:
    script = (ROOT / "scripts" / "run_human_mask_annotation_merge_queue.sh").read_text()
    assert "LEGACY OPTIONAL DIAGNOSTIC ONLY" in script
    assert "ANNOTATION_MANIFEST_A" in script
    assert "ANNOTATION_MANIFEST_B" in script
    assert "ANNOTATION_MANIFEST_C" in script
    assert "gravityspy-mask-annotation-merge" in script
    assert "--minimum-annotators 3" in script
    assert "completed human annotation manifest is immutable" in script


def test_automatic_mask_publication_chain_has_no_human_or_gpu_dependency() -> None:
    script = (
        ROOT / "scripts" / "run_automatic_mask_publication_evidence.sh"
    ).read_text()
    audit = script.index("automatic-mask-policy-audit")
    binding = script.index(
        "candidate-search-raw-mask-automatic-endpoint-bind"
    )
    assert audit < binding
    assert "OVERLAP_VALIDATION_MANIFEST" in script
    assert "human_annotation_required" in script
    assert "COMPLETED_ANNOTATION_MANIFEST" not in script
    assert "nvidia-smi" not in script

    queue = (
        ROOT / "scripts" / "run_automatic_mask_publication_queue.sh"
    ).read_text()
    assert "OVERLAP_VALIDATION_MANIFEST" in queue
    assert "RAW_MASK_ENDPOINT" in queue
    assert "COMPLETED_ANNOTATION_MANIFEST" not in queue
    assert "nvidia-smi" not in queue
    assert "run_automatic_mask_publication_evidence.sh" in queue
