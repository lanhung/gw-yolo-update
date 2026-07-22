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
