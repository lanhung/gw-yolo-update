from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_locked_evaluation_suite_reduction.sh"
)


def test_locked_suite_reduction_requires_all_shards_before_every_endpoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    required_order = (
        "locked-o4b-streaming-completion-audit",
        "locked-o4b-post-dq-injection-weights",
        "locked-o4b-streaming-suite-inputs-merge",
        "run_locked_search_endpoints.sh",
        "run_locked_ood_endpoint.sh",
        "run_locked_pe_endpoints.sh",
        "run_locked_catalog_endpoint.sh",
        "locked-evaluation-suite-finalize",
    )
    positions = [source.index(token) for token in required_order]
    assert positions == sorted(positions)
    for token in (
        "LOCKED_SHARD_RECEIPT_MANIFEST",
        "all_predeclared_shards_reduced",
        "completed_shards",
        "expected_shards",
        "negative_and_null_results_retained",
        "raw_mask_shared_physical_denominator",
        "--streaming-completion-audit",
        "GWYOLO_CODE_COMMIT",
    ):
        assert token in source
    assert "evaluation-corpus-open-once" not in source
