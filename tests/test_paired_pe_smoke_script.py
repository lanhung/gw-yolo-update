from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_paired_pe_smoke_fails_before_work_when_paths_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "GWYOLO_PYTHON" in completed.stderr


def test_paired_pe_smoke_binds_the_selected_model_configuration() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    assert "GWYOLO_MODEL_CONFIG" in source
    assert '"config_file_sha256"' in source
    assert '--config "$GWYOLO_MODEL_CONFIG"' in source
    assert "--config configs/physical_overlap_finetune.yaml" not in source


def test_paired_pe_smoke_records_independent_source_receipts() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    for variable in (
        "GWYOLO_MODEL_SELECTION_OVERLAP_MANIFEST",
        "GWYOLO_MODEL_SELECTION_VALIDATION_MANIFEST",
        "GWYOLO_INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "GWYOLO_INDEPENDENT_PE_OVERLAP_REPORT",
        "GWYOLO_INDEPENDENT_OVERLAP_AUDIT",
    ):
        assert variable in source
    assert '"source_receipts": source_receipts' in source
    assert '"evaluation_tier": evaluation_tier' in source
    assert "minimum_publication_validation_injections" in source
