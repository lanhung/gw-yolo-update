from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_publication_validation_ledger.sh"


def test_publication_validation_ledger_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_publication_validation_ledger_binds_exactly_ten_requirements() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    identifiers = (
        "source_safe_corpus",
        "independent_validation_endpoint",
        "five_seed_model",
        "group_safe_data_scaling",
        "continuous_candidate_calibration",
        "paired_raw_mask_vt",
        "calibration_perturbation_robustness",
        "detector_set_ood_transfer",
        "paired_dingo_amplfi_pe_portfolio",
        "locked_corpus_unopened",
    )
    for identifier in identifiers:
        assert source.count(f'--evidence "{identifier}=') == 1
    assert "--require-ready" in source
    assert 'required_total") != 10' in source
    assert 'required_passed") != 10' in source
    assert "locked_final_evidence_complete" in source
    assert "scientific_claim_allowed" in source
    assert "--required-split test" not in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
