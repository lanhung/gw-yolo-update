from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_promoted_paired_pe_smoke_fails_closed_when_inputs_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_promoted_paired_pe_smoke_binds_every_selected_artifact() -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    for field in (
        "selected_checkpoint_sha256",
        "finetune_reports",
        "config_file_sha256",
        "overlap_validation_manifest_sha256",
        "clean_validation_manifest_sha256",
        "MODEL_SELECTION_OVERLAP_MANIFEST",
        "MODEL_SELECTION_VALIDATION_MANIFEST",
        "INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "INDEPENDENT_PE_OVERLAP_REPORT",
        "INDEPENDENT_OVERLAP_AUDIT",
        "verified_independent_validation_pe_overlap",
        "passed_physical_overlap_group_audit",
        "test_data_opened",
    ):
        assert field in source
    assert 'export GWYOLO_MODEL_CONFIG="${selection[1]}"' in source
    assert "promoted paired PE model selection failed" in source
    assert "readarray -t selection < <(" not in source


def test_promoted_paired_pe_smoke_embedded_python_compiles() -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    import re

    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{script.name}:heredoc-{index}", "exec")


def test_promoted_paired_pe_smoke_propagates_selector_failure(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    code = tmp_path / "code"
    (code / "src" / "gwyolo").mkdir(parents=True)
    inputs = {}
    for name in (
        "five_seed_summary",
        "uniform_config",
        "family_balanced_config",
        "model_selection_overlap_manifest",
        "model_selection_validation_manifest",
        "independent_validation_endpoint_report",
        "independent_pe_overlap_report",
        "independent_overlap_audit",
        "overlap_manifest",
        "injection_manifest",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        inputs[name.upper()] = str(path)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TASK_CODE_DIR": str(code),
            "GWYOLO_CODE_COMMIT": "commit",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            **inputs,
        }
    )
    completed = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "promoted paired PE model selection failed" in completed.stderr
    assert "unbound variable" not in completed.stderr
