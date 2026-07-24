from __future__ import annotations

import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_amplfi_engineering_smoke.sh"


def test_amplfi_engineering_smoke_is_real_but_non_publication() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "CUDA_VISIBLE_DEVICES",
        "engineering_smoke",
        "amplfi-training-stage-freeze",
        "amplfi-common-prior-audit",
        "AMPLFI_CLI",
        "run_pe_model_load_smoke.py",
        "verified_amplfi_engineering_smoke",
        '"publication_candidate": False',
        '"deterministic_policy": "seeded_warn_on_unsupported_cuda_operations"',
        '"scientific_claim_allowed": False',
        '"test_rows_read": 0',
    ):
        assert token in source
    assert "--split test" not in source
    assert "evaluation-corpus-open-once" not in source
    assert "publication_stage_1" not in source


def test_amplfi_engineering_smoke_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 4
    for snippet in snippets:
        compile(snippet, SCRIPT.name, "exec")
