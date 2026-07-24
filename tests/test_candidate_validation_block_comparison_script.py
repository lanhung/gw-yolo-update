from __future__ import annotations

import re
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_candidate_validation_block_comparison.sh"
)


def test_block_comparison_requires_same_frozen_endpoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "INDEPENDENT_VALIDATION_ENDPOINT_REPORT" in source
    assert "frozen_gps_and_purpose_disjoint_validation_endpoint" in source
    assert 'get("background_manifest_sha256")' in source
    assert 'get("injection_manifest_sha256")' in source
    assert 'get("code_commit")' in source


def test_block_comparison_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], SCRIPT.name, "exec")
