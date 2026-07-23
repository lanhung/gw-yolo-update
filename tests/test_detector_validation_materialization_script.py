from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_detector_validation_physical_materialization.sh"
)


def test_detector_materialization_projects_and_audits_all_ifos() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    ordered = (
        "waveform-validate",
        "injection-materialize",
        "injection-snr-annotate",
        "injection-arrival-annotate",
        "detector-validation-materialization-audit",
    )
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    for token in (
        "import lal; import lalsimulation; import pycbc",
        "--storage-mode signal_scaled_float16",
        "--split val",
        '"candidate_scores_inspected"',
        '"test_rows_read"',
    ):
        assert token in source
    assert "trigger-score" not in source
    assert "evaluation-corpus-open-once" not in source
    assert "O4b" not in source
