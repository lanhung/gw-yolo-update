from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_independent_validation_injections.sh"


def test_independent_validation_runner_is_group_disjoint_and_test_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "background-disjoint-subset" in source
    assert "background-purpose-partition" in source
    assert '--exclude-manifest "$BASELINE_TRAIN_MANIFEST"' in source
    assert '--exclude-manifest "$BASELINE_VALIDATION_MANIFEST"' in source
    assert '--split val' in source
    assert '--test-count 0' in source
    assert "selected_exclusion_gps_block_overlap" in source
    assert "purpose_gps_block_overlap" in source
    assert "MINIMUM_PURPOSE_GPS_BLOCKS" in source
    assert "waveform-validate" in source
    assert 'WAVEFORM_PYTHON=${WAVEFORM_PYTHON:-$TASK_PYTHON}' in source
    assert "import lal; import lalsimulation; import pycbc" in source
    assert '"$WAVEFORM_PYTHON" -m gwyolo.cli waveform-validate' in source
    assert '"$WAVEFORM_PYTHON" -m gwyolo.cli injection-materialize' in source
    assert '"$WAVEFORM_PYTHON" -m gwyolo.cli injection-snr-annotate' in source
    assert '"$WAVEFORM_PYTHON" -m gwyolo.cli injection-arrival-annotate' in source
    assert "signal_scaled_float16" in source
    assert "injection-snr-annotate" in source
    assert "injection-arrival-annotate" in source
    assert "independent-validation-endpoint-freeze" in source


def test_independent_validation_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 7
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_independent_validation_runner_fails_on_missing_background(tmp_path: Path) -> None:
    environment = os.environ.copy()
    task_python = tmp_path / "python"
    task_python.write_text("", encoding="utf-8")
    code = tmp_path / "code" / "src" / "gwyolo"
    code.mkdir(parents=True)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    environment.update(
        {
            "TASK_PYTHON": str(task_python),
            "TASK_CODE_DIR": str(tmp_path / "code"),
            "GWYOLO_CODE_COMMIT": "commit",
            "ACQUISITION_ROOT": str(tmp_path / "missing-acquisition"),
            "BASELINE_TRAIN_MANIFEST": str(train),
            "BASELINE_VALIDATION_MANIFEST": str(validation),
            "OUTPUT_ROOT": str(tmp_path / "output"),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "required input is absent" in completed.stderr
