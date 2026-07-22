from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_waveform_runtime_setup_is_isolated_pinned_and_receipted() -> None:
    script = (ROOT / "scripts/setup_waveform_runtime.sh").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-waveforms.txt").read_text(encoding="utf-8")
    assert "pycbc==2.9.0" in requirements
    assert 'venv_dir="$output_dir/venv"' in script
    assert '"pip", "freeze", "--all"' in script
    assert "verified_isolated_waveform_runtime" in script
    assert "requirements_sha256" in script
    assert "pip_freeze_sha256" in script
    assert "SEOBNRv5PHM" in script
    assert "IMRPhenomXPHM" in script
