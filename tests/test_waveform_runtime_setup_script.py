from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_waveform_runtime_setup_is_isolated_pinned_and_receipted() -> None:
    script = (ROOT / "scripts/setup_waveform_runtime.sh").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-waveforms.txt").read_text(encoding="utf-8")
    assert "pycbc==2.9.0" in requirements
    assert "numpy==1.26.4" in requirements
    assert "scipy==1.13.1" in requirements
    assert "astropy==6.1.7" in requirements
    assert "lalsuite==7.26.15" in requirements
    assert "setuptools==80.9.0" in requirements
    assert "pip==26.1.2 setuptools==80.9.0 wheel==0.47.0" in script.replace(
        "\\\n", ""
    )
    assert 'venv_dir="$output_dir/venv"' in script
    assert '"pip", "freeze", "--all"' in script
    assert "verified_isolated_waveform_runtime" in script
    assert "requirements_sha256" in script
    assert "pip_freeze_sha256" in script
    assert "IMRPhenomPv3HM" in script
    assert "IMRPhenomXPHM" in script
    assert "GetStringFromApproximant" in script
    assert "SimInspiralImplementedFDApproximants" in script
    assert "SimInspiralImplementedTDApproximants" in script
