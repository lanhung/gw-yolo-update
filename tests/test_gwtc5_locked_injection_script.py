from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_gwtc5_locked_injection_plan.sh"
)


def test_locked_injection_script_runs_plan_waveform_freeze_and_replay() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gwtc5-locked-injection-plan" in source
    assert "gwtc5-locked-corpus-freeze" in source
    assert "waveform-validate" in source
    assert "WAVEFORM_PYTHON" in source
    assert "WAVEFORM_RUNTIME_RECEIPT" in source
    assert "--selection-mode family_approximant" in source
    assert "--include-alternatives" in source
    assert '--runtime-receipt "$waveform_runtime_receipt"' in source
    assert "publication-evidence-audit" in source
    assert '--inventory-report "$inventory_report"' in source
    assert 'len(gate.get("artifact_replay", [])) != 8' in source
    assert 'inventory.get("pre_access_vt_weights_assigned") is not False' in source
    assert 'inventory.get("post_access_dq_replacement_allowed") is not False' in source
    assert 'access_log.exists()' in source
