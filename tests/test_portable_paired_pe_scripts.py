from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_adapter_paired_pe_queue_is_validation_gated_and_publication_sized() -> None:
    script = (ROOT / "scripts/queue_adapter_paired_pe_input_bundle.sh").read_text()
    assert 'value.get("test_data_opened") is False' in script
    assert 'value.get("promoted_arm") == "glitch_adapter"' in script
    assert 'PE_VALIDATION_LIMIT:-100' in script
    assert 'PE_MINIMUM_GPS_BLOCKS:-25' in script
    assert "pe-input-bundle-export" in script
    assert "adapter_paired_pe_input_queue_negative_validation" in script


def test_portable_backend_wrappers_import_shared_inputs_and_export_evidence() -> None:
    expectations = {
        "run_amplfi_portable_paired_pe.sh": (
            "run_amplfi_within_backend_paired_smoke.sh",
            "amplfi_within_backend_paired_smoke_summary.json",
        ),
        "run_dingo_portable_paired_pe.sh": (
            "run_dingo_official_native_paired_smoke.sh",
            "dingo_official_native_paired_smoke_summary.json",
        ),
    }
    for filename, required in expectations.items():
        script = (ROOT / "scripts" / filename).read_text()
        assert "pe-input-bundle-import" in script
        assert "pe-within-backend-bundle-export" in script
        assert all(value in script for value in required)


def test_portable_portfolio_reprojects_both_backends_before_evaluation() -> None:
    script = (ROOT / "scripts/run_portable_paired_pe_portfolio.sh").read_text()
    assert script.count("pe-within-backend-bundle-import") == 2
    assert "within_backend_summary.projected.json" in script
    assert "run_paired_pe_portfolio_validation.sh" in script
