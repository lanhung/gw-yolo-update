from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_physical_overlap_scaling_hard_endpoint.sh"
)


def test_hard_endpoint_successor_freezes_before_reading_scaling_metrics() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    freeze = source.index("physical-overlap-scale-hard-subset-freeze")
    scaling_read = source.index('summary = json.loads(summary_path.read_text')
    assert freeze < scaling_read
    assert "GWYOLO_CODE_COMMIT" in source
    assert "test_rows_read" in source
    assert "candidate_scores" not in source
    assert "physical-overlap-scale-hard-endpoint-cell" in source
    assert "physical-overlap-scale-hard-endpoint-bind" in source
    assert "NEXT_PHYSICAL_SCALE" in source
