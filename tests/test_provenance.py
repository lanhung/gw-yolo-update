from __future__ import annotations

from dataclasses import replace

from gwyolo.provenance import SceneRecipe, audit_provenance


def recipe(split: str, index: int, kind: str = "overlap") -> SceneRecipe:
    has_chirp = kind in {"chirp_only", "overlap"}
    has_glitch = kind in {"noise_only", "overlap"}
    return SceneRecipe(
        split=split,
        scene_type=kind,
        observing_run="O4a",
        gps_start=1_360_000_000 + index,
        duration=4.0,
        sample_rate=1024,
        ifos=("H1", "L1"),
        q_values=(4.0, 8.0),
        seed=index,
        waveform_id=f"w-{split}-{index}" if has_chirp else None,
        injection_id=f"i-{split}-{index}" if has_chirp else None,
        glitch_id=f"g-{split}-{index}" if has_glitch else None,
        glitch_ifo="H1" if has_glitch else None,
        source_family="BBH" if has_chirp else None,
        target_snr=10.0 if has_chirp else None,
    )


def test_audit_accepts_disjoint_physical_ids() -> None:
    report = audit_provenance([recipe("train", 1), recipe("val", 2), recipe("test", 3)])
    assert report["passed"]
    assert report["cross_split_overlap_count"] == 0


def test_audit_rejects_reused_injection_even_when_scene_differs() -> None:
    train = recipe("train", 1)
    val = replace(recipe("val", 2), injection_id=train.injection_id)
    report = audit_provenance([train, val])
    assert not report["passed"]
    assert report["cross_split_overlaps"]["injection_id"]["train:val"] == ["i-train-1"]


def test_audit_rejects_reused_gps_background() -> None:
    train = recipe("train", 1, "quiet")
    val = replace(recipe("val", 2, "quiet"), gps_start=train.gps_start)
    report = audit_provenance([train, val])
    assert not report["passed"]
    assert report["cross_split_overlap_count"] == 1
