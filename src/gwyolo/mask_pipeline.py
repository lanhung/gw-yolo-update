from __future__ import annotations

from pathlib import Path
from typing import Any

from .injection_score import score_materialized_injections
from .io import atomic_write_json, file_sha256
from .learned_deglitch import (
    run_learned_background_deglitch,
    run_learned_deglitch,
)
from .runtime import execution_provenance
from .search import run_mask_search_validation
from .trigger import score_background_manifest


def run_mask_search_validation_pipeline(
    background_manifest: str | Path,
    clean_injection_manifest: str | Path,
    contaminated_injection_manifest: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    maximum_validation_false_alarms: int,
    strength: float = 0.9,
    clean_noninferiority_margin: float = 0.01,
    minimum_contaminated_efficiency_gain: float = 0.05,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
) -> dict[str, Any]:
    """Run all six validation arms needed by the frozen mask promotion table."""

    if maximum_validation_false_alarms < 0:
        raise ValueError("Maximum validation false alarms cannot be negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    background_raw = score_background_manifest(
        background_manifest,
        checkpoint_path,
        config_path,
        output / "background_raw",
        model_ifos,
        q_values,
        target_sample_rate,
        context_duration,
        save_probabilities=True,
        required_split="val",
    )
    background_cleaning = run_learned_background_deglitch(
        background_manifest,
        background_raw["triggers_path"],
        output / "background_cleaning",
        strength,
        model_ifos,
        target_sample_rate,
        context_duration,
        "val",
    )
    background_mask = score_background_manifest(
        background_cleaning["manifest_path"],
        checkpoint_path,
        config_path,
        output / "background_mask",
        model_ifos,
        q_values,
        target_sample_rate,
        context_duration,
        required_split="val",
    )

    clean_raw = score_materialized_injections(
        clean_injection_manifest,
        checkpoint_path,
        config_path,
        output / "clean_raw",
        model_ifos,
        q_values,
        target_sample_rate,
        save_probabilities=True,
        required_split="val",
    )
    clean_cleaning = run_learned_deglitch(
        clean_injection_manifest,
        clean_raw["triggers_path"],
        output / "clean_cleaning",
        strength,
    )
    clean_mask = score_materialized_injections(
        clean_cleaning["manifest_path"],
        checkpoint_path,
        config_path,
        output / "clean_mask",
        model_ifos,
        q_values,
        target_sample_rate,
        required_split="val",
    )

    contaminated_raw = score_materialized_injections(
        contaminated_injection_manifest,
        checkpoint_path,
        config_path,
        output / "contaminated_raw",
        model_ifos,
        q_values,
        target_sample_rate,
        save_probabilities=True,
        required_split="val",
    )
    contaminated_cleaning = run_learned_deglitch(
        contaminated_injection_manifest,
        contaminated_raw["triggers_path"],
        output / "contaminated_cleaning",
        strength,
    )
    contaminated_mask = score_materialized_injections(
        contaminated_cleaning["manifest_path"],
        checkpoint_path,
        config_path,
        output / "contaminated_mask",
        model_ifos,
        q_values,
        target_sample_rate,
        save_probabilities=True,
        required_split="val",
    )

    comparison_path = output / "mask_search_validation.json"
    comparison = run_mask_search_validation(
        background_raw["triggers_path"],
        background_mask["triggers_path"],
        clean_raw["triggers_path"],
        clean_mask["triggers_path"],
        contaminated_raw["triggers_path"],
        contaminated_mask["triggers_path"],
        comparison_path,
        maximum_validation_false_alarms,
        clean_noninferiority_margin,
        minimum_contaminated_efficiency_gain,
        "ranking_score",
        bootstrap_replicates,
        seed,
    )
    stages = {
        "background_raw": background_raw,
        "background_cleaning": background_cleaning,
        "background_mask": background_mask,
        "clean_raw": clean_raw,
        "clean_cleaning": clean_cleaning,
        "clean_mask": clean_mask,
        "contaminated_raw": contaminated_raw,
        "contaminated_cleaning": contaminated_cleaning,
        "contaminated_mask": contaminated_mask,
    }
    result = {
        "status": "validation_only_end_to_end_mask_search_pipeline",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "test_rows_read": 0,
        "development_gates_passed": comparison["development_gates_passed"],
        "scientific_blocker": (
            "passing validation gates still requires continuous clustered time-slide background, "
            "five seeds and one-time locked evaluation"
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "input_manifests": {
            "background": {
                "path": str(background_manifest),
                "sha256": file_sha256(background_manifest),
            },
            "clean_injections": {
                "path": str(clean_injection_manifest),
                "sha256": file_sha256(clean_injection_manifest),
            },
            "contaminated_injections": {
                "path": str(contaminated_injection_manifest),
                "sha256": file_sha256(contaminated_injection_manifest),
            },
        },
        "strength": strength,
        "maximum_validation_false_alarms": maximum_validation_false_alarms,
        "clean_noninferiority_margin": clean_noninferiority_margin,
        "minimum_contaminated_efficiency_gain": minimum_contaminated_efficiency_gain,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "stage_reports": {
            name: {
                "status": report["status"],
                "manifest_path": report["manifest_path"],
                "manifest_sha256": report["manifest_sha256"],
            }
            for name, report in stages.items()
        },
        "comparison_path": str(comparison_path),
        "comparison_sha256": file_sha256(comparison_path),
        "comparison": comparison,
        "test_evaluation": None,
        **execution_provenance(),
    }
    atomic_write_json(output / "mask_search_pipeline_report.json", result)
    return result
