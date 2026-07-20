from __future__ import annotations

import argparse
import json
from typing import Any

from .catalog import evaluate_catalog_predictions
from .config import load_config
from .data import audit_and_split, scan_sources
from .factory import run_data_factory
from .gwosc import (
    run_gwosc_batch_download,
    run_gwosc_event_exclusions,
    run_gwosc_pilot,
    run_gwosc_run_plan,
    run_gwosc_verification,
)
from .pipeline import run_pipeline
from .prediction import predict_catalog
from .provenance import create_recipe_subset
from .search import (
    run_frozen_search_evaluation,
    run_search_benchmark,
    run_search_calibration,
    run_search_comparison,
    run_validation_injection_diagnostic,
)
from .scaling import run_curve_fit, run_scale_plan
from .training import evaluate_checkpoint, train_candidate


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gwyolo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("audit", "split", "pipeline"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--candidate", type=int, default=0)
    train.add_argument("--dataset-yaml", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset-yaml", required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--name", default="evaluation")

    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--confidence", type=float, default=0.25)

    catalog = subparsers.add_parser("catalog-eval")
    catalog.add_argument("--predictions", required=True)
    catalog.add_argument("--api-url", required=True)
    catalog.add_argument("--output", required=True)

    search = subparsers.add_parser("search-eval")
    search.add_argument("--validation-background", required=True)
    search.add_argument("--test-background", required=True)
    search.add_argument("--test-injections", required=True)
    search.add_argument("--validation-live-time-years", required=True, type=float)
    search.add_argument("--test-live-time-years", required=True, type=float)
    search.add_argument("--target-far-per-year", required=True, type=float)
    search.add_argument("--output", required=True)

    search_compare = subparsers.add_parser("search-compare")
    search_compare.add_argument("--validation-background", required=True)
    search_compare.add_argument("--test-background", required=True)
    search_compare.add_argument("--test-injections", required=True)
    search_compare.add_argument("--validation-live-time-years", required=True, type=float)
    search_compare.add_argument("--test-live-time-years", required=True, type=float)
    search_compare.add_argument("--target-far-per-year", required=True, type=float)
    search_compare.add_argument("--score-field-a", required=True)
    search_compare.add_argument("--score-field-b", required=True)
    search_compare.add_argument("--bootstrap-replicates", type=int, default=2000)
    search_compare.add_argument("--seed", type=int, default=20260719)
    search_compare.add_argument("--output", required=True)

    search_calibrate = subparsers.add_parser("search-calibrate")
    search_calibrate.add_argument("--validation-background", required=True)
    search_calibrate.add_argument("--validation-live-time-years", required=True, type=float)
    search_calibrate.add_argument("--target-far-per-year", required=True, type=float)
    search_calibrate.add_argument("--score-field", default="ranking_score")
    search_calibrate.add_argument("--output", required=True)

    search_frozen = subparsers.add_parser("search-evaluate-frozen")
    search_frozen.add_argument("--calibration-report", required=True)
    search_frozen.add_argument("--test-background", required=True)
    search_frozen.add_argument("--test-injections", required=True)
    search_frozen.add_argument("--test-live-time-years", required=True, type=float)
    search_frozen.add_argument("--bootstrap-replicates", type=int, default=2000)
    search_frozen.add_argument("--seed", type=int, default=20260719)
    search_frozen.add_argument("--output", required=True)

    search_validation = subparsers.add_parser("search-validation-injections")
    search_validation.add_argument("--calibration-report", required=True)
    search_validation.add_argument("--validation-injections", required=True)
    search_validation.add_argument("--bootstrap-replicates", type=int, default=2000)
    search_validation.add_argument("--seed", type=int, default=20260719)
    search_validation.add_argument("--output", required=True)

    scaling = subparsers.add_parser("scale-plan")
    scaling.add_argument("--manifest", required=True)
    scaling.add_argument("--output", required=True)
    scaling.add_argument("--baseline-target", type=int, default=10_000)
    scaling.add_argument("--research-target", type=int, default=200_000)
    scaling.add_argument("--seeds", type=int, default=3)

    factory = subparsers.add_parser("data-factory")
    factory.add_argument("--config", required=True)
    factory.add_argument("--output-dir", required=True)
    factory.add_argument("--limit", type=int)

    gwosc = subparsers.add_parser("gwosc-pilot")
    gwosc.add_argument("--event", required=True)
    gwosc.add_argument("--cache-dir", required=True)
    gwosc.add_argument("--output-dir", required=True)
    gwosc.add_argument("--detectors", nargs="+")
    gwosc.add_argument("--context-duration", type=float, default=64.0)
    gwosc.add_argument("--output-duration", type=float, default=8.0)
    gwosc.add_argument("--target-sample-rate", type=int, default=1024)
    gwosc.add_argument("--download-workers", type=int, default=4)
    gwosc.add_argument("--allow-locked-evaluation-data", action="store_true")

    gwosc_verify = subparsers.add_parser("gwosc-verify")
    gwosc_verify.add_argument("--event", required=True)
    gwosc_verify.add_argument(
        "--file", action="append", required=True, metavar="IFO=PATH"
    )
    gwosc_verify.add_argument("--output", required=True)
    gwosc_verify.add_argument("--chunk-samples", type=int, default=1_048_576)

    gwosc_run_plan = subparsers.add_parser("gwosc-run-plan")
    gwosc_run_plan.add_argument("--run", required=True)
    gwosc_run_plan.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    gwosc_run_plan.add_argument("--sample-rate-khz", type=int, default=4)
    gwosc_run_plan.add_argument("--maximum-pairs", type=int)
    gwosc_run_plan.add_argument("--seed", type=int, default=20260719)
    gwosc_run_plan.add_argument("--output", required=True)

    gwosc_batch = subparsers.add_parser("gwosc-batch-download")
    gwosc_batch.add_argument("--plan", required=True)
    gwosc_batch.add_argument("--cache-dir", required=True)
    gwosc_batch.add_argument("--output-dir", required=True)
    gwosc_batch.add_argument("--maximum-pairs", type=int)
    gwosc_batch.add_argument("--download-workers", type=int, default=8)
    gwosc_batch.add_argument("--chunk-samples", type=int, default=1_048_576)

    gwosc_exclusions = subparsers.add_parser("gwosc-event-exclusions")
    gwosc_exclusions.add_argument("--run", required=True)
    gwosc_exclusions.add_argument("--padding-seconds", type=float, default=16.0)
    gwosc_exclusions.add_argument("--workers", type=int, default=4)
    gwosc_exclusions.add_argument("--output", required=True)

    numeric = subparsers.add_parser("numeric-train")
    numeric.add_argument("--config", required=True)
    numeric.add_argument("--manifest", required=True)
    numeric.add_argument("--output-dir", required=True)
    numeric.add_argument("--seed", type=int)

    numeric_multiseed = subparsers.add_parser("numeric-multiseed")
    numeric_multiseed.add_argument("--config", required=True)
    numeric_multiseed.add_argument("--manifest", required=True)
    numeric_multiseed.add_argument("--output-dir", required=True)
    numeric_multiseed.add_argument("--seeds", nargs="+", type=int, required=True)
    numeric_multiseed.add_argument(
        "--reuse-run", action="append", default=[], help="SEED=/path/numeric_training_report.json"
    )

    numeric_evaluate = subparsers.add_parser("numeric-evaluate")
    numeric_evaluate.add_argument("--config", required=True)
    numeric_evaluate.add_argument("--manifest", required=True)
    numeric_evaluate.add_argument("--checkpoint", required=True)
    numeric_evaluate.add_argument("--split", choices=("val", "test"), required=True)
    numeric_evaluate.add_argument("--chirp-threshold", required=True, type=float)
    numeric_evaluate.add_argument("--glitch-threshold", required=True, type=float)
    numeric_evaluate.add_argument("--output", required=True)

    physical_finetune = subparsers.add_parser("physical-finetune")
    physical_finetune.add_argument("--config", required=True)
    physical_finetune.add_argument("--train-manifest", required=True)
    physical_finetune.add_argument("--validation-manifest", required=True)
    physical_finetune.add_argument("--pretrained-checkpoint", required=True)
    physical_finetune.add_argument("--output-dir", required=True)
    physical_finetune.add_argument("--seed", type=int)

    snr_curriculum = subparsers.add_parser("physical-snr-curriculum")
    snr_curriculum.add_argument("--manifest", required=True)
    snr_curriculum.add_argument("--output-dir", required=True)
    snr_curriculum.add_argument("--minimum-snr", type=float, default=4.0)
    snr_curriculum.add_argument("--rescale-upper-snr", type=float, default=8.0)
    snr_curriculum.add_argument("--seed", type=int, default=20260720)

    snr_quota = subparsers.add_parser("physical-snr-quota")
    snr_quota.add_argument("--manifest", required=True)
    snr_quota.add_argument("--output-dir", required=True)
    snr_quota.add_argument(
        "--snr-bin",
        action="append",
        metavar="LOWER:UPPER:FRACTION",
        help="repeat for custom non-overlapping bins; default is 4:8:0.4, 8:15:0.35, 15:30:0.2, 30:50:0.05",
    )
    snr_quota.add_argument("--seed", type=int, default=20260720)

    physical_audit = subparsers.add_parser("physical-checkpoint-audit")
    physical_audit.add_argument("--config", required=True)
    physical_audit.add_argument("--validation-manifest", required=True)
    physical_audit.add_argument("--checkpoint", required=True)
    physical_audit.add_argument("--chirp-threshold", type=float, required=True)
    physical_audit.add_argument("--output", required=True)

    subset = subparsers.add_parser("recipe-subset")
    subset.add_argument("--manifest", required=True)
    subset.add_argument("--output", required=True)
    subset.add_argument("--train-count", required=True, type=int)
    subset.add_argument("--val-count", required=True, type=int)
    subset.add_argument("--test-count", required=True, type=int)

    gravityspy = subparsers.add_parser("gravityspy-index")
    gravityspy.add_argument("--record-id", type=int, default=5_649_212)
    gravityspy.add_argument("--filenames", nargs="+", required=True)
    gravityspy.add_argument("--cache-dir", required=True)
    gravityspy.add_argument("--output-dir", required=True)
    gravityspy.add_argument("--minimum-confidence", type=float, default=0.9)
    gravityspy.add_argument("--per-label", type=int, default=100)
    gravityspy.add_argument("--seed", type=int, default=20260719)
    gravityspy.add_argument("--download-workers", type=int, default=8)

    gravityspy_split = subparsers.add_parser("gravityspy-split")
    gravityspy_split.add_argument("--manifest", required=True)
    gravityspy_split.add_argument("--output-dir", required=True)
    gravityspy_split.add_argument("--validation-fraction", type=float, default=0.1)
    gravityspy_split.add_argument("--test-fraction", type=float, default=0.1)
    gravityspy_split.add_argument("--seed", type=int, default=20260720)

    gravityspy_strain = subparsers.add_parser("gravityspy-strain-plan")
    gravityspy_strain.add_argument("--manifest", required=True)
    gravityspy_strain.add_argument("--output-dir", required=True)
    gravityspy_strain.add_argument("--sample-rate-khz", type=int, default=4)
    gravityspy_strain.add_argument("--context-duration", type=float, default=64.0)

    gravityspy_shard = subparsers.add_parser("gravityspy-strain-shard")
    gravityspy_shard.add_argument("--manifest", required=True)
    gravityspy_shard.add_argument("--output-dir", required=True)
    gravityspy_shard.add_argument("--files-per-shard", type=int, default=32)
    gravityspy_shard.add_argument("--seed", type=int, default=20260720)

    gravityspy_materialize = subparsers.add_parser("gravityspy-strain-materialize")
    gravityspy_materialize.add_argument("--manifest", required=True)
    gravityspy_materialize.add_argument("--shard", required=True, type=int)
    gravityspy_materialize.add_argument("--config", required=True)
    gravityspy_materialize.add_argument("--cache-dir", required=True)
    gravityspy_materialize.add_argument("--output-dir", required=True)
    gravityspy_materialize.add_argument("--output-duration", type=float, default=8.0)
    gravityspy_materialize.add_argument("--download-workers", type=int, default=8)
    gravityspy_materialize.add_argument("--chunk-samples", type=int, default=1_048_576)

    gravityspy_merge = subparsers.add_parser("gravityspy-numeric-merge")
    gravityspy_merge.add_argument("--report", action="append", required=True)
    gravityspy_merge.add_argument("--output-dir", required=True)
    gravityspy_merge.add_argument("--split", choices=("train", "val", "test"), required=True)

    gravityspy_evict = subparsers.add_parser("gravityspy-strain-evict")
    gravityspy_evict.add_argument("--materialization-report", required=True)
    gravityspy_evict.add_argument("--cache-dir", required=True)
    gravityspy_evict.add_argument("--output", required=True)

    curve = subparsers.add_parser("fit-curve")
    curve.add_argument("--points", required=True)
    curve.add_argument("--output", required=True)

    background = subparsers.add_parser("background-plan")
    background.add_argument("--file", action="append", required=True, help="IFO=/path/file.hdf5")
    background.add_argument("--source-verification-report", required=True)
    background.add_argument("--output-dir", required=True)
    background.add_argument("--window-duration", type=int, default=8)
    background.add_argument("--stride", type=int, default=8)
    background.add_argument("--block-duration", type=int, default=256)
    background.add_argument("--required-context-duration", type=int, default=64)
    background.add_argument("--required-dq-bits", type=int, default=1)
    background.add_argument("--required-injection-bits", type=int, default=23)
    background.add_argument("--exclude", action="append", default=[], help="GPS_START:GPS_END")
    background.add_argument("--validation-fraction", type=float, default=0.2)
    background.add_argument("--test-fraction", type=float, default=0.2)
    background.add_argument("--seed", type=int, default=20260719)

    background_batch = subparsers.add_parser("background-batch-plan")
    background_batch.add_argument("--batch-report", action="append", required=True)
    background_batch.add_argument("--event-exclusions", required=True)
    background_batch.add_argument("--output-dir", required=True)
    background_batch.add_argument("--window-duration", type=int, default=8)
    background_batch.add_argument("--stride", type=int, default=8)
    background_batch.add_argument("--block-duration", type=int, default=256)
    background_batch.add_argument("--required-context-duration", type=int, default=64)
    background_batch.add_argument("--required-dq-bits", type=int, default=1)
    background_batch.add_argument("--required-injection-bits", type=int, default=23)
    background_batch.add_argument("--validation-fraction", type=float, default=0.2)
    background_batch.add_argument("--test-fraction", type=float, default=0.2)
    background_batch.add_argument("--seed", type=int, default=20260719)

    deglitch = subparsers.add_parser("oracle-deglitch")
    deglitch.add_argument("--input", required=True)
    deglitch.add_argument("--output", required=True)
    deglitch.add_argument("--report", required=True)
    deglitch.add_argument("--strength", type=float, default=0.9)

    deglitch_benchmark = subparsers.add_parser("oracle-deglitch-benchmark")
    deglitch_benchmark.add_argument("--factory-report", required=True)
    deglitch_benchmark.add_argument("--output", required=True)
    deglitch_benchmark.add_argument("--strength", type=float, default=0.9)

    learned_deglitch = subparsers.add_parser("learned-deglitch")
    learned_deglitch.add_argument("--materialized-manifest", required=True)
    learned_deglitch.add_argument("--scored-manifest", required=True)
    learned_deglitch.add_argument("--output-dir", required=True)
    learned_deglitch.add_argument("--strength", type=float, default=0.9)

    trigger = subparsers.add_parser("trigger-score")
    trigger.add_argument("--manifest", required=True)
    trigger.add_argument("--checkpoint", required=True)
    trigger.add_argument("--config", required=True)
    trigger.add_argument("--output-dir", required=True)
    trigger.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    trigger.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    trigger.add_argument("--target-sample-rate", type=int, default=1024)
    trigger.add_argument("--context-duration", type=float, default=64.0)
    trigger.add_argument("--save-probabilities", action="store_true")

    candidates = subparsers.add_parser("candidate-extract")
    candidates.add_argument("--triggers", required=True)
    candidates.add_argument("--output-dir", required=True)
    candidates.add_argument("--chirp-threshold", type=float, default=0.3)
    candidates.add_argument("--minimum-bins", type=int, default=1)

    candidate_slides = subparsers.add_parser("candidate-time-slides")
    candidate_slides.add_argument("--candidates", required=True)
    candidate_slides.add_argument("--background-manifest", required=True)
    candidate_slides.add_argument("--output-dir", required=True)
    candidate_slides.add_argument("--split", choices=("val", "test"), required=True)
    candidate_slides.add_argument("--reference-ifo", default="H1")
    candidate_slides.add_argument("--shifted-ifo", default="L1")
    candidate_slides.add_argument("--slide-count", type=int, required=True)
    candidate_slides.add_argument("--step-seconds", type=float, required=True)
    candidate_slides.add_argument("--coincidence-window-seconds", type=float, required=True)
    candidate_slides.add_argument("--cluster-window-seconds", type=float, default=0.1)

    time_slide = subparsers.add_parser("time-slide-background")
    time_slide.add_argument("--triggers", required=True)
    time_slide.add_argument("--output-dir", required=True)
    time_slide.add_argument("--split", choices=("val", "test"), required=True)
    time_slide.add_argument("--reference-ifo", default="H1")
    time_slide.add_argument("--shifted-ifo", default="L1")
    time_slide.add_argument("--slide-count", type=int, required=True)
    time_slide.add_argument("--step-seconds", type=float, required=True)
    time_slide.add_argument("--coincidence-window-seconds", type=float)

    injection = subparsers.add_parser("injection-plan")
    injection.add_argument("--background-manifest", required=True)
    injection.add_argument("--background-report", required=True)
    injection.add_argument("--output-dir", required=True)
    injection.add_argument("--train-count", type=int, default=0)
    injection.add_argument("--validation-count", type=int, default=5000)
    injection.add_argument("--test-count", type=int, default=20000)
    injection.add_argument("--seed", type=int, default=20260719)

    materialize = subparsers.add_parser("injection-materialize")
    materialize.add_argument("--recipes", required=True)
    materialize.add_argument("--background-manifest", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--sample-rate", type=int, default=2048)
    materialize.add_argument("--context-duration", type=float, default=64.0)
    materialize.add_argument(
        "--storage-mode",
        choices=("signal_only", "signal_scaled_float16", "full"),
        default="signal_only",
    )
    materialize.add_argument("--split", choices=("train", "val", "test"))
    materialize.add_argument("--limit", type=int)
    materialize.add_argument("--backend-validation-report")

    waveform_validate = subparsers.add_parser("waveform-validate")
    waveform_validate.add_argument("--recipes", required=True)
    waveform_validate.add_argument("--output", required=True)
    waveform_validate.add_argument("--sample-rate", type=int, default=2048)
    waveform_validate.add_argument("--reference-duration", type=float, default=128.0)
    waveform_validate.add_argument("--per-family", type=int, default=5)

    snr_annotate = subparsers.add_parser("injection-snr-annotate")
    snr_annotate.add_argument("--manifest", required=True)
    snr_annotate.add_argument("--output-dir", required=True)
    snr_annotate.add_argument("--low-frequency", type=float, default=20.0)
    snr_annotate.add_argument("--high-frequency", type=float, default=500.0)
    snr_annotate.add_argument("--psd-segment-seconds", type=float, default=8.0)
    snr_annotate.add_argument("--psd-stride-seconds", type=float, default=4.0)

    injection_score = subparsers.add_parser("injection-score")
    injection_score.add_argument("--manifest", required=True)
    injection_score.add_argument("--checkpoint", required=True)
    injection_score.add_argument("--config", required=True)
    injection_score.add_argument("--output-dir", required=True)
    injection_score.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    injection_score.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    injection_score.add_argument("--target-sample-rate", type=int, default=1024)
    injection_score.add_argument("--save-probabilities", action="store_true")

    pe = subparsers.add_parser("pe-evaluate")
    pe.add_argument("--manifest", required=True)
    pe.add_argument("--output", required=True)
    pe.add_argument("--credible-level", type=float, default=0.9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"audit", "split", "pipeline", "train"}:
        config = load_config(args.config)
    if args.command == "audit":
        samples, report = scan_sources(config["data"]["sources"])
        report["sample_preview"] = [sample.sample_id for sample in samples[:5]]
        _print(report)
    elif args.command == "split":
        _print(audit_and_split(config, config["project"]["output_dir"]))
    elif args.command == "pipeline":
        _print(run_pipeline(config))
    elif args.command == "train":
        candidate = config["training"]["candidates"][args.candidate]
        _print(
            train_candidate(
                candidate,
                config["training"].get("common", {}),
                args.dataset_yaml,
                config["project"]["output_dir"],
                selection_metric=str(config["quality"]["metric"]),
            )
        )
    elif args.command == "evaluate":
        _print(
            evaluate_checkpoint(
                args.checkpoint,
                args.dataset_yaml,
                args.split,
                args.output_dir,
                args.name,
            )
        )
    elif args.command == "predict":
        _print(predict_catalog(args.checkpoint, args.source, args.output_dir, args.confidence))
    elif args.command == "catalog-eval":
        _print(evaluate_catalog_predictions(args.predictions, args.api_url, args.output))
    elif args.command == "search-eval":
        _print(
            run_search_benchmark(
                args.validation_background,
                args.test_background,
                args.test_injections,
                args.validation_live_time_years,
                args.test_live_time_years,
                args.target_far_per_year,
                args.output,
            )
        )
    elif args.command == "search-compare":
        _print(
            run_search_comparison(
                args.validation_background,
                args.test_background,
                args.test_injections,
                args.validation_live_time_years,
                args.test_live_time_years,
                args.target_far_per_year,
                args.score_field_a,
                args.score_field_b,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "search-calibrate":
        _print(
            run_search_calibration(
                args.validation_background,
                args.validation_live_time_years,
                args.target_far_per_year,
                args.score_field,
                args.output,
            )
        )
    elif args.command == "search-evaluate-frozen":
        _print(
            run_frozen_search_evaluation(
                args.calibration_report,
                args.test_background,
                args.test_injections,
                args.test_live_time_years,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "search-validation-injections":
        _print(
            run_validation_injection_diagnostic(
                args.calibration_report,
                args.validation_injections,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "scale-plan":
        _print(
            run_scale_plan(
                args.manifest,
                args.output,
                args.baseline_target,
                args.research_target,
                args.seeds,
            )
        )
    elif args.command == "data-factory":
        _print(run_data_factory(args.config, args.output_dir, args.limit))
    elif args.command == "gwosc-pilot":
        _print(
            run_gwosc_pilot(
                event=args.event,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                detectors=args.detectors,
                context_duration=args.context_duration,
                output_duration=args.output_duration,
                target_sample_rate=args.target_sample_rate,
                download_workers=args.download_workers,
                allow_locked_evaluation_data=args.allow_locked_evaluation_data,
            )
        )
    elif args.command == "gwosc-verify":
        files = {}
        for item in args.file:
            if "=" not in item:
                raise ValueError(f"Expected IFO=PATH for --file, received {item!r}")
            detector, path = item.split("=", 1)
            detector = detector.strip().upper()
            if not detector or not path or detector in files:
                raise ValueError(f"Invalid or duplicate --file value: {item!r}")
            files[detector] = path
        _print(run_gwosc_verification(args.event, files, args.output, args.chunk_samples))
    elif args.command == "gwosc-run-plan":
        _print(
            run_gwosc_run_plan(
                args.run,
                args.detectors,
                args.output,
                args.sample_rate_khz,
                args.maximum_pairs,
                args.seed,
            )
        )
    elif args.command == "gwosc-batch-download":
        _print(
            run_gwosc_batch_download(
                args.plan,
                args.cache_dir,
                args.output_dir,
                args.maximum_pairs,
                args.download_workers,
                args.chunk_samples,
            )
        )
    elif args.command == "gwosc-event-exclusions":
        _print(
            run_gwosc_event_exclusions(
                args.run,
                args.output,
                args.padding_seconds,
                args.workers,
            )
        )
    elif args.command == "numeric-train":
        from .numeric import train_numeric_model

        _print(train_numeric_model(args.config, args.manifest, args.output_dir, args.seed))
    elif args.command == "numeric-multiseed":
        from .multiseed import run_numeric_multiseed

        reuse_runs = {
            int(seed): path for seed, path in (item.split("=", 1) for item in args.reuse_run)
        }
        _print(
            run_numeric_multiseed(
                args.config,
                args.manifest,
                args.output_dir,
                args.seeds,
                reuse_runs,
            )
        )
    elif args.command == "numeric-evaluate":
        from .numeric import evaluate_numeric_checkpoint

        _print(
            evaluate_numeric_checkpoint(
                args.config,
                args.manifest,
                args.checkpoint,
                args.split,
                (args.chirp_threshold, args.glitch_threshold),
                args.output,
            )
        )
    elif args.command == "physical-finetune":
        from .physical_training import run_physical_finetune

        _print(
            run_physical_finetune(
                args.config,
                args.train_manifest,
                args.validation_manifest,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "physical-snr-curriculum":
        from .physical_training import build_snr_curriculum_manifest

        _print(
            build_snr_curriculum_manifest(
                args.manifest,
                args.output_dir,
                args.minimum_snr,
                args.rescale_upper_snr,
                args.seed,
            )
        )
    elif args.command == "physical-snr-quota":
        from .physical_training import build_snr_quota_manifest

        bins = (
            [tuple(float(value) for value in item.split(":")) for item in args.snr_bin]
            if args.snr_bin
            else None
        )
        if bins is not None and any(len(item) != 3 for item in bins):
            raise ValueError("Each --snr-bin must be LOWER:UPPER:FRACTION")
        _print(build_snr_quota_manifest(args.manifest, args.output_dir, bins, args.seed))
    elif args.command == "physical-checkpoint-audit":
        from .physical_training import audit_physical_checkpoint

        _print(
            audit_physical_checkpoint(
                args.config,
                args.validation_manifest,
                args.checkpoint,
                args.chirp_threshold,
                args.output,
            )
        )
    elif args.command == "recipe-subset":
        _print(
            create_recipe_subset(
                args.manifest,
                args.output,
                args.train_count,
                args.val_count,
                args.test_count,
            )
        )
    elif args.command == "gravityspy-index":
        from .gravityspy import run_gravityspy_index

        _print(
            run_gravityspy_index(
                args.record_id,
                args.filenames,
                args.cache_dir,
                args.output_dir,
                args.minimum_confidence,
                args.per_label,
                args.seed,
                args.download_workers,
            )
        )
    elif args.command == "gravityspy-split":
        from .gravityspy import split_gravityspy_anchors

        _print(
            split_gravityspy_anchors(
                args.manifest,
                args.output_dir,
                args.validation_fraction,
                args.test_fraction,
                args.seed,
            )
        )
    elif args.command == "gravityspy-strain-plan":
        from .gravityspy import plan_gravityspy_strain

        _print(
            plan_gravityspy_strain(
                args.manifest,
                args.output_dir,
                args.sample_rate_khz,
                args.context_duration,
            )
        )
    elif args.command == "gravityspy-strain-shard":
        from .gravityspy import shard_gravityspy_strain_plan

        _print(
            shard_gravityspy_strain_plan(
                args.manifest,
                args.output_dir,
                args.files_per_shard,
                args.seed,
            )
        )
    elif args.command == "gravityspy-strain-materialize":
        from .gravityspy import materialize_gravityspy_strain_shard

        _print(
            materialize_gravityspy_strain_shard(
                args.manifest,
                args.shard,
                args.config,
                args.cache_dir,
                args.output_dir,
                args.output_duration,
                args.download_workers,
                args.chunk_samples,
            )
        )
    elif args.command == "gravityspy-numeric-merge":
        from .gravityspy import merge_gravityspy_numeric_manifests

        _print(merge_gravityspy_numeric_manifests(args.report, args.output_dir, args.split))
    elif args.command == "gravityspy-strain-evict":
        from .gravityspy import evict_gravityspy_verified_sources

        _print(
            evict_gravityspy_verified_sources(
                args.materialization_report, args.cache_dir, args.output
            )
        )
    elif args.command == "fit-curve":
        _print(run_curve_fit(args.points, args.output))
    elif args.command == "background-plan":
        from .background import run_background_plan

        files = dict(item.split("=", 1) for item in args.file)
        exclusions = [tuple(float(value) for value in item.split(":", 1)) for item in args.exclude]
        _print(
            run_background_plan(
                files,
                args.output_dir,
                source_verification_report=args.source_verification_report,
                window_duration=args.window_duration,
                stride=args.stride,
                block_duration=args.block_duration,
                required_context_duration=args.required_context_duration,
                required_dq_bits=args.required_dq_bits,
                required_injection_bits=args.required_injection_bits,
                excluded_intervals=exclusions,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
        )
    elif args.command == "background-batch-plan":
        from .background import run_batch_background_plan

        _print(
            run_batch_background_plan(
                args.batch_report,
                args.event_exclusions,
                args.output_dir,
                args.window_duration,
                args.stride,
                args.block_duration,
                args.required_context_duration,
                args.required_dq_bits,
                args.required_injection_bits,
                args.validation_fraction,
                args.test_fraction,
                args.seed,
            )
        )
    elif args.command == "oracle-deglitch":
        from .deglitch import run_oracle_deglitch

        _print(run_oracle_deglitch(args.input, args.output, args.report, args.strength))
    elif args.command == "oracle-deglitch-benchmark":
        from .deglitch import run_oracle_deglitch_benchmark

        _print(
            run_oracle_deglitch_benchmark(
                args.factory_report, args.output, args.strength
            )
        )
    elif args.command == "learned-deglitch":
        from .learned_deglitch import run_learned_deglitch

        _print(
            run_learned_deglitch(
                args.materialized_manifest,
                args.scored_manifest,
                args.output_dir,
                args.strength,
            )
        )
    elif args.command == "trigger-score":
        from .trigger import score_background_manifest

        _print(
            score_background_manifest(
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                output_dir=args.output_dir,
                model_ifos=tuple(args.model_ifos),
                q_values=tuple(args.q_values),
                target_sample_rate=args.target_sample_rate,
                context_duration=args.context_duration,
                save_probabilities=args.save_probabilities,
            )
        )
    elif args.command == "candidate-extract":
        from .candidates import run_candidate_extraction

        _print(
            run_candidate_extraction(
                args.triggers,
                args.output_dir,
                args.chirp_threshold,
                args.minimum_bins,
            )
        )
    elif args.command == "candidate-time-slides":
        from .candidates import run_candidate_time_slides

        _print(
            run_candidate_time_slides(
                args.candidates,
                args.background_manifest,
                args.output_dir,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.slide_count,
                args.step_seconds,
                args.coincidence_window_seconds,
                args.cluster_window_seconds,
            )
        )
    elif args.command == "time-slide-background":
        from .timeslides import run_window_time_slides

        _print(
            run_window_time_slides(
                args.triggers,
                args.output_dir,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.slide_count,
                args.step_seconds,
                args.coincidence_window_seconds,
            )
        )
    elif args.command == "injection-plan":
        from .injections import run_injection_plan

        _print(
            run_injection_plan(
                background_manifest=args.background_manifest,
                background_report=args.background_report,
                output_dir=args.output_dir,
                validation_count=args.validation_count,
                test_count=args.test_count,
                seed=args.seed,
                training_count=args.train_count,
            )
        )
    elif args.command == "injection-materialize":
        from .waveforms import run_injection_materialization

        _print(
            run_injection_materialization(
                recipe_manifest=args.recipes,
                background_manifest=args.background_manifest,
                output_dir=args.output_dir,
                sample_rate=args.sample_rate,
                split=args.split,
                limit=args.limit,
                backend_validation_report=args.backend_validation_report,
                context_duration=args.context_duration,
                storage_mode=args.storage_mode,
            )
        )
    elif args.command == "waveform-validate":
        from .waveforms import validate_waveform_backend

        _print(
            validate_waveform_backend(
                args.recipes,
                args.output,
                args.sample_rate,
                args.reference_duration,
                args.per_family,
            )
        )
    elif args.command == "injection-snr-annotate":
        from .waveforms import annotate_materialized_optimal_snr

        _print(
            annotate_materialized_optimal_snr(
                args.manifest,
                args.output_dir,
                args.low_frequency,
                args.high_frequency,
                args.psd_segment_seconds,
                args.psd_stride_seconds,
            )
        )
    elif args.command == "injection-score":
        from .injection_score import score_materialized_injections

        _print(
            score_materialized_injections(
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                output_dir=args.output_dir,
                model_ifos=tuple(args.model_ifos),
                q_values=tuple(args.q_values),
                target_sample_rate=args.target_sample_rate,
                save_probabilities=args.save_probabilities,
            )
        )
    elif args.command == "pe-evaluate":
        from .pe import run_pe_evaluation

        _print(run_pe_evaluation(args.manifest, args.output, args.credible_level))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
