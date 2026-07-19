from __future__ import annotations

import argparse
import json
from typing import Any

from .catalog import evaluate_catalog_predictions
from .config import load_config
from .data import audit_and_split, scan_sources
from .factory import run_data_factory
from .gwosc import run_gwosc_pilot
from .pipeline import run_pipeline
from .prediction import predict_catalog
from .provenance import create_recipe_subset
from .search import run_search_benchmark, run_search_comparison
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

    numeric = subparsers.add_parser("numeric-train")
    numeric.add_argument("--config", required=True)
    numeric.add_argument("--manifest", required=True)
    numeric.add_argument("--output-dir", required=True)

    numeric_evaluate = subparsers.add_parser("numeric-evaluate")
    numeric_evaluate.add_argument("--config", required=True)
    numeric_evaluate.add_argument("--manifest", required=True)
    numeric_evaluate.add_argument("--checkpoint", required=True)
    numeric_evaluate.add_argument("--split", choices=("val", "test"), required=True)
    numeric_evaluate.add_argument("--chirp-threshold", required=True, type=float)
    numeric_evaluate.add_argument("--glitch-threshold", required=True, type=float)
    numeric_evaluate.add_argument("--output", required=True)

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

    curve = subparsers.add_parser("fit-curve")
    curve.add_argument("--points", required=True)
    curve.add_argument("--output", required=True)

    background = subparsers.add_parser("background-plan")
    background.add_argument("--file", action="append", required=True, help="IFO=/path/file.hdf5")
    background.add_argument("--output-dir", required=True)
    background.add_argument("--window-duration", type=int, default=8)
    background.add_argument("--stride", type=int, default=8)
    background.add_argument("--block-duration", type=int, default=256)
    background.add_argument("--required-dq-bits", type=int, default=1)
    background.add_argument("--required-injection-bits", type=int, default=0)
    background.add_argument("--exclude", action="append", default=[], help="GPS_START:GPS_END")
    background.add_argument("--validation-fraction", type=float, default=0.2)
    background.add_argument("--test-fraction", type=float, default=0.2)
    background.add_argument("--seed", type=int, default=20260719)

    deglitch = subparsers.add_parser("oracle-deglitch")
    deglitch.add_argument("--input", required=True)
    deglitch.add_argument("--output", required=True)
    deglitch.add_argument("--report", required=True)
    deglitch.add_argument("--strength", type=float, default=0.9)

    deglitch_benchmark = subparsers.add_parser("oracle-deglitch-benchmark")
    deglitch_benchmark.add_argument("--factory-report", required=True)
    deglitch_benchmark.add_argument("--output", required=True)
    deglitch_benchmark.add_argument("--strength", type=float, default=0.9)

    trigger = subparsers.add_parser("trigger-score")
    trigger.add_argument("--manifest", required=True)
    trigger.add_argument("--checkpoint", required=True)
    trigger.add_argument("--config", required=True)
    trigger.add_argument("--output-dir", required=True)
    trigger.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    trigger.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    trigger.add_argument("--target-sample-rate", type=int, default=1024)
    trigger.add_argument("--context-duration", type=float, default=64.0)

    injection = subparsers.add_parser("injection-plan")
    injection.add_argument("--background-manifest", required=True)
    injection.add_argument("--background-report", required=True)
    injection.add_argument("--output-dir", required=True)
    injection.add_argument("--validation-count", type=int, default=5000)
    injection.add_argument("--test-count", type=int, default=20000)
    injection.add_argument("--seed", type=int, default=20260719)
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
    elif args.command == "numeric-train":
        from .numeric import train_numeric_model

        _print(train_numeric_model(args.config, args.manifest, args.output_dir))
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
                window_duration=args.window_duration,
                stride=args.stride,
                block_duration=args.block_duration,
                required_dq_bits=args.required_dq_bits,
                required_injection_bits=args.required_injection_bits,
                excluded_intervals=exclusions,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
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
    elif args.command == "trigger-score":
        from .trigger import score_background_manifest

        _print(
            score_background_manifest(
                args.manifest,
                args.checkpoint,
                args.config,
                args.output_dir,
                tuple(args.model_ifos),
                tuple(args.q_values),
                args.target_sample_rate,
                args.context_duration,
            )
        )
    elif args.command == "injection-plan":
        from .injections import run_injection_plan

        _print(
            run_injection_plan(
                args.background_manifest,
                args.background_report,
                args.output_dir,
                args.validation_count,
                args.test_count,
                args.seed,
            )
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
