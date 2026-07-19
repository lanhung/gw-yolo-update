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
from .search import run_search_benchmark
from .scaling import run_scale_plan
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
    gwosc.add_argument("--allow-locked-evaluation-data", action="store_true")
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
                allow_locked_evaluation_data=args.allow_locked_evaluation_data,
            )
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
