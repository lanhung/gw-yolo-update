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
    run_candidate_search_calibration,
    run_frozen_candidate_search_evaluation,
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

    candidate_search_calibrate = subparsers.add_parser("candidate-search-calibrate")
    candidate_search_calibrate.add_argument("--validation-time-slide-report", required=True)
    candidate_search_calibrate.add_argument(
        "--validation-injection-ranking-report", required=True
    )
    candidate_search_calibrate.add_argument("--target-far-per-year", type=float, required=True)
    candidate_search_calibrate.add_argument("--output", required=True)
    candidate_search_calibrate.add_argument("--bootstrap-replicates", type=int, default=2000)
    candidate_search_calibrate.add_argument("--seed", type=int, default=20260720)

    candidate_search_frozen = subparsers.add_parser("candidate-search-evaluate-frozen")
    candidate_search_frozen.add_argument("--calibration-report", required=True)
    candidate_search_frozen.add_argument("--test-time-slide-report", required=True)
    candidate_search_frozen.add_argument("--test-injection-ranking-report", required=True)
    candidate_search_frozen.add_argument(
        "--minimum-test-live-time-years", type=float, required=True
    )
    candidate_search_frozen.add_argument("--minimum-test-injections", type=int, required=True)
    candidate_search_frozen.add_argument("--bootstrap-replicates", type=int, default=10000)
    candidate_search_frozen.add_argument("--seed", type=int, default=20260721)
    candidate_search_frozen.add_argument("--output", required=True)

    search_validation = subparsers.add_parser("search-validation-injections")
    search_validation.add_argument("--calibration-report", required=True)
    search_validation.add_argument("--validation-injections", required=True)
    search_validation.add_argument("--bootstrap-replicates", type=int, default=2000)
    search_validation.add_argument("--seed", type=int, default=20260719)
    search_validation.add_argument("--output", required=True)

    physical_validation = subparsers.add_parser("physical-validation-endpoint")
    physical_validation.add_argument("--training-report", required=True)
    physical_validation.add_argument("--background-score-report", required=True)
    physical_validation.add_argument("--injection-score-report", required=True)
    physical_validation.add_argument(
        "--maximum-validation-false-alarms", required=True, type=int
    )
    physical_validation.add_argument("--bootstrap-replicates", type=int, default=2000)
    physical_validation.add_argument("--seed", type=int, default=20260719)
    physical_validation.add_argument("--output", required=True)

    physical_validation_summary = subparsers.add_parser(
        "physical-validation-summarize"
    )
    physical_validation_summary.add_argument(
        "--endpoint-report", action="append", required=True
    )
    physical_validation_summary.add_argument("--scale-subset-report", required=True)
    physical_validation_summary.add_argument("--bootstrap-replicates", type=int, default=10000)
    physical_validation_summary.add_argument("--seed", type=int, default=20260720)
    physical_validation_summary.add_argument("--output", required=True)

    detector_subset_summary = subparsers.add_parser(
        "detector-subset-summarize"
    )
    detector_subset_summary.add_argument(
        "--endpoint-report", action="append", required=True
    )
    detector_subset_summary.add_argument(
        "--reference-ifos", nargs="+", default=["H1", "L1", "V1"]
    )
    detector_subset_summary.add_argument(
        "--relative-noninferiority-margin", type=float, default=0.1
    )
    detector_subset_summary.add_argument("--bootstrap-replicates", type=int, default=10000)
    detector_subset_summary.add_argument("--seed", type=int, default=20260720)
    detector_subset_summary.add_argument("--output", required=True)

    physical_endpoint_series = subparsers.add_parser(
        "physical-validation-score-series"
    )
    physical_endpoint_series.add_argument("--training-series-dir", required=True)
    physical_endpoint_series.add_argument("--background-manifest", required=True)
    physical_endpoint_series.add_argument("--injection-manifest", required=True)
    physical_endpoint_series.add_argument("--config", required=True)
    physical_endpoint_series.add_argument("--scale-subset-report", required=True)
    physical_endpoint_series.add_argument("--output-dir", required=True)
    physical_endpoint_series.add_argument(
        "--maximum-validation-false-alarms", required=True, type=int
    )
    physical_endpoint_series.add_argument("--context-duration", type=float, default=64.0)
    physical_endpoint_series.add_argument("--bootstrap-replicates", type=int, default=10000)
    physical_endpoint_series.add_argument("--seed", type=int, default=20260720)

    coherence_compare = subparsers.add_parser("coherence-validation-compare")
    coherence_compare.add_argument("--background-score-report", required=True)
    coherence_compare.add_argument("--injection-score-report", required=True)
    coherence_compare.add_argument(
        "--maximum-validation-false-alarms", required=True, type=int
    )
    coherence_compare.add_argument("--bootstrap-replicates", type=int, default=10000)
    coherence_compare.add_argument("--seed", type=int, default=20260720)
    coherence_compare.add_argument("--output", required=True)

    mask_search = subparsers.add_parser("mask-search-validation")
    mask_search.add_argument("--background-raw", required=True)
    mask_search.add_argument("--background-mask", required=True)
    mask_search.add_argument("--clean-raw", required=True)
    mask_search.add_argument("--clean-mask", required=True)
    mask_search.add_argument("--contaminated-raw", required=True)
    mask_search.add_argument("--contaminated-mask", required=True)
    mask_search.add_argument("--maximum-validation-false-alarms", required=True, type=int)
    mask_search.add_argument("--clean-noninferiority-margin", type=float, default=0.01)
    mask_search.add_argument("--minimum-contaminated-efficiency-gain", type=float, default=0.05)
    mask_search.add_argument("--score-field", default="ranking_score")
    mask_search.add_argument("--bootstrap-replicates", type=int, default=10000)
    mask_search.add_argument("--seed", type=int, default=20260720)
    mask_search.add_argument("--output", required=True)

    mask_search_pipeline = subparsers.add_parser("mask-search-validation-pipeline")
    mask_search_pipeline.add_argument("--background-manifest", required=True)
    mask_search_pipeline.add_argument("--clean-injection-manifest", required=True)
    mask_search_pipeline.add_argument("--contaminated-injection-manifest", required=True)
    mask_search_pipeline.add_argument("--checkpoint", required=True)
    mask_search_pipeline.add_argument("--config", required=True)
    mask_search_pipeline.add_argument("--output-dir", required=True)
    mask_search_pipeline.add_argument(
        "--maximum-validation-false-alarms", required=True, type=int
    )
    mask_search_pipeline.add_argument("--strength", type=float, default=0.9)
    mask_search_pipeline.add_argument("--clean-noninferiority-margin", type=float, default=0.01)
    mask_search_pipeline.add_argument(
        "--minimum-contaminated-efficiency-gain", type=float, default=0.05
    )
    mask_search_pipeline.add_argument("--bootstrap-replicates", type=int, default=10000)
    mask_search_pipeline.add_argument("--seed", type=int, default=20260720)
    mask_search_pipeline.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    mask_search_pipeline.add_argument(
        "--q-values", nargs="+", type=float, default=[4, 8, 16]
    )
    mask_search_pipeline.add_argument("--target-sample-rate", type=int, default=1024)
    mask_search_pipeline.add_argument("--context-duration", type=float, default=64.0)

    split_manifest = subparsers.add_parser("manifest-select-split")
    split_manifest.add_argument("--manifest", required=True)
    split_manifest.add_argument("--split", required=True, choices=["train", "val", "test"])
    split_manifest.add_argument("--output-dir", required=True)

    scaling = subparsers.add_parser("scale-plan")
    scaling.add_argument("--manifest", required=True)
    scaling.add_argument("--output", required=True)
    scaling.add_argument("--baseline-target", type=int, default=10_000)
    scaling.add_argument("--research-target", type=int, default=200_000)
    scaling.add_argument("--seeds", type=int, default=3)

    physical_scale_summary = subparsers.add_parser("physical-scale-summarize")
    physical_scale_summary.add_argument("--scale-subset-report", required=True)
    physical_scale_summary.add_argument("--report", action="append", required=True)
    physical_scale_summary.add_argument("--output", required=True)

    physical_scale_series = subparsers.add_parser("physical-scale-series")
    physical_scale_series.add_argument("--config", required=True)
    physical_scale_series.add_argument("--scale-subset-report", required=True)
    physical_scale_series.add_argument("--pretrained-checkpoint", required=True)
    physical_scale_series.add_argument("--output-dir", required=True)
    physical_scale_series.add_argument("--seed", action="append", type=int, required=True)
    physical_scale_series.add_argument("--validation-feature-cache-dir")

    physical_scale_epoch_series = subparsers.add_parser("physical-scale-epoch-series")
    physical_scale_epoch_series.add_argument("--config", required=True)
    physical_scale_epoch_series.add_argument("--scale-subset-report", required=True)
    physical_scale_epoch_series.add_argument("--pretrained-checkpoint", required=True)
    physical_scale_epoch_series.add_argument("--output-dir", required=True)
    physical_scale_epoch_series.add_argument("--seed", action="append", type=int, required=True)
    physical_scale_epoch_series.add_argument("--validation-feature-cache-dir")

    physical_domain_compare = subparsers.add_parser("physical-data-domain-compare")
    physical_domain_compare.add_argument("--config", required=True)
    physical_domain_compare.add_argument("--data-domain-audit", required=True)
    physical_domain_compare.add_argument(
        "--fixed-update-training", action="append", required=True
    )
    physical_domain_compare.add_argument(
        "--fixed-update-audit", action="append", required=True
    )
    physical_domain_compare.add_argument(
        "--fixed-epoch-training", action="append", required=True
    )
    physical_domain_compare.add_argument(
        "--fixed-epoch-audit", action="append", required=True
    )
    physical_domain_compare.add_argument("--output", required=True)

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

    gwosc_plan_extend = subparsers.add_parser("gwosc-plan-extend")
    gwosc_plan_extend.add_argument("--base-plan", required=True)
    gwosc_plan_extend.add_argument("--target-pairs", type=int, required=True)
    gwosc_plan_extend.add_argument("--extension-seed", type=int)
    gwosc_plan_extend.add_argument("--output", required=True)

    gwosc_plan_disjoint = subparsers.add_parser("gwosc-plan-disjoint")
    gwosc_plan_disjoint.add_argument("--run", required=True)
    gwosc_plan_disjoint.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    gwosc_plan_disjoint.add_argument("--sample-rate-khz", type=int, default=4)
    gwosc_plan_disjoint.add_argument("--exclude-plan", action="append", required=True)
    gwosc_plan_disjoint.add_argument("--target-pairs", type=int, required=True)
    gwosc_plan_disjoint.add_argument("--seed", type=int, default=20260727)
    gwosc_plan_disjoint.add_argument("--output", required=True)

    gwosc_plan_shard = subparsers.add_parser("gwosc-plan-shard")
    gwosc_plan_shard.add_argument("--plan", required=True)
    gwosc_plan_shard.add_argument("--shard-index", type=int, required=True)
    gwosc_plan_shard.add_argument("--pairs-per-shard", type=int, default=1)
    gwosc_plan_shard.add_argument("--output", required=True)

    gwosc_batch = subparsers.add_parser("gwosc-batch-download")
    gwosc_batch.add_argument("--plan", required=True)
    gwosc_batch.add_argument("--cache-dir", required=True)
    gwosc_batch.add_argument("--output-dir", required=True)
    gwosc_batch.add_argument("--maximum-pairs", type=int)
    gwosc_batch.add_argument("--download-workers", type=int, default=8)
    gwosc_batch.add_argument("--chunk-samples", type=int, default=1_048_576)
    gwosc_batch.add_argument(
        "--verified-source-inventory", action="append", default=[]
    )

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
    physical_finetune.add_argument("--validation-feature-cache-dir")

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

    scale_subsets = subparsers.add_parser("physical-scale-subsets")
    scale_subsets.add_argument("--manifest", required=True)
    scale_subsets.add_argument("--validation-manifest", required=True)
    scale_subsets.add_argument("--output-dir", required=True)
    scale_subsets.add_argument("--scale", action="append", type=int)
    scale_subsets.add_argument("--seed", type=int, default=20260720)

    physical_audit = subparsers.add_parser("physical-checkpoint-audit")
    physical_audit.add_argument("--config", required=True)
    physical_audit.add_argument("--validation-manifest", required=True)
    physical_audit.add_argument("--checkpoint", required=True)
    physical_audit.add_argument("--chirp-threshold", type=float, required=True)
    physical_audit.add_argument("--output", required=True)

    physical_timing = subparsers.add_parser("physical-timing-train")
    physical_timing.add_argument("--config", required=True)
    physical_timing.add_argument("--train-manifest", required=True)
    physical_timing.add_argument("--validation-manifest", required=True)
    physical_timing.add_argument("--pretrained-checkpoint", required=True)
    physical_timing.add_argument("--output-dir", required=True)
    physical_timing.add_argument("--seed", type=int)

    detector_arrival_timing = subparsers.add_parser("detector-arrival-timing-train")
    detector_arrival_timing.add_argument("--config", required=True)
    detector_arrival_timing.add_argument("--train-manifest", required=True)
    detector_arrival_timing.add_argument("--validation-manifest", required=True)
    detector_arrival_timing.add_argument("--output-dir", required=True)
    detector_arrival_timing.add_argument("--seed", type=int)

    endpoint_proposal = subparsers.add_parser("detector-endpoint-proposal-train")
    endpoint_proposal.add_argument("--config", required=True)
    endpoint_proposal.add_argument("--train-manifest", required=True)
    endpoint_proposal.add_argument("--validation-manifest", required=True)
    endpoint_proposal.add_argument("--pretrained-checkpoint", required=True)
    endpoint_proposal.add_argument("--output-dir", required=True)
    endpoint_proposal.add_argument("--seed", type=int)

    endpoint_proposal_evaluate = subparsers.add_parser(
        "detector-endpoint-proposal-evaluate"
    )
    endpoint_proposal_evaluate.add_argument("--config", required=True)
    endpoint_proposal_evaluate.add_argument("--validation-manifest", required=True)
    endpoint_proposal_evaluate.add_argument("--checkpoint", required=True)
    endpoint_proposal_evaluate.add_argument("--output-dir", required=True)

    endpoint_proposal_apply = subparsers.add_parser(
        "detector-endpoint-proposal-apply"
    )
    endpoint_proposal_apply.add_argument("--config", required=True)
    endpoint_proposal_apply.add_argument("--manifest", required=True)
    endpoint_proposal_apply.add_argument("--checkpoint", required=True)
    endpoint_proposal_apply.add_argument("--threshold", type=float, required=True)
    endpoint_proposal_apply.add_argument(
        "--required-split", choices=("train", "val"), required=True
    )
    endpoint_proposal_apply.add_argument("--shard-size", type=int, default=256)
    endpoint_proposal_apply.add_argument("--output-dir", required=True)

    candidate_refiner_plan = subparsers.add_parser("candidate-refiner-plan")
    candidate_refiner_plan.add_argument("--train-injection-manifest", required=True)
    candidate_refiner_plan.add_argument("--train-candidate-manifest", required=True)
    candidate_refiner_plan.add_argument(
        "--validation-injection-manifest", required=True
    )
    candidate_refiner_plan.add_argument(
        "--validation-candidate-manifest", required=True
    )
    candidate_refiner_plan.add_argument("--output-dir", required=True)
    candidate_refiner_plan.add_argument(
        "--positive-padding-seconds", type=float, default=0.5
    )
    candidate_refiner_plan.add_argument(
        "--validation-selection-fraction", type=float, default=0.2
    )
    candidate_refiner_plan.add_argument("--seed", type=int, default=20260720)

    candidate_refiner_train = subparsers.add_parser("candidate-refiner-train")
    candidate_refiner_train.add_argument("--config", required=True)
    candidate_refiner_train.add_argument("--train-injection-manifest", required=True)
    candidate_refiner_train.add_argument("--train-candidate-manifest", required=True)
    candidate_refiner_train.add_argument(
        "--validation-injection-manifest", required=True
    )
    candidate_refiner_train.add_argument(
        "--validation-selection-candidate-manifest", required=True
    )
    candidate_refiner_train.add_argument(
        "--validation-calibration-candidate-manifest", required=True
    )
    candidate_refiner_train.add_argument("--output-dir", required=True)
    candidate_refiner_train.add_argument("--seed", type=int)
    candidate_refiner_train.add_argument("--pretrained-endpoint-checkpoint")

    candidate_refiner_validation = subparsers.add_parser(
        "candidate-refiner-validation"
    )
    candidate_refiner_validation.add_argument("--config", required=True)
    candidate_refiner_validation.add_argument("--checkpoint", required=True)
    candidate_refiner_validation.add_argument(
        "--validation-injection-manifest", required=True
    )
    candidate_refiner_validation.add_argument(
        "--validation-candidate-manifest", required=True
    )
    candidate_refiner_validation.add_argument("--output-dir", required=True)

    candidate_network_set = subparsers.add_parser("candidate-network-set-audit")
    candidate_network_set.add_argument("--config", required=True)
    candidate_network_set.add_argument("--injection-manifest", required=True)
    candidate_network_set.add_argument("--candidate-manifest", required=True)
    candidate_network_set.add_argument("--output-dir", required=True)

    candidate_pair_ranker = subparsers.add_parser("candidate-pair-ranker-train")
    candidate_pair_ranker.add_argument("--config", required=True)
    candidate_pair_ranker.add_argument("--train-injection-manifest", required=True)
    candidate_pair_ranker.add_argument("--train-candidate-manifest", required=True)
    candidate_pair_ranker.add_argument(
        "--validation-injection-manifest", required=True
    )
    candidate_pair_ranker.add_argument(
        "--validation-selection-candidate-manifest", required=True
    )
    candidate_pair_ranker.add_argument("--output-dir", required=True)
    candidate_pair_ranker.add_argument("--seed", type=int)

    candidate_pair_scaling = subparsers.add_parser("candidate-pair-scaling-plan")
    candidate_pair_scaling.add_argument("--train-injection-manifest", required=True)
    candidate_pair_scaling.add_argument("--train-candidate-manifest", required=True)
    candidate_pair_scaling.add_argument(
        "--scale-manifest", action="append", required=True, metavar="SIZE=PATH"
    )
    candidate_pair_scaling.add_argument("--output-dir", required=True)

    candidate_pair_scaling_evaluate = subparsers.add_parser(
        "candidate-pair-scaling-evaluate"
    )
    candidate_pair_scaling_evaluate.add_argument("--config", required=True)
    candidate_pair_scaling_evaluate.add_argument(
        "--scaling-plan-report", required=True
    )
    candidate_pair_scaling_evaluate.add_argument(
        "--fixed-update-report", action="append", required=True, metavar="SIZE=PATH"
    )
    candidate_pair_scaling_evaluate.add_argument(
        "--fixed-epoch-report", action="append", required=True, metavar="SIZE=PATH"
    )
    candidate_pair_scaling_evaluate.add_argument("--output", required=True)

    candidate_pair_representation_evaluate = subparsers.add_parser(
        "candidate-pair-representation-evaluate"
    )
    candidate_pair_representation_evaluate.add_argument("--config", required=True)
    candidate_pair_representation_evaluate.add_argument(
        "--baseline-report", required=True
    )
    candidate_pair_representation_evaluate.add_argument(
        "--candidate-report", required=True
    )
    candidate_pair_representation_evaluate.add_argument("--output", required=True)

    detector_arrival_stratify = subparsers.add_parser(
        "detector-arrival-timing-validation-stratify"
    )
    detector_arrival_stratify.add_argument("--config", required=True)
    detector_arrival_stratify.add_argument("--validation-manifest", required=True)
    detector_arrival_stratify.add_argument("--checkpoint", required=True)
    detector_arrival_stratify.add_argument("--output", required=True)
    detector_arrival_stratify.add_argument("--predictions-output", required=True)

    detector_arrival_compare = subparsers.add_parser(
        "detector-arrival-timing-validation-compare"
    )
    detector_arrival_compare.add_argument("--config", required=True)
    detector_arrival_compare.add_argument("--reference-predictions", required=True)
    detector_arrival_compare.add_argument("--candidate-predictions", required=True)
    detector_arrival_compare.add_argument("--output", required=True)

    glitch_finetune = subparsers.add_parser("gravityspy-glitch-finetune")
    glitch_finetune.add_argument("--config", required=True)
    glitch_finetune.add_argument("--glitch-train-manifest", required=True)
    glitch_finetune.add_argument("--glitch-validation-manifest", required=True)
    glitch_finetune.add_argument("--chirp-validation-manifest", required=True)
    glitch_finetune.add_argument("--pretrained-checkpoint", required=True)
    glitch_finetune.add_argument("--output-dir", required=True)
    glitch_finetune.add_argument("--seed", type=int)

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

    gravityspy_network = subparsers.add_parser("gravityspy-network-strain-plan")
    gravityspy_network.add_argument("--manifest", required=True)
    gravityspy_network.add_argument("--output-dir", required=True)
    gravityspy_network.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"])
    gravityspy_network.add_argument("--sample-rate-khz", type=int, default=4)
    gravityspy_network.add_argument("--context-duration", type=float, default=64.0)
    gravityspy_network.add_argument("--minimum-detectors", type=int, default=2)

    gravityspy_network_materialize = subparsers.add_parser(
        "gravityspy-network-strain-materialize"
    )
    gravityspy_network_materialize.add_argument("--manifest", required=True)
    gravityspy_network_materialize.add_argument("--config", required=True)
    gravityspy_network_materialize.add_argument("--cache-dir", required=True)
    gravityspy_network_materialize.add_argument("--output-dir", required=True)
    gravityspy_network_materialize.add_argument("--output-duration", type=float, default=8.0)
    gravityspy_network_materialize.add_argument("--download-workers", type=int, default=8)
    gravityspy_network_materialize.add_argument("--chunk-samples", type=int, default=1_048_576)
    gravityspy_network_materialize.add_argument("--shard", type=int)
    gravityspy_network_materialize.add_argument(
        "--verified-source-inventory", action="append", default=[]
    )

    gravityspy_network_recovery = subparsers.add_parser(
        "gravityspy-network-recovery-plan"
    )
    gravityspy_network_recovery.add_argument("--source-manifest", required=True)
    gravityspy_network_recovery.add_argument(
        "--materialization-report", action="append", required=True
    )
    gravityspy_network_recovery.add_argument("--output-dir", required=True)

    gravityspy_network_select = subparsers.add_parser(
        "gravityspy-network-strain-select"
    )
    gravityspy_network_select.add_argument("--manifest", required=True)
    gravityspy_network_select.add_argument("--output-dir", required=True)
    gravityspy_network_select.add_argument("--per-label", type=int, required=True)
    gravityspy_network_select.add_argument(
        "--maximum-source-files", type=int, required=True
    )
    gravityspy_network_select.add_argument("--seed", type=int, default=20260720)
    gravityspy_network_select.add_argument("--existing-manifest")

    gravityspy_network_shard = subparsers.add_parser("gravityspy-network-strain-shard")
    gravityspy_network_shard.add_argument("--manifest", required=True)
    gravityspy_network_shard.add_argument("--output-dir", required=True)
    gravityspy_network_shard.add_argument("--files-per-shard", type=int, default=16)
    gravityspy_network_shard.add_argument("--seed", type=int, default=20260720)

    gravityspy_select = subparsers.add_parser("gravityspy-strain-select")
    gravityspy_select.add_argument("--manifest", required=True)
    gravityspy_select.add_argument("--output-dir", required=True)
    gravityspy_select.add_argument("--per-label", type=int, required=True)
    gravityspy_select.add_argument("--maximum-files", type=int, required=True)
    gravityspy_select.add_argument("--seed", type=int, default=20260720)
    gravityspy_select.add_argument("--existing-manifest")

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

    gravityspy_network_merge = subparsers.add_parser("gravityspy-network-numeric-merge")
    gravityspy_network_merge.add_argument("--report", action="append", required=True)
    gravityspy_network_merge.add_argument("--output-dir", required=True)
    gravityspy_network_merge.add_argument(
        "--split", choices=("train", "val", "test"), required=True
    )

    gravityspy_network_corpus_audit = subparsers.add_parser(
        "gravityspy-network-corpus-audit"
    )
    gravityspy_network_corpus_audit.add_argument("--train-report", required=True)
    gravityspy_network_corpus_audit.add_argument("--validation-report", required=True)
    gravityspy_network_corpus_audit.add_argument("--output", required=True)

    gravityspy_network_resplit = subparsers.add_parser(
        "gravityspy-network-corpus-resplit"
    )
    gravityspy_network_resplit.add_argument("--report", action="append", required=True)
    gravityspy_network_resplit.add_argument("--output-dir", required=True)
    gravityspy_network_resplit.add_argument("--validation-fraction", type=float, default=0.2)
    gravityspy_network_resplit.add_argument("--seed", type=int, default=20260720)

    gravityspy_evict = subparsers.add_parser("gravityspy-strain-evict")
    gravityspy_evict.add_argument("--materialization-report", required=True)
    gravityspy_evict.add_argument("--cache-dir", required=True)
    gravityspy_evict.add_argument("--output", required=True)

    physical_overlap = subparsers.add_parser("physical-overlap-materialize")
    physical_overlap.add_argument("--gravityspy-manifest", required=True)
    physical_overlap.add_argument("--injection-manifest", required=True)
    physical_overlap.add_argument("--config", required=True)
    physical_overlap.add_argument("--output-dir", required=True)
    physical_overlap.add_argument("--split", required=True, choices=["train", "val", "test"])
    physical_overlap.add_argument("--seed", type=int, default=20260720)
    physical_overlap.add_argument("--limit", type=int)
    physical_overlap.add_argument("--gravityspy-corpus-audit")

    physical_overlap_audit = subparsers.add_parser("physical-overlap-audit")
    physical_overlap_audit.add_argument("--manifest", action="append", required=True)
    physical_overlap_audit.add_argument("--output", required=True)

    physical_overlap_contamination = subparsers.add_parser(
        "physical-overlap-contamination"
    )
    physical_overlap_contamination.add_argument("--overlap-manifest", required=True)
    physical_overlap_contamination.add_argument("--injection-manifest", required=True)
    physical_overlap_contamination.add_argument("--output-dir", required=True)
    physical_overlap_contamination.add_argument(
        "--required-split", required=True, choices=["train", "val", "test"]
    )

    physical_overlap_train = subparsers.add_parser("physical-overlap-finetune")
    physical_overlap_train.add_argument("--config", required=True)
    physical_overlap_train.add_argument("--overlap-train-manifest", required=True)
    physical_overlap_train.add_argument("--overlap-validation-manifest", required=True)
    physical_overlap_train.add_argument("--clean-train-manifest", required=True)
    physical_overlap_train.add_argument("--clean-validation-manifest", required=True)
    physical_overlap_train.add_argument("--pretrained-checkpoint", required=True)
    physical_overlap_train.add_argument("--output-dir", required=True)
    physical_overlap_train.add_argument("--seed", type=int)
    physical_overlap_train.add_argument("--clean-validation-feature-cache-dir")

    overlap_sampling_promotion = subparsers.add_parser(
        "physical-overlap-sampling-promote"
    )
    overlap_sampling_promotion.add_argument("--uniform-report", required=True)
    overlap_sampling_promotion.add_argument("--family-balanced-report", required=True)
    overlap_sampling_promotion.add_argument("--overlap-train-manifest", required=True)
    overlap_sampling_promotion.add_argument("--overlap-validation-manifest", required=True)
    overlap_sampling_promotion.add_argument("--gravityspy-corpus-audit", required=True)
    overlap_sampling_promotion.add_argument("--config", required=True)
    overlap_sampling_promotion.add_argument("--output", required=True)

    overlap_five_seed_summary = subparsers.add_parser(
        "physical-overlap-five-seed-summarize"
    )
    overlap_five_seed_summary.add_argument("--promotion-report", required=True)
    overlap_five_seed_summary.add_argument("--report", action="append", required=True)
    overlap_five_seed_summary.add_argument("--output", required=True)

    mask_audit_plan = subparsers.add_parser("gravityspy-mask-audit-plan")
    mask_audit_plan.add_argument("--manifest", required=True)
    mask_audit_plan.add_argument("--output-dir", required=True)
    mask_audit_plan.add_argument("--per-label", type=int, default=5)
    mask_audit_plan.add_argument("--seed", type=int, default=20260720)

    mask_audit_evaluate = subparsers.add_parser("gravityspy-mask-audit-evaluate")
    mask_audit_evaluate.add_argument("--tasks", required=True)
    mask_audit_evaluate.add_argument("--annotations", required=True)
    mask_audit_evaluate.add_argument("--output", required=True)

    mask_consensus = subparsers.add_parser("gravityspy-mask-consensus-materialize")
    mask_consensus.add_argument("--tasks", required=True)
    mask_consensus.add_argument("--annotations", required=True)
    mask_consensus.add_argument("--audit-report", required=True)
    mask_consensus.add_argument("--output-dir", required=True)

    mask_predict = subparsers.add_parser("gravityspy-mask-segmentation-predict")
    mask_predict.add_argument("--gold-report", required=True)
    mask_predict.add_argument("--selection-report", required=True)
    mask_predict.add_argument("--config", required=True)
    mask_predict.add_argument("--output-dir", required=True)

    mask_segmentation = subparsers.add_parser("gravityspy-mask-segmentation-evaluate")
    mask_segmentation.add_argument("--gold-report", required=True)
    mask_segmentation.add_argument("--predictions", required=True)
    mask_segmentation.add_argument("--output", required=True)
    mask_segmentation.add_argument("--bootstrap-replicates", type=int, default=10000)
    mask_segmentation.add_argument("--bootstrap-seed", type=int, default=20260720)

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
    background.add_argument(
        "--split-strategy",
        choices=("balanced_rank_v1", "hash_threshold_v1"),
        default="balanced_rank_v1",
    )

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
    background_batch.add_argument(
        "--split-strategy",
        choices=("balanced_rank_v1", "hash_threshold_v1"),
        default="balanced_rank_v1",
    )

    background_disjoint = subparsers.add_parser("background-disjoint-subset")
    background_disjoint.add_argument("--background-manifest", required=True)
    background_disjoint.add_argument("--background-report", required=True)
    background_disjoint.add_argument("--exclude-manifest", action="append", required=True)
    background_disjoint.add_argument("--output-dir", required=True)
    background_disjoint.add_argument("--split", choices=("train", "val"), default="val")

    background_purpose = subparsers.add_parser("background-purpose-partition")
    background_purpose.add_argument("--background-manifest", required=True)
    background_purpose.add_argument("--background-report", required=True)
    background_purpose.add_argument("--output-dir", required=True)
    background_purpose.add_argument("--injection-fraction", type=float, default=0.5)
    background_purpose.add_argument("--seed", type=int, default=20260725)

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

    learned_background_deglitch = subparsers.add_parser("learned-background-deglitch")
    learned_background_deglitch.add_argument("--background-manifest", required=True)
    learned_background_deglitch.add_argument("--scored-manifest", required=True)
    learned_background_deglitch.add_argument("--output-dir", required=True)
    learned_background_deglitch.add_argument("--strength", type=float, default=0.9)
    learned_background_deglitch.add_argument(
        "--model-ifos", nargs="+", default=["H1", "L1", "V1"]
    )
    learned_background_deglitch.add_argument("--target-sample-rate", type=int, default=1024)
    learned_background_deglitch.add_argument("--context-duration", type=float, default=64.0)
    learned_background_deglitch.add_argument(
        "--required-split", choices=["train", "val", "test"]
    )

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
    trigger.add_argument("--required-split", choices=["train", "val", "test"])
    trigger.add_argument("--enabled-ifos", nargs="+", choices=["H1", "L1", "V1"])
    trigger.add_argument("--coherence-config")

    candidates = subparsers.add_parser("candidate-extract")
    candidates.add_argument("--triggers", required=True)
    candidates.add_argument("--output-dir", required=True)
    candidates.add_argument("--chirp-threshold", type=float, default=0.3)
    candidates.add_argument("--minimum-bins", type=int, default=1)

    timing_calibration = subparsers.add_parser("candidate-timing-calibrate")
    timing_calibration.add_argument("--injection-triggers", required=True)
    timing_calibration.add_argument("--output", required=True)
    timing_calibration.add_argument("--chirp-threshold", type=float, default=0.3)
    timing_calibration.add_argument("--minimum-bins", type=int, default=1)
    timing_calibration.add_argument("--association-window-seconds", type=float, default=0.25)
    timing_calibration.add_argument("--uncertainty-quantile", type=float, default=0.99)
    timing_calibration.add_argument("--minimum-matches-per-method", type=int, default=30)
    timing_calibration.add_argument(
        "--maximum-empirical-timing-uncertainty-seconds", type=float, default=0.01
    )

    timing_apply = subparsers.add_parser("candidate-timing-apply")
    timing_apply.add_argument("--candidates", required=True)
    timing_apply.add_argument("--calibration-report", required=True)
    timing_apply.add_argument("--output", required=True)

    injection_candidates = subparsers.add_parser("injection-candidate-extract")
    injection_candidates.add_argument("--injection-triggers", required=True)
    injection_candidates.add_argument("--output-dir", required=True)
    injection_candidates.add_argument("--chirp-threshold", type=float, default=0.3)
    injection_candidates.add_argument("--minimum-bins", type=int, default=1)

    proposal_audit = subparsers.add_parser("candidate-proposal-audit")
    proposal_audit.add_argument("--injection-manifest", required=True)
    proposal_audit.add_argument("--candidate-manifest", required=True)
    proposal_audit.add_argument("--output", required=True)
    proposal_audit.add_argument("--padding-seconds", type=float, default=0.5)

    proposal_select = subparsers.add_parser("candidate-proposal-sweep-select")
    proposal_select.add_argument("--config", required=True)
    proposal_select.add_argument("--audit-report", action="append", required=True)
    proposal_select.add_argument("--output", required=True)

    injection_candidate_rank = subparsers.add_parser("injection-candidate-rank")
    injection_candidate_rank.add_argument("--injection-triggers", required=True)
    injection_candidate_rank.add_argument("--candidates", required=True)
    injection_candidate_rank.add_argument("--output-dir", required=True)
    injection_candidate_rank.add_argument("--split", choices=("val", "test"), required=True)
    injection_candidate_rank.add_argument("--reference-ifo", default="H1")
    injection_candidate_rank.add_argument("--second-ifo", default="L1")
    injection_candidate_rank.add_argument("--physical-delay-limit-seconds", type=float, required=True)
    injection_candidate_rank.add_argument(
        "--empirical-timing-uncertainty-seconds", type=float, required=True
    )
    injection_candidate_rank.add_argument(
        "--truth-association-window-seconds", type=float, default=0.25
    )

    candidate_slides = subparsers.add_parser("candidate-time-slides")
    candidate_slides.add_argument("--candidates", required=True)
    candidate_slides.add_argument("--background-manifest", required=True)
    candidate_slides.add_argument("--output-dir", required=True)
    candidate_slides.add_argument("--split", choices=("val", "test"), required=True)
    candidate_slides.add_argument("--reference-ifo", default="H1")
    candidate_slides.add_argument("--shifted-ifo", default="L1")
    candidate_slides.add_argument("--slide-count", type=int, required=True)
    candidate_slides.add_argument("--slide-start-index", type=int, default=1)
    candidate_slides.add_argument("--slide-schedule")
    candidate_slides.add_argument("--schedule-offset", type=int, default=0)
    candidate_slides.add_argument("--step-seconds", type=float, required=True)
    candidate_slides.add_argument("--coincidence-window-seconds", type=float, required=True)
    candidate_slides.add_argument("--cluster-window-seconds", type=float, default=0.1)
    candidate_slides.add_argument("--physical-delay-limit-seconds", type=float)
    candidate_slides.add_argument("--empirical-timing-uncertainty-seconds", type=float)

    candidate_block_permutations = subparsers.add_parser("candidate-block-permutations")
    candidate_block_permutations.add_argument("--candidates", required=True)
    candidate_block_permutations.add_argument("--background-manifest", required=True)
    candidate_block_permutations.add_argument("--schedule", required=True)
    candidate_block_permutations.add_argument("--output-dir", required=True)
    candidate_block_permutations.add_argument(
        "--split", choices=("val", "test"), required=True
    )
    candidate_block_permutations.add_argument("--reference-ifo", default="H1")
    candidate_block_permutations.add_argument("--shifted-ifo", default="L1")
    candidate_block_permutations.add_argument(
        "--coincidence-window-seconds", type=float, required=True
    )
    candidate_block_permutations.add_argument(
        "--cluster-window-seconds", type=float, default=0.1
    )
    candidate_block_permutations.add_argument(
        "--physical-delay-limit-seconds", type=float, required=True
    )
    candidate_block_permutations.add_argument(
        "--empirical-timing-uncertainty-seconds", type=float, required=True
    )

    candidate_slide_merge = subparsers.add_parser("candidate-time-slide-merge")
    candidate_slide_merge.add_argument("--report", action="append", required=True)
    candidate_slide_merge.add_argument("--output-dir", required=True)
    candidate_slide_merge.add_argument("--split", choices=("val", "test"), required=True)

    candidate_slide_schedule = subparsers.add_parser(
        "candidate-time-slide-schedule-freeze"
    )
    candidate_slide_schedule.add_argument("--background-manifest", required=True)
    candidate_slide_schedule.add_argument("--output", required=True)
    candidate_slide_schedule.add_argument(
        "--split", choices=("val", "test"), required=True
    )
    candidate_slide_schedule.add_argument("--reference-ifo", default="H1")
    candidate_slide_schedule.add_argument("--shifted-ifo", default="L1")
    candidate_slide_schedule.add_argument("--step-seconds", type=float, required=True)
    candidate_slide_schedule.add_argument(
        "--slide-index", nargs="+", type=int, required=True
    )
    candidate_slide_schedule.add_argument(
        "--target-far-per-year", type=float, required=True
    )
    candidate_slide_schedule.add_argument(
        "--zero-count-confidence", type=float, default=0.90
    )

    candidate_slide_range_schedule = subparsers.add_parser(
        "candidate-time-slide-range-schedule-freeze"
    )
    candidate_slide_range_schedule.add_argument("--background-manifest", required=True)
    candidate_slide_range_schedule.add_argument("--output", required=True)
    candidate_slide_range_schedule.add_argument(
        "--split", choices=("val", "test"), required=True
    )
    candidate_slide_range_schedule.add_argument("--reference-ifo", default="H1")
    candidate_slide_range_schedule.add_argument("--shifted-ifo", default="L1")
    candidate_slide_range_schedule.add_argument("--step-seconds", type=float, required=True)
    candidate_slide_range_schedule.add_argument(
        "--slide-start-index", type=int, default=1
    )
    candidate_slide_range_schedule.add_argument(
        "--slide-stop-index-exclusive", type=int, required=True
    )
    candidate_slide_range_schedule.add_argument(
        "--target-far-per-year", type=float, required=True
    )
    candidate_slide_range_schedule.add_argument(
        "--zero-count-confidence", type=float, default=0.90
    )

    candidate_block_schedule = subparsers.add_parser(
        "candidate-block-permutation-schedule-freeze"
    )
    candidate_block_schedule.add_argument("--background-manifest", required=True)
    candidate_block_schedule.add_argument("--output", required=True)
    candidate_block_schedule.add_argument(
        "--split", choices=("val", "test"), required=True
    )
    candidate_block_schedule.add_argument("--reference-ifo", default="H1")
    candidate_block_schedule.add_argument("--shifted-ifo", default="L1")
    candidate_block_schedule.add_argument(
        "--target-far-per-year", type=float, required=True
    )
    candidate_block_schedule.add_argument(
        "--zero-count-confidence", type=float, default=0.90
    )
    candidate_block_schedule.add_argument("--maximum-shifts", type=int)

    candidate_block_capacity = subparsers.add_parser(
        "candidate-block-permutation-capacity-forecast"
    )
    candidate_block_capacity.add_argument("--pilot-schedule", required=True)
    candidate_block_capacity.add_argument("--pilot-background-report", required=True)
    candidate_block_capacity.add_argument("--planned-parent-plan", required=True)
    candidate_block_capacity.add_argument("--output", required=True)
    candidate_block_capacity.add_argument("--safety-factor", type=float, default=1.5)
    candidate_block_capacity.add_argument("--allow-insufficient", action="store_true")

    candidate_block_extension = subparsers.add_parser(
        "candidate-block-permutation-capacity-extension-freeze"
    )
    candidate_block_extension.add_argument("--base-forecast", required=True)
    candidate_block_extension.add_argument("--extended-plan", required=True)
    candidate_block_extension.add_argument("--extended-forecast", required=True)
    candidate_block_extension.add_argument("--output", required=True)

    candidate_pipeline = subparsers.add_parser("candidate-search-validation-pipeline")
    candidate_pipeline.add_argument("--background-manifest", required=True)
    candidate_pipeline.add_argument("--injection-manifest", required=True)
    candidate_pipeline.add_argument("--checkpoint", required=True)
    candidate_pipeline.add_argument("--config", required=True)
    candidate_pipeline.add_argument("--coherence-config", required=True)
    candidate_pipeline.add_argument("--output-dir", required=True)
    candidate_pipeline.add_argument("--reference-ifo", default="H1")
    candidate_pipeline.add_argument("--second-ifo", default="L1")
    candidate_pipeline.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    candidate_pipeline.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    candidate_pipeline.add_argument("--target-sample-rate", type=int, default=1024)
    candidate_pipeline.add_argument("--context-duration", type=float, default=64.0)
    candidate_pipeline.add_argument("--chirp-threshold", type=float, default=0.3)
    candidate_pipeline.add_argument("--minimum-bins", type=int, default=1)
    candidate_pipeline.add_argument(
        "--timing-association-window-seconds", type=float, default=0.25
    )
    candidate_pipeline.add_argument("--timing-uncertainty-quantile", type=float, default=0.99)
    candidate_pipeline.add_argument("--minimum-timing-matches", type=int, default=30)
    candidate_pipeline.add_argument(
        "--maximum-timing-uncertainty-seconds", type=float, default=0.01
    )
    candidate_pipeline.add_argument(
        "--truth-association-window-seconds", type=float, default=0.25
    )
    candidate_pipeline.add_argument("--slide-count", type=int, default=512)
    candidate_pipeline.add_argument("--slide-step-seconds", type=float, default=8.0)
    candidate_pipeline.add_argument("--cluster-window-seconds", type=float, default=0.1)
    candidate_pipeline.add_argument("--target-far-per-year", type=float, default=100.0)
    candidate_pipeline.add_argument("--bootstrap-replicates", type=int, default=10000)
    candidate_pipeline.add_argument("--seed", type=int, default=20260720)
    candidate_pipeline.add_argument("--model-selection-report")

    candidate_pipeline_compare = subparsers.add_parser(
        "candidate-search-validation-compare"
    )
    candidate_pipeline_compare.add_argument("--baseline-report", required=True)
    candidate_pipeline_compare.add_argument("--promoted-report", required=True)
    candidate_pipeline_compare.add_argument("--config", required=True)
    candidate_pipeline_compare.add_argument("--output", required=True)

    candidate_pipeline_block = subparsers.add_parser(
        "candidate-search-validation-block-recalibrate"
    )
    candidate_pipeline_block.add_argument("--pipeline-report", required=True)
    candidate_pipeline_block.add_argument("--background-manifest", required=True)
    candidate_pipeline_block.add_argument(
        "--calibrated-candidate-manifest", required=True
    )
    candidate_pipeline_block.add_argument("--injection-ranking-report", required=True)
    candidate_pipeline_block.add_argument("--output-dir", required=True)
    candidate_pipeline_block.add_argument(
        "--zero-count-confidence", type=float, default=0.90
    )

    exposure_plan = subparsers.add_parser("candidate-exposure-plan")
    exposure_plan.add_argument("--background-manifest", required=True)
    exposure_plan.add_argument("--output", required=True)
    exposure_plan.add_argument("--split", choices=("val", "test"), required=True)
    exposure_plan.add_argument("--reference-ifo", default="H1")
    exposure_plan.add_argument("--shifted-ifo", default="L1")
    exposure_plan.add_argument("--slide-count", type=int, required=True)
    exposure_plan.add_argument("--slide-start-index", type=int, default=1)
    exposure_plan.add_argument("--step-seconds", type=float, required=True)
    exposure_plan.add_argument("--target-far-per-year", type=float, required=True)
    exposure_plan.add_argument("--zero-count-confidence", type=float, default=0.90)

    probability_evict = subparsers.add_parser("candidate-probability-evict")
    probability_evict.add_argument("--candidate-report", required=True)
    probability_evict.add_argument("--score-report", required=True)
    probability_evict.add_argument("--probability-root", required=True)
    probability_evict.add_argument("--output", required=True)

    source_evict = subparsers.add_parser("background-source-evict")
    source_evict.add_argument("--batch-report", required=True)
    source_evict.add_argument("--background-report", required=True)
    source_evict.add_argument("--score-report", action="append", default=[])
    source_evict.add_argument("--candidate-report", action="append", default=[])
    source_evict.add_argument("--cache-root", required=True)
    source_evict.add_argument("--output", required=True)

    amplfi_source_evict = subparsers.add_parser("amplfi-background-source-evict")
    amplfi_source_evict.add_argument("--batch-report", required=True)
    amplfi_source_evict.add_argument("--background-report", required=True)
    amplfi_source_evict.add_argument("--export-report", required=True)
    amplfi_source_evict.add_argument("--cache-root", required=True)
    amplfi_source_evict.add_argument("--output", required=True)

    stream_shard = subparsers.add_parser("background-stream-shard")
    stream_shard.add_argument("--parent-plan", required=True)
    stream_shard.add_argument("--event-exclusions", required=True)
    stream_shard.add_argument("--timing-calibration-report", required=True)
    stream_shard.add_argument("--checkpoint", required=True)
    stream_shard.add_argument("--config", required=True)
    stream_shard.add_argument("--coherence-config", required=True)
    stream_shard.add_argument("--cache-root", required=True)
    stream_shard.add_argument("--output-dir", required=True)
    stream_shard.add_argument("--shard-index", type=int, required=True)
    stream_shard.add_argument("--pairs-per-shard", type=int, default=1)
    stream_shard.add_argument("--validation-fraction", type=float, default=0.2)
    stream_shard.add_argument("--test-fraction", type=float, default=0.2)
    stream_shard.add_argument("--seed", type=int, default=20260720)
    stream_shard.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    stream_shard.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    stream_shard.add_argument("--target-sample-rate", type=int, default=1024)
    stream_shard.add_argument("--context-duration", type=float, default=64.0)
    stream_shard.add_argument("--chirp-threshold", type=float, default=0.3)
    stream_shard.add_argument("--minimum-bins", type=int, default=1)
    stream_shard.add_argument("--download-workers", type=int, default=8)
    stream_shard.add_argument(
        "--verified-source-inventory", action="append", default=[]
    )

    morphology_stream_shard = subparsers.add_parser(
        "background-morphology-stream-shard"
    )
    morphology_stream_shard.add_argument("--parent-plan", required=True)
    morphology_stream_shard.add_argument("--event-exclusions", required=True)
    morphology_stream_shard.add_argument("--checkpoint", required=True)
    morphology_stream_shard.add_argument("--config", required=True)
    morphology_stream_shard.add_argument("--coherence-config", required=True)
    morphology_stream_shard.add_argument("--cache-root", required=True)
    morphology_stream_shard.add_argument("--output-dir", required=True)
    morphology_stream_shard.add_argument("--shard-index", type=int, required=True)
    morphology_stream_shard.add_argument("--pairs-per-shard", type=int, default=1)
    morphology_stream_shard.add_argument("--validation-fraction", type=float, default=0.2)
    morphology_stream_shard.add_argument("--seed", type=int, default=20260720)
    morphology_stream_shard.add_argument(
        "--model-ifos", nargs="+", default=["H1", "L1", "V1"]
    )
    morphology_stream_shard.add_argument(
        "--q-values", nargs="+", type=float, default=[4, 8, 16]
    )
    morphology_stream_shard.add_argument("--target-sample-rate", type=int, default=1024)
    morphology_stream_shard.add_argument("--context-duration", type=float, default=64.0)
    morphology_stream_shard.add_argument("--chirp-threshold", type=float, default=0.3)
    morphology_stream_shard.add_argument("--minimum-bins", type=int, default=1)
    morphology_stream_shard.add_argument("--download-workers", type=int, default=8)
    morphology_stream_shard.add_argument(
        "--verified-source-inventory", action="append", default=[]
    )

    stream_merge = subparsers.add_parser("background-stream-merge")
    stream_merge.add_argument("--shard-report", action="append", required=True)
    stream_merge.add_argument("--parent-plan")
    stream_merge.add_argument("--output-dir", required=True)

    morphology_calibrate = subparsers.add_parser("background-morphology-calibrate")
    morphology_calibrate.add_argument("--merge-report", required=True)
    morphology_calibrate.add_argument(
        "--target-rate-per-detector-year", required=True, type=float
    )
    morphology_calibrate.add_argument("--output", required=True)

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

    injection_scale = subparsers.add_parser("injection-scale-plan")
    injection_scale.add_argument("--base-recipes", required=True)
    injection_scale.add_argument("--background-manifest", required=True)
    injection_scale.add_argument("--background-report", required=True)
    injection_scale.add_argument("--output-dir", required=True)
    injection_scale.add_argument("--scale", action="append", type=int)
    injection_scale.add_argument("--supplement-seed", type=int, default=20260722)

    injection_remap = subparsers.add_parser("injection-background-remap")
    injection_remap.add_argument("--source-recipes", required=True)
    injection_remap.add_argument("--target-background-manifest", required=True)
    injection_remap.add_argument("--validation-manifest", required=True)
    injection_remap.add_argument("--output-dir", required=True)
    injection_remap.add_argument("--split", choices=("train", "val"), default="train")
    injection_remap.add_argument("--seed", type=int, default=20260724)

    injection_domain_audit = subparsers.add_parser("injection-domain-pair-audit")
    injection_domain_audit.add_argument("--baseline-manifest", required=True)
    injection_domain_audit.add_argument("--independent-gps-manifest", required=True)
    injection_domain_audit.add_argument("--validation-manifest", required=True)
    injection_domain_audit.add_argument("--output", required=True)

    independent_validation_freeze = subparsers.add_parser(
        "independent-validation-endpoint-freeze"
    )
    independent_validation_freeze.add_argument("--purpose-partition-report", required=True)
    independent_validation_freeze.add_argument("--injection-plan-report", required=True)
    independent_validation_freeze.add_argument("--waveform-validation-report", required=True)
    independent_validation_freeze.add_argument("--materialization-report", required=True)
    independent_validation_freeze.add_argument("--snr-annotation-report", required=True)
    independent_validation_freeze.add_argument("--arrival-annotation-report", required=True)
    independent_validation_freeze.add_argument("--output", required=True)

    evaluation_freeze = subparsers.add_parser("evaluation-corpus-freeze")
    evaluation_freeze.add_argument("--manifest", required=True)
    evaluation_freeze.add_argument("--output", required=True)
    evaluation_freeze.add_argument("--access-log", required=True)
    evaluation_freeze.add_argument("--corpus-label", required=True)
    evaluation_freeze.add_argument("--expected-split", default="test")
    evaluation_freeze.add_argument("--minimum-rows", type=int, default=1)
    evaluation_freeze.add_argument(
        "--group-field",
        action="append",
        default=[],
        help="repeat to freeze physical group counts; defaults to injection/waveform/GPS/family",
    )

    evaluation_open = subparsers.add_parser("evaluation-corpus-open-once")
    evaluation_open.add_argument("--freeze-report", required=True)
    evaluation_open.add_argument("--code-commit", required=True)
    evaluation_open.add_argument(
        "--artifact",
        action="append",
        required=True,
        help=(
            "repeat LABEL=PATH; config, model, threshold_calibration and ood_policy "
            "are mandatory"
        ),
    )
    evaluation_open.add_argument(
        "--comparison-manifest", action="append", required=True
    )
    evaluation_open.add_argument("--evaluation-output", required=True)
    evaluation_open.add_argument("--evaluation-command", required=True)
    evaluation_open.add_argument("--overlap-field", action="append", default=[])

    background_bank = subparsers.add_parser("background-bank-materialize")
    background_bank.add_argument("--background-manifest", required=True)
    background_bank.add_argument("--output-dir", required=True)
    background_bank.add_argument("--target-sample-rate", type=int, default=1024)
    background_bank.add_argument("--context-duration", type=float, default=64.0)
    background_bank.add_argument("--split", choices=("train", "val", "test"))
    background_bank.add_argument("--limit", type=int)
    background_bank.add_argument("--maximum-windows-per-gps-block", type=int)

    background_bank_evict = subparsers.add_parser("background-bank-evict-sources")
    background_bank_evict.add_argument("--background-bank-report", required=True)
    background_bank_evict.add_argument("--cache-root", required=True)
    background_bank_evict.add_argument("--output", required=True)

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

    arrival_annotate = subparsers.add_parser("injection-arrival-annotate")
    arrival_annotate.add_argument("--manifest", required=True)
    arrival_annotate.add_argument("--output-dir", required=True)

    injection_score = subparsers.add_parser("injection-score")
    injection_score.add_argument("--manifest", required=True)
    injection_score.add_argument("--checkpoint", required=True)
    injection_score.add_argument("--config", required=True)
    injection_score.add_argument("--output-dir", required=True)
    injection_score.add_argument("--model-ifos", nargs="+", default=["H1", "L1", "V1"])
    injection_score.add_argument("--q-values", nargs="+", type=float, default=[4, 8, 16])
    injection_score.add_argument("--target-sample-rate", type=int, default=1024)
    injection_score.add_argument("--save-probabilities", action="store_true")
    injection_score.add_argument("--required-split", choices=["train", "val", "test"])
    injection_score.add_argument("--enabled-ifos", nargs="+", choices=["H1", "L1", "V1"])
    injection_score.add_argument("--coherence-config")

    pe = subparsers.add_parser("pe-evaluate")
    pe.add_argument("--manifest", required=True)
    pe.add_argument("--output", required=True)
    pe.add_argument("--credible-level", type=float, default=0.9)
    pe.add_argument("--bootstrap-replicates", type=int, default=2000)
    pe.add_argument("--bootstrap-seed", type=int, default=20260719)
    pe.add_argument("--require-publication-provenance", action="store_true")

    pe_robustness = subparsers.add_parser("pe-robustness-evaluate")
    pe_robustness.add_argument("--manifest", required=True)
    pe_robustness.add_argument("--output", required=True)
    pe_robustness.add_argument("--credible-level", type=float, default=0.9)
    pe_robustness.add_argument("--bootstrap-replicates", type=int, default=2000)
    pe_robustness.add_argument("--bootstrap-seed", type=int, default=20260720)
    pe_robustness.add_argument(
        "--allow-incomplete-provenance", action="store_true"
    )

    pe_joint = subparsers.add_parser("pe-robustness-joint-evaluate")
    pe_joint.add_argument("--dingo-batch-report", required=True)
    pe_joint.add_argument("--amplfi-batch-report", required=True)
    pe_joint.add_argument("--manifest-output", required=True)
    pe_joint.add_argument("--output", required=True)
    pe_joint.add_argument("--credible-level", type=float, default=0.9)
    pe_joint.add_argument("--bootstrap-replicates", type=int, default=2000)
    pe_joint.add_argument("--bootstrap-seed", type=int, default=20260720)

    pe_promotion = subparsers.add_parser("pe-robustness-promote")
    pe_promotion.add_argument("--joint-report", required=True)
    pe_promotion.add_argument("--config", required=True)
    pe_promotion.add_argument("--output", required=True)

    pe_inputs = subparsers.add_parser("pe-input-materialize")
    pe_inputs.add_argument("--clean-manifest", required=True)
    pe_inputs.add_argument("--contaminated-manifest", required=True)
    pe_inputs.add_argument("--mask-conditioned-manifest", required=True)
    pe_inputs.add_argument("--common-prior", required=True)
    pe_inputs.add_argument("--mask-model", required=True)
    pe_inputs.add_argument("--mask-policy", required=True)
    pe_inputs.add_argument("--output-dir", required=True)
    pe_inputs.add_argument("--required-split", choices=["val", "test"], required=True)
    pe_inputs.add_argument("--required-ifos", nargs="+", default=["H1", "L1"])
    pe_inputs.add_argument("--source-sample-rate-hz", type=int, default=4096)
    pe_inputs.add_argument("--source-duration-seconds", type=float, default=16.0)
    pe_inputs.add_argument("--source-post-trigger-seconds", type=float, default=2.0)
    pe_inputs.add_argument("--analysis-high-frequency-hz", type=float, default=1024.0)
    pe_inputs.add_argument("--asd-segment-seconds", type=float, default=8.0)
    pe_inputs.add_argument("--asd-stride-seconds", type=float, default=4.0)
    pe_inputs.add_argument("--asd-guard-seconds", type=float, default=2.0)
    pe_inputs.add_argument("--limit", type=int)
    pe_inputs.add_argument("--selection-seed", type=int, default=20260721)

    pe_conditioning = subparsers.add_parser("pe-native-condition")
    pe_conditioning.add_argument("--source-manifest", required=True)
    pe_conditioning.add_argument("--config", required=True)
    pe_conditioning.add_argument("--output-dir", required=True)
    pe_conditioning.add_argument("--required-split", choices=["val", "test"], required=True)

    dingo_batch = subparsers.add_parser("dingo-common-batch")
    dingo_batch.add_argument("--native-manifest", required=True)
    dingo_batch.add_argument("--model-metadata", required=True)
    dingo_batch.add_argument("--native-prior", required=True)
    dingo_batch.add_argument("--model-init", required=True)
    dingo_batch.add_argument("--python-executable", required=True)
    dingo_batch.add_argument("--runner-script", default="scripts/run_dingo_common_event.py")
    dingo_batch.add_argument("--output-dir", required=True)
    dingo_batch.add_argument("--required-split", choices=["val", "test"], required=True)
    dingo_batch.add_argument("--num-samples", type=int, default=10000)
    dingo_batch.add_argument("--batch-size", type=int, default=1000)
    dingo_batch.add_argument("--num-gnpe-iterations", type=int, default=30)
    dingo_batch.add_argument("--device", default="cuda")
    dingo_batch.add_argument("--seed", type=int, default=20260721)
    dingo_batch.add_argument(
        "--comparison-mode",
        choices=("common_prior", "official_native"),
        default="common_prior",
    )

    dingo_official_metadata = subparsers.add_parser("dingo-official-native-model-freeze")
    dingo_official_metadata.add_argument("--source-config", required=True)
    dingo_official_metadata.add_argument("--acquisition-report", required=True)
    dingo_official_metadata.add_argument("--model-load-receipt", required=True)
    dingo_official_metadata.add_argument("--native-conditioning-config", required=True)
    dingo_official_metadata.add_argument("--output", required=True)

    amplfi_batch = subparsers.add_parser("amplfi-common-batch")
    amplfi_batch.add_argument("--native-manifest", required=True)
    amplfi_batch.add_argument("--model-metadata", required=True)
    amplfi_batch.add_argument("--native-prior", required=True)
    amplfi_batch.add_argument("--python-executable", required=True)
    amplfi_batch.add_argument(
        "--runner-script", default="scripts/run_amplfi_common_event.py"
    )
    amplfi_batch.add_argument("--output-dir", required=True)
    amplfi_batch.add_argument("--required-split", choices=["val", "test"], required=True)
    amplfi_batch.add_argument("--num-samples", type=int, default=10000)
    amplfi_batch.add_argument("--sample-batch-size", type=int, default=1000)
    amplfi_batch.add_argument("--device", default="cuda")
    amplfi_batch.add_argument("--seed", type=int, default=20260721)

    pe_backend = subparsers.add_parser("pe-backend-lock-audit")
    pe_backend.add_argument("--config", required=True)
    pe_backend.add_argument("--output", required=True)
    pe_backend.add_argument("--allow-incomplete", action="store_true")

    pe_model = subparsers.add_parser("pe-backend-model-freeze")
    pe_model.add_argument("--backend", required=True, choices=["DINGO", "AMPLFI"])
    pe_model.add_argument("--model", required=True)
    pe_model.add_argument("--training-config", required=True)
    pe_model.add_argument("--training-data-manifest", required=True)
    pe_model.add_argument("--analysis-prior", required=True)
    pe_model.add_argument("--selection-report", required=True)
    pe_model.add_argument("--native-conditioning-config", required=True)
    pe_model.add_argument("--native-prior")
    pe_model.add_argument("--prior-projection-report")
    pe_model.add_argument("--initialization-model")
    pe_model.add_argument("--output", required=True)
    pe_model.add_argument("--population", default="BBH", choices=["BBH"])
    pe_model.add_argument("--source-ifos", nargs="+", default=["H1", "L1"])
    pe_model.add_argument("--source-sample-rate-hz", type=float, default=2048)
    pe_model.add_argument("--source-duration-seconds", type=float, default=8)
    pe_model.add_argument("--source-post-trigger-seconds", type=float, default=2)
    pe_model.add_argument("--analysis-waveform-approximant", required=True)
    pe_model.add_argument("--native-model-waveform-approximant", required=True)
    pe_model.add_argument("--model-training-backend-version", required=True)
    pe_model.add_argument("--native-inference-parameters", nargs="+", required=True)
    pe_model.add_argument("--reported-parameter-mapping", nargs="+", required=True)

    pe_checkpoint = subparsers.add_parser("pe-lightning-checkpoint-select")
    pe_checkpoint.add_argument("--training-config", required=True)
    pe_checkpoint.add_argument("--training-data-manifest", required=True)
    pe_checkpoint.add_argument("--metrics-csv", required=True)
    pe_checkpoint.add_argument("--checkpoint-index", required=True)
    pe_checkpoint.add_argument("--output", required=True)
    pe_checkpoint.add_argument("--selection-metric", default="valid_loss")
    pe_checkpoint.add_argument("--selection-metric-mode", choices=("min", "max"), default="min")
    pe_checkpoint.add_argument("--minimum-publication-epochs", type=int, default=100)
    pe_checkpoint.add_argument("--minimum-validation-points", type=int, default=50)

    pe_sources = subparsers.add_parser("pe-model-sources-acquire")
    pe_sources.add_argument("--config", required=True)
    pe_sources.add_argument("--output-dir", required=True)
    pe_sources.add_argument("--report", required=True)
    pe_sources.add_argument("--download", action="store_true")
    pe_sources.add_argument("--minimum-free-bytes", type=int, default=0)
    pe_sources.add_argument("--transfer-attempts", type=int, default=40)
    pe_sources.add_argument("--retry-delay-seconds", type=float, default=5.0)
    pe_sources.add_argument("--maximum-stalled-attempts", type=int, default=5)

    dingo_failure = subparsers.add_parser("dingo-runtime-failure-adjudicate")
    dingo_failure.add_argument("--failure-receipt", required=True)
    dingo_failure.add_argument("--policy", required=True)
    dingo_failure.add_argument("--output", required=True)

    amplfi_background = subparsers.add_parser("amplfi-background-export")
    amplfi_background.add_argument("--manifest", required=True)
    amplfi_background.add_argument("--output-dir", required=True)
    amplfi_background.add_argument("--report", required=True)
    amplfi_background.add_argument("--target-sample-rate", type=int, default=2048)
    amplfi_background.add_argument("--minimum-segment-seconds", type=int, default=16)

    amplfi_capacity = subparsers.add_parser("amplfi-background-capacity-audit")
    amplfi_capacity.add_argument("--manifest", required=True)
    amplfi_capacity.add_argument("--policy", required=True)
    amplfi_capacity.add_argument("--output", required=True)

    amplfi_stage = subparsers.add_parser("amplfi-training-stage-freeze")
    amplfi_stage.add_argument("--base-config", required=True)
    amplfi_stage.add_argument("--stage-policy", required=True)
    amplfi_stage.add_argument("--stage", required=True)
    amplfi_stage.add_argument("--output-config", required=True)
    amplfi_stage.add_argument("--output-report", required=True)

    amplfi_prior = subparsers.add_parser("amplfi-common-prior-audit")
    amplfi_prior.add_argument("--canonical-prior", required=True)
    amplfi_prior.add_argument("--amplfi-prior", required=True)
    amplfi_prior.add_argument("--training-config", required=True)
    amplfi_prior.add_argument("--output", required=True)

    dingo_prior = subparsers.add_parser("dingo-common-prior-audit")
    dingo_prior.add_argument("--canonical-prior", required=True)
    dingo_prior.add_argument("--dingo-prior-config", required=True)
    dingo_prior.add_argument("--training-config", required=True)
    dingo_prior.add_argument("--output", required=True)

    ood = subparsers.add_parser("ood-abstention-evaluate")
    ood.add_argument("--calibration-manifest", required=True)
    ood.add_argument("--evaluation-manifest", required=True)
    ood.add_argument("--output", required=True)
    ood.add_argument("--maximum-known-abstention-rate", type=float, default=0.05)
    ood.add_argument("--score-field", default="ood_score")

    ood_split = subparsers.add_parser("gravityspy-ood-split")
    ood_split.add_argument("--train-manifest", required=True)
    ood_split.add_argument("--validation-manifest", required=True)
    ood_split.add_argument("--held-out-family", required=True)
    ood_split.add_argument("--output-dir", required=True)
    ood_split.add_argument("--seed", type=int, default=20260720)

    ood_family = subparsers.add_parser("gravityspy-ood-family-freeze")
    ood_family.add_argument("--train-manifest", required=True)
    ood_family.add_argument("--validation-manifest", required=True)
    ood_family.add_argument("--output", required=True)
    ood_family.add_argument("--exclude-family", action="append", default=[])
    ood_family.add_argument("--minimum-train-rows", type=int, default=20)
    ood_family.add_argument("--minimum-validation-rows", type=int, default=20)
    ood_family.add_argument("--minimum-validation-gps-blocks", type=int, default=5)

    ood_train = subparsers.add_parser("glitch-ood-train")
    ood_train.add_argument("--config", required=True)
    ood_train.add_argument("--known-train-manifest", required=True)
    ood_train.add_argument("--known-calibration-manifest", required=True)
    ood_train.add_argument("--heldout-evaluation-manifest", required=True)
    ood_train.add_argument("--output-dir", required=True)
    ood_train.add_argument("--seed", type=int)
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
                args.model_selection_report,
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
    elif args.command == "candidate-search-calibrate":
        _print(
            run_candidate_search_calibration(
                args.validation_time_slide_report,
                args.validation_injection_ranking_report,
                args.target_far_per_year,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "candidate-search-evaluate-frozen":
        _print(
            run_frozen_candidate_search_evaluation(
                args.calibration_report,
                args.test_time_slide_report,
                args.test_injection_ranking_report,
                args.output,
                args.minimum_test_live_time_years,
                args.minimum_test_injections,
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
    elif args.command == "physical-validation-endpoint":
        from .search import run_physical_validation_endpoint

        _print(
            run_physical_validation_endpoint(
                args.training_report,
                args.background_score_report,
                args.injection_score_report,
                args.maximum_validation_false_alarms,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "physical-validation-summarize":
        from .search import summarize_physical_validation_endpoints

        _print(
            summarize_physical_validation_endpoints(
                args.endpoint_report,
                args.scale_subset_report,
                args.output,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "detector-subset-summarize":
        from .search import summarize_detector_subset_endpoints

        _print(
            summarize_detector_subset_endpoints(
                args.endpoint_report,
                args.output,
                tuple(args.reference_ifos),
                args.relative_noninferiority_margin,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "physical-validation-score-series":
        from .search import score_physical_training_series

        _print(
            score_physical_training_series(
                args.training_series_dir,
                args.background_manifest,
                args.injection_manifest,
                args.config,
                args.scale_subset_report,
                args.output_dir,
                args.maximum_validation_false_alarms,
                args.context_duration,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "coherence-validation-compare":
        from .search import run_coherence_validation_comparison

        _print(
            run_coherence_validation_comparison(
                args.background_score_report,
                args.injection_score_report,
                args.output,
                args.maximum_validation_false_alarms,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "mask-search-validation":
        from .search import run_mask_search_validation

        _print(
            run_mask_search_validation(
                args.background_raw,
                args.background_mask,
                args.clean_raw,
                args.clean_mask,
                args.contaminated_raw,
                args.contaminated_mask,
                args.output,
                args.maximum_validation_false_alarms,
                args.clean_noninferiority_margin,
                args.minimum_contaminated_efficiency_gain,
                args.score_field,
                args.bootstrap_replicates,
                args.seed,
            )
        )
    elif args.command == "mask-search-validation-pipeline":
        from .mask_pipeline import run_mask_search_validation_pipeline

        _print(
            run_mask_search_validation_pipeline(
                args.background_manifest,
                args.clean_injection_manifest,
                args.contaminated_injection_manifest,
                args.checkpoint,
                args.config,
                args.output_dir,
                args.maximum_validation_false_alarms,
                args.strength,
                args.clean_noninferiority_margin,
                args.minimum_contaminated_efficiency_gain,
                args.bootstrap_replicates,
                args.seed,
                tuple(args.model_ifos),
                tuple(args.q_values),
                args.target_sample_rate,
                args.context_duration,
            )
        )
    elif args.command == "manifest-select-split":
        from .manifests import select_jsonl_split

        _print(select_jsonl_split(args.manifest, args.split, args.output_dir))
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
    elif args.command == "physical-scale-summarize":
        from .scaling import summarize_physical_scale_reports

        _print(
            summarize_physical_scale_reports(
                args.report, args.scale_subset_report, args.output
            )
        )
    elif args.command == "physical-scale-series":
        from .scaling import run_physical_fixed_update_series

        _print(
            run_physical_fixed_update_series(
                args.config,
                args.scale_subset_report,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
                args.validation_feature_cache_dir,
            )
        )
    elif args.command == "physical-scale-epoch-series":
        from .scaling import run_physical_fixed_epoch_series

        _print(
            run_physical_fixed_epoch_series(
                args.config,
                args.scale_subset_report,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
                args.validation_feature_cache_dir,
            )
        )
    elif args.command == "physical-data-domain-compare":
        from .data_domain import summarize_physical_data_domain_comparison

        _print(
            summarize_physical_data_domain_comparison(
                args.config,
                args.data_domain_audit,
                args.fixed_update_training,
                args.fixed_update_audit,
                args.fixed_epoch_training,
                args.fixed_epoch_audit,
                args.output,
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
    elif args.command == "gwosc-plan-shard":
        from .gwosc import run_gwosc_plan_shard

        _print(
            run_gwosc_plan_shard(
                args.plan,
                args.output,
                args.shard_index,
                args.pairs_per_shard,
            )
        )
    elif args.command == "gwosc-plan-extend":
        from .gwosc import extend_gwosc_run_plan

        _print(
            extend_gwosc_run_plan(
                args.base_plan,
                args.output,
                args.target_pairs,
                args.extension_seed,
            )
        )
    elif args.command == "gwosc-plan-disjoint":
        from .gwosc import run_disjoint_gwosc_run_plan

        _print(
            run_disjoint_gwosc_run_plan(
                args.run,
                args.detectors,
                args.exclude_plan,
                args.output,
                target_pairs=args.target_pairs,
                sample_rate_khz=args.sample_rate_khz,
                seed=args.seed,
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
                args.verified_source_inventory,
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
                args.validation_feature_cache_dir,
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
    elif args.command == "physical-scale-subsets":
        from .physical_training import build_physical_scale_subsets

        _print(
            build_physical_scale_subsets(
                args.manifest,
                args.validation_manifest,
                args.output_dir,
                tuple(args.scale) if args.scale else (2_000, 5_000, 10_000),
                args.seed,
            )
        )
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
    elif args.command == "physical-timing-train":
        from .timing import run_physical_timing_training

        _print(
            run_physical_timing_training(
                args.config,
                args.train_manifest,
                args.validation_manifest,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "detector-arrival-timing-train":
        from .arrival_timing import run_detector_arrival_timing_training

        _print(
            run_detector_arrival_timing_training(
                args.config,
                args.train_manifest,
                args.validation_manifest,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "detector-arrival-timing-validation-stratify":
        from .arrival_timing import (
            run_detector_arrival_timing_validation_stratification,
        )

        _print(
            run_detector_arrival_timing_validation_stratification(
                args.config,
                args.validation_manifest,
                args.checkpoint,
                args.output,
                args.predictions_output,
            )
        )
    elif args.command == "detector-endpoint-proposal-train":
        from .endpoint_proposal import run_detector_endpoint_proposal_training

        _print(
            run_detector_endpoint_proposal_training(
                args.config,
                args.train_manifest,
                args.validation_manifest,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "detector-endpoint-proposal-evaluate":
        from .endpoint_proposal import run_detector_endpoint_proposal_evaluation

        _print(
            run_detector_endpoint_proposal_evaluation(
                args.config,
                args.validation_manifest,
                args.checkpoint,
                args.output_dir,
            )
        )
    elif args.command == "detector-endpoint-proposal-apply":
        from .endpoint_proposal import run_detector_endpoint_proposal_application

        _print(
            run_detector_endpoint_proposal_application(
                args.config,
                args.manifest,
                args.checkpoint,
                args.threshold,
                args.required_split,
                args.output_dir,
                args.shard_size,
            )
        )
    elif args.command == "candidate-refiner-plan":
        from .candidate_refiner import run_candidate_refiner_plan

        _print(
            run_candidate_refiner_plan(
                args.train_injection_manifest,
                args.train_candidate_manifest,
                args.validation_injection_manifest,
                args.validation_candidate_manifest,
                args.output_dir,
                args.positive_padding_seconds,
                args.validation_selection_fraction,
                args.seed,
            )
        )
    elif args.command == "candidate-refiner-train":
        from .candidate_refiner import run_candidate_local_refiner_training

        _print(
            run_candidate_local_refiner_training(
                args.config,
                args.train_injection_manifest,
                args.train_candidate_manifest,
                args.validation_injection_manifest,
                args.validation_selection_candidate_manifest,
                args.validation_calibration_candidate_manifest,
                args.output_dir,
                args.seed,
                args.pretrained_endpoint_checkpoint,
            )
        )
    elif args.command == "candidate-refiner-validation":
        from .candidate_refiner import run_candidate_local_refiner_validation

        _print(
            run_candidate_local_refiner_validation(
                args.config,
                args.checkpoint,
                args.validation_injection_manifest,
                args.validation_candidate_manifest,
                args.output_dir,
            )
        )
    elif args.command == "candidate-network-set-audit":
        from .candidate_refiner import run_candidate_network_set_audit

        _print(
            run_candidate_network_set_audit(
                args.config,
                args.injection_manifest,
                args.candidate_manifest,
                args.output_dir,
            )
        )
    elif args.command == "candidate-pair-ranker-train":
        from .candidate_set_training import run_candidate_pair_ranker_training

        _print(
            run_candidate_pair_ranker_training(
                args.config,
                args.train_injection_manifest,
                args.train_candidate_manifest,
                args.validation_injection_manifest,
                args.validation_selection_candidate_manifest,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "candidate-pair-scaling-plan":
        from .candidate_set_training import run_candidate_pair_scaling_plan

        _print(
            run_candidate_pair_scaling_plan(
                args.train_injection_manifest,
                args.train_candidate_manifest,
                args.scale_manifest,
                args.output_dir,
            )
        )
    elif args.command == "candidate-pair-scaling-evaluate":
        from .candidate_set_training import run_candidate_pair_scaling_evaluation

        _print(
            run_candidate_pair_scaling_evaluation(
                args.config,
                args.scaling_plan_report,
                args.fixed_update_report,
                args.fixed_epoch_report,
                args.output,
            )
        )
    elif args.command == "candidate-pair-representation-evaluate":
        from .candidate_set_training import (
            run_candidate_pair_representation_evaluation,
        )

        _print(
            run_candidate_pair_representation_evaluation(
                args.config,
                args.baseline_report,
                args.candidate_report,
                args.output,
            )
        )
    elif args.command == "detector-arrival-timing-validation-compare":
        from .arrival_timing import run_detector_arrival_timing_validation_comparison

        _print(
            run_detector_arrival_timing_validation_comparison(
                args.config,
                args.reference_predictions,
                args.candidate_predictions,
                args.output,
            )
        )
    elif args.command == "gravityspy-glitch-finetune":
        from .glitch_training import run_gravityspy_glitch_finetune

        _print(
            run_gravityspy_glitch_finetune(
                args.config,
                args.glitch_train_manifest,
                args.glitch_validation_manifest,
                args.chirp_validation_manifest,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
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
    elif args.command == "gravityspy-network-strain-plan":
        from .gravityspy import plan_gravityspy_network_strain

        _print(
            plan_gravityspy_network_strain(
                args.manifest,
                args.output_dir,
                args.detectors,
                args.sample_rate_khz,
                args.context_duration,
                args.minimum_detectors,
            )
        )
    elif args.command == "gravityspy-network-strain-materialize":
        from .gravityspy import materialize_gravityspy_network_strain

        _print(
            materialize_gravityspy_network_strain(
                args.manifest,
                args.config,
                args.cache_dir,
                args.output_dir,
                args.output_duration,
                args.download_workers,
                args.chunk_samples,
                args.shard,
                args.verified_source_inventory,
            )
        )
    elif args.command == "gravityspy-network-strain-shard":
        from .gravityspy import shard_gravityspy_network_strain_plan

        _print(
            shard_gravityspy_network_strain_plan(
                args.manifest, args.output_dir, args.files_per_shard, args.seed
            )
        )
    elif args.command == "gravityspy-network-recovery-plan":
        from .gravityspy import plan_gravityspy_network_recovery

        _print(
            plan_gravityspy_network_recovery(
                args.source_manifest,
                args.materialization_report,
                args.output_dir,
            )
        )
    elif args.command == "gravityspy-network-strain-select":
        from .gravityspy import select_gravityspy_network_source_components

        _print(
            select_gravityspy_network_source_components(
                args.manifest,
                args.output_dir,
                args.per_label,
                args.maximum_source_files,
                args.seed,
                args.existing_manifest,
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
    elif args.command == "gravityspy-strain-select":
        from .gravityspy import select_gravityspy_source_files

        _print(
            select_gravityspy_source_files(
                args.manifest,
                args.output_dir,
                args.per_label,
                args.maximum_files,
                args.seed,
                args.existing_manifest,
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
    elif args.command == "gravityspy-network-numeric-merge":
        from .gravityspy import merge_gravityspy_network_numeric_manifests

        _print(
            merge_gravityspy_network_numeric_manifests(
                args.report, args.output_dir, args.split
            )
        )
    elif args.command == "gravityspy-network-corpus-audit":
        from .glitch_training import audit_gravityspy_network_numeric_corpus

        _print(
            audit_gravityspy_network_numeric_corpus(
                args.train_report, args.validation_report, args.output
            )
        )
    elif args.command == "gravityspy-network-corpus-resplit":
        from .gravityspy import resplit_gravityspy_network_numeric_corpus

        _print(
            resplit_gravityspy_network_numeric_corpus(
                args.report, args.output_dir, args.validation_fraction, args.seed
            )
        )
    elif args.command == "gravityspy-strain-evict":
        from .gravityspy import evict_gravityspy_verified_sources

        _print(
            evict_gravityspy_verified_sources(
                args.materialization_report, args.cache_dir, args.output
            )
        )
    elif args.command == "physical-overlap-materialize":
        from .overlaps import materialize_physical_overlaps

        _print(
            materialize_physical_overlaps(
                args.gravityspy_manifest,
                args.injection_manifest,
                args.config,
                args.output_dir,
                args.split,
                args.seed,
                args.limit,
                args.gravityspy_corpus_audit,
            )
        )
    elif args.command == "physical-overlap-audit":
        from .overlaps import audit_physical_overlap_manifests

        _print(audit_physical_overlap_manifests(args.manifest, args.output))
    elif args.command == "physical-overlap-sampling-promote":
        from .overlap_training import promote_overlap_sampling_arm

        _print(
            promote_overlap_sampling_arm(
                args.uniform_report,
                args.family_balanced_report,
                args.overlap_train_manifest,
                args.overlap_validation_manifest,
                args.gravityspy_corpus_audit,
                args.config,
                args.output,
            )
        )
    elif args.command == "physical-overlap-five-seed-summarize":
        from .overlap_training import summarize_overlap_five_seed_promotion

        _print(
            summarize_overlap_five_seed_promotion(
                args.promotion_report, args.report, args.output
            )
        )
    elif args.command == "physical-overlap-contamination":
        from .overlaps import build_contaminated_injection_overrides

        _print(
            build_contaminated_injection_overrides(
                args.overlap_manifest,
                args.injection_manifest,
                args.output_dir,
                args.required_split,
            )
        )
    elif args.command == "physical-overlap-finetune":
        from .overlap_training import run_physical_overlap_finetune

        _print(
            run_physical_overlap_finetune(
                args.config,
                args.overlap_train_manifest,
                args.overlap_validation_manifest,
                args.clean_train_manifest,
                args.clean_validation_manifest,
                args.pretrained_checkpoint,
                args.output_dir,
                args.seed,
                args.clean_validation_feature_cache_dir,
            )
        )
    elif args.command == "gravityspy-mask-audit-plan":
        from .mask_audit import plan_gravityspy_mask_audit

        _print(
            plan_gravityspy_mask_audit(
                args.manifest, args.output_dir, args.per_label, args.seed
            )
        )
    elif args.command == "gravityspy-mask-audit-evaluate":
        from .mask_audit import evaluate_gravityspy_mask_audit

        _print(evaluate_gravityspy_mask_audit(args.tasks, args.annotations, args.output))
    elif args.command == "gravityspy-mask-consensus-materialize":
        from .mask_audit import materialize_gravityspy_mask_consensus

        _print(
            materialize_gravityspy_mask_consensus(
                args.tasks,
                args.annotations,
                args.audit_report,
                args.output_dir,
            )
        )
    elif args.command == "gravityspy-mask-segmentation-predict":
        from .mask_audit import predict_gravityspy_mask_segmentation

        _print(
            predict_gravityspy_mask_segmentation(
                args.gold_report,
                args.selection_report,
                args.config,
                args.output_dir,
            )
        )
    elif args.command == "gravityspy-mask-segmentation-evaluate":
        from .mask_audit import evaluate_gravityspy_mask_segmentation

        _print(
            evaluate_gravityspy_mask_segmentation(
                args.gold_report,
                args.predictions,
                args.output,
                args.bootstrap_replicates,
                args.bootstrap_seed,
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
                split_strategy=args.split_strategy,
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
                args.split_strategy,
            )
        )
    elif args.command == "background-disjoint-subset":
        from .background import run_disjoint_background_subset

        _print(
            run_disjoint_background_subset(
                args.background_manifest,
                args.background_report,
                args.exclude_manifest,
                args.output_dir,
                args.split,
            )
        )
    elif args.command == "background-purpose-partition":
        from .background import run_background_purpose_partition

        _print(
            run_background_purpose_partition(
                args.background_manifest,
                args.background_report,
                args.output_dir,
                args.injection_fraction,
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
    elif args.command == "learned-background-deglitch":
        from .learned_deglitch import run_learned_background_deglitch

        _print(
            run_learned_background_deglitch(
                args.background_manifest,
                args.scored_manifest,
                args.output_dir,
                args.strength,
                tuple(args.model_ifos),
                args.target_sample_rate,
                args.context_duration,
                args.required_split,
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
                required_split=args.required_split,
                enabled_ifos=(tuple(args.enabled_ifos) if args.enabled_ifos else None),
                coherence_config_path=args.coherence_config,
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
    elif args.command == "candidate-timing-calibrate":
        from .candidates import run_candidate_timing_calibration

        _print(
            run_candidate_timing_calibration(
                args.injection_triggers,
                args.output,
                args.chirp_threshold,
                args.minimum_bins,
                args.association_window_seconds,
                args.uncertainty_quantile,
                args.minimum_matches_per_method,
                args.maximum_empirical_timing_uncertainty_seconds,
            )
        )
    elif args.command == "candidate-timing-apply":
        from .candidates import run_apply_candidate_timing_calibration

        _print(
            run_apply_candidate_timing_calibration(
                args.candidates, args.calibration_report, args.output
            )
        )
    elif args.command == "injection-candidate-extract":
        from .candidates import run_injection_candidate_extraction

        _print(
            run_injection_candidate_extraction(
                args.injection_triggers,
                args.output_dir,
                args.chirp_threshold,
                args.minimum_bins,
            )
        )
    elif args.command == "candidate-proposal-audit":
        from .candidates import run_candidate_proposal_coverage_audit

        _print(
            run_candidate_proposal_coverage_audit(
                args.injection_manifest,
                args.candidate_manifest,
                args.output,
                args.padding_seconds,
            )
        )
    elif args.command == "candidate-proposal-sweep-select":
        from .candidates import run_candidate_proposal_threshold_selection

        _print(
            run_candidate_proposal_threshold_selection(
                args.config,
                args.audit_report,
                args.output,
            )
        )
    elif args.command == "injection-candidate-rank":
        from .candidates import run_injection_candidate_rankings

        _print(
            run_injection_candidate_rankings(
                args.injection_triggers,
                args.candidates,
                args.output_dir,
                args.split,
                args.reference_ifo,
                args.second_ifo,
                args.physical_delay_limit_seconds,
                args.empirical_timing_uncertainty_seconds,
                args.truth_association_window_seconds,
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
                args.physical_delay_limit_seconds,
                args.empirical_timing_uncertainty_seconds,
                args.slide_start_index,
                args.slide_schedule,
                args.schedule_offset,
            )
        )
    elif args.command == "candidate-time-slide-schedule-freeze":
        from .exposure import freeze_candidate_time_slide_schedule

        _print(
            freeze_candidate_time_slide_schedule(
                args.background_manifest,
                args.output,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.step_seconds,
                args.slide_index,
                args.target_far_per_year,
                args.zero_count_confidence,
            )
        )
    elif args.command == "candidate-block-permutations":
        from .candidates import run_candidate_block_permutations

        _print(
            run_candidate_block_permutations(
                args.candidates,
                args.background_manifest,
                args.schedule,
                args.output_dir,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.coincidence_window_seconds,
                args.cluster_window_seconds,
                args.physical_delay_limit_seconds,
                args.empirical_timing_uncertainty_seconds,
            )
        )
    elif args.command == "candidate-time-slide-range-schedule-freeze":
        from .exposure import freeze_candidate_time_slide_range_schedule

        _print(
            freeze_candidate_time_slide_range_schedule(
                args.background_manifest,
                args.output,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.step_seconds,
                args.slide_start_index,
                args.slide_stop_index_exclusive,
                args.target_far_per_year,
                args.zero_count_confidence,
            )
        )
    elif args.command == "candidate-block-permutation-schedule-freeze":
        from .exposure import freeze_candidate_block_permutation_schedule

        _print(
            freeze_candidate_block_permutation_schedule(
                args.background_manifest,
                args.output,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.target_far_per_year,
                args.zero_count_confidence,
                args.maximum_shifts,
            )
        )
    elif args.command == "candidate-block-permutation-capacity-forecast":
        from .exposure import run_candidate_block_permutation_capacity_forecast

        _print(
            run_candidate_block_permutation_capacity_forecast(
                args.pilot_schedule,
                args.pilot_background_report,
                args.planned_parent_plan,
                args.output,
                args.safety_factor,
                args.allow_insufficient,
            )
        )
    elif args.command == "candidate-block-permutation-capacity-extension-freeze":
        from .exposure import freeze_candidate_block_capacity_extension_decision

        _print(
            freeze_candidate_block_capacity_extension_decision(
                args.base_forecast,
                args.extended_plan,
                args.extended_forecast,
                args.output,
            )
        )
    elif args.command == "candidate-search-validation-pipeline":
        from .candidate_pipeline import run_candidate_validation_pipeline

        _print(
            run_candidate_validation_pipeline(
                args.background_manifest,
                args.injection_manifest,
                args.checkpoint,
                args.config,
                args.coherence_config,
                args.output_dir,
                args.reference_ifo,
                args.second_ifo,
                tuple(args.model_ifos),
                tuple(args.q_values),
                args.target_sample_rate,
                args.context_duration,
                args.chirp_threshold,
                args.minimum_bins,
                args.timing_association_window_seconds,
                args.timing_uncertainty_quantile,
                args.minimum_timing_matches,
                args.maximum_timing_uncertainty_seconds,
                args.truth_association_window_seconds,
                args.slide_count,
                args.slide_step_seconds,
                args.cluster_window_seconds,
                args.target_far_per_year,
                args.bootstrap_replicates,
                args.seed,
                args.model_selection_report,
            )
        )
    elif args.command == "candidate-search-validation-compare":
        from .candidate_pipeline import compare_candidate_validation_pipelines

        _print(
            compare_candidate_validation_pipelines(
                args.baseline_report,
                args.promoted_report,
                args.config,
                args.output,
            )
        )
    elif args.command == "candidate-search-validation-block-recalibrate":
        from .candidate_pipeline import (
            recalibrate_candidate_validation_pipeline_with_block_permutations,
        )

        _print(
            recalibrate_candidate_validation_pipeline_with_block_permutations(
                args.pipeline_report,
                args.background_manifest,
                args.calibrated_candidate_manifest,
                args.injection_ranking_report,
                args.output_dir,
                args.zero_count_confidence,
            )
        )
    elif args.command == "candidate-time-slide-merge":
        from .candidates import merge_candidate_time_slide_shards

        _print(
            merge_candidate_time_slide_shards(
                args.report,
                args.output_dir,
                args.split,
            )
        )
    elif args.command == "candidate-exposure-plan":
        from .exposure import run_candidate_background_exposure_plan

        _print(
            run_candidate_background_exposure_plan(
                args.background_manifest,
                args.output,
                args.split,
                args.reference_ifo,
                args.shifted_ifo,
                args.slide_count,
                args.step_seconds,
                args.target_far_per_year,
                args.zero_count_confidence,
                args.slide_start_index,
            )
        )
    elif args.command == "candidate-probability-evict":
        from .streaming import evict_candidate_probability_artifacts

        _print(
            evict_candidate_probability_artifacts(
                args.candidate_report,
                args.score_report,
                args.probability_root,
                args.output,
            )
        )
    elif args.command == "background-source-evict":
        from .streaming import evict_scored_background_batch_sources

        _print(
            evict_scored_background_batch_sources(
                args.batch_report,
                args.background_report,
                args.score_report,
                args.candidate_report,
                args.cache_root,
                args.output,
            )
        )
    elif args.command == "amplfi-background-source-evict":
        from .streaming import evict_amplfi_background_batch_sources

        _print(
            evict_amplfi_background_batch_sources(
                args.batch_report,
                args.background_report,
                args.export_report,
                args.cache_root,
                args.output,
            )
        )
    elif args.command == "background-stream-shard":
        from .streaming import run_streamed_background_shard

        _print(
            run_streamed_background_shard(
                args.parent_plan,
                args.event_exclusions,
                args.timing_calibration_report,
                args.checkpoint,
                args.config,
                args.coherence_config,
                args.cache_root,
                args.output_dir,
                args.shard_index,
                args.pairs_per_shard,
                args.validation_fraction,
                args.test_fraction,
                args.seed,
                tuple(args.model_ifos),
                tuple(args.q_values),
                args.target_sample_rate,
                args.context_duration,
                args.chirp_threshold,
                args.minimum_bins,
                args.download_workers,
                False,
                args.verified_source_inventory,
            )
        )
    elif args.command == "background-morphology-stream-shard":
        from .streaming import run_streamed_morphology_background_shard

        _print(
            run_streamed_morphology_background_shard(
                args.parent_plan,
                args.event_exclusions,
                args.checkpoint,
                args.config,
                args.coherence_config,
                args.cache_root,
                args.output_dir,
                args.shard_index,
                args.pairs_per_shard,
                args.validation_fraction,
                args.seed,
                tuple(args.model_ifos),
                tuple(args.q_values),
                args.target_sample_rate,
                args.context_duration,
                args.chirp_threshold,
                args.minimum_bins,
                args.download_workers,
                args.verified_source_inventory,
            )
        )
    elif args.command == "background-stream-merge":
        from .streaming import merge_streamed_background_shards

        _print(
            merge_streamed_background_shards(
                args.shard_report, args.output_dir, args.parent_plan
            )
        )
    elif args.command == "background-morphology-calibrate":
        from .streaming import calibrate_streamed_morphology_candidate_rate

        _print(
            calibrate_streamed_morphology_candidate_rate(
                args.merge_report,
                args.target_rate_per_detector_year,
                args.output,
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
    elif args.command == "injection-scale-plan":
        from .injections import run_nested_injection_scale_plan

        _print(
            run_nested_injection_scale_plan(
                args.base_recipes,
                args.background_manifest,
                args.background_report,
                args.output_dir,
                tuple(args.scale) if args.scale else (10_000, 25_000, 50_000),
                args.supplement_seed,
            )
        )
    elif args.command == "injection-background-remap":
        from .injections import run_paired_background_remap

        _print(
            run_paired_background_remap(
                args.source_recipes,
                args.target_background_manifest,
                args.validation_manifest,
                args.output_dir,
                args.split,
                args.seed,
            )
        )
    elif args.command == "injection-domain-pair-audit":
        from .injections import audit_paired_data_domain_manifests

        _print(
            audit_paired_data_domain_manifests(
                args.baseline_manifest,
                args.independent_gps_manifest,
                args.validation_manifest,
                args.output,
            )
        )
    elif args.command == "independent-validation-endpoint-freeze":
        from .injections import freeze_independent_validation_endpoint

        _print(
            freeze_independent_validation_endpoint(
                args.purpose_partition_report,
                args.injection_plan_report,
                args.waveform_validation_report,
                args.materialization_report,
                args.snr_annotation_report,
                args.arrival_annotation_report,
                args.output,
            )
        )
    elif args.command == "evaluation-corpus-freeze":
        from .evaluation_lock import freeze_evaluation_corpus

        _print(
            freeze_evaluation_corpus(
                args.manifest,
                args.output,
                args.access_log,
                args.corpus_label,
                args.expected_split,
                args.minimum_rows,
                tuple(args.group_field)
                if args.group_field
                else (
                    "injection_id",
                    "waveform_id",
                    "gps_block",
                    "source_family",
                ),
            )
        )
    elif args.command == "evaluation-corpus-open-once":
        from .evaluation_lock import open_evaluation_corpus_once

        artifacts = {}
        for value in args.artifact:
            label, separator, path = value.partition("=")
            if not separator or not label or not path or label in artifacts:
                raise ValueError(
                    "--artifact must use a unique non-empty LABEL=PATH value"
                )
            artifacts[label] = path
        _print(
            open_evaluation_corpus_once(
                args.freeze_report,
                args.code_commit,
                artifacts,
                tuple(args.comparison_manifest),
                args.evaluation_output,
                args.evaluation_command,
                tuple(args.overlap_field)
                if args.overlap_field
                else ("injection_id", "waveform_id", "gps_block", "glitch_id"),
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
    elif args.command == "background-bank-materialize":
        from .waveforms import materialize_background_bank

        _print(
            materialize_background_bank(
                args.background_manifest,
                args.output_dir,
                args.target_sample_rate,
                args.context_duration,
                args.split,
                args.limit,
                args.maximum_windows_per_gps_block,
            )
        )
    elif args.command == "background-bank-evict-sources":
        from .waveforms import evict_verified_background_bank_sources

        _print(
            evict_verified_background_bank_sources(
                args.background_bank_report, args.cache_root, args.output
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
    elif args.command == "injection-arrival-annotate":
        from .waveforms import run_detector_arrival_annotation

        _print(run_detector_arrival_annotation(args.manifest, args.output_dir))
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
                required_split=args.required_split,
                enabled_ifos=(tuple(args.enabled_ifos) if args.enabled_ifos else None),
                coherence_config_path=args.coherence_config,
            )
        )
    elif args.command == "pe-evaluate":
        from .pe import run_pe_evaluation

        _print(
            run_pe_evaluation(
                args.manifest,
                args.output,
                args.credible_level,
                args.bootstrap_replicates,
                args.bootstrap_seed,
                args.require_publication_provenance,
            )
        )
    elif args.command == "pe-robustness-evaluate":
        from .pe import run_pe_robustness_evaluation

        _print(
            run_pe_robustness_evaluation(
                args.manifest,
                args.output,
                args.credible_level,
                args.bootstrap_replicates,
                args.bootstrap_seed,
                not args.allow_incomplete_provenance,
            )
        )
    elif args.command == "pe-robustness-joint-evaluate":
        from .pe import run_joint_pe_robustness_evaluation

        _print(
            run_joint_pe_robustness_evaluation(
                args.dingo_batch_report,
                args.amplfi_batch_report,
                args.manifest_output,
                args.output,
                args.credible_level,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            )
        )
    elif args.command == "pe-robustness-promote":
        from .pe import promote_pe_robustness_validation

        _print(
            promote_pe_robustness_validation(
                args.joint_report,
                args.config,
                args.output,
            )
        )
    elif args.command == "pe-input-materialize":
        from .pe_inputs import materialize_common_pe_inputs

        _print(
            materialize_common_pe_inputs(
                clean_manifest=args.clean_manifest,
                contaminated_manifest=args.contaminated_manifest,
                mask_conditioned_manifest=args.mask_conditioned_manifest,
                common_prior_path=args.common_prior,
                mask_model_path=args.mask_model,
                mask_policy_path=args.mask_policy,
                output_dir=args.output_dir,
                required_split=args.required_split,
                required_ifos=tuple(args.required_ifos),
                source_sample_rate_hz=args.source_sample_rate_hz,
                source_duration_seconds=args.source_duration_seconds,
                source_post_trigger_seconds=args.source_post_trigger_seconds,
                analysis_high_frequency_hz=args.analysis_high_frequency_hz,
                asd_segment_seconds=args.asd_segment_seconds,
                asd_stride_seconds=args.asd_stride_seconds,
                asd_guard_seconds=args.asd_guard_seconds,
                limit=args.limit,
                selection_seed=args.selection_seed,
            )
        )
    elif args.command == "pe-backend-lock-audit":
        from .pe_backend import run_pe_backend_lock_audit

        _print(
            run_pe_backend_lock_audit(
                args.config,
                args.output,
                args.allow_incomplete,
            )
        )
    elif args.command == "pe-native-condition":
        from .pe_conditioning import materialize_native_pe_conditioning

        _print(
            materialize_native_pe_conditioning(
                args.source_manifest,
                args.config,
                args.output_dir,
                args.required_split,
            )
        )
    elif args.command == "dingo-common-batch":
        from .dingo_adapter import run_dingo_common_batch

        _print(
            run_dingo_common_batch(
                args.native_manifest,
                args.model_metadata,
                args.native_prior,
                args.model_init,
                args.python_executable,
                args.runner_script,
                args.output_dir,
                args.required_split,
                args.num_samples,
                args.batch_size,
                args.num_gnpe_iterations,
                args.device,
                args.seed,
                args.comparison_mode,
            )
        )
    elif args.command == "dingo-official-native-model-freeze":
        from .dingo_adapter import freeze_official_dingo_native_model_metadata

        _print(
            freeze_official_dingo_native_model_metadata(
                args.source_config,
                args.acquisition_report,
                args.model_load_receipt,
                args.native_conditioning_config,
                args.output,
            )
        )
    elif args.command == "amplfi-common-batch":
        from .amplfi_adapter import run_amplfi_common_batch

        _print(
            run_amplfi_common_batch(
                args.native_manifest,
                args.model_metadata,
                args.native_prior,
                args.python_executable,
                args.runner_script,
                args.output_dir,
                args.required_split,
                args.num_samples,
                args.sample_batch_size,
                args.device,
                args.seed,
            )
        )
    elif args.command == "pe-backend-model-freeze":
        from .pe_backend import freeze_pe_backend_model_metadata

        _print(
            freeze_pe_backend_model_metadata(
                backend=args.backend,
                model_path=args.model,
                training_config_path=args.training_config,
                training_data_manifest_path=args.training_data_manifest,
                analysis_prior_path=args.analysis_prior,
                selection_report_path=args.selection_report,
                native_conditioning_config_path=args.native_conditioning_config,
                native_prior_path=args.native_prior,
                prior_projection_report_path=args.prior_projection_report,
                initialization_model_path=args.initialization_model,
                output_path=args.output,
                population=args.population,
                source_ifos=args.source_ifos,
                source_sample_rate_hz=args.source_sample_rate_hz,
                source_duration_seconds=args.source_duration_seconds,
                source_post_trigger_seconds=args.source_post_trigger_seconds,
                analysis_waveform_approximant=args.analysis_waveform_approximant,
                native_model_waveform_approximant=args.native_model_waveform_approximant,
                model_training_backend_version=args.model_training_backend_version,
                native_inference_parameters=args.native_inference_parameters,
                reported_parameter_mapping=args.reported_parameter_mapping,
            )
        )
    elif args.command == "pe-model-sources-acquire":
        from .external_models import run_external_model_source_acquisition

        _print(
            run_external_model_source_acquisition(
                args.config,
                args.output_dir,
                args.report,
                download=args.download,
                minimum_free_bytes=args.minimum_free_bytes,
                transfer_attempts=args.transfer_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
                maximum_stalled_attempts=args.maximum_stalled_attempts,
            )
        )
    elif args.command == "pe-lightning-checkpoint-select":
        from .pe_backend import select_lightning_validation_checkpoint

        _print(
            select_lightning_validation_checkpoint(
                training_config_path=args.training_config,
                training_data_manifest_path=args.training_data_manifest,
                metrics_csv_path=args.metrics_csv,
                checkpoint_index_path=args.checkpoint_index,
                output_path=args.output,
                selection_metric=args.selection_metric,
                selection_metric_mode=args.selection_metric_mode,
                minimum_publication_epochs=args.minimum_publication_epochs,
                minimum_validation_points=args.minimum_validation_points,
            )
        )
    elif args.command == "dingo-runtime-failure-adjudicate":
        from .pe_compatibility import run_dingo_runtime_failure_adjudication

        _print(
            run_dingo_runtime_failure_adjudication(
                args.failure_receipt,
                args.policy,
                args.output,
            )
        )
    elif args.command == "amplfi-background-export":
        from .amplfi_adapter import run_amplfi_group_safe_background_export

        _print(
            run_amplfi_group_safe_background_export(
                args.manifest,
                args.output_dir,
                args.report,
                target_sample_rate=args.target_sample_rate,
                minimum_segment_seconds=args.minimum_segment_seconds,
            )
        )
    elif args.command == "amplfi-background-capacity-audit":
        from .amplfi_adapter import run_amplfi_background_capacity_audit

        _print(
            run_amplfi_background_capacity_audit(
                args.manifest,
                args.policy,
                args.output,
            )
        )
    elif args.command == "amplfi-training-stage-freeze":
        from .amplfi_adapter import freeze_amplfi_training_stage_config

        _print(
            freeze_amplfi_training_stage_config(
                args.base_config,
                args.stage_policy,
                args.stage,
                args.output_config,
                args.output_report,
            )
        )
    elif args.command == "amplfi-common-prior-audit":
        from .amplfi_adapter import run_amplfi_common_prior_audit

        _print(
            run_amplfi_common_prior_audit(
                args.canonical_prior,
                args.amplfi_prior,
                args.training_config,
                args.output,
            )
        )
    elif args.command == "dingo-common-prior-audit":
        from .dingo_adapter import run_dingo_common_prior_audit

        _print(
            run_dingo_common_prior_audit(
                args.canonical_prior,
                args.dingo_prior_config,
                args.training_config,
                args.output,
            )
        )
    elif args.command == "ood-abstention-evaluate":
        from .ood import run_ood_abstention_evaluation

        _print(
            run_ood_abstention_evaluation(
                args.calibration_manifest,
                args.evaluation_manifest,
                args.output,
                args.maximum_known_abstention_rate,
                args.score_field,
            )
        )
    elif args.command == "gravityspy-ood-split":
        from .ood import build_leave_one_family_out_split

        _print(
            build_leave_one_family_out_split(
                args.train_manifest,
                args.validation_manifest,
                args.held_out_family,
                args.output_dir,
                args.seed,
            )
        )
    elif args.command == "gravityspy-ood-family-freeze":
        from .ood import freeze_ood_held_family_protocol

        _print(
            freeze_ood_held_family_protocol(
                args.train_manifest,
                args.validation_manifest,
                args.output,
                args.exclude_family,
                args.minimum_train_rows,
                args.minimum_validation_rows,
                args.minimum_validation_gps_blocks,
            )
        )
    elif args.command == "glitch-ood-train":
        from .ood import run_glitch_ood_embedding

        _print(
            run_glitch_ood_embedding(
                args.config,
                args.known_train_manifest,
                args.known_calibration_manifest,
                args.heldout_evaluation_manifest,
                args.output_dir,
                args.seed,
            )
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
