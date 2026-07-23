from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from gwyolo.amplfi_adapter import (
    audit_amplfi_background_capacity,
    audit_amplfi_common_prior_projection,
    export_amplfi_group_safe_background,
    freeze_amplfi_training_bank,
    freeze_amplfi_training_stage_config,
    merge_amplfi_streamed_background_extension,
    run_amplfi_background_capacity_audit,
    run_amplfi_common_batch,
)
from gwyolo.io import file_sha256
from gwyolo.pe import PAIRED_PE_LATENCY_SCOPE_V1
from gwyolo.pe_conditioning import materialize_native_pe_conditioning
from test_pe_conditioning import _common_sources


def _source(path: Path, ifo: str, start: int = 1000, rate: int = 4) -> None:
    values = np.arange(64 * rate, dtype=np.float64) + (1 if ifo == "L1" else 0)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("strain/Strain", data=values)
        dataset.attrs["Xspacing"] = 1 / rate
        dataset.attrs["Xstart"] = start


def _rows(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    sources = {ifo: tmp_path / f"{ifo}.hdf5" for ifo in ("H1", "L1")}
    for ifo, path in sources.items():
        _source(path, ifo)
    identities = {
        ifo: {"path": str(path), "sha256": file_sha256(path)}
        for ifo, path in sources.items()
    }
    rows = []
    for split, block_start in (("train", 1000), ("val", 1032)):
        for offset in (0, 8, 16, 24):
            rows.append(
                {
                    "split": split,
                    "ifos": ["H1", "L1"],
                    "gps_block": f"gps:{block_start}:32",
                    "pair_id": f"pair-{block_start}",
                    "gps_start": block_start + offset,
                    "gps_end": block_start + offset + 8,
                    "source_files": identities,
                }
            )
    manifest = tmp_path / "background.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest, sources


def test_amplfi_background_export_preserves_group_safe_splits(tmp_path: Path) -> None:
    manifest, _ = _rows(tmp_path)
    report = export_amplfi_group_safe_background(
        manifest,
        tmp_path / "amplfi",
        target_sample_rate=4,
        minimum_segment_seconds=16,
    )
    assert report["split_file_counts"] == {"train": 1, "val": 1, "test": 0}
    assert report["split_duration_seconds"] == {"train": 32.0, "val": 32.0, "test": 0.0}
    assert report["cross_split_gps_block_overlap"] == 0
    validation = Path(report["files"][1]["path"])
    assert validation.parts[-3:-1] == ("validation", "background")
    with h5py.File(validation) as handle:
        assert handle["H1"].shape == (128,)
        assert handle["H1"].attrs["dx"] == 0.25
        assert handle.attrs["gps_block"] == "gps:1032:32"


def test_amplfi_background_export_rejects_cross_split_gps_block(
    tmp_path: Path,
) -> None:
    manifest, _ = _rows(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[-1]["gps_block"] = "gps:1000:32"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="cross AMPLFI export splits"):
        export_amplfi_group_safe_background(manifest, tmp_path / "amplfi", target_sample_rate=4)


def _capacity_policy(tmp_path: Path, duration: int = 32) -> Path:
    policy = tmp_path / "capacity.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "required_ifos": ["H1", "L1"],
                "minimum_contiguous_segment_seconds": 16,
                "minimum_duration_seconds": {"train": duration, "val": duration},
                "minimum_gps_blocks": {"train": 1, "val": 1},
            }
        ),
        encoding="utf-8",
    )
    return policy


def test_amplfi_background_capacity_uses_physical_duration_and_excludes_test(
    tmp_path: Path,
) -> None:
    manifest, _ = _rows(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    test_row = {**rows[0]}
    test_row.update(
        {
            "split": "test",
            "gps_block": "gps:1064:32",
            "pair_id": "pair-1064",
            "gps_start": 1064,
            "gps_end": 1096,
        }
    )
    rows.append(test_row)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = audit_amplfi_background_capacity(
        manifest, _capacity_policy(tmp_path)
    )
    assert report["passed"] is True
    assert report["checks"]["train"]["duration_seconds"] == 32
    assert report["checks"]["val"]["duration_seconds"] == 32
    assert report["test_metadata_rows_excluded"] == 1
    assert report["test_strain_rows_read"] == 0
    assert report["strain_arrays_read"] == 0


def test_amplfi_background_capacity_writes_failure_report_before_nonzero(
    tmp_path: Path,
) -> None:
    manifest, _ = _rows(tmp_path)
    output = tmp_path / "capacity-report.json"
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        run_amplfi_background_capacity_audit(
            manifest,
            _capacity_policy(tmp_path, duration=33),
            output,
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "amplfi_background_capacity_insufficient"
    assert report["checks"]["train"]["duration_passed"] is False
    assert report["checks"]["val"]["duration_passed"] is False


def test_amplfi_stream_extension_is_source_disjoint_and_capacity_auditable(
    tmp_path: Path,
) -> None:
    base_manifest = tmp_path / "base.jsonl"
    base_rows = [
        {
            "window_id": "base-train",
            "split": "train",
            "ifos": ["H1", "L1"],
            "gps_block": "gps:1000:32",
            "pair_id": "base-pair",
            "gps_start": 1000,
            "gps_end": 1032,
        },
        {
            "window_id": "base-val",
            "split": "val",
            "ifos": ["H1", "L1"],
            "gps_block": "gps:1100:32",
            "pair_id": "base-val-pair",
            "gps_start": 1100,
            "gps_end": 1132,
        },
    ]
    base_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows),
        encoding="utf-8",
    )
    parent_plan = tmp_path / "base-plan.json"
    parent_plan.write_text('{"frozen":true}\n', encoding="utf-8")
    base_merge = tmp_path / "base-merge.json"
    base_merge.write_text(
        json.dumps(
            {
                "status": "verified_streamed_amplfi_background_bank",
                "passed": True,
                "recoverable": True,
                "test_strain_rows_read": 0,
                "test_rows_exported": 0,
                "parent_plan_sha256": file_sha256(parent_plan),
                "background_manifest_path": str(base_manifest),
                "background_manifest_sha256": file_sha256(base_manifest),
            }
        ),
        encoding="utf-8",
    )
    extension_plan = tmp_path / "extension-plan.json"
    extension_plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "selection_rule": "stratified_exclusion_complement_v1",
                "candidate_scores_inspected": False,
                "test_data_opened": False,
                "locked_evaluation_data": False,
                "run": "O4a",
                "detectors": ["H1", "L1"],
                "selected_pairs": 1,
                "pairs": [{"pair_id": "extension-pair", "gps_start": 2000}],
                "exclusion_plans": [
                    {"sha256": file_sha256(parent_plan)}
                ],
            }
        ),
        encoding="utf-8",
    )
    shard = tmp_path / "shard-0"
    (shard / "download").mkdir(parents=True)
    (shard / "background").mkdir()
    shard_plan = shard / "acquisition_plan.json"
    shard_plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "locked_evaluation_data": False,
                "parent_plan_sha256": file_sha256(extension_plan),
                "pairs": [{"pair_id": "extension-pair", "gps_start": 2000}],
            }
        ),
        encoding="utf-8",
    )
    batch = shard / "download" / "batch_download_report.json"
    batch.write_text(
        json.dumps(
            {
                "status": "verified_development_strain_batch",
                "passed": True,
                "plan_sha256": file_sha256(shard_plan),
            }
        ),
        encoding="utf-8",
    )
    extension_manifest = shard / "background" / "background.jsonl"
    extension_manifest.write_text(
        json.dumps(
            {
                "window_id": "extension-train",
                "split": "train",
                "ifos": ["H1", "L1"],
                "gps_block": "gps:2000:32",
                "pair_id": "extension-pair",
                "gps_start": 2000,
                "gps_end": 2032,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    background = shard / "background" / "background_plan_report.json"
    background.write_text(
        json.dumps(
            {
                "status": "verified_multi_segment_development_background",
                "passed": True,
                "split_strategy": "hash_threshold_v1",
                "splits": {"test": {"windows": 0}},
                "source_batch_report_sha256s": [file_sha256(batch)],
                "manifest_path": str(extension_manifest),
                "manifest_sha256": file_sha256(extension_manifest),
            }
        ),
        encoding="utf-8",
    )
    exported = shard / "exported.hdf5"
    exported.write_bytes(b"verified exported strain")
    export = shard / "amplfi_export_report.json"
    export.write_text(
        json.dumps(
            {
                "status": "group_safe_amplfi_background",
                "manifest_sha256": file_sha256(extension_manifest),
                "split_file_counts": {"train": 1, "val": 0, "test": 0},
                "files": [
                    {"path": str(exported), "sha256": file_sha256(exported)}
                ],
            }
        ),
        encoding="utf-8",
    )
    eviction = shard / "source_eviction_report.json"
    eviction.write_text(
        json.dumps(
            {
                "status": "verified_exported_amplfi_source_eviction",
                "recoverable": True,
                "amplfi_export_report_sha256": file_sha256(export),
            }
        ),
        encoding="utf-8",
    )

    result = merge_amplfi_streamed_background_extension(
        base_merge,
        extension_plan,
        [shard],
        tmp_path / "merged",
    )

    assert result["passed"] is True
    assert result["extension_source_pairs"] == 1
    assert result["background_windows"] == 3
    assert result["test_rows_exported"] == 0
    capacity = audit_amplfi_background_capacity(
        result["background_manifest_path"],
        _capacity_policy(tmp_path),
    )
    assert capacity["passed"] is True


def test_amplfi_stream_extension_rejects_plan_without_base_exclusion(
    tmp_path: Path,
) -> None:
    base_manifest = tmp_path / "base.jsonl"
    base_manifest.write_text(
        json.dumps(
            {
                "window_id": "base",
                "split": "train",
                "ifos": ["H1", "L1"],
                "gps_block": "gps:1000:32",
                "pair_id": "base-pair",
                "gps_start": 1000,
                "gps_end": 1032,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "status": "verified_streamed_amplfi_background_bank",
                "passed": True,
                "recoverable": True,
                "test_strain_rows_read": 0,
                "test_rows_exported": 0,
                "parent_plan_sha256": "required-parent",
                "background_manifest_path": str(base_manifest),
                "background_manifest_sha256": file_sha256(base_manifest),
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "selection_rule": "stratified_exclusion_complement_v1",
                "candidate_scores_inspected": False,
                "test_data_opened": False,
                "locked_evaluation_data": False,
                "run": "O4a",
                "detectors": ["H1", "L1"],
                "pairs": [{"pair_id": "new", "gps_start": 2000}],
                "exclusion_plans": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not exclude"):
        merge_amplfi_streamed_background_extension(
            base, plan, [tmp_path / "not-read"], tmp_path / "output"
        )


def test_amplfi_training_bank_freezes_base_and_extension_without_copy(
    tmp_path: Path,
) -> None:
    base_source = tmp_path / "base-train.hdf5"
    extension_source = tmp_path / "extension-val.hdf5"
    base_source.write_bytes(b"base train strain")
    extension_source.write_bytes(b"extension validation strain")

    def export_report(path: Path, source: Path, split: str) -> dict[str, str]:
        payload = {
            "status": "group_safe_amplfi_background",
            "split_file_counts": {
                "train": int(split == "train"),
                "val": int(split == "val"),
                "test": 0,
            },
            "files": [
                {
                    "path": str(source),
                    "sha256": file_sha256(source),
                    "split": split,
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": str(path), "sha256": file_sha256(path)}

    base_export = export_report(tmp_path / "base-export.json", base_source, "train")
    extension_export = export_report(
        tmp_path / "extension-export.json", extension_source, "val"
    )
    base_manifest = tmp_path / "base-manifest.jsonl"
    base_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    base_merge = tmp_path / "base-merge.json"
    base_merge.write_text(
        json.dumps(
            {
                "status": "verified_streamed_amplfi_background_bank",
                "passed": True,
                "test_strain_rows_read": 0,
                "test_rows_exported": 0,
                "background_manifest_path": str(base_manifest),
                "background_manifest_sha256": file_sha256(base_manifest),
                "shards": [{"export": base_export}],
            }
        ),
        encoding="utf-8",
    )
    merged_manifest = tmp_path / "merged-manifest.jsonl"
    merged_manifest.write_text(
        '{"split":"train"}\n{"split":"val"}\n', encoding="utf-8"
    )
    extension_merge = tmp_path / "extension-merge.json"
    extension_merge.write_text(
        json.dumps(
            {
                "status": "verified_extended_streamed_amplfi_background_bank",
                "passed": True,
                "test_strain_rows_read": 0,
                "test_rows_exported": 0,
                "base_merge_report_path": str(base_merge),
                "base_merge_report_sha256": file_sha256(base_merge),
                "extension_shards": [{"export": extension_export}],
                "background_manifest_path": str(merged_manifest),
                "background_manifest_sha256": file_sha256(merged_manifest),
            }
        ),
        encoding="utf-8",
    )
    capacity = tmp_path / "capacity.json"
    capacity.write_text(
        json.dumps(
            {
                "status": "amplfi_background_capacity_ready",
                "passed": True,
                "test_strain_rows_read": 0,
                "manifest_sha256": file_sha256(merged_manifest),
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "extension-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "verified_capacity_ready_amplfi_background_extension",
                "passed": True,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "stream_merge_report_path": str(extension_merge),
                "stream_merge_report_sha256": file_sha256(extension_merge),
                "capacity_report_path": str(capacity),
                "capacity_report_sha256": file_sha256(capacity),
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "training-bank"
    report = freeze_amplfi_training_bank(receipt, output)

    assert report["status"] == "frozen_hash_bound_amplfi_training_bank"
    assert report["file_counts"] == {"train": 1, "val": 1}
    assert report["source_bytes"] == base_source.stat().st_size + extension_source.stat().st_size
    links = [output / item["relative_path"] for item in report["files"]]
    assert all(path.is_symlink() for path in links)
    assert {path.read_bytes() for path in links} == {
        base_source.read_bytes(),
        extension_source.read_bytes(),
    }
    assert (output / "amplfi_training_bank_report.json").is_file()


def test_amplfi_training_stage_freezes_hand_calculated_compute_budget(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    output_config = tmp_path / "stage.yaml"
    output_report = tmp_path / "stage.json"
    report = freeze_amplfi_training_stage_config(
        root / "configs/amplfi_common_bbh_publication.yaml",
        root / "configs/amplfi_training_stage_policy.yaml",
        "publication_stage_1",
        output_config,
        output_report,
    )
    resolved = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert resolved["trainer"]["max_epochs"] == 100
    assert resolved["data"]["init_args"]["batches_per_epoch"] == 200
    assert resolved["data"]["init_args"]["batch_size"] == 256
    assert resolved["data"]["init_args"]["min_valid_duration"] == 50000
    assert resolved["trainer"]["logger"]["init_args"]["version"] == (
        "gwyolo_publication_stage_1"
    )
    assert report["compute_budget"]["updates"] == 20000
    assert report["compute_budget"]["online_waveform_examples"] == 5_120_000
    assert report["publication_candidate"] is True
    assert report["test_rows_read"] == 0
    assert report["resolved_config_sha256"] == file_sha256(output_config)


def test_amplfi_training_stage_rejects_unbound_base_config(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    base = tmp_path / "changed.yaml"
    base.write_bytes((root / "configs/amplfi_common_bbh_publication.yaml").read_bytes())
    with base.open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    with pytest.raises(ValueError, match="does not bind"):
        freeze_amplfi_training_stage_config(
            base,
            root / "configs/amplfi_training_stage_policy.yaml",
            "publication_stage_1",
            tmp_path / "resolved.yaml",
            tmp_path / "report.json",
        )


def test_amplfi_common_prior_projection_matches_every_native_distribution() -> None:
    root = Path(__file__).parents[1]
    report = audit_amplfi_common_prior_projection(
        root / "configs/pe_common_bbh_analysis_prior.yaml",
        root / "configs/amplfi_common_bbh_training_prior.yaml",
        root / "configs/amplfi_common_bbh_publication.yaml",
    )
    assert report["publication_ready"] is True
    assert len(report["checks"]) == 14
    assert report["checks"]["luminosity_distance"]["native_bounds"] == [100.0, 3100.0]


def _amplfi_native_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = _common_sources(tmp_path)
    config = tmp_path / "amplfi-conditioning.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "backend": "AMPLFI",
                "ifos": ["H1", "L1"],
                "source_sample_rate_hz": 16,
                "source_duration_seconds": 4,
                "source_post_trigger_seconds": 1,
                "native_sample_rate_hz": 8,
                "native_kernel_seconds": 1,
                "native_whitening_duration_seconds": 1,
                "native_highpass_hz": 1,
                "native_right_pad_seconds": 0.25,
                "resampling": {
                    "method": "scipy_signal_resample_poly",
                    "window": ["kaiser", 8.6],
                },
                "asd": {
                    "source": "common_source_artifact",
                    "condition_invariant_required": True,
                    "runtime_whitening_must_not_reestimate_psd": True,
                },
            }
        ),
        encoding="utf-8",
    )
    report = materialize_native_pe_conditioning(
        source, config, tmp_path / "amplfi-native", "val"
    )
    return Path(report["manifest_path"]), config


def _fake_amplfi_runner(path: Path) -> None:
    path.write_text(
        """import argparse,hashlib,json,numpy as np,pathlib
p=argparse.ArgumentParser()
for x in ('event','model','model_config','native_prior','posterior_output','result_output','report_output','expected_event_sha256','expected_model_sha256','expected_model_config_sha256','expected_native_prior_sha256','num_samples','sample_batch_size','device','seed'):
 p.add_argument('--'+x.replace('_','-'))
a=p.parse_args()
def sha(path):
 return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
np.savez(a.posterior_output, chirp_mass=np.array([20.,21.]), mass_ratio=np.array([.5,.6]), luminosity_distance=np.array([900.,1100.]), theta_jn=np.array([.6,.8]), ra=np.array([.9,1.1]), dec=np.array([.1,.3]), psi=np.array([.4,.6]))
pathlib.Path(a.result_output).write_bytes(b'native-amplfi-result')
r={'status':'real_amplfi_flow_posterior_complete','backend':'AMPLFI','backend_version':'0.6.0','event_sha256':a.expected_event_sha256,'model_sha256':a.expected_model_sha256,'model_config_sha256':a.expected_model_config_sha256,'native_prior_path':str(pathlib.Path(a.native_prior).resolve()),'native_prior_sha256':a.expected_native_prior_sha256,'posterior_path':str(pathlib.Path(a.posterior_output).resolve()),'posterior_sha256':sha(a.posterior_output),'native_result_path':str(pathlib.Path(a.result_output).resolve()),'native_result_sha256':sha(a.result_output),'latency_seconds':0.7,'latency_scope':'model-load-and-event-preprocessing-through-posterior-and-native-result-write_v1_excludes-artifact-verification-imports-and-mask-generation','latency_components_seconds':{'model_load':.1,'event_preprocessing':.1,'posterior_sampling':.4,'posterior_postprocessing_and_write':.05},'effective_sample_size':2.0,'environment':{'hostname':'gpu-node','gpu':'RTX 4090','python':'3.11','torch':'2','cuda':'12'}}
pathlib.Path(a.report_output).write_text(json.dumps(r))
""",
        encoding="utf-8",
    )


def test_amplfi_common_batch_runs_and_resumes_real_runner_contract(
    tmp_path: Path,
) -> None:
    native, conditioning_config = _amplfi_native_manifest(tmp_path)
    model = tmp_path / "amplfi.ckpt"
    model.write_bytes(b"model")
    training_config = tmp_path / "amplfi-training.yaml"
    training_config.write_text("model: frozen\n", encoding="utf-8")
    training_manifest = tmp_path / "amplfi-training.jsonl"
    training_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    selection_report = tmp_path / "selection.json"
    selection_report.write_text(
        json.dumps(
            {
                "status": "validation_selected_checkpoint",
                "publication_eligible": True,
                "selection_split": "validation",
                "selection_metric": "validation_loss",
                "selected_checkpoint_sha256": file_sha256(model),
            }
        ),
        encoding="utf-8",
    )
    analysis_prior = tmp_path / "analysis-prior.yaml"
    analysis_prior.write_text("prior: common\n", encoding="utf-8")
    native_prior = tmp_path / "amplfi-prior.yaml"
    native_prior.write_text("prior: frozen\n", encoding="utf-8")
    prior_projection = tmp_path / "prior-projection.json"
    prior_projection.write_text(
        json.dumps(
            {
                "status": "passed",
                "publication_ready": True,
                "canonical_prior_sha256": file_sha256(analysis_prior),
                "amplfi_prior_sha256": file_sha256(native_prior),
                "amplfi_training_config_sha256": file_sha256(training_config),
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "backend": "AMPLFI",
                "selection_split": "validation",
                "model_path": str(model),
                "model_sha256": file_sha256(model),
                "source_input": {
                    "ifos": ["H1", "L1"],
                    "common_asd_required": True,
                    "sample_rate_hz": 16,
                    "duration_seconds": 4,
                    "post_trigger_seconds": 1,
                },
                "analysis_waveform_approximant": "IMRPhenomXPHM",
                "selection_metric": "validation_loss",
                "artifacts": {
                    "training_config": {
                        "path": str(training_config),
                        "sha256": file_sha256(training_config),
                    },
                    "native_conditioning_config": {
                        "path": str(conditioning_config),
                        "sha256": file_sha256(conditioning_config),
                    },
                    "training_data_manifest": {
                        "path": str(training_manifest),
                        "sha256": file_sha256(training_manifest),
                    },
                    "selection_report": {
                        "path": str(selection_report),
                        "sha256": file_sha256(selection_report),
                    },
                    "analysis_prior": {
                        "path": str(analysis_prior),
                        "sha256": file_sha256(analysis_prior),
                    },
                    "native_prior": {
                        "path": str(native_prior),
                        "sha256": file_sha256(native_prior),
                    },
                    "prior_projection_report": {
                        "path": str(prior_projection),
                        "sha256": file_sha256(prior_projection),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "amplfi-runner.py"
    _fake_amplfi_runner(runner)
    kwargs = dict(
        native_manifest=native,
        model_metadata_path=metadata,
        native_prior_path=native_prior,
        python_executable=sys.executable,
        runner_script=runner,
        output_dir=tmp_path / "amplfi-posterior",
        required_split="val",
        num_samples=2,
        sample_batch_size=1,
        device="cpu",
    )
    report = run_amplfi_common_batch(**kwargs)
    assert report["rows"] == 3
    rows = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    assert all(row["sky_area_90_deg2"] > 0 for row in rows)
    assert all(row["latency_scope"] == PAIRED_PE_LATENCY_SCOPE_V1 for row in rows)
    assert all(
        set(row["backend_native_latency_components_seconds"])
        == {
            "model_load",
            "event_preprocessing",
            "posterior_sampling",
            "posterior_postprocessing_and_write",
        }
        for row in rows
    )
    assert all(
        row["sky_area_estimator"]["method"]
        == "fixed_equal_solid_angle_histogram_v1"
        for row in rows
    )
    rows = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    assert len({row["source_event_hash"] for row in rows}) == 1
    assert all(row["backend_version"] == "0.6.0" for row in rows)
    assert all(row["hardware"] == {"hostname": "gpu-node", "gpu": "RTX 4090"} for row in rows)
    assert all(file_sha256(row["posterior_path"]) == row["posterior_sha256"] for row in rows)

    resumed = run_amplfi_common_batch(**kwargs)
    assert resumed["manifest_sha256"] == report["manifest_sha256"]

    other_prior = tmp_path / "other-prior.yaml"
    other_prior.write_text("prior: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime native prior differs"):
        run_amplfi_common_batch(**{**kwargs, "native_prior_path": other_prior})
