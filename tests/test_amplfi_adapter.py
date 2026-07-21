from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from gwyolo.amplfi_adapter import (
    audit_amplfi_common_prior_projection,
    export_amplfi_group_safe_background,
    run_amplfi_common_batch,
)
from gwyolo.io import file_sha256
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
r={'status':'real_amplfi_flow_posterior_complete','backend':'AMPLFI','backend_version':'0.6.0','event_sha256':a.expected_event_sha256,'model_sha256':a.expected_model_sha256,'model_config_sha256':a.expected_model_config_sha256,'native_prior_path':str(pathlib.Path(a.native_prior).resolve()),'native_prior_sha256':a.expected_native_prior_sha256,'posterior_path':str(pathlib.Path(a.posterior_output).resolve()),'posterior_sha256':sha(a.posterior_output),'native_result_path':str(pathlib.Path(a.result_output).resolve()),'native_result_sha256':sha(a.result_output),'latency_seconds':0.7,'latency_scope':'native','effective_sample_size':2.0,'environment':{'hostname':'gpu-node','gpu':'RTX 4090','python':'3.11','torch':'2','cuda':'12'}}
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
                "artifacts": {
                    "training_config": {
                        "path": str(training_config),
                        "sha256": file_sha256(training_config),
                    },
                    "native_conditioning_config": {
                        "path": str(conditioning_config),
                        "sha256": file_sha256(conditioning_config),
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
