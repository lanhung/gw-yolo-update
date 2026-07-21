from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from gwyolo.dingo_adapter import (
    audit_dingo_common_prior_projection,
    run_dingo_common_batch,
    run_dingo_common_prior_audit,
)
from gwyolo.io import file_sha256
from gwyolo.pe import PAIRED_PE_LATENCY_SCOPE_V1
from gwyolo.pe_conditioning import materialize_native_pe_conditioning
from test_pe_conditioning import _common_sources


def _native_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = _common_sources(tmp_path)
    config = tmp_path / "dingo-config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "backend": "DINGO",
                "ifos": ["H1", "L1"],
                "source_sample_rate_hz": 16,
                "source_duration_seconds": 4,
                "source_post_trigger_seconds": 1,
                "window": {"type": "tukey", "roll_off_seconds": 0.5},
                "frequency_domain": {
                    "minimum_frequency_hz": 1,
                    "maximum_frequency_hz": 4,
                    "delta_frequency_hz": 0.25,
                    "fourier_convention": "numpy_rfft_times_delta_t",
                    "time_translation": "exp_minus_2pi_i_f_post_trigger",
                },
                "asd": {
                    "source": "common_source_artifact",
                    "condition_invariant_required": True,
                    "below_minimum_frequency_value": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    report = materialize_native_pe_conditioning(source, config, tmp_path / "native", "val")
    return Path(report["manifest_path"]), config


def _fake_runner(path: Path) -> None:
    path.write_text(
        """import argparse,hashlib,json,numpy as np,pathlib
p=argparse.ArgumentParser()
for x in ('event','model','model_init','posterior_output','result_output','report_output','expected_event_sha256','expected_model_sha256','expected_model_init_sha256','num_samples','batch_size','num_gnpe_iterations','device','seed'):
 p.add_argument('--'+x.replace('_','-'))
a=p.parse_args()
def sha(path):
 return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
np.savez(a.posterior_output, chirp_mass=np.array([20.,21.]), mass_ratio=np.array([.5,.6]), luminosity_distance=np.array([900.,1100.]), theta_jn=np.array([.6,.8]), ra=np.array([.9,1.1]), dec=np.array([.1,.3]), psi=np.array([.4,.6]))
pathlib.Path(a.result_output).write_bytes(b'native-result')
r={'status':'real_dingo_gnpe_posterior_complete','backend':'DINGO','backend_version':'0.9.8','event_sha256':a.expected_event_sha256,'model_sha256':a.expected_model_sha256,'model_init_sha256':a.expected_model_init_sha256,'posterior_path':str(pathlib.Path(a.posterior_output).resolve()),'posterior_sha256':sha(a.posterior_output),'native_result_path':str(pathlib.Path(a.result_output).resolve()),'native_result_sha256':sha(a.result_output),'latency_seconds':1.5,'latency_scope':'model-load-and-event-preprocessing-through-posterior-and-native-result-write_v1_excludes-artifact-verification-imports-and-mask-generation','latency_components_seconds':{'model_load':.2,'event_preprocessing':.1,'posterior_sampling':1.0,'posterior_postprocessing_and_write':.1},'effective_sample_size':2.0,'environment':{'hostname':'gpu-node','gpu':'RTX 4090','python':'3.11','torch':'2','cuda':'12'}}
pathlib.Path(a.report_output).write_text(json.dumps(r))
""",
        encoding="utf-8",
    )


def _compatible_dingo_settings() -> dict:
    return {
        "dataset_settings": {
            "intrinsic_prior": {
                "chirp_mass": "bilby.core.prior.Uniform(minimum=15.0, maximum=100.0)",
                "mass_ratio": "bilby.core.prior.Uniform(minimum=0.125, maximum=0.999)",
                "phase": "default",
                "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
                "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
                "tilt_1": "default",
                "tilt_2": "default",
                "phi_12": "default",
                "phi_jl": "default",
                "theta_jn": "default",
                "luminosity_distance": 100.0,
            }
        },
        "train_settings": {
            "data": {
                "detectors": ["H1", "L1"],
                "extrinsic_prior": {
                    "luminosity_distance": (
                        "bilby.core.prior.Uniform(minimum=100.0, maximum=3100.0)"
                    ),
                    "ra": "default",
                    "dec": "default",
                    "psi": "default",
                },
                "inference_parameters": [
                    "chirp_mass",
                    "mass_ratio",
                    "luminosity_distance",
                    "theta_jn",
                    "ra",
                    "dec",
                    "psi",
                ],
                "window": {"f_s": 4096, "T": 16.0},
                "domain_update": {"f_min": 20.0, "f_max": 1024.0},
            }
        },
    }


def test_dingo_prior_audit_parses_model_inspection_and_rejects_official_mismatch(
    tmp_path: Path,
) -> None:
    canonical = Path(__file__).parents[1] / "configs/pe_common_bbh_analysis_prior.yaml"
    compatible = tmp_path / "compatible-settings.yaml"
    compatible.write_text(
        "Extracting information about torch model.\n\n"
        "Version: dingo=0.9.8\n\n"
        "Model metadata:\n"
        + yaml.safe_dump(_compatible_dingo_settings(), sort_keys=False),
        encoding="utf-8",
    )
    passed = audit_dingo_common_prior_projection(canonical, compatible, compatible)
    assert passed["publication_ready"] is True
    assert passed["failures"] == []
    assert set(passed["checks"]) == {
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "theta_jn",
        "phase",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_jl",
        "phi_12",
        "ra",
        "dec",
        "psi",
    }

    official = _compatible_dingo_settings()
    intrinsic = official["dataset_settings"]["intrinsic_prior"]
    intrinsic["mass_1"] = "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)"
    intrinsic["mass_2"] = "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)"
    intrinsic["chirp_mass"] = (
        "bilby.gw.prior.UniformInComponentsChirpMass(minimum=15.0, maximum=150.0)"
    )
    intrinsic["mass_ratio"] = (
        "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)"
    )
    extrinsic = official["train_settings"]["data"]["extrinsic_prior"]
    extrinsic["luminosity_distance"] = (
        "bilby.core.prior.Uniform(minimum=100.0, maximum=10000.0)"
    )
    extrinsic["geocent_time"] = (
        "bilby.core.prior.Uniform(minimum=-0.10, maximum=0.10)"
    )
    official_path = tmp_path / "official-settings.yaml"
    official_path.write_text(
        "Extracting information about torch model.\n\n"
        "Version: dingo=0.5.8\n\n"
        "Model metadata:\n"
        + yaml.safe_dump(official, sort_keys=False),
        encoding="utf-8",
    )
    failed = audit_dingo_common_prior_projection(canonical, official_path, official_path)
    assert failed["publication_ready"] is False
    assert "mass_1" in failed["extra_native_constraints"]
    assert failed["extra_native_stochastic_parameters"] == ["geocent_time"]
    assert any("prior family mismatch for chirp_mass" == value for value in failed["failures"])
    assert any("prior bounds mismatch for luminosity_distance" == value for value in failed["failures"])
    output = tmp_path / "official-prior-audit.json"
    with pytest.raises(RuntimeError, match="prior projection failed"):
        run_dingo_common_prior_audit(canonical, official_path, official_path, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["dingo_prior_config_sha256"] == file_sha256(official_path)


def test_dingo_common_batch_runs_and_resumes_real_runner_contract(tmp_path: Path) -> None:
    native, conditioning_config = _native_manifest(tmp_path)
    native_rows = [json.loads(line) for line in native.read_text().splitlines()]
    model = tmp_path / "model.pt"
    model_init = tmp_path / "model-init.pt"
    model.write_bytes(b"model")
    model_init.write_bytes(b"init")
    training_config = tmp_path / "training.yaml"
    training_config.write_text("backend: DINGO\n", encoding="utf-8")
    training_manifest = tmp_path / "training.jsonl"
    training_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    analysis_prior = Path(native_rows[0]["common_prior_path"])
    native_prior = tmp_path / "dingo-native-prior.yaml"
    native_prior.write_text("native: dingo-prior\n", encoding="utf-8")
    prior_projection = tmp_path / "dingo-prior-projection.json"
    prior_projection.write_text(
        json.dumps(
            {
                "status": "passed",
                "publication_ready": True,
                "canonical_prior_sha256": file_sha256(analysis_prior),
                "dingo_prior_config_sha256": file_sha256(native_prior),
                "dingo_training_config_sha256": file_sha256(training_config),
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
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
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "backend": "DINGO",
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
                    "training_data_manifest": {
                        "path": str(training_manifest),
                        "sha256": file_sha256(training_manifest),
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
                    "selection_report": {
                        "path": str(selection_report),
                        "sha256": file_sha256(selection_report),
                    },
                    "native_conditioning_config": {
                        "path": str(conditioning_config),
                        "sha256": file_sha256(conditioning_config),
                    },
                    "initialization_model": {
                        "path": str(model_init),
                        "sha256": file_sha256(model_init),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    _fake_runner(runner)
    kwargs = dict(
        native_manifest=native,
        model_metadata_path=metadata,
        native_prior_path=native_prior,
        model_init_path=model_init,
        python_executable=sys.executable,
        runner_script=runner,
        output_dir=tmp_path / "posterior",
        required_split="val",
        num_samples=2,
        batch_size=1,
        num_gnpe_iterations=2,
        device="cpu",
    )
    report = run_dingo_common_batch(**kwargs)
    assert report["rows"] == 3
    rows = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert len({row["source_event_hash"] for row in rows}) == 1
    assert all(row["backend_version"] == "0.9.8" for row in rows)
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
    assert all(row["hardware"] == {"hostname": "gpu-node", "gpu": "RTX 4090"} for row in rows)
    assert all(file_sha256(row["posterior_path"]) == row["posterior_sha256"] for row in rows)
    assert report["run_identity"]["native_prior_sha256"] == file_sha256(native_prior)
    assert report["run_identity"]["prior_projection_report_sha256"] == file_sha256(
        prior_projection
    )

    resumed = run_dingo_common_batch(**kwargs)
    assert resumed["manifest_sha256"] == report["manifest_sha256"]

    other_init = tmp_path / "other-init.pt"
    other_init.write_bytes(b"changed-init")
    with pytest.raises(ValueError, match="runtime initialization model differs"):
        run_dingo_common_batch(**{**kwargs, "model_init_path": other_init})

    other_prior = tmp_path / "other-prior.yaml"
    other_prior.write_text("native: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime native prior differs"):
        run_dingo_common_batch(**{**kwargs, "native_prior_path": other_prior})
