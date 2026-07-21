from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .injection_score import apply_analysis_override
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .overlaps import _fft_upsample
from .runtime import execution_provenance
from .waveforms import _atomic_save_npz, load_materialized_context


CONDITIONS = ("clean", "contaminated", "mask_conditioned")
IDENTITY_FIELDS = (
    "waveform_id",
    "source_family",
    "split",
    "gps_block",
    "gps_time",
    "mass_1_detector_msun",
    "mass_2_detector_msun",
    "luminosity_distance_mpc",
    "right_ascension",
    "declination",
    "polarization",
    "inclination",
    "coalescence_phase",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"PE source manifest is empty: {path}")
    return rows


def _by_injection(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["injection_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} manifest repeats an injection ID")
    return result


def _canonical_truth(row: dict[str, Any]) -> dict[str, float]:
    mass_1 = float(row["mass_1_detector_msun"])
    mass_2 = float(row["mass_2_detector_msun"])
    if mass_1 < mass_2 or mass_2 <= 0:
        raise ValueError("PE truth requires detector-frame mass_1 >= mass_2 > 0")
    chirp_mass = (mass_1 * mass_2) ** (3.0 / 5.0) / (mass_1 + mass_2) ** (1.0 / 5.0)
    return {
        "chirp_mass": float(chirp_mass),
        "mass_ratio": float(mass_2 / mass_1),
        "luminosity_distance": float(row["luminosity_distance_mpc"]),
        "theta_jn": float(row["inclination"]),
        "ra": float(row["right_ascension"]),
        "dec": float(row["declination"]),
        "psi": float(row["polarization"]),
        "phase": float(row["coalescence_phase"]),
        "geocent_time": float(row["gps_time"]),
    }


def _prior_support(truth: dict[str, float], prior: dict[str, Any]) -> list[str]:
    failures = []
    distributions = prior.get("distributions")
    if not isinstance(distributions, dict) or not distributions:
        raise ValueError("Common PE prior requires a non-empty distributions mapping")
    for parameter, specification in distributions.items():
        if parameter not in truth:
            raise ValueError(f"PE truth cannot evaluate common-prior parameter {parameter}")
        if not isinstance(specification, dict):
            raise ValueError(f"Common-prior parameter {parameter} is not a mapping")
        value = truth[parameter]
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        if not minimum <= value <= maximum:
            failures.append(parameter)
    return failures


def _selection_key(injection_id: str, seed: int) -> str:
    return hashlib.sha256(f"pe_source_v1\0{seed}\0{injection_id}".encode()).hexdigest()


def _crop_and_resample(
    context: dict[str, Any],
    required_ifos: tuple[str, ...],
    duration_seconds: float,
    target_sample_rate: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    ifos = list(context["ifos"])
    if any(ifo not in ifos for ifo in required_ifos):
        raise ValueError("Materialized context lacks a required PE detector")
    source_rate = int(context["sample_rate"])
    if target_sample_rate < source_rate or target_sample_rate % source_rate:
        raise ValueError("PE source rate must be an integer multiple of the materialized rate")
    source_samples = int(round(duration_seconds * source_rate))
    if source_samples <= 0 or not np.isclose(
        source_samples / source_rate, duration_seconds, rtol=0.0, atol=1e-12
    ):
        raise ValueError("PE source duration must contain an integer number of samples")
    analysis_start = int(context["analysis_start_index"])
    analysis_stop = int(context["analysis_stop_index"])
    center = (analysis_start + analysis_stop) // 2
    start = center - source_samples // 2
    stop = start + source_samples
    if start < 0 or stop > context["mixture"].shape[1]:
        raise ValueError("PE source crop falls outside the materialized context")
    selected = np.stack(
        [np.asarray(context["mixture"][ifos.index(ifo), start:stop], dtype=np.float64) for ifo in required_ifos]
    )
    if not np.isfinite(selected).all():
        raise ValueError("PE source crop contains non-finite strain")
    if target_sample_rate != source_rate:
        selected = np.stack(
            [_fft_upsample(values, source_rate, target_sample_rate) for values in selected]
        )
    gps_start = float(context["context_gps_start"]) + start / source_rate
    return selected, gps_start, {
        "materialized_sample_rate_hz": source_rate,
        "output_sample_rate_hz": target_sample_rate,
        "resampling": (
            "none"
            if source_rate == target_sample_rate
            else "FFT bandlimited integer upsampling; no information added above the materialized Nyquist"
        ),
        "information_nyquist_hz": source_rate / 2.0,
    }


def _validate_existing_source(
    path: Path,
    expected_ifos: tuple[str, ...],
    expected_rate: int,
    expected_samples: int,
    expected_condition: str,
    expected_injection: str,
) -> None:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"strain", "ifos", "sample_rate", "gps_start", "condition", "injection_id"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"Existing PE source lacks fields: {sorted(missing)}")
        strain = np.asarray(arrays["strain"])
        ifos = tuple(str(value) for value in arrays["ifos"].tolist())
        rate = int(arrays["sample_rate"])
        condition = str(arrays["condition"].item())
        injection = str(arrays["injection_id"].item())
    if strain.shape != (len(expected_ifos), expected_samples) or not np.isfinite(strain).all():
        raise ValueError("Existing PE source has invalid strain shape/content")
    if (ifos, rate, condition, injection) != (
        expected_ifos,
        expected_rate,
        expected_condition,
        expected_injection,
    ):
        raise ValueError("Existing PE source belongs to a different run identity")


def materialize_common_pe_inputs(
    clean_manifest: str | Path,
    contaminated_manifest: str | Path,
    mask_conditioned_manifest: str | Path,
    common_prior_path: str | Path,
    mask_model_path: str | Path,
    mask_policy_path: str | Path,
    output_dir: str | Path,
    required_split: str,
    required_ifos: tuple[str, ...] = ("H1", "L1"),
    source_sample_rate_hz: int = 4096,
    source_duration_seconds: float = 16.0,
    analysis_high_frequency_hz: float = 1024.0,
    limit: int | None = None,
    selection_seed: int = 20260721,
) -> dict[str, Any]:
    """Build one backend-neutral strain artifact per injection and PE condition."""

    if required_split not in {"val", "test"}:
        raise ValueError("Paired PE source materialization is restricted to val or test")
    if len(required_ifos) < 2 or len(set(required_ifos)) != len(required_ifos):
        raise ValueError("PE source detector set must contain at least two unique IFOs")
    if source_sample_rate_hz <= 0 or source_duration_seconds <= 0:
        raise ValueError("PE source sample rate and duration must be positive")
    if analysis_high_frequency_hz <= 0 or analysis_high_frequency_hz > source_sample_rate_hz / 2:
        raise ValueError("PE analysis high frequency exceeds the output Nyquist")
    if limit is not None and limit <= 0:
        raise ValueError("PE source limit must be positive")

    paths = {
        "clean": Path(clean_manifest).resolve(),
        "contaminated": Path(contaminated_manifest).resolve(),
        "mask_conditioned": Path(mask_conditioned_manifest).resolve(),
    }
    loaded = {condition: _read_jsonl(path) for condition, path in paths.items()}
    indexed = {condition: _by_injection(rows, condition) for condition, rows in loaded.items()}
    id_sets = {condition: set(rows) for condition, rows in indexed.items()}
    if len({frozenset(values) for values in id_sets.values()}) != 1:
        raise ValueError("Clean, contaminated and mask-conditioned PE manifests differ in IDs")
    if any(str(row.get("split")) != required_split for rows in loaded.values() for row in rows):
        raise ValueError("A PE source manifest contains a different split")

    prior = load_yaml(common_prior_path)
    if str(prior.get("population")) != "BBH":
        raise ValueError("Common PE input materialization currently supports the BBH contract")
    mask_model = Path(mask_model_path).resolve()
    mask_policy = Path(mask_policy_path).resolve()
    for artifact in (mask_model, mask_policy, *paths.values(), Path(common_prior_path)):
        if not artifact.is_file():
            raise FileNotFoundError(f"PE provenance artifact is absent: {artifact}")
    mask_policy_config = load_yaml(mask_policy)
    if mask_policy_config.get("mode") != "cleaned_strain":
        raise ValueError("PE mask policy must declare cleaned_strain mode")
    if mask_policy_config.get("algorithm") != "hamming_stft_overlap_add":
        raise ValueError("PE mask policy declares an unsupported cleaning algorithm")
    suppression_strength = float(mask_policy_config["suppression_strength"])
    if not 0 <= suppression_strength <= 1:
        raise ValueError("PE mask-policy suppression strength must lie in [0, 1]")
    if mask_policy_config.get("auxiliary_channel_veto") is not False:
        raise ValueError("PE mask policy cannot silently enable an auxiliary-channel veto")

    rejection_counts: Counter[str] = Counter()
    eligible = []
    truths: dict[str, dict[str, float]] = {}
    for injection_id in sorted(id_sets["clean"]):
        clean = indexed["clean"][injection_id]
        contaminated = indexed["contaminated"][injection_id]
        masked = indexed["mask_conditioned"][injection_id]
        for field in IDENTITY_FIELDS:
            if any(row.get(field) != clean.get(field) for row in (contaminated, masked)):
                raise ValueError(f"PE condition identity mismatch for {injection_id}: {field}")
        if str(clean.get("source_family")) != "BBH":
            rejection_counts["non_bbh"] += 1
            continue
        truth = _canonical_truth(clean)
        support_failures = _prior_support(truth, prior)
        if support_failures:
            for parameter in support_failures:
                rejection_counts[f"outside_prior:{parameter}"] += 1
            continue
        if clean.get("analysis_override_path") or clean.get("analysis_override_sha256"):
            raise ValueError("Clean PE input unexpectedly declares an analysis override")
        if contaminated.get("analysis_override_kind") != "real_glitch_contaminated":
            raise ValueError("Contaminated PE input lacks real-glitch override lineage")
        if masked.get("analysis_override_kind") != "mask_conditioned":
            raise ValueError("Mask-conditioned PE input lacks learned-mask override lineage")
        if str(masked.get("input_analysis_override_sha256")) != str(
            contaminated.get("analysis_override_sha256")
        ):
            raise ValueError("Mask-conditioned input is not derived from its contaminated pair")
        if str(masked.get("glitch_id")) != str(contaminated.get("glitch_id")):
            raise ValueError("PE contaminated and mask-conditioned glitch IDs differ")
        if masked.get("deglitch_algorithm") != mask_policy_config["algorithm"] or not np.isclose(
            float(masked.get("deglitch_strength", -1.0)),
            suppression_strength,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Mask-conditioned input differs from the frozen mask policy")
        probability_path = masked.get("probability_path")
        probability_sha = masked.get("probability_sha256")
        if not probability_path or not probability_sha:
            raise ValueError("Mask-conditioned input lacks its numeric probability artifact")
        if file_sha256(probability_path) != str(probability_sha):
            raise ValueError("Mask probability artifact hash mismatch")
        truths[injection_id] = truth
        eligible.append(injection_id)

    eligible.sort(key=lambda injection_id: _selection_key(injection_id, selection_seed))
    selected_ids = eligible[:limit] if limit is not None else eligible
    if not selected_ids:
        raise ValueError("No paired BBH PE inputs satisfy the common-prior support contract")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": "common_pe_source_v1",
        "manifest_sha256": {condition: file_sha256(path) for condition, path in paths.items()},
        "common_prior_sha256": file_sha256(common_prior_path),
        "mask_model_sha256": file_sha256(mask_model),
        "mask_policy_sha256": file_sha256(mask_policy),
        "required_split": required_split,
        "required_ifos": list(required_ifos),
        "source_sample_rate_hz": source_sample_rate_hz,
        "source_duration_seconds": source_duration_seconds,
        "analysis_high_frequency_hz": analysis_high_frequency_hz,
        "limit": limit,
        "selection_seed": selection_seed,
        "selected_ids_hash": canonical_hash(selected_ids, 64),
    }
    state_path = output / "pe_input_materialization_state.json"
    if state_path.is_file():
        prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        if prior_state.get("run_identity") != run_identity:
            raise ValueError("Existing PE source output belongs to a different run")
    else:
        atomic_write_json(
            state_path,
            {"status": "in_progress", "run_identity": run_identity, "completed": 0},
        )

    records = []
    resampling_records = []
    verified_background_hashes: dict[str, str] = {}
    expected_samples = int(round(source_duration_seconds * source_sample_rate_hz))
    for injection_id in selected_ids:
        condition_strain_digests = {}
        for condition in CONDITIONS:
            row = indexed[condition][injection_id]
            context = load_materialized_context(row, verified_background_hashes)
            context, override = apply_analysis_override(row, context)
            if condition == "clean" and override:
                raise AssertionError("Clean override passed validation unexpectedly")
            strain, gps_start, resampling = _crop_and_resample(
                context,
                required_ifos,
                source_duration_seconds,
                source_sample_rate_hz,
            )
            if resampling["information_nyquist_hz"] < analysis_high_frequency_hz:
                raise ValueError(
                    "Materialized strain Nyquist is below the declared PE analysis band"
                )
            source_path = output / "arrays" / condition / f"{injection_id}.npz"
            if source_path.is_file():
                _validate_existing_source(
                    source_path,
                    required_ifos,
                    source_sample_rate_hz,
                    expected_samples,
                    condition,
                    injection_id,
                )
            else:
                _atomic_save_npz(
                    source_path,
                    strain=strain.astype(np.float32),
                    ifos=np.asarray(required_ifos),
                    sample_rate=np.asarray(source_sample_rate_hz, dtype=np.int64),
                    gps_start=np.asarray(gps_start, dtype=np.float64),
                    geocent_time=np.asarray(float(row["gps_time"]), dtype=np.float64),
                    condition=np.asarray(condition),
                    injection_id=np.asarray(injection_id),
                )
            source_sha = file_sha256(source_path)
            condition_strain_digests[condition] = hashlib.sha256(
                np.asarray(strain, dtype="<f4").tobytes(order="C")
            ).hexdigest()
            record = {
                "injection_id": injection_id,
                "waveform_id": row["waveform_id"],
                "source_family": row["source_family"],
                "split": required_split,
                "condition": condition,
                "truth": truths[injection_id],
                "analysis_input_path": str(source_path),
                "analysis_input_sha256": source_sha,
                "input_sample_rate_hz": source_sample_rate_hz,
                "input_duration_seconds": source_duration_seconds,
                "input_ifos": list(required_ifos),
                "base_injection_manifest_path": str(paths["clean"]),
                "base_injection_manifest_sha256": run_identity["manifest_sha256"]["clean"],
                "common_prior_path": str(Path(common_prior_path).resolve()),
                "common_prior_sha256": run_identity["common_prior_sha256"],
                "source_waveform_approximant": row.get("waveform_approximant"),
                "source_materialized_sample_rate_hz": resampling[
                    "materialized_sample_rate_hz"
                ],
                "source_information_nyquist_hz": resampling["information_nyquist_hz"],
                "resampling": resampling["resampling"],
                "gps_block": row["gps_block"],
                "gps_time": row["gps_time"],
            }
            if condition in {"contaminated", "mask_conditioned"}:
                record.update(
                    {
                        "glitch_id": row["glitch_id"],
                        "glitch_gps_block": row.get("glitch_gps_block"),
                        "glitch_label": row.get("glitch_label"),
                        "contamination_manifest_path": str(paths["contaminated"]),
                        "contamination_manifest_sha256": run_identity["manifest_sha256"][
                            "contaminated"
                        ],
                    }
                )
            if condition == "mask_conditioned":
                record.update(
                    {
                        "mask_conditioning_mode": "cleaned_strain",
                        "mask_artifact_path": str(Path(row["probability_path"]).resolve()),
                        "mask_artifact_sha256": row["probability_sha256"],
                        "mask_model_path": str(mask_model),
                        "mask_model_sha256": run_identity["mask_model_sha256"],
                        "mask_policy_path": str(mask_policy),
                        "mask_policy_sha256": run_identity["mask_policy_sha256"],
                    }
                )
            records.append(record)
            resampling_records.append(resampling)
        if condition_strain_digests["clean"] == condition_strain_digests["contaminated"]:
            raise ValueError("Clean and contaminated PE source strain is identical")
        atomic_write_json(
            state_path,
            {
                "status": "in_progress",
                "run_identity": run_identity,
                "completed": len(records) // len(CONDITIONS),
            },
        )

    manifest_path = output / "common_pe_inputs.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
    )
    report = {
        "status": "backend_neutral_paired_pe_inputs_materialized",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "DINGO and AMPLFI posteriors, validation-frozen model selection and paired coverage/"
            "bias/width intervals have not yet been evaluated"
        ),
        "backend_input_contract": "the exact analysis_input bytes are consumed by both backends",
        "rows": len(records),
        "paired_injections": len(selected_ids),
        "conditions": list(CONDITIONS),
        "required_split": required_split,
        "required_ifos": list(required_ifos),
        "source_sample_rate_hz": source_sample_rate_hz,
        "source_duration_seconds": source_duration_seconds,
        "analysis_high_frequency_hz": analysis_high_frequency_hz,
        "eligible_before_limit": len(eligible),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selection_seed": selection_seed,
        "selected_ids_hash": run_identity["selected_ids_hash"],
        "source_waveform_approximants": sorted(
            {str(indexed["clean"][item].get("waveform_approximant")) for item in selected_ids}
        ),
        "input_population_matches_analysis_prior_distribution": False,
        "input_population_note": (
            "all selected truths lie inside common-prior support, but the detection-injection "
            "population is not claimed to have been sampled from that analysis prior"
        ),
        "bandlimited_upsampling_used": any(
            item["materialized_sample_rate_hz"] != source_sample_rate_hz
            for item in resampling_records
        ),
        "bandlimited_upsampling_note": (
            "upsampling preserves the <= materialized-Nyquist series and adds no high-frequency "
            "information; the declared analysis band is checked against that original Nyquist"
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "run_identity": run_identity,
        **execution_provenance(),
    }
    atomic_write_json(output / "common_pe_inputs_report.json", report)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "run_identity": run_identity,
            "completed": len(selected_ids),
            "manifest_sha256": report["manifest_sha256"],
        },
    )
    return report
