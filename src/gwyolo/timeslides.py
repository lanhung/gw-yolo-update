from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, atomic_write_text, file_sha256
from .trigger import network_ranking


def _time_key(value: float) -> int:
    return int(round(float(value) * 1_000_000_000))


def build_window_time_slides(
    trigger_rows: Iterable[dict[str, Any]],
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    coincidence_window_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build non-cyclic, window-level two-IFO slides for pipeline integration tests."""
    if reference_ifo == shifted_ifo:
        raise ValueError("reference and shifted IFOs must differ")
    if slide_count <= 0 or step_seconds <= 0:
        raise ValueError("slide count and step must be positive")
    if coincidence_window_seconds is not None and coincidence_window_seconds <= 0:
        raise ValueError("coincidence window must be positive")
    rows = [row for row in trigger_rows if str(row["split"]) == split]
    if not rows:
        raise ValueError(f"No trigger rows for split {split}")
    starts = [_time_key(row["gps_start"]) for row in rows]
    if len(starts) != len(set(starts)):
        raise ValueError("Selected trigger rows contain duplicate GPS starts")
    durations = {float(row["gps_end"]) - float(row["gps_start"]) for row in rows}
    if len(durations) != 1:
        raise ValueError("Window-level slides require a common window duration")
    duration = next(iter(durations))
    if step_seconds < duration:
        raise ValueError("Window-level slide step must be at least one window duration")
    required_ifos = {reference_ifo, shifted_ifo}
    missing = [
        str(row.get("window_id"))
        for row in rows
        if not required_ifos.issubset(set(row.get("valid_ifos", [])))
    ]
    if missing:
        raise ValueError(f"Trigger rows lack required coincident IFOs: {missing[:10]}")
    by_start = {_time_key(row["gps_start"]): row for row in rows}
    output = []
    slide_exposure = []
    for slide_index in range(1, slide_count + 1):
        offset_seconds = slide_index * step_seconds
        offset_key = _time_key(offset_seconds)
        intervals = []
        pairs = 0
        coincident_candidates = 0
        for reference in sorted(rows, key=lambda row: float(row["gps_start"])):
            shifted = by_start.get(_time_key(reference["gps_start"]) + offset_key)
            if shifted is None:
                continue
            chirp_scores = {
                reference_ifo: float(reference["chirp_scores"][reference_ifo]),
                shifted_ifo: float(shifted["chirp_scores"][shifted_ifo]),
            }
            glitch_scores = {
                reference_ifo: float(reference["glitch_scores"][reference_ifo]),
                shifted_ifo: float(shifted["glitch_scores"][shifted_ifo]),
            }
            ranking = network_ranking(
                chirp_scores, glitch_scores, [reference_ifo, shifted_ifo]
            )
            intervals.append((float(reference["gps_start"]), float(reference["gps_end"])))
            pairs += 1
            peak_separation = None
            if coincidence_window_seconds is not None:
                try:
                    reference_peak = float(reference["peak_times"]["chirp"][reference_ifo]["gps"])
                    shifted_peak = float(shifted["peak_times"]["chirp"][shifted_ifo]["gps"])
                except KeyError as exc:
                    raise ValueError(
                        "Peak-coincidence slides require trigger peak_times for both IFOs"
                    ) from exc
                shifted_peak_in_reference_time = shifted_peak - offset_seconds
                peak_separation = abs(reference_peak - shifted_peak_in_reference_time)
                if peak_separation > coincidence_window_seconds:
                    continue
            coincident_candidates += 1
            output.append(
                {
                    "slide_id": f"slide-{slide_index:06d}",
                    "slide_index": slide_index,
                    "split": split,
                    "offset_seconds": {reference_ifo: 0.0, shifted_ifo: offset_seconds},
                    "reference_gps_start": float(reference["gps_start"]),
                    "reference_gps_end": float(reference["gps_end"]),
                    "source_window_ids": {
                        reference_ifo: reference["window_id"],
                        shifted_ifo: shifted["window_id"],
                    },
                    "source_gps_blocks": {
                        reference_ifo: reference["gps_block"],
                        shifted_ifo: shifted["gps_block"],
                    },
                    "peak_separation_seconds": peak_separation,
                    "chirp_scores": chirp_scores,
                    "glitch_scores": glitch_scores,
                    **ranking,
                }
            )
        exposure = _union_duration(intervals)
        slide_exposure.append(
            {
                "slide_index": slide_index,
                "offset_seconds": offset_seconds,
                "coincident_windows": pairs,
                "coincident_candidates": coincident_candidates,
                "live_time_seconds": exposure,
            }
        )
    total_exposure = sum(item["live_time_seconds"] for item in slide_exposure)
    report = {
        "status": "window_level_time_slide_integration_only",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "requires sub-window clustered trigger times, continuous independent segments, "
            "predeclared shifts and adequate equivalent live time"
        ),
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "input_windows": len(rows),
        "window_duration_seconds": duration,
        "slide_count": slide_count,
        "step_seconds": step_seconds,
        "coincidence_window_seconds": coincidence_window_seconds,
        "nonzero_noncyclic_slides": True,
        "background_rows": len(output),
        "slide_exposure": slide_exposure,
        "equivalent_live_time_seconds": total_exposure,
        "equivalent_live_time_years": total_exposure / SECONDS_PER_YEAR,
    }
    return output, report


def run_window_time_slides(
    triggers_path: str | Path,
    output_dir: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    coincidence_window_seconds: float | None = None,
) -> dict[str, Any]:
    with Path(triggers_path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    background, report = build_window_time_slides(
        rows,
        split,
        reference_ifo,
        shifted_ifo,
        slide_count,
        step_seconds,
        coincidence_window_seconds,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{split}_time_slide_background.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in background),
    )
    result = {
        **report,
        "trigger_manifest_sha256": file_sha256(triggers_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(output / f"{split}_time_slide_report.json", result)
    return result
