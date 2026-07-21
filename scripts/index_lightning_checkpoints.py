#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> float | str | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()  # type: ignore[union-attr]
    if isinstance(value, (int, float, str)):
        return value
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.checkpoint_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Lightning checkpoint indices are immutable")
    checkpoints = sorted(root.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no Lightning checkpoints found below {root}")
    records = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "epoch" not in payload or "global_step" not in payload:
            raise ValueError(f"Lightning checkpoint lacks epoch/global_step: {path}")
        callbacks = []
        for name, state in (payload.get("callbacks") or {}).items():
            if not isinstance(state, dict):
                continue
            callbacks.append(
                {
                    "name": str(name),
                    "best_model_path": state.get("best_model_path"),
                    "best_model_score": scalar(state.get("best_model_score")),
                    "current_score": scalar(state.get("current_score")),
                    "monitor": state.get("monitor"),
                    "mode": state.get("mode"),
                }
            )
        records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "epoch": int(payload["epoch"]),
                "global_step": int(payload["global_step"]),
                "callbacks": callbacks,
            }
        )
    result = {
        "status": "indexed_lightning_checkpoints",
        "scientific_claim_allowed": False,
        "checkpoint_root": str(root),
        "checkpoints": records,
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
