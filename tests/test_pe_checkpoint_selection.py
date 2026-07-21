from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.pe_backend import select_lightning_validation_checkpoint


def _metrics(path: Path, include_test: bool = False) -> None:
    fields = ["epoch", "step", "valid_loss"]
    if include_test:
        fields.append("test_loss")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, value in enumerate((3.0, 1.0, 2.0)):
            row = {"epoch": epoch, "step": (epoch + 1) * 10, "valid_loss": value}
            if include_test and epoch == 2:
                row["test_loss"] = 0.5
            writer.writerow(row)


def test_lightning_checkpoint_index_and_validation_selection(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    indexed_checkpoints = []
    for epoch in range(3):
        path = checkpoints / f"epoch={epoch}-step={(epoch + 1) * 10}.ckpt"
        path.write_bytes(f"checkpoint-{epoch}".encode())
        indexed_checkpoints.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "epoch": epoch,
                "global_step": (epoch + 1) * 10,
                "callbacks": [],
            }
        )
    last = checkpoints / "last.ckpt"
    last.write_bytes(b"last-checkpoint")
    indexed_checkpoints.append(
        {
            "path": str(last.resolve()),
            "sha256": file_sha256(last),
            "bytes": last.stat().st_size,
            "epoch": 2,
            "global_step": 30,
            "callbacks": [],
        }
    )
    index = tmp_path / "checkpoint-index.json"
    index.write_text(
        json.dumps(
            {
                "status": "indexed_lightning_checkpoints",
                "scientific_claim_allowed": False,
                "checkpoint_root": str(checkpoints.resolve()),
                "checkpoints": indexed_checkpoints,
            }
        ),
        encoding="utf-8",
    )
    indexed = json.loads(index.read_text(encoding="utf-8"))
    assert indexed["status"] == "indexed_lightning_checkpoints"
    assert len(indexed["checkpoints"]) == 4
    assert all(
        file_sha256(row["path"]) == row["sha256"] for row in indexed["checkpoints"]
    )

    config = tmp_path / "training.yaml"
    config.write_text("trainer:\n  max_epochs: 3\n", encoding="utf-8")
    manifest = tmp_path / "training.jsonl"
    manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    _metrics(metrics)
    report = select_lightning_validation_checkpoint(
        training_config_path=config,
        training_data_manifest_path=manifest,
        metrics_csv_path=metrics,
        checkpoint_index_path=index,
        output_path=tmp_path / "selection.json",
        minimum_publication_epochs=3,
        minimum_validation_points=3,
    )
    assert report["status"] == "validation_selected_checkpoint"
    assert report["publication_eligible"] is True
    assert report["selected_epoch"] == 1
    assert report["selected_global_step"] == 20
    assert report["selected_metric_value"] == 1.0
    assert Path(report["selected_checkpoint_path"]).name == "epoch=1-step=20.ckpt"

    engineering = select_lightning_validation_checkpoint(
        training_config_path=config,
        training_data_manifest_path=manifest,
        metrics_csv_path=metrics,
        checkpoint_index_path=index,
        output_path=tmp_path / "engineering-selection.json",
        minimum_publication_epochs=4,
        minimum_validation_points=3,
    )
    assert engineering["publication_eligible"] is False
    assert "below the publication minimum" in engineering["blockers"][0]

    contaminated_metrics = tmp_path / "metrics-with-test.csv"
    _metrics(contaminated_metrics, include_test=True)
    with pytest.raises(ValueError, match="test-set"):
        select_lightning_validation_checkpoint(
            training_config_path=config,
            training_data_manifest_path=manifest,
            metrics_csv_path=contaminated_metrics,
            checkpoint_index_path=index,
            output_path=tmp_path / "invalid-selection.json",
            minimum_publication_epochs=3,
            minimum_validation_points=3,
        )
