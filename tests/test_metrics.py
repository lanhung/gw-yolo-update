from gwyolo.metrics import binary_metrics, snr_binned_hit_rate, wilson_interval
from gwyolo.training import _target_checkpoint_callback


def test_pdf_confusion_matrix_is_recomputed_exactly():
    metrics = binary_metrics(tp=61, fp=7, fn=22, tn=35)
    assert metrics["recall"] == 61 / 83
    assert metrics["precision"] == 61 / 68
    assert metrics["accuracy"] == 96 / 125
    assert metrics["f1"] == 122 / 151


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(61, 83)
    assert lower < 61 / 83 < upper
    assert round(lower, 3) == 0.631
    assert round(upper, 3) == 0.818


def test_snr_bins():
    rows = [
        {"snr": 9.0, "hit": True},
        {"snr": 9.5, "hit": False},
        {"snr": 12.0, "hit": True},
        {"snr": None, "hit": True},
    ]
    result = snr_binned_hit_rate(rows)
    eight_to_ten = next(row for row in result if row["snr_min"] == 8.0)
    twelve_to_fifteen = next(row for row in result if row["snr_min"] == 12.0)
    assert (eight_to_ten["hits"], eight_to_ten["total"]) == (1, 2)
    assert (twelve_to_fifteen["hits"], twelve_to_fifteen["total"]) == (1, 1)


def test_target_checkpoint_callback_tracks_requested_metric(tmp_path):
    source = tmp_path / "last.pt"
    target = tmp_path / "best_target.pt"
    source.write_bytes(b"epoch-one")

    class Trainer:
        last = source
        epoch = 0
        metrics = {"metrics/mAP50(M)": 0.6}

    callback, state = _target_checkpoint_callback("metrics/mAP50(M)", target)
    trainer = Trainer()
    callback(trainer)
    assert target.read_bytes() == b"epoch-one"
    assert state == {"value": 0.6, "epoch": 1}

    source.write_bytes(b"worse")
    trainer.epoch = 1
    trainer.metrics = {"metrics/mAP50(M)": 0.5}
    callback(trainer)
    assert target.read_bytes() == b"epoch-one"

    source.write_bytes(b"epoch-three")
    trainer.epoch = 2
    trainer.metrics = {"metrics/mAP50(M)": 0.7}
    callback(trainer)
    assert target.read_bytes() == b"epoch-three"
    assert state == {"value": 0.7, "epoch": 3}
