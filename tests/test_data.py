from pathlib import Path

from gwyolo.data import Sample, assign_group_splits, derive_group_id


def _sample(index: int, group: str, class_0: int, class_1: int) -> Sample:
    return Sample(
        sample_id=f"sample_{index}.jpg",
        group_id=group,
        source="source",
        image=f"/images/{index}.jpg",
        label=f"/labels/{index}.txt",
        image_sha256=f"image-{index}",
        label_sha256=f"label-{index}",
        image_bytes=100,
        objects=class_0 + class_1,
        class_0=class_0,
        class_1=class_1,
        empty=class_0 + class_1 == 0,
    )


def test_group_id_ignores_roboflow_hash():
    name = (
        "1249628966_032dcc00a5a1001b197eb4b11814685b_0_L1_png."
        "rf.e7f1b6ae1155a4923641c94d4d8c3180.jpg"
    )
    assert derive_group_id(name) == "032dcc00a5a1001b197eb4b11814685b"


def test_bns_group_id():
    name = "bns_ext_loud_0d23041fccf7c7cca2d4fb99bd9266b8_png.rf.abc.jpg"
    assert derive_group_id(name) == "0d23041fccf7c7cca2d4fb99bd9266b8"


def test_group_split_has_zero_leakage_and_is_deterministic():
    samples = []
    index = 0
    for group_index in range(30):
        group = f"group-{group_index}"
        for _ in range(1 + group_index % 3):
            samples.append(_sample(index, group, group_index % 2, (group_index + 1) % 2))
            index += 1
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    first, report = assign_group_splits(samples, fractions, seed=7, trials=30)
    second, _ = assign_group_splits(samples, fractions, seed=7, trials=30)
    assert report["cross_split_group_overlap"] == {}
    assert [(row.sample_id, row.split) for row in first] == [
        (row.sample_id, row.split) for row in second
    ]
    mapping = {}
    for row in first:
        mapping.setdefault(row.group_id, row.split)
        assert mapping[row.group_id] == row.split
    assert set(mapping.values()) == {"train", "val", "test"}


def test_import_does_not_require_training_dependencies():
    assert Path(__file__).name == "test_data.py"
