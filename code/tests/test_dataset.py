"""Small CPU-only checks for the DeepWeeds data pipeline."""
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from dataset import DeepWeedsDataset, build_transforms, check_split, load_split, make_loader


@pytest.fixture
def sample(tmp_path):
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    labels_dir.mkdir()
    images_dir.mkdir()
    rows = {
        "train": [("a.jpg", 0, "Chinee Apple"), ("b.jpg", 8, "Negatives")],
        "val": [("c.jpg", 1, "Lantana")],
        "test": [("d.jpg", 7, "Snake Weed")],
    }
    for split, items in rows.items():
        pd.DataFrame(items, columns=["Filename", "Label", "Species"]).to_csv(
            labels_dir / f"{split}_subset0.csv", index=False
        )
        for filename, _, _ in items:
            Image.new("RGB", (32, 32), (80, 120, 160)).save(images_dir / filename, "JPEG")
    return labels_dir, images_dir


def test_load_split_reads_three_csvs(sample):
    labels_dir, images_dir = sample
    train, val, test = load_split(labels_dir)
    assert [len(train), len(val), len(test)] == [2, 1, 1]
    assert list(train.columns) == ["Filename", "Label", "Species"]
    assert list(val["Filename"]) == ["c.jpg"]
    assert check_split(train, val, test, images_dir)["union"] == 4


def test_missing_csv_and_column_report_source(sample):
    labels_dir, _ = sample
    (labels_dir / "test_subset0.csv").unlink()
    with pytest.raises(FileNotFoundError, match="test_subset0.csv"):
        load_split(labels_dir)
    pd.DataFrame({"Filename": ["d.jpg"], "Label": [7]}).to_csv(
        labels_dir / "test_subset0.csv", index=False
    )
    with pytest.raises(ValueError, match="Species"):
        load_split(labels_dir)


def test_overlap_is_rejected(sample):
    labels_dir, _ = sample
    train, val, test = load_split(labels_dir)
    val.loc[0, "Filename"] = "a.jpg"
    with pytest.raises(ValueError, match="a.jpg"):
        check_split(train, val, test)


def test_missing_image_names_filename(sample):
    labels_dir, images_dir = sample
    train, val, test = load_split(labels_dir)
    (images_dir / "d.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="d.jpg"):
        check_split(train, val, test, images_dir)


def test_dataset_tensor_label_and_read_error(sample):
    labels_dir, images_dir = sample
    train, _, _ = load_split(labels_dir)
    ds = DeepWeedsDataset(train, images_dir, build_transforms(False, img_size=16))
    image, label, filename = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 16)
    assert isinstance(label, int) and label == 0
    assert filename == "a.jpg"
    (images_dir / "b.jpg").write_bytes(b"broken jpeg")
    with pytest.raises(ValueError, match="b.jpg"):
        ds[1]


def test_validation_transform_is_deterministic(sample):
    _, images_dir = sample
    transform = build_transforms(False, img_size=16, aug="color")
    with Image.open(images_dir / "a.jpg") as image:
        assert torch.equal(transform(image), transform(image))
    with pytest.raises(ValueError, match="aug"):
        build_transforms(True, aug="unknown")


def test_loader_yields_cpu_batch_in_csv_order(sample):
    labels_dir, images_dir = sample
    train, _, _ = load_split(labels_dir)
    loader = make_loader(train, images_dir, build_transforms(False, img_size=16),
                         batch_size=2, train=False, num_workers=0, shuffle=False,
                         pin_memory=False, seed=7)
    images, labels, filenames = next(iter(loader))
    assert images.shape == (2, 3, 16, 16)
    assert labels.dtype == torch.int64
    assert labels.device.type == "cpu"
    assert labels.tolist() == [0, 8]
    assert list(filenames) == ["a.jpg", "b.jpg"]


def test_bad_label_and_duplicate_inside_split(sample):
    labels_dir, _ = sample
    train, val, test = load_split(labels_dir)
    train.loc[1, "Label"] = 9
    with pytest.raises(ValueError, match="Label"):
        check_split(train, val, test)
    train.loc[1, "Label"] = 8
    train.loc[1, "Filename"] = "a.jpg"
    with pytest.raises(ValueError, match="a.jpg"):
        check_split(train, val, test)
