"""CPU tests for training utilities and a synthetic end-to-end run."""
import copy
import json
import math
import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import eval as evaluation
import train
from model import freeze_backbone


class TinyClassifier(nn.Module):
    """Nine-class model with a real backbone/head boundary for optimizer tests."""

    def __init__(self, batch_norm=False):
        super().__init__()
        layers = [nn.Conv2d(3, 4, 3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(4))
        layers.extend([nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(4, 9)

    def get_classifier(self):
        return self.classifier

    def forward(self, images):
        return self.classifier(self.backbone(images))


def _batch(names=("a.jpg", "b.jpg")):
    return (torch.randn(len(names), 3, 16, 16),
            torch.tensor(list(range(len(names)))), list(names))


def test_set_seed_repeats_python_numpy_and_torch():
    train.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(2))
    train.set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_optimizer_has_exactly_three_groups_and_unique_trainable_params():
    model = TinyClassifier(batch_norm=True)
    cfg = train.Config(lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
    optimizer = train.build_optimizer(model, cfg)
    groups = optimizer.param_groups
    assert [group["name"] for group in groups] == [
        "backbone_decay", "backbone_no_decay", "head"
    ]
    assert [(group["lr"], group["weight_decay"]) for group in groups] == [
        (1e-4, 0.05), (1e-4, 0.0), (1e-3, 0.05)
    ]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert {id(model.backbone[1].weight), id(model.backbone[1].bias)} <= {
        id(parameter) for parameter in groups[1]["params"]
    }
    freeze_backbone(model)
    frozen_groups = train.build_optimizer(model, cfg).param_groups
    assert not frozen_groups[0]["params"] and not frozen_groups[1]["params"]
    assert {id(p) for p in frozen_groups[2]["params"]} == {
        id(p) for p in model.classifier.parameters()
    }


def test_scheduler_warmup_cosine_and_zero_warmup():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.9)
    cfg = train.Config(epochs=2, warmup_epochs=1.0)
    scheduler = train.build_scheduler(optimizer, cfg, steps_per_epoch=3)
    applied = []
    for _ in range(6):
        applied.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert applied[:3] == pytest.approx([0.3, 0.6, 0.9])
    assert applied[3] >= applied[4] >= applied[5] >= 0
    assert all(math.isfinite(value) and value >= 0 for value in applied)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0)

    zero_optimizer = torch.optim.SGD([parameter], lr=0.9)
    train.build_scheduler(zero_optimizer, train.Config(epochs=1, warmup_epochs=0), 2)
    assert zero_optimizer.param_groups[0]["lr"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="steps_per_epoch"):
        train.build_scheduler(zero_optimizer, cfg, 0)


def test_ema_independent_update_buffers_and_serialization():
    model = TinyClassifier(batch_norm=True)
    ema = train.EMA(model, 0.5)
    assert all(id(a) != id(b) for a, b in zip(model.parameters(), ema.model.parameters()))
    assert not any(p.requires_grad for p in ema.model.parameters())
    before_weight = ema.model.classifier.weight.detach().clone()
    before_mean = ema.model.backbone[1].running_mean.detach().clone()
    with torch.no_grad():
        model.classifier.weight.add_(2)
        model.backbone[1].running_mean.add_(4)
        model.backbone[1].num_batches_tracked.fill_(7)
    ema.update(model)
    assert torch.allclose(ema.model.classifier.weight, before_weight + 1)
    assert torch.allclose(ema.model.backbone[1].running_mean, before_mean + 2)
    assert ema.model.backbone[1].num_batches_tracked.item() == 7
    assert not torch.equal(ema.model.classifier.weight, model.classifier.weight)
    replica = copy.deepcopy(ema)
    assert replica.state_dict().keys() == ema.state_dict().keys()
    other = train.EMA(model, 0.5)
    other.load_state_dict(ema.state_dict())
    assert torch.equal(other.model.classifier.weight, ema.model.classifier.weight)
    assert model.backbone[1].num_batches_tracked.item() == 7
    with pytest.raises(ValueError, match="decay"):
        train.EMA(model, 1.0)


@pytest.mark.parametrize("mix", [None, "mixup", "cutmix"])
def test_train_one_epoch_tiny_cpu_and_frozen_batchnorm(mix):
    model = TinyClassifier(batch_norm=True)
    freeze_backbone(model)
    cfg = train.Config(init="frozen", mix=mix, amp=True, epochs=1,
                       warmup_epochs=0, lr_head=1e-3)
    optimizer = train.build_optimizer(model, cfg)
    scheduler = train.build_scheduler(optimizer, cfg, steps_per_epoch=1)
    stats = train.train_one_epoch(model, [_batch()], nn.CrossEntropyLoss(), optimizer,
                                  scheduler, None, cfg, torch.device("cpu"))
    assert model.training
    assert not model.backbone[1].training
    assert stats["n_batches"] == 1 and stats["n_samples"] == 2
    assert math.isfinite(stats["train_loss"]) and stats["train_loss"] > 0
    assert stats["lr_head"] == pytest.approx(cfg.lr_head)
    assert stats["lr_backbone"] == pytest.approx(cfg.lr_backbone)


def test_evaluate_preserves_order_shape_and_reports_empty_loader():
    model = TinyClassifier()
    batches = [_batch(("b.jpg", "a.jpg")), _batch(("c.jpg",))]
    names, labels, logits, loss = train.evaluate(model, batches, nn.CrossEntropyLoss(), "cpu")
    assert names == ["b.jpg", "a.jpg", "c.jpg"]
    assert labels.tolist() == [0, 1, 0] and labels.dtype == np.int64
    assert logits.shape == (3, 9) and np.issubdtype(logits.dtype, np.floating)
    assert math.isfinite(loss) and loss > 0
    assert not model.training
    with pytest.raises(ValueError, match="loader"):
        train.evaluate(model, [], nn.CrossEntropyLoss(), "cpu")


def test_plot_curves_creates_nonempty_png(tmp_path):
    path = tmp_path / "curves" / "T01_seed0.png"
    history = [
        {"epoch": 1, "train_loss": 2.2, "val_loss": 2.1,
         "macro_f1_val": 0.1, "lr_backbone": 1e-4},
        {"epoch": 2, "train_loss": 1.8, "val_loss": 2.0,
         "macro_f1_val": 0.2, "lr_backbone": 5e-5},
    ]
    train.plot_curves(history, path, "synthetic")
    assert path.is_file() and path.stat().st_size > 100


def test_parse_overrides_types_and_bad_values():
    parsed = train.parse_overrides([
        "seed=7", "warmup_epochs=0.5", "amp=FALSE", "backbone=resnet18",
        "ema_decay=0.99", "sampler=NULL", "save_test_predictions=true",
        "expected_total=none",
    ])
    assert parsed == {
        "seed": 7, "warmup_epochs": 0.5, "amp": False, "backbone": "resnet18",
        "ema_decay": 0.99, "sampler": None, "save_test_predictions": True,
        "expected_total": None,
    }
    for pairs in (["missing"], ["unknown=1"], ["seed=1", "seed=2"],
                  ["seed=1.5"], ["amp=yes"], ["epochs=none"], ["lr_head=nan"]):
        with pytest.raises(ValueError):
            train.parse_overrides(pairs)


def test_path_contracts(tmp_path):
    cfg = train.Config(exp_id="T01", seed=3, out_dir=str(tmp_path / "runs"),
                       pred_dir=str(tmp_path / "predictions"))
    assert train.run_dir(cfg) == tmp_path / "runs" / "T01" / "seed3"
    assert train.pred_path(cfg, "val") == tmp_path / "predictions" / "T01_seed3_val.csv"
    assert train.pred_path(cfg, "test") == tmp_path / "predictions" / "T01_seed3_test.csv"


@pytest.fixture
def synthetic_split(tmp_path):
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    labels_dir.mkdir()
    images_dir.mkdir()
    splits = {
        "train": [(f"train_{label}.jpg", label, f"species_{label}") for label in range(9)],
        "val": [("val_0.jpg", 0, "species_0"), ("val_8.jpg", 8, "species_8")],
        "test": [("test_1.jpg", 1, "species_1"), ("test_7.jpg", 7, "species_7")],
    }
    metadata_rows = [row for rows in splits.values() for row in rows]
    assert len({filename for filename, _, _ in metadata_rows}) == len(metadata_rows)
    pd.DataFrame(metadata_rows, columns=["Filename", "Label", "Species"]).to_csv(
        labels_dir / "labels.csv", index=False
    )
    for split, rows in splits.items():
        pd.DataFrame(rows, columns=["Filename", "Label", "Species"]).to_csv(
            labels_dir / f"{split}_subset0.csv", index=False
        )
        for index, (filename, _, _) in enumerate(rows):
            Image.new("RGB", (20, 20), (20 + index * 10, 80, 140)).save(
                images_dir / filename, "JPEG"
            )
    return labels_dir, images_dir


@pytest.mark.parametrize("save_test_predictions", [False, True])
def test_run_synthetic_smoke(tmp_path, synthetic_split, monkeypatch, save_test_predictions):
    labels_dir, images_dir = synthetic_split
    monkeypatch.setattr(train.models, "build_model", lambda *args, **kwargs: TinyClassifier())
    monkeypatch.setattr(train.models, "count_gmacs", lambda *args, **kwargs: 0.001)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    loader_calls = []
    original_make_loader = train.data.make_loader

    def tracked_make_loader(*args, **kwargs):
        loader_calls.append(kwargs["train"])
        return original_make_loader(*args, **kwargs)

    monkeypatch.setattr(train.data, "make_loader", tracked_make_loader)
    evaluation_calls = []
    original_evaluate = train.evaluate

    def tracked_evaluate(model, loader, criterion, device):
        evaluation_calls.append(tuple(loader.dataset.df["Filename"]))
        return original_evaluate(model, loader, criterion, device)

    monkeypatch.setattr(train, "evaluate", tracked_evaluate)
    cfg = train.Config(
        exp_id="T01" if not save_test_predictions else "T02", init="scratch",
        images_dir=str(images_dir), labels_dir=str(labels_dir),
        out_dir=str(tmp_path / "runs"), pred_dir=str(tmp_path / "predictions"),
        curve_dir=str(tmp_path / "curves"), expected_total=None,
        epochs=1, batch_size=3, img_size=16, num_workers=0, amp=False,
        warmup_epochs=0, save_test_predictions=save_test_predictions,
        ema_decay=0.5,
    )
    summary = train.run(cfg)
    assert loader_calls == ([True, False, False] if save_test_predictions else [True, False])
    assert evaluation_calls.count(("val_0.jpg", "val_8.jpg")) == 2
    assert evaluation_calls.count(("test_1.jpg", "test_7.jpg")) == int(save_test_predictions)
    destination = train.run_dir(cfg)
    assert summary["best_epoch"] == 1
    assert summary["run_dir"] == str(destination)
    assert summary["val_prediction_path"] == str(train.pred_path(cfg, "val"))
    assert all((destination / name).is_file() for name in (
        "config.json", "split_stats.json", "history.csv", "best.pt", "val_logits.npy"
    ))
    assert (tmp_path / "curves" / f"{cfg.exp_id}_seed0.png").stat().st_size > 100
    assert json.loads((destination / "config.json").read_text(encoding="utf-8"))["expected_total"] is None
    history = pd.read_csv(destination / "history.csv")
    assert len(history) == 1 and history.loc[0, "selection_model"] == "ema"
    assert np.load(destination / "val_logits.npy").shape == (2, 9)
    val_pred = evaluation.read_pred(str(train.pred_path(cfg, "val")))
    assert val_pred.filenames.tolist() == ["val_0.jpg", "val_8.jpg"]
    assert "test_macro_f1" not in summary
    if save_test_predictions:
        assert summary["test_prediction_path"] == str(train.pred_path(cfg, "test"))
        assert np.load(destination / "test_logits.npy").shape == (2, 9)
        test_pred = evaluation.read_pred(str(train.pred_path(cfg, "test")))
        assert test_pred.filenames.tolist() == ["test_1.jpg", "test_7.jpg"]
    else:
        assert summary["test_prediction_path"] is None
        assert not (destination / "test_logits.npy").exists()
        assert not train.pred_path(cfg, "test").exists()
    with pytest.raises(FileExistsError, match="exp_id/seed"):
        train.run(cfg)
