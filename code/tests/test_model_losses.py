"""CPU tests for model setup and loss arithmetic; no data or weight downloads."""
import copy
import importlib.util

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from losses import (FocalLoss, LabelSmoothingCE, build_criterion, class_weights,
                    mix_batch, mixed_loss)
from model import (build_model, count_gmacs, count_params, freeze_backbone,
                   param_groups, set_frozen_backbone_eval)


def _small_model():
    return build_model("mobilenetv3_large_100", pretrained=False, num_classes=9)


def test_build_model_forward_and_invalid_name():
    model = _small_model().cpu().eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 9)
    with pytest.raises(ValueError, match="name"):
        build_model("not_a_timm_model", pretrained=False)


def test_freeze_backbone_and_batchnorm_helper():
    model = build_model("mobilenetv3_large_100", pretrained=False,
                        num_classes=9, init="frozen")
    classifier_ids = {id(p) for p in model.get_classifier().parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert classifier_ids and classifier_ids <= trainable_ids
    assert any(not p.requires_grad for p in model.parameters())
    backbone_bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert backbone_bn and all(not m.training for m in backbone_bn)
    model.train()
    assert all(m.training for m in backbone_bn)
    set_frozen_backbone_eval(model)
    assert all(not m.training for m in backbone_bn)
    model.eval()
    assert all(not m.training for m in backbone_bn)


@pytest.mark.parametrize("init", ["finetune", "scratch"])
def test_unfrozen_model_batchnorm_train_eval_and_deepcopy(init):
    model = build_model("mobilenetv3_large_100", pretrained=False,
                        num_classes=9, init=init)
    backbone_bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert backbone_bn
    model.eval()
    assert all(not m.training for m in backbone_bn)
    model.train()
    assert all(m.training for m in backbone_bn)
    assert all(p.requires_grad for p in model.parameters())
    before = model.state_dict()
    replica = copy.deepcopy(model)
    after = replica.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert replica.training and model.training


def test_frozen_model_state_dict_and_deepcopy():
    model = build_model("mobilenetv3_large_100", pretrained=False,
                        num_classes=9, init="frozen")
    replica = copy.deepcopy(model)
    original_state = model.state_dict()
    copied_state = replica.state_dict()
    assert original_state.keys() == copied_state.keys()
    assert all(torch.equal(original_state[key], copied_state[key])
               for key in original_state)
    assert [p.requires_grad for p in replica.parameters()] == [
        p.requires_grad for p in model.parameters()
    ]
    assert all(not m.training for m in replica.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))


@pytest.mark.parametrize("name", ["resnet18", "convnext_tiny"])
def test_freeze_handles_direct_and_nested_classifier(name):
    model = build_model(name, pretrained=False, num_classes=9, init="frozen")
    classifier_ids = {id(p) for p in model.get_classifier().parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert classifier_ids and classifier_ids <= trainable_ids
    assert len(trainable_ids) < len(list(model.parameters()))
    if name == "convnext_tiny":
        head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
        assert head_params and all(p.requires_grad for p in head_params)


def test_param_groups_cover_each_trainable_parameter_once():
    model = _small_model()
    groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
    assert [group["name"] for group in groups] == [
        "backbone_decay", "backbone_no_decay", "head"
    ]
    ids = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) == len(expected)
    assert set(ids) == expected
    assert [(group["lr"], group["weight_decay"]) for group in groups] == [
        (1e-4, 0.05), (1e-4, 0.0), (1e-3, 0.05)
    ]
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    decay_ids = {id(p) for p in groups[0]["params"]}
    assert head_ids <= {id(p) for p in groups[2]["params"]}
    assert all(id(p) in no_decay_ids for name, p in model.named_parameters()
               if id(p) not in head_ids and (name.endswith(".bias") or p.ndim <= 1))
    norm_ids = {id(p) for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
                for p in module.parameters(recurse=False)}
    assert norm_ids and norm_ids <= no_decay_ids
    assert all(p.ndim > 1 for p in groups[0]["params"])
    assert decay_ids.isdisjoint(no_decay_ids)
    optimizer = torch.optim.AdamW(groups)
    assert len(optimizer.param_groups) == len(groups)


def test_frozen_groups_exclude_backbone_and_count_all_parameters():
    model = _small_model()
    before = count_params(model)
    freeze_backbone(model)
    groups = param_groups(model, 1e-4, 1e-3, 0.05)
    assert len(groups) == 3
    assert groups[0]["params"] == []
    assert groups[1]["params"] == []
    assert groups[2]["params"]
    assert groups[2]["lr"] == 1e-3 and groups[2]["weight_decay"] == 0.05
    assert isinstance(before, float) and before > 0
    assert count_params(model) == before


def test_gmacs_or_clear_optional_dependency_error():
    model = _small_model().cpu()
    original_training = model.training
    if importlib.util.find_spec("fvcore") is None:
        with pytest.raises(ImportError, match="fvcore"):
            count_gmacs(model, img_size=64)
    else:
        gmacs = count_gmacs(model, img_size=64)
        assert isinstance(gmacs, float) and gmacs > 0
    assert model.training == original_training


def test_ce_equivalences_and_backward():
    target = torch.tensor([0, 3, 8, 1])
    reference = torch.randn(4, 9)
    ce = F.cross_entropy(reference, target)
    assert torch.allclose(LabelSmoothingCE(0)(reference, target), ce, atol=1e-6)
    assert torch.allclose(FocalLoss(gamma=0)(reference, target), ce, atol=1e-6)
    for criterion in (build_criterion("ce"), build_criterion("ls", smoothing=0.1),
                      build_criterion("focal", gamma=2),
                      build_criterion("ce_weighted", weight=torch.ones(9))):
        logits = reference.clone().requires_grad_()
        loss = criterion(logits, target)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_class_weights_and_focal_alpha():
    counts = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    for beta in (0, 0.99):
        weights = class_weights(counts, beta=beta)
        assert weights.shape == (9,)
        assert torch.isfinite(weights).all() and (weights > 0).all()
        assert weights.mean().item() == pytest.approx(1, abs=1e-6)
        assert weights[0] > weights[-1]
    logits = torch.randn(4, 9, requires_grad=True)
    loss = FocalLoss(gamma=2, alpha=class_weights(counts))(logits, torch.tensor([0, 1, 2, 8]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_mixup_keeps_shape_and_input_unchanged():
    torch.manual_seed(7)
    x = torch.randn(4, 3, 8, 8)
    original = x.clone()
    y = torch.tensor([0, 1, 2, 3])
    mixed, (y_a, y_b, lam) = mix_batch(x, y, alpha=0.5, mode="mixup")
    assert mixed.shape == x.shape
    assert 0 <= lam <= 1
    assert torch.equal(x, original)
    assert torch.equal(y_a, y)
    assert sorted(y_b.tolist()) == sorted(y.tolist())


def test_cutmix_corrects_lambda_using_clipped_area(monkeypatch):
    monkeypatch.setattr(torch.distributions.Beta, "sample", lambda self: torch.tensor(0.5))
    monkeypatch.setattr(torch, "randperm",
                        lambda n, device=None: torch.arange(n - 1, -1, -1, device=device))
    x = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
    original = x.clone()
    y = torch.tensor([0, 1])
    mixed, (y_a, y_b, lam) = mix_batch(x, y, mode="cutmix")
    assert mixed.shape == x.shape and 0 <= lam <= 1
    assert torch.equal(x, original)
    assert torch.equal(y_a, y) and torch.equal(y_b, torch.tensor([1, 0]))
    assert 1 - lam == pytest.approx(mixed[0, 0].mean().item())


def test_mixed_loss_backward_and_lambda_one():
    logits = torch.randn(4, 9, requires_grad=True)
    y_a = torch.tensor([0, 1, 2, 3])
    y_b = torch.tensor([3, 2, 1, 0])
    criterion = nn.CrossEntropyLoss()
    assert torch.allclose(mixed_loss(criterion, logits, (y_a, y_b, 1.0)),
                          criterion(logits, y_a))
    loss = mixed_loss(criterion, logits, (y_a, y_b, 0.3))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_invalid_options():
    with pytest.raises(ValueError, match="kind"):
        build_criterion("unknown")
    with pytest.raises(ValueError, match="weight"):
        build_criterion("ce_weighted")
    with pytest.raises(ValueError, match="smoothing"):
        LabelSmoothingCE(-0.1)
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(-1)
    with pytest.raises(ValueError, match="counts"):
        class_weights([1, 2])
    with pytest.raises(ValueError, match="mode"):
        mix_batch(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]), mode="unknown")
    with pytest.raises(ValueError, match="alpha"):
        mix_batch(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]), alpha=0)
