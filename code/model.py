"""Build timm classifiers and prepare their parameters for DeepWeeds."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _head_prefix(model: nn.Module) -> str:
    """Find the classifier in timm's module tree, including nested head.fc."""
    if not hasattr(model, "get_classifier"):
        raise ValueError("Model cần cung cấp get_classifier() để xác định classifier/head")
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module) or isinstance(classifier, nn.Identity):
        raise ValueError("get_classifier() không trả về classifier/head có tham số")
    matches = [name for name, module in model.named_modules() if module is classifier]
    if len(matches) != 1 or not matches[0]:
        raise ValueError("Không xác định được đường dẫn duy nhất của classifier/head")
    prefix = matches[0]
    # ConvNeXt-style head.fc: include its adjacent head norm/scale in the head.
    if prefix.startswith("head."):
        prefix = "head"
    if not any(name == prefix or name.startswith(prefix + ".") for name, _ in model.named_parameters()):
        raise ValueError("Classifier/head không có tham số")
    return prefix


def _is_head(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Create a timm classifier; scratch always disables pretrained weights.

    Frozen permits pretrained=False for offline tests. The actual weight
    configuration remains available in model.pretrained_cfg.
    """
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"init không hợp lệ: {init!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name phải là tên model timm không rỗng")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes phải là số nguyên dương")
    if not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError("drop_rate phải nằm trong [0, 1)")
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Cần cài gói timm để gọi build_model") from exc
    if not timm.is_model(name):
        raise ValueError(f"Model name không hợp lệ trong timm: {name!r}")
    model = timm.create_model(
        name, pretrained=bool(pretrained and init != "scratch"),
        num_classes=num_classes, drop_rate=drop_rate,
    )
    _head_prefix(model)
    if init == "frozen":
        freeze_backbone(model)
    return model


def set_frozen_backbone_eval(model: nn.Module) -> None:
    """Call after each model.train() to keep frozen backbone BatchNorm in eval."""
    prefix = _head_prefix(model)
    for name, module in model.named_modules():
        if not _is_head(name, prefix) and isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def freeze_backbone(model) -> None:
    """Freeze every backbone parameter and leave the final classifier trainable.

    After subsequent calls to model.train(), call set_frozen_backbone_eval(model)
    again before the training forward pass.
    """
    prefix = _head_prefix(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_head(name, prefix))
    set_frozen_backbone_eval(model)


def _norm_parameter_ids(model: nn.Module) -> set[int]:
    ids = set()
    for module in model.modules():
        if "norm" in type(module).__name__.lower():
            ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    return ids


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Return exactly three optimizer groups from the starter contract.

    Backbone matrix/kernel weights use decay; backbone norm and bias do not.
    Every trainable head parameter, including head norm and bias, uses lr_head
    and the supplied weight_decay.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (lr_backbone, lr_head, weight_decay)):
        raise ValueError("LR và weight_decay phải hữu hạn, không âm")
    prefix = _head_prefix(model)
    norm_ids = _norm_parameter_ids(model)
    groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay,
                           "name": "backbone_decay"},
        "backbone_no_decay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0,
                              "name": "backbone_no_decay"},
        "head": {"params": [], "lr": lr_head, "weight_decay": weight_decay,
                 "name": "head"},
    }
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Tham số trainable trùng: {name}")
        seen.add(id(parameter))
        if _is_head(name, prefix):
            group_name = "head"
        elif parameter.ndim <= 1 or name.endswith(".bias") or id(parameter) in norm_ids:
            group_name = "backbone_no_decay"
        else:
            group_name = "backbone_decay"
        groups[group_name]["params"].append(parameter)
    return list(groups.values())


def count_params(model) -> float:
    """Count all parameters, including frozen ones, in millions."""
    return float(sum(parameter.numel() for parameter in model.parameters()) / 1_000_000)


def count_gmacs(model, img_size: int = 224) -> float:
    """Count batch-1 multiply-accumulates with fvcore on a CPU copy.

    fvcore counts one fused multiply-add as one operation. We sum its conv,
    linear and matrix-product operators as MACs; unsupported operations are
    excluded, so the reported GMAC is an approximate architecture cost.
    """
    if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size <= 0:
        raise ValueError("img_size phải là số nguyên dương")
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise ImportError("Cần cài gói fvcore để tính GMAC: pip install fvcore") from exc
    replica = copy.deepcopy(model).cpu().eval()
    image = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        analysis = FlopCountAnalysis(replica, image)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        operators = analysis.by_operator()
    macs = sum(operators.get(op, 0) for op in ("conv", "linear", "matmul", "bmm", "einsum"))
    if macs <= 0:
        raise ValueError("fvcore không đếm được phép MAC của model này")
    return float(macs / 1_000_000_000)
