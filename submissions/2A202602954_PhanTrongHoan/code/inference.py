"""Inference views, probability aggregation, validation calibration and BN fusion."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _matrix(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là ma trận số (N, C)") from exc
    if array.ndim != 2 or min(array.shape) <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} phải là ma trận (N, C) không rỗng, hữu hạn")
    return array.copy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    if not np.isfinite(probs).all():
        raise ValueError("Không tạo được probability hữu hạn")
    return probs


def _image_batch(x, name="x") -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError(f"{name} phải là tensor ảnh (N, C, H, W) với các chiều dương")


def predict_logits(model, loader, device, view=None):
    """Return filenames list, int64 labels and float32 raw logits as NumPy arrays.

    The view maps one image batch to one tensor batch. For several views, call
    this function separately, then aggregate the logits. Autocast is off here;
    a caller may wrap this call in a CUDA autocast context explicitly.
    """
    device = torch.device(device)
    model.eval()
    names_out, labels_out, logits_out = [], [], []
    with torch.inference_mode():
        for images, labels, names in loader:
            _image_batch(images, "images")
            if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                raise ValueError("labels phải là tensor nhãn nguyên (N,)")
            n = images.shape[0]
            try:
                batch_names = [str(name) for name in names]
            except TypeError as exc:
                raise ValueError("filenames phải là sequence") from exc
            if len(batch_names) != n or labels.shape[0] != n:
                raise ValueError("Số ảnh, nhãn và Filename trong batch không khớp")
            images = images.to(device)
            if view is not None:
                images = view(images)
                _image_batch(images, "view(images)")
                if images.shape[0] != n:
                    raise ValueError("view phải giữ nguyên số ảnh trong batch")
            logits = model(images)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != n:
                raise ValueError("Model phải trả logits tensor shape (N, C)")
            if logits.shape[1] <= 0 or not torch.isfinite(logits).all():
                raise ValueError("Logits phải có số lớp dương và hữu hạn")
            names_out.extend(batch_names)
            labels_out.append(labels.detach().cpu().numpy().astype(np.int64, copy=True))
            logits_out.append(logits.detach().float().cpu().numpy().copy())
    if not names_out:
        raise ValueError("Inference loader rỗng")
    if len({item.shape[1] for item in logits_out}) != 1:
        raise ValueError("Số lớp trong logits thay đổi giữa các batch")
    return names_out, np.concatenate(labels_out), np.concatenate(logits_out)


def view_identity(x):
    """Return x unchanged."""
    return x


def view_hflip(x):
    """Flip width of an image batch without mutating x."""
    _image_batch(x)
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return five crops in order: top-left, top-right, bottom-left, bottom-right, center."""
    _image_batch(x)
    if isinstance(crop, bool) or not isinstance(crop, int) or crop <= 0:
        raise ValueError("crop phải là số nguyên dương")
    height, width = x.shape[-2:]
    if crop > height or crop > width:
        raise ValueError("crop không được lớn hơn chiều cao hoặc chiều rộng ảnh")
    positions = ((0, 0), (0, width - crop), (height - crop, 0),
                 (height - crop, width - crop),
                 ((height - crop) // 2, (width - crop) // 2))
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    """Return square resized batches in sizes order, preserving device and dtype.

    CNNs with global pooling usually accept varied sizes. ViT/Swin may need
    special positional embedding or window handling at new input sizes.
    """
    _image_batch(x)
    if isinstance(sizes, (str, bytes)):
        raise ValueError("sizes phải là danh sách không rỗng các số nguyên dương")
    try:
        sizes = list(sizes)
    except TypeError as exc:
        raise ValueError("sizes phải là danh sách không rỗng các số nguyên dương") from exc
    if not sizes or any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in sizes):
        raise ValueError("sizes phải là danh sách không rỗng các số nguyên dương")
    if not x.is_floating_point():
        raise ValueError("x phải là tensor ảnh floating point để resize")
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Return float64 NumPy probabilities (N,C) for aligned views.

    prob: mean of per-view softmax; logit: softmax of mean logits.
    Caller must ensure identical Filename row order across views.
    """
    if not isinstance(space, str) or space not in {"prob", "logit"}:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    values = [_matrix(v, "logits_per_view") for v in logits_per_view]
    if not values:
        raise ValueError("logits_per_view không được rỗng")
    if any(v.shape != values[0].shape for v in values[1:]):
        raise ValueError("Các view phải có cùng shape (N, C)")
    if space == "prob":
        result = np.mean([_softmax(v) for v in values], axis=0)
        return result / result.sum(axis=1, keepdims=True)
    return _softmax(np.mean(values, axis=0))


def ensemble_probs(list_of_probs):
    """Return float64 NumPy mean probabilities for identically ordered images.

    Caller must confirm every probability row has the same Filename in the
    same order; this function has no filenames to check.
    """
    values = [_matrix(v, "list_of_probs") for v in list_of_probs]
    if not values:
        raise ValueError("list_of_probs không được rỗng")
    for v in values:
        if v.shape != values[0].shape:
            raise ValueError("Các probability phải có cùng shape (N, C)")
        if (v < -1e-8).any() or (v > 1 + 1e-8).any():
            raise ValueError("Probability phải nằm trong [0, 1]")
        if not np.allclose(v.sum(axis=1), 1, atol=1e-3, rtol=0):
            raise ValueError("Tổng probability mỗi hàng phải gần 1")
    mean = np.clip(np.mean(values, axis=0), 0, None)
    return mean / mean.sum(axis=1, keepdims=True)


def _validation_targets(labels, n: int, classes: int) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("val_labels phải là vector nhãn nguyên shape (N,)")
    if not np.isfinite(labels).all() or (labels != np.floor(labels)).any():
        raise ValueError("val_labels phải gồm số nguyên hữu hạn")
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError("val_labels nằm ngoài khoảng lớp của val_logits")
    return labels.astype(np.int64, copy=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Minimize validation NLL over log(T) in [log(0.05), log(100)].

    Golden-section search gives T>0. T=1 is retained when fitting does not
    improve validation NLL. Never fit this function on test data.
    """
    logits = _matrix(val_logits, "val_logits")
    labels = _validation_targets(val_labels, *logits.shape)
    row = np.arange(logits.shape[0])

    def nll(log_t: float) -> float:
        scaled = logits / math.exp(log_t)
        centered = scaled - scaled.max(axis=1, keepdims=True)
        return float((np.log(np.exp(centered).sum(axis=1)) - centered[row, labels]).mean())

    baseline = nll(0.0)
    lower, upper = math.log(0.05), math.log(100.0)
    golden = (math.sqrt(5) - 1) / 2
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_loss, right_loss = nll(left), nll(right)
    for _ in range(90):
        if left_loss <= right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - golden * (upper - lower)
            left_loss = nll(left)
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + golden * (upper - lower)
            right_loss = nll(right)
    optimum = (lower + upper) / 2
    best = nll(optimum)
    if not math.isfinite(best) or best >= baseline - 1e-12:
        return 1.0
    return float(math.exp(optimum))


def apply_temperature(logits, T: float):
    """Return float64 NumPy softmax(logits/T) without changing logits."""
    if isinstance(T, bool) or not isinstance(T, (int, float)) or not math.isfinite(T) or T <= 0:
        raise ValueError("T phải là số hữu hạn và dương")
    return _softmax(_matrix(logits, "logits") / T)


def fuse_conv_bn(model):
    """Return an eval deepcopy with adjacent Conv2d-BatchNorm2d in Sequential fused.

    Other parent modules may define another forward order and are left alone.
    The copy carries ``_conv_bn_fusion_count`` (zero when no safe pair exists).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model phải là torch.nn.Module")
    if model.training:
        raise ValueError("Đặt model.eval() trước khi fuse Conv-BN")
    replica = copy.deepcopy(model)
    count = 0

    def visit(parent: nn.Module) -> None:
        nonlocal count
        if isinstance(parent, nn.Sequential):
            names = list(parent._modules)
            for index in range(len(names) - 1):
                conv = parent._modules[names[index]]
                bn = parent._modules[names[index + 1]]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    if conv.training or bn.training:
                        raise ValueError("Mọi Conv-BN cần ở chế độ eval trước khi fuse")
                    if bn.running_mean is None or bn.running_var is None:
                        continue
                    parent._modules[names[index]] = fuse_conv_bn_eval(conv, bn)
                    parent._modules[names[index + 1]] = nn.Identity()
                    count += 1
        for child in parent.children():
            visit(child)

    with torch.no_grad():
        visit(replica)
    replica._conv_bn_fusion_count = count
    return replica
