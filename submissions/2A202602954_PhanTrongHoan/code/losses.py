"""Classification losses and batch Mixup/CutMix for DeepWeeds."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_CLASSES = 9


def build_criterion(kind: str = "ce", **kw):
    """Build CE, label smoothing, focal, or class-weighted CE."""
    if kind == "ce":
        return nn.CrossEntropyLoss(**kw)
    if kind == "ls":
        return LabelSmoothingCE(**kw)
    if kind == "focal":
        return FocalLoss(**kw)
    if kind == "ce_weighted":
        weight = kw.pop("weight", None)
        if weight is None:
            raise ValueError("ce_weighted cần truyền weight từ class_weights(train_counts)")
        return nn.CrossEntropyLoss(weight=weight, **kw)
    raise ValueError(f"kind không hợp lệ: {kind!r}; chọn ce, ls, focal hoặc ce_weighted")


class LabelSmoothingCE(nn.Module):
    """CE with q'(k) = (1 - eps) 1[k=y] + eps / K."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
            raise ValueError("smoothing phải nằm trong [0, 1]")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multiclass focal loss, averaged over the batch.

    alpha is an optional vector of per-class multipliers. Unlike weighted CE,
    these multipliers are applied after per-example CE and the focal factor.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma phải hữu hạn và không âm")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            weights = torch.as_tensor(alpha, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() == 0 or not torch.isfinite(weights).all():
                raise ValueError("alpha phải là vector trọng số lớp hữu hạn")
            if (weights <= 0).any():
                raise ValueError("alpha phải có trọng số dương")
            self.register_buffer("alpha", weights.clone())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.alpha is not None and self.alpha.numel() != logits.shape[1]:
            raise ValueError(f"alpha có {self.alpha.numel()} lớp; logits có {logits.shape[1]} lớp")
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Weights from TRAIN class counts; inverse frequency or effective number.

    beta=0 selects 1/n. For 0<beta<1, use (1-beta)/(1-beta**n).
    The nine weights are normalized to mean one.
    """
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta phải nằm trong [0, 1)")
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or values.numel() != NUM_CLASSES:
        raise ValueError(f"counts phải có đúng {NUM_CLASSES} lớp từ tập train")
    if not torch.isfinite(values).all() or (values <= 0).any() or (values != values.round()).any():
        raise ValueError("counts phải là các số nguyên dương hữu hạn")
    if beta == 0:
        weights = values.reciprocal()
    else:
        weights = (1 - beta) / (-torch.expm1(values * math.log(beta)))
    return (weights / weights.mean()).to(dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Return new images and (original labels, permuted labels, corrected lambda).

    All random draws use the PyTorch RNG, so torch.manual_seed controls them.
    """
    if mode not in {"mixup", "cutmix"}:
        raise ValueError(f"mode không hợp lệ: {mode!r}; chọn mixup hoặc cutmix")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha phải hữu hạn và dương")
    if x.ndim != 4 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("Cần x dạng (N,C,H,W) và y dạng (N,)")
    if x.shape[2] <= 0 or x.shape[3] <= 0:
        raise ValueError("Chiều cao và rộng của ảnh phải dương")
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.shape[0], device=x.device)
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        height, width = x.shape[-2:]
        cut_ratio = math.sqrt(1 - lam)
        cut_width = int(width * cut_ratio)
        cut_height = int(height * cut_ratio)
        center_x = int(torch.randint(width, (1,)).item())
        center_y = int(torch.randint(height, (1,)).item())
        x1 = max(center_x - cut_width // 2, 0)
        x2 = min(center_x + (cut_width + 1) // 2, width)
        y1 = max(center_y - cut_height // 2, 0)
        y2 = min(center_y + (cut_height + 1) // 2, height)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (width * height)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Combine criterion on both targets using the returned Mixup/CutMix lambda."""
    try:
        y_a, y_b, lam = targets
    except (TypeError, ValueError) as exc:
        raise ValueError("targets phải là (y_a, y_b, lam)") from exc
    if not 0 <= lam <= 1:
        raise ValueError("lam phải nằm trong [0, 1]")
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
