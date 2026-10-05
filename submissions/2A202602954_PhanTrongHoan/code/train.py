"""One training pipeline for DeepWeeds B, T, and F experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from importlib import metadata
from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pandas as pd
import torch

import dataset as data
import losses
import model as models

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import eval as evaluation  # noqa: E402


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    curve_dir: str = "curves"
    expected_total: int | None = 17509


def run_dir(cfg: Config) -> Path:
    """Directory for one exp_id and seed."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Prediction CSV path for val or test."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch; some CUDA operators remain nondeterministic."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed phải là số nguyên không âm")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW with the three parameter groups from model.param_groups."""
    groups = models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if len(groups) != 3 or not any(group["params"] for group in groups):
        raise ValueError("Optimizer cần ba nhóm và ít nhất một tham số trainable")
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Step-wise linear warmup followed by cosine decay to zero.

    The initial LR is the LR for optimizer step 1. Call scheduler.step()
    after each successful optimizer step.
    """
    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch phải là số nguyên dương")
    if isinstance(cfg.epochs, bool) or not isinstance(cfg.epochs, int) or cfg.epochs <= 0:
        raise ValueError("epochs phải là số nguyên dương")
    if not math.isfinite(cfg.warmup_epochs) or not 0 <= cfg.warmup_epochs <= cfg.epochs:
        raise ValueError("warmup_epochs phải nằm trong [0, epochs]")
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = min(total_steps, int(round(cfg.warmup_epochs * steps_per_epoch)))

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps == warmup_steps:
            return 1.0
        # At the first post-warmup optimizer step, use the first cosine value.
        # With no warmup, start exactly at the configured base learning rate.
        cosine_step = step if warmup_steps == 0 else step - warmup_steps + 1
        progress = min(1.0, max(0.0, cosine_step / (total_steps - warmup_steps)))
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class EMA:
    """Independent exponential moving average of model parameters and buffers."""

    def __init__(self, model, decay: float):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay phải nằm trong [0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source_params = dict(model.named_parameters())
        target_params = dict(self.model.named_parameters())
        source_buffers = dict(model.named_buffers())
        target_buffers = dict(self.model.named_buffers())
        if source_params.keys() != target_params.keys() or source_buffers.keys() != target_buffers.keys():
            raise ValueError("EMA và model huấn luyện không cùng cấu trúc")
        for name, target in target_params.items():
            target.lerp_(source_params[name].detach(), 1 - self.decay)
        for name, target in target_buffers.items():
            source = source_buffers[name].detach()
            if target.is_floating_point():
                target.lerp_(source, 1 - self.decay)
            else:
                target.copy_(source)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        return self.model.load_state_dict(state)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.model.state_dict())


def _amp_enabled(cfg: Config, device) -> bool:
    return bool(cfg.amp and torch.device(device).type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Train one epoch, reporting sample-weighted loss and the last applied LRs."""
    model.train()
    if cfg.init == "frozen":
        models.set_frozen_backbone_eval(model)
    use_amp = _amp_enabled(cfg, device)
    total_loss = 0.0
    n_samples = 0
    n_batches = 0
    last_lrs = [group["lr"] for group in optimizer.param_groups]
    for images, labels, _filenames in loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        if cfg.mix is not None:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with context:
            logits = model(images)
            loss = (losses.mixed_loss(criterion, logits, targets)
                    if cfg.mix is not None else criterion(logits, labels))
        if not torch.isfinite(loss):
            raise ValueError("Train loss không hữu hạn")
        last_lrs = [group["lr"] for group in optimizer.param_groups]
        if use_amp and scaler is not None and scaler.is_enabled():
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= old_scale
        else:
            loss.backward()
            optimizer.step()
            stepped = True
        if stepped:
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        n_samples += batch_size
        n_batches += 1
    if n_samples == 0:
        raise ValueError("Train loader rỗng; kiểm tra batch_size và drop_last")
    return {
        "train_loss": total_loss / n_samples,
        "lr_backbone": last_lrs[0],
        "lr_head": last_lrs[2],
        "n_batches": n_batches,
        "n_samples": n_samples,
    }


def evaluate(model, loader, criterion, device):
    """Return ordered filenames, integer labels, raw logits, and mean loss."""
    model.eval()
    filenames: list[str] = []
    labels_parts = []
    logits_parts = []
    total_loss = 0.0
    n_samples = 0
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss) * batch_size
            n_samples += batch_size
            filenames.extend(str(name) for name in names)
            labels_parts.append(labels.cpu().numpy().astype(np.int64, copy=True))
            logits_parts.append(logits.float().cpu().numpy().copy())
    if n_samples == 0:
        raise ValueError("Evaluation loader rỗng")
    return filenames, np.concatenate(labels_parts), np.concatenate(logits_parts), total_loss / n_samples


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Save noninteractive loss, validation F1 and LR curves."""
    if not history:
        raise ValueError("history rỗng")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epochs = [row["epoch"] for row in history]
    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    loss_ax, metric_ax = figure.subplots(1, 2)
    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="val loss")
    loss_ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)
    metric_ax.plot(epochs, [row["macro_f1_val"] for row in history], marker="o",
                   label="val macro-F1")
    metric_ax.set(xlabel="Epoch", ylabel="Macro-F1", title="Validation")
    metric_ax.grid(alpha=0.3)
    if all("lr_backbone" in row for row in history):
        lr_ax = metric_ax.twinx()
        lr_ax.plot(epochs, [row["lr_backbone"] for row in history], linestyle="--",
                   color="gray", label="backbone LR")
        lr_ax.set_ylabel("Learning rate")
        lr_ax.legend(loc="lower right")
    metric_ax.legend(loc="upper left")
    figure.suptitle(title)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    figure.clear()


def _validate_config(cfg: Config) -> None:
    if not isinstance(cfg.exp_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", cfg.exp_id):
        raise ValueError("exp_id phải bắt đầu bằng chữ và chỉ gồm chữ, số, _ hoặc -")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed phải là số nguyên không âm")
    if isinstance(cfg.fold, bool) or not isinstance(cfg.fold, int) or cfg.fold not in range(5):
        raise ValueError("fold phải nằm trong 0..4")
    if isinstance(cfg.epochs, bool) or not isinstance(cfg.epochs, int) or cfg.epochs <= 0:
        raise ValueError("epochs phải là số nguyên dương")
    for name in ("batch_size", "img_size"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} phải là số nguyên dương")
    if isinstance(cfg.num_workers, bool) or not isinstance(cfg.num_workers, int) or cfg.num_workers < 0:
        raise ValueError("num_workers phải là số nguyên không âm")
    if not isinstance(cfg.backbone, str) or not cfg.backbone.strip():
        raise ValueError("backbone không được rỗng")
    if cfg.init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init phải là scratch, frozen hoặc finetune")
    if cfg.aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError("aug phải là basic, color, trivial hoặc randaug")
    if cfg.sampler not in (None, "balanced"):
        raise ValueError("sampler phải là None hoặc balanced")
    if cfg.mix not in (None, "mixup", "cutmix"):
        raise ValueError("mix phải là None, mixup hoặc cutmix")
    if cfg.loss not in {"ce", "ls", "focal", "ce_weighted"}:
        raise ValueError("loss phải là ce, ls, focal hoặc ce_weighted")
    for name in ("lr_backbone", "lr_head", "weight_decay", "warmup_epochs",
                 "drop_rate", "label_smoothing", "focal_gamma"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} phải hữu hạn và không âm")
    if cfg.warmup_epochs > cfg.epochs:
        raise ValueError("warmup_epochs không được lớn hơn epochs")
    if cfg.drop_rate >= 1 or cfg.label_smoothing > 1:
        raise ValueError("drop_rate cần < 1 và label_smoothing cần <= 1")
    if cfg.mix is not None and (not math.isfinite(cfg.mix_alpha) or cfg.mix_alpha <= 0):
        raise ValueError("mix_alpha phải hữu hạn và dương")
    if cfg.ema_decay is not None and (not math.isfinite(cfg.ema_decay) or
                                       not 0 <= cfg.ema_decay < 1):
        raise ValueError("ema_decay phải nằm trong [0, 1)")
    if cfg.class_weight_beta is not None and (not math.isfinite(cfg.class_weight_beta) or
                                                not 0 <= cfg.class_weight_beta < 1):
        raise ValueError("class_weight_beta phải nằm trong [0, 1)")
    if cfg.expected_total is not None and (isinstance(cfg.expected_total, bool) or
                                            not isinstance(cfg.expected_total, int) or
                                            cfg.expected_total <= 0):
        raise ValueError("expected_total phải là số nguyên dương hoặc None")
    if not isinstance(cfg.save_test_predictions, bool) or not isinstance(cfg.amp, bool):
        raise ValueError("amp và save_test_predictions phải là bool")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _criterion(cfg: Config, train_df: pd.DataFrame, device):
    weights = None
    if cfg.loss == "ce_weighted" or (cfg.loss == "focal" and cfg.class_weight_beta is not None):
        counts = train_df["Label"].value_counts().reindex(range(data.NUM_CLASSES), fill_value=0)
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        weights = losses.class_weights(counts.tolist(), beta=beta).to(device)
    if cfg.loss == "ls":
        criterion = losses.build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = losses.build_criterion("focal", gamma=cfg.focal_gamma, alpha=weights)
    elif cfg.loss == "ce_weighted":
        criterion = losses.build_criterion("ce_weighted", weight=weights)
    else:
        criterion = losses.build_criterion("ce")
    return criterion.to(device)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.from_numpy(logits).softmax(dim=1).numpy()


def run(cfg: Config) -> dict:
    """Run one experiment; select and save its checkpoint using validation only."""
    _validate_config(cfg)
    destination = run_dir(cfg)
    curve_path = Path(cfg.curve_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png"
    planned_outputs = [destination, pred_path(cfg, "val"), curve_path]
    if cfg.save_test_predictions:
        planned_outputs.append(pred_path(cfg, "test"))
    existing = [str(path) for path in planned_outputs if path.exists()]
    if existing:
        raise FileExistsError("Run đã tồn tại; chọn exp_id/seed khác: " + ", ".join(existing))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    destination.mkdir(parents=True, exist_ok=False)
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "determinism_note": "Một số CUDA operator vẫn có thể không deterministic.",
    }
    (destination / "config.json").write_text(
        json.dumps({**asdict(cfg), "environment": environment}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_df, val_df, test_df = data.load_split(cfg.labels_dir, cfg.fold)
    split_stats = data.check_split(train_df, val_df, test_df, cfg.images_dir,
                                   expected_total=cfg.expected_total)
    (destination / "split_stats.json").write_text(
        json.dumps(split_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_transform = data.build_transforms(True, cfg.img_size, cfg.aug)
    eval_transform = data.build_transforms(False, cfg.img_size)
    train_loader = data.make_loader(train_df, cfg.images_dir, train_transform, cfg.batch_size,
                                    train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
                                    pin_memory=device.type == "cuda", seed=cfg.seed)
    val_loader = data.make_loader(val_df, cfg.images_dir, eval_transform, cfg.batch_size,
                                  train=False, num_workers=cfg.num_workers, shuffle=False,
                                  pin_memory=device.type == "cuda", seed=cfg.seed)
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = data.make_loader(test_df, cfg.images_dir, eval_transform, cfg.batch_size,
                                       train=False, num_workers=cfg.num_workers, shuffle=False,
                                       pin_memory=device.type == "cuda", seed=cfg.seed)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train/validation loader rỗng")

    net = models.build_model(cfg.backbone, pretrained=cfg.init != "scratch",
                             num_classes=data.NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init)
    params_m = models.count_params(net)
    gmacs = models.count_gmacs(net, cfg.img_size)
    net = net.to(device)
    criterion = _criterion(cfg, train_df, device)
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = _amp_enabled(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    history = []
    best_score = -math.inf
    best_epoch = None
    checkpoint_path = destination / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(net, train_loader, criterion, optimizer, scheduler,
                                      scaler, cfg, device, ema)
        selected = ema.model if ema is not None else net
        _names, val_y, val_logits, val_loss = evaluate(selected, val_loader, criterion, device)
        probs = _probabilities(val_logits)
        metrics = evaluation.compute_metrics(val_y, probs.argmax(axis=1), probs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_loss,
            "macro_f1_val": metrics["macro_f1"],
            "top1_val": metrics["top1"],
            "balanced_acc_val": metrics["balanced_acc"],
            "epoch_seconds": seconds,
            "selection_model": "ema" if ema is not None else "raw",
        }
        history.append(row)
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        if not math.isfinite(metrics["macro_f1"]):
            raise ValueError("macro-F1 validation không hữu hạn")
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            torch.save({
                "model": net.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "macro_f1_val": best_score,
                "config": asdict(cfg),
                "selection_model": row["selection_model"],
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model"])
    if ema is not None:
        ema.load_state_dict(checkpoint["ema"])
    selected = ema.model if ema is not None else net
    val_names, val_y, val_logits, _val_loss = evaluate(selected, val_loader, criterion, device)
    val_probs = _probabilities(val_logits)
    val_metrics = evaluation.compute_metrics(val_y, val_probs.argmax(axis=1), val_probs)
    np.save(destination / "val_logits.npy", val_logits)
    val_prediction = evaluation.save_predictions(pred_path(cfg, "val"), val_names, val_y, val_probs)

    test_prediction = None
    if cfg.save_test_predictions:
        test_names, test_y, test_logits, _test_loss = evaluate(selected, test_loader,
                                                               criterion, device)
        np.save(destination / "test_logits.npy", test_logits)
        test_prediction = evaluation.save_predictions(
            pred_path(cfg, "test"), test_names, test_y, _probabilities(test_logits)
        )

    plot_curves(history, curve_path, f"{cfg.exp_id} seed {cfg.seed} - {cfg.backbone}")
    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "macro_f1_val": val_metrics["macro_f1"],
        "top1_val": val_metrics["top1"],
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "params_m": params_m,
        "gmacs": gmacs,
        "run_dir": str(destination),
        "val_prediction_path": str(val_prediction),
        "test_prediction_path": str(test_prediction) if test_prediction is not None else None,
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Parse KEY=VALUE against Config annotations, including optional types."""
    annotations = get_type_hints(Config)
    allowed = {field.name for field in fields(Config)}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Thiếu dấu = trong override: {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in allowed:
            raise ValueError(f"Config không có key: {key!r}")
        if key in result:
            raise ValueError(f"Override key trùng lặp: {key!r}")
        expected = annotations[key]
        variants = get_args(expected)
        optional = type(None) in variants
        value_type = next((part for part in variants if part is not type(None)), expected)
        lowered = raw.strip().lower()
        if lowered in {"none", "null"}:
            if not optional:
                raise ValueError(f"{key} không chấp nhận None")
            value = None
        elif value_type is bool:
            if lowered not in {"true", "false"}:
                raise ValueError(f"{key} cần true hoặc false, nhận {raw!r}")
            value = lowered == "true"
        elif value_type is int:
            if not re.fullmatch(r"[+-]?\d+", raw.strip()):
                raise ValueError(f"{key} cần số nguyên, nhận {raw!r}")
            value = int(raw)
        elif value_type is float:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{key} cần số thực, nhận {raw!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key} cần số thực hữu hạn")
        elif value_type is str:
            value = raw
        else:
            raise TypeError(f"Kiểu Config chưa hỗ trợ cho {key}: {expected}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DeepWeeds training experiment")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="Override Config fields, for example exp_id=B01 seed=0")
    args = parser.parse_args()
    try:
        cfg = Config(**parse_overrides(args.set))
        summary = run(cfg)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
