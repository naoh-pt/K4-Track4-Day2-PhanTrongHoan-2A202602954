"""Measured forward latency with warmup, synchronization, and metadata."""
from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext

import numpy as np
import torch


def _count(value, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} phải là số nguyên >= {minimum}")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Time fn in ms, excluding warmup; sync immediately before and after each call."""
    _count(warmup, "warmup", 0)
    _count(iters, "iters", 1)
    if sync is not None and not callable(sync):
        raise ValueError("sync phải là hàm hoặc None")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1000)
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Không đo được độ trễ hữu hạn, không âm")
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()), "n": iters}


def _prepare(model, batch_size, img_size, dtype, device, warmup, iters):
    _count(batch_size, "batch_size", 1)
    _count(img_size, "img_size", 1)
    _count(warmup, "warmup", 10)
    _count(iters, "iters", 50)
    if not isinstance(dtype, str) or dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device phải là cpu, cuda hoặc torch.device hợp lệ") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Chỉ hỗ trợ device cpu hoặc cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu CUDA nhưng CUDA không khả dụng")
    if dtype == "fp16" and device.type != "cuda":
        raise ValueError("fp16 benchmark chỉ hỗ trợ CUDA; CPU không được báo cáo là fp16")
    replica = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        replica = replica.half()
    else:
        replica = replica.float()
    image_dtype = torch.float16 if dtype == "fp16" else torch.float32
    image = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=image_dtype)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return replica, image, device, sync, gpu


def _measure(model, batch_size, img_size, dtype, device, warmup, iters, views):
    replica, image, device, sync, gpu = _prepare(
        model, batch_size, img_size, dtype, device, warmup, iters
    )

    def forward():
        for _ in range(views):
            replica(image)

    autocast = (torch.amp.autocast(device_type=device.type,
                                  dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)
                if dtype == "amp" else nullcontext())
    with torch.inference_mode(), autocast:
        stats = bench(forward, warmup=warmup, iters=iters, sync=sync)
    p50 = stats["p50"]
    if not math.isfinite(p50) or p50 <= 0:
        raise ValueError("p50 phải dương để tính thông lượng")
    return {"device": str(device), "gpu": gpu, "dtype": dtype,
            "batch": batch_size, "img_size": img_size,
            **stats, "images_per_s": float(batch_size * 1000 / p50),
            "torch": torch.__version__, "includes_preprocessing": False,
            "fused_bn_pairs": int(getattr(model, "_conv_bn_fusion_count", 0))}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Measure one eval forward; model/input copies preserve the caller's model.

    Input creation and preprocessing are excluded. Warmup >=10 and iters >=50
    enforce the reporting protocol; CUDA is synchronized on both timer sides.
    """
    return _measure(model, batch_size, img_size, dtype, device, warmup, iters, views=1)


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure one call containing K forwards and compare with a measured single view.

    No preprocessing is included. Defaults for omitted keyword arguments are
    batch_size=1, img_size=224 and the latency_report defaults.
    """
    _count(k_views, "k_views", 1)
    options = {"batch_size": 1, "img_size": 224, "dtype": "fp32",
               "device": "cuda", "warmup": 10, "iters": 100}
    unknown = set(kw) - set(options)
    if unknown:
        raise ValueError(f"tta_latency không hỗ trợ key: {sorted(unknown)}")
    options.update(kw)
    single = _measure(model, **options, views=1)
    combined = _measure(model, **options, views=k_views)
    combined.update({"k_views": k_views,
                     "single_view_p50": single["p50"],
                     "expected_k_times_p50": k_views * single["p50"],
                     "ratio_to_single": combined["p50"] / single["p50"]})
    return combined
