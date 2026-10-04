"""CPU-only synthetic checks for inference, calibration, fusion and latency."""
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

import benchmark
import eval as evaluation
import inference


class TinyModel(nn.Module):
    forwards = 0

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(2, 9))

    def forward(self, x):
        type(self).forwards += 1
        return self.net(x)


def _batch(names):
    n = len(names)
    return torch.randn(n, 3, 4, 4), torch.arange(n, dtype=torch.int64), list(names)


def _probabilities(probs):
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2 and np.isfinite(probs).all()
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-8)


def test_predict_logits_preserves_order_and_raw_values():
    model = TinyModel().train()
    batches = [_batch(("b.jpg", "a.jpg")), _batch(("c.jpg",))]
    names, labels, logits = inference.predict_logits(model, batches, "cpu", inference.view_hflip)
    assert names == ["b.jpg", "a.jpg", "c.jpg"]
    assert labels.tolist() == [0, 1, 0] and labels.dtype == np.int64
    assert logits.shape == (3, 9) and np.issubdtype(logits.dtype, np.floating)
    assert not model.training
    with pytest.raises(ValueError, match="loader"):
        inference.predict_logits(model, [], "cpu")
    with pytest.raises(ValueError, match="view"):
        inference.predict_logits(model, batches, "cpu", lambda x: [x, x])


def test_predict_logits_rejects_bad_model_output_and_misaligned_names():
    class BadModel(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="logits"):
        inference.predict_logits(BadModel(), [_batch(("one.jpg",))], "cpu")
    images, labels, _ = _batch(("one.jpg", "two.jpg"))
    with pytest.raises(ValueError, match="không khớp"):
        inference.predict_logits(TinyModel(), [(images, labels, ["one.jpg"])], "cpu")


def test_identity_hflip_and_five_crop_positions():
    x = torch.arange(5 * 7).reshape(1, 1, 5, 7).float()
    original = x.clone()
    assert inference.view_identity(x) is x
    assert torch.equal(inference.view_hflip(x), x.flip(-1))
    crops = inference.views_multicrop(x, 3)
    assert len(crops) == 5 and all(crop.shape == (1, 1, 3, 3) for crop in crops)
    expected = (x[:, :, :3, :3], x[:, :, :3, -3:],
                x[:, :, -3:, :3], x[:, :, -3:, -3:], x[:, :, 1:4, 2:5])
    assert all(torch.equal(actual, target) for actual, target in zip(crops, expected))
    assert torch.equal(x, original)
    square = torch.randn(2, 3, 4, 4)
    assert all(torch.equal(crop, square) for crop in inference.views_multicrop(square, 4))
    with pytest.raises(ValueError, match="crop"):
        inference.views_multicrop(x, 8)
    with pytest.raises(ValueError, match="tensor"):
        inference.view_hflip(torch.zeros(3, 4, 4))


def test_multiscale_shapes_order_dtype_and_validation():
    x = torch.randn(2, 3, 5, 7, dtype=torch.float32)
    views = inference.views_multiscale(x, [3, 8, 5])
    assert [tuple(v.shape) for v in views] == [(2, 3, 3, 3), (2, 3, 8, 8), (2, 3, 5, 5)]
    assert all(v.dtype == x.dtype and v.device == x.device for v in views)
    for bad in ([], [0], [True], [2.5], "4"):
        with pytest.raises(ValueError, match="sizes"):
            inference.views_multiscale(x, bad)


def test_aggregate_probability_and_logit_spaces():
    first = np.array([[4., 0.], [0., 4.]])
    second = np.array([[0., 2.], [3., 0.]])
    prob = inference.aggregate_views([first, second], "prob")
    logit = inference.aggregate_views([first, second], "logit")
    _probabilities(prob)
    _probabilities(logit)
    np.testing.assert_allclose(prob, (inference.apply_temperature(first, 1) +
                                      inference.apply_temperature(second, 1)) / 2)
    np.testing.assert_allclose(logit, inference.apply_temperature((first + second) / 2, 1))
    assert not np.allclose(prob, logit)
    for bad in ([], [first, np.ones((3, 2))]):
        with pytest.raises(ValueError):
            inference.aggregate_views(bad)
    with pytest.raises(ValueError, match="space"):
        inference.aggregate_views([first], "unknown")


def test_ensemble_probabilities_and_invalid_input():
    a = np.array([[0.7, 0.3], [0.2, 0.8]])
    b = np.array([[0.4, 0.6], [0.9, 0.1]])
    mean = inference.ensemble_probs([a, b])
    _probabilities(mean)
    np.testing.assert_allclose(mean, (a + b) / 2)
    for bad in ([], [a, np.ones((3, 2))], [a, np.array([[1.2, -0.2], [0.5, 0.5]])],
                [a, np.array([[0.2, 0.2], [0.4, 0.6]])]):
        with pytest.raises(ValueError):
            inference.ensemble_probs(bad)


def test_temperature_positive_improves_validation_nll_and_keeps_argmax():
    logits = np.array([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
    labels = np.array([0, 1, 1, 0])
    original = logits.copy()
    temperature = inference.fit_temperature(logits, labels)
    assert isinstance(temperature, float) and math.isfinite(temperature) and temperature > 0
    assert temperature > 1
    before = inference.apply_temperature(logits, 1)
    after = inference.apply_temperature(logits, temperature)
    _probabilities(after)
    assert -np.log(after[np.arange(4), labels]).mean() <= (
        -np.log(before[np.arange(4), labels]).mean() + 1e-8
    )
    assert evaluation.ece_score(after, labels) < evaluation.ece_score(before, labels)
    assert np.array_equal(before.argmax(1), after.argmax(1))
    assert np.array_equal(logits, original)
    with pytest.raises(ValueError, match="T"):
        inference.apply_temperature(logits, 0)
    with pytest.raises(ValueError, match="val_labels"):
        inference.fit_temperature(logits, [0, 1, 2, 0])


@pytest.mark.parametrize("bias", [False, True])
def test_fuse_conv_bn_matches_original_and_preserves_state(bias):
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=bias),
                          nn.BatchNorm2d(4), nn.ReLU()).eval()
    original_state = copy.deepcopy(model.state_dict())
    image = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        expected = model(image)
    fused = inference.fuse_conv_bn(model)
    with torch.inference_mode():
        actual = fused(image)
    assert fused._conv_bn_fusion_count == 1
    assert isinstance(fused[1], nn.Identity)
    assert isinstance(model[1], nn.BatchNorm2d)
    assert (actual - expected).abs().max().item() < 1e-5
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original_state.items())
    replica = copy.deepcopy(fused)
    replica.load_state_dict(fused.state_dict())
    assert torch.allclose(replica(image), actual, atol=1e-5)


def test_fuse_without_bn_and_train_mode_rejected():
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU()).eval()
    fused = inference.fuse_conv_bn(model)
    assert fused is not model and fused._conv_bn_fusion_count == 0
    image = torch.randn(1, 3, 4, 4)
    with torch.inference_mode():
        assert torch.allclose(model(image), fused(image))
    nonadjacent = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU(), nn.BatchNorm2d(3)).eval()
    assert inference.fuse_conv_bn(nonadjacent)._conv_bn_fusion_count == 0
    with pytest.raises(ValueError, match="eval"):
        inference.fuse_conv_bn(model.train())


def test_bench_counts_warmup_measurements_and_sync():
    calls = {"fn": 0, "sync": 0}

    def fn():
        calls["fn"] += 1

    def sync():
        calls["sync"] += 1

    stats = benchmark.bench(fn, warmup=2, iters=5, sync=sync)
    assert calls == {"fn": 7, "sync": 10}
    assert stats["n"] == 5
    assert all(math.isfinite(stats[key]) and stats[key] >= 0
               for key in ("p50", "p95", "p99", "mean"))
    assert stats["p50"] <= stats["p95"] <= stats["p99"]
    with pytest.raises(ValueError, match="iters"):
        benchmark.bench(fn, warmup=0, iters=0)


def test_latency_report_cpu_metadata_and_original_unchanged():
    model = TinyModel().train()
    state = copy.deepcopy(model.state_dict())
    report = benchmark.latency_report(model, batch_size=1, img_size=4,
                                      dtype="fp32", device="cpu", warmup=10, iters=50)
    assert report["device"] == "cpu" and report["gpu"] is None
    assert report["dtype"] == "fp32" and report["batch"] == 1 and report["img_size"] == 4
    assert report["n"] == 50 and report["torch"] == torch.__version__
    assert report["includes_preprocessing"] is False
    assert report["fused_bn_pairs"] == 0
    assert report["images_per_s"] > 0
    assert all(math.isfinite(report[key]) and report[key] >= 0
               for key in ("p50", "p95", "p99", "mean"))
    assert model.training and next(model.parameters()).dtype == torch.float32
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())


def test_latency_validation_cuda_unavailable_and_cpu_fp16(monkeypatch):
    model = TinyModel()
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        benchmark.latency_report(model, 1, 4, device="cuda")
    with pytest.raises(ValueError, match="fp16"):
        benchmark.latency_report(model, 1, 4, dtype="fp16", device="cpu")
    with pytest.raises(ValueError, match="warmup"):
        benchmark.latency_report(model, 1, 4, device="cpu", warmup=9)
    with pytest.raises(ValueError, match="iters"):
        benchmark.latency_report(model, 1, 4, device="cpu", iters=49)
    with pytest.raises(ValueError, match="dtype"):
        benchmark.latency_report(model, 1, 4, device="cpu", dtype="int8")


def test_tta_latency_measures_k_forwards_per_iteration():
    TinyModel.forwards = 0
    model = TinyModel().train()
    report = benchmark.tta_latency(model, 3, batch_size=1, img_size=4,
                                   device="cpu", warmup=10, iters=50)
    assert TinyModel.forwards == (1 + 3) * (10 + 50)
    assert report["k_views"] == 3 and report["n"] == 50
    assert report["single_view_p50"] > 0
    assert report["expected_k_times_p50"] == pytest.approx(3 * report["single_view_p50"])
    assert report["ratio_to_single"] == pytest.approx(report["p50"] / report["single_view_p50"])
    assert report["includes_preprocessing"] is False
    assert model.training and next(model.parameters()).dtype == torch.float32
    with pytest.raises(ValueError, match="k_views"):
        benchmark.tta_latency(model, 0, batch_size=1, img_size=4, device="cpu")
