"""CNN kernel: correct gradients, data-parallel additivity (distributed == centralized), JSON-safety."""

import json
from typing import Any

import numpy as np
import pytest

from proofnet_kernels.core import cnn
from proofnet_kernels.core.serialize import canonical_json, payload_sha256

TINY = {"input": [14, 14, 1], "conv1": 2, "conv2": 3, "dense": 5, "classes": 3}
SMALL = {"input": [28, 28, 1], "conv1": 8, "conv2": 16, "dense": 64, "classes": 10}


def batch(n: int, arch: dict[str, Any], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w, c = arch["input"]
    return (
        rng.integers(0, 256, size=(n, h, w, c)).astype(np.uint8),
        rng.integers(0, arch["classes"], size=n),
    )


def test_param_count_and_layout() -> None:
    assert cnn.n_params(SMALL) == 27_562
    w = cnn.init_weights(SMALL, 0)
    assert w.shape == (27_562,) and w.dtype == np.float32
    p = cnn.unflatten(w, SMALL)
    assert p["W1"].shape == (8, 9) and p["Wd"].shape == (400, 64) and p["Wo"].shape == (64, 10)
    assert np.array_equal(cnn.flatten(p, SMALL), w)
    assert np.array_equal(cnn.init_weights(SMALL, 0), w)  # deterministic
    assert not np.array_equal(cnn.init_weights(SMALL, 1), w)
    with pytest.raises(ValueError):
        cnn.unflatten(w[:-1], SMALL)
    with pytest.raises(ValueError):
        cnn.arch_dims({**SMALL, "input": [6, 6, 1]})  # too small for two conv+pool stages


def test_gradients_match_finite_differences() -> None:
    """Backprop vs numerical gradient (float64) on random parameters of a tiny network."""
    x, y = batch(6, TINY, seed=1)
    w = cnn.init_weights(TINY, 3, dtype=np.float64)
    # nudge biases off zero so ReLU kinks are not exactly at 0
    w += np.random.default_rng(5).normal(0, 0.05, size=w.shape)
    loss0, _, grad = cnn.loss_and_grad_sum(w, x, y, TINY)
    rng = np.random.default_rng(2)
    eps = 1e-6
    for i in rng.choice(w.size, size=60, replace=False):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        lp, _, _ = cnn.loss_and_grad_sum(wp, x, y, TINY)
        lm, _, _ = cnn.loss_and_grad_sum(wm, x, y, TINY)
        numeric = (lp - lm) / (2 * eps)
        assert abs(numeric - grad[i]) <= 1e-5 * max(1.0, abs(numeric)), (i, numeric, grad[i])
    assert loss0 > 0


@pytest.mark.parametrize("splits", [[64], [30, 34], [1, 63], [10, 20, 34], [16, 16, 16, 16]])
def test_gradient_sums_are_additive_so_distributed_equals_centralized(splits: list[int]) -> None:
    x, y = batch(64, SMALL, seed=4)
    w = cnn.init_weights(SMALL, 7)
    loss_all, correct_all, g_all = cnn.loss_and_grad_sum(w, x, y, SMALL)
    parts, pos = [], 0
    for n in splits:
        parts.append(cnn.map(x[pos : pos + n], y[pos : pos + n], w, SMALL))
        pos += n
    merged = cnn.merge([json.loads(json.dumps(p, allow_nan=False)) for p in parts])  # JSON hop
    g = cnn.decode_vector(merged["grad_sum_b64"], "<f8")
    rel = np.abs(g - g_all).max() / np.abs(g_all).max()
    assert rel < 1e-5, rel
    assert merged["n"] == 64 and merged["correct"] == correct_all
    assert merged["loss_sum"] == pytest.approx(loss_all, rel=1e-5)


def test_payload_is_json_safe_with_stable_digest() -> None:
    x, y = batch(8, SMALL, seed=2)
    p = cnn.map(x, y, cnn.init_weights(SMALL, 0), SMALL)
    assert isinstance(p["grad_b64"], str) and p["n_params"] == 27_562
    text = canonical_json(p)  # allow_nan=False: raises on NaN/inf
    assert payload_sha256(json.loads(text)) == payload_sha256(p)
    assert len(cnn.decode_vector(p["grad_b64"], "<f4")) == 27_562


def test_maxpool_ties_and_odd_sizes_do_not_break_backprop() -> None:
    """All-zero image: every ReLU/pool window ties; gradients must stay finite."""
    x = np.zeros((4, 14, 14, 1), dtype=np.uint8)
    y = np.array([0, 1, 2, 0])
    _, _, g = cnn.loss_and_grad_sum(cnn.init_weights(TINY, 1), x, y, TINY)
    assert np.isfinite(g).all()
    odd = {"input": [15, 15, 3], "conv1": 2, "conv2": 2, "dense": 4, "classes": 2}  # odd -> floors
    xo, yo = batch(5, odd, seed=3)
    _, _, g2 = cnn.loss_and_grad_sum(cnn.init_weights(odd, 1), xo, yo, odd)
    assert g2.shape == (cnn.n_params(odd),) and np.isfinite(g2).all()


def test_sgd_on_separable_data_reduces_loss() -> None:
    rng = np.random.default_rng(0)
    n = 96
    y = rng.integers(0, 2, size=n)
    x = rng.integers(0, 30, size=(n, 14, 14, 1)).astype(np.uint8)
    x[y == 1, 3:9, 3:9, 0] = 220  # class 1 has a bright square
    arch = {"input": [14, 14, 1], "conv1": 4, "conv2": 4, "dense": 8, "classes": 2}
    w = cnn.init_weights(arch, 0)
    first, _, _ = cnn.loss_and_grad_sum(w, x, y, arch)
    for _ in range(40):
        _, _, g = cnn.loss_and_grad_sum(w, x, y, arch)
        w = (w - 0.1 * g / n).astype(np.float32)
    last, correct, _ = cnn.loss_and_grad_sum(w, x, y, arch)
    assert last < 0.3 * first and correct >= int(0.95 * n)


def test_predict_logits_batches_consistently() -> None:
    x, _ = batch(37, SMALL, seed=6)
    w = cnn.init_weights(SMALL, 2)
    a = cnn.predict_logits(w, x, SMALL, batch=8)
    b = cnn.predict_logits(w, x, SMALL, batch=256)
    assert a.shape == (37, 10) and np.allclose(a, b, atol=1e-5)
