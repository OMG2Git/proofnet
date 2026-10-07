"""Small CNN in NumPy: conv3x3 -> ReLU -> maxpool2 -> conv3x3 -> ReLU -> maxpool2 -> dense -> ReLU
-> dense(softmax). Forward, backward, and the data-parallel map/merge used by ProofNet.

Runs unchanged on CPython and Pyodide (NumPy + standard library only, ProofNet-authored: no user
code, no deep-learning framework). Images are channels-last uint8 (N, H, W, C).

Data-parallel training (ARCHITECTURE 4.5): a worker computes the SUM over its mini-batch slice of the
per-sample loss and gradient. Sums are additive across slices, so
    sum over devices of grad_sum  ==  gradient of the same global batch computed on one machine
(up to floating-point summation order), and the backend divides by the global batch size.
"""

import base64
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KERNEL = "cnn_image_train"
VERSION = "1"
KSZ = 3  # convolution kernel size (valid padding)

Arch = dict[str, Any]


# ------------------------------------------------------------------ architecture / parameters
def arch_dims(arch: Arch) -> dict[str, int]:
    h, w, c = (int(v) for v in arch["input"])
    f1, f2, dense, k = (int(arch[x]) for x in ("conv1", "conv2", "dense", "classes"))
    h1, w1 = h - KSZ + 1, w - KSZ + 1
    p1h, p1w = h1 // 2, w1 // 2
    h2, w2 = p1h - KSZ + 1, p1w - KSZ + 1
    p2h, p2w = h2 // 2, w2 // 2
    if min(h1, w1, p1h, p1w, h2, w2, p2h, p2w) < 1:
        raise ValueError("input too small for this architecture")
    return {
        "h": h, "w": w, "c": c, "f1": f1, "f2": f2, "dense": dense, "k": k,
        "p2h": p2h, "p2w": p2w, "flat": p2h * p2w * f2,
    }  # fmt: skip


def param_shapes(arch: Arch) -> list[tuple[str, tuple[int, ...]]]:
    d = arch_dims(arch)
    return [
        ("W1", (d["f1"], d["c"] * KSZ * KSZ)),
        ("b1", (d["f1"],)),
        ("W2", (d["f2"], d["f1"] * KSZ * KSZ)),
        ("b2", (d["f2"],)),
        ("Wd", (d["flat"], d["dense"])),
        ("bd", (d["dense"],)),
        ("Wo", (d["dense"], d["k"])),
        ("bo", (d["k"],)),
    ]


def n_params(arch: Arch) -> int:
    return int(sum(int(np.prod(s)) for _, s in param_shapes(arch)))


def unflatten(vec: np.ndarray, arch: Arch) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    pos = 0
    for name, shape in param_shapes(arch):
        size = int(np.prod(shape))
        out[name] = vec[pos : pos + size].reshape(shape)
        pos += size
    if pos != vec.size:
        raise ValueError(f"expected {pos} parameters, got {vec.size}")
    return out


def flatten(params: dict[str, np.ndarray], arch: Arch, dtype: Any = np.float32) -> np.ndarray:
    return np.concatenate([params[name].reshape(-1) for name, _ in param_shapes(arch)]).astype(
        dtype
    )


def init_weights(arch: Arch, seed: int, dtype: Any = np.float32) -> np.ndarray:
    """He-normal weights, zero biases; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for name, shape in param_shapes(arch):
        if name.startswith("b"):
            parts.append(np.zeros(shape))
        else:
            fan_in = shape[1] if name in ("W1", "W2") else shape[0]
            parts.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape))
    return np.concatenate([p.reshape(-1) for p in parts]).astype(dtype)


# ------------------------------------------------------------------ layers
def _conv_fwd(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    win = sliding_window_view(x, (KSZ, KSZ), axis=(1, 2))  # (N, H', W', C, 3, 3)
    n, h, ww, c = win.shape[:4]
    cols = np.ascontiguousarray(win).reshape(n * h * ww, c * KSZ * KSZ)
    out = cols @ w.T + b
    return out.reshape(n, h, ww, -1), cols


def _conv_bwd(
    dout: np.ndarray, cols: np.ndarray, w: np.ndarray, x_shape: tuple[int, ...], need_dx: bool
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    n, h, ww, f = dout.shape
    d2 = dout.reshape(n * h * ww, f)
    dw = d2.T @ cols
    db = d2.sum(axis=0)
    if not need_dx:
        return None, dw, db
    c = x_shape[3]
    dcols = (d2 @ w).reshape(n, h, ww, c, KSZ, KSZ)
    dx = np.zeros(x_shape, dtype=dout.dtype)
    for i in range(KSZ):
        for j in range(KSZ):
            dx[:, i : i + h, j : j + ww, :] += dcols[..., i, j]
    return dx, dw, db


def _pool_fwd(x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, tuple[int, ...]]]:
    n, h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    xr = x[:, : h2 * 2, : w2 * 2, :].reshape(n, h2, 2, w2, 2, c)
    out = xr.max(axis=(2, 4))
    mask = xr == out[:, :, None, :, None, :]
    return out, (mask, x.shape)


def _pool_bwd(dout: np.ndarray, cache: tuple[np.ndarray, tuple[int, ...]]) -> np.ndarray:
    mask, shape = cache
    n, h, w, c = shape
    h2, w2 = h // 2, w // 2
    cnt = mask.sum(axis=(2, 4), keepdims=True)
    g = mask * (dout[:, :, None, :, None, :] / cnt)
    dx = np.zeros(shape, dtype=dout.dtype)
    dx[:, : h2 * 2, : w2 * 2, :] = g.reshape(n, h2 * 2, w2 * 2, c)
    return dx


# ------------------------------------------------------------------ forward / loss / backward
def _prep(x_u8: np.ndarray, dtype: Any) -> np.ndarray:
    return np.asarray(x_u8.astype(dtype) / dtype(255.0))


def forward(
    p: dict[str, np.ndarray], x: np.ndarray, arch: Arch, keep: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    cache: dict[str, Any] = {}
    z1, cols1 = _conv_fwd(x, p["W1"], p["b1"])
    a1 = np.maximum(z1, 0)
    q1, pc1 = _pool_fwd(a1)
    z2, cols2 = _conv_fwd(q1, p["W2"], p["b2"])
    a2 = np.maximum(z2, 0)
    q2, pc2 = _pool_fwd(a2)
    flat = q2.reshape(q2.shape[0], -1)
    zd = flat @ p["Wd"] + p["bd"]
    ad = np.maximum(zd, 0)
    logits = ad @ p["Wo"] + p["bo"]
    if keep:
        cache = dict(
            x_shape=x.shape, cols1=cols1, z1=z1, pc1=pc1, q1=q1, q1_shape=q1.shape, cols2=cols2,
            z2=z2, pc2=pc2, q2_shape=q2.shape, flat=flat, zd=zd, ad=ad,
        )  # fmt: skip
    return logits, cache


def predict_logits(
    weights: np.ndarray, x_u8: np.ndarray, arch: Arch, batch: int = 256
) -> np.ndarray:
    p = unflatten(weights, arch)
    dtype = weights.dtype.type
    outs = [
        forward(p, _prep(x_u8[i : i + batch], dtype), arch)[0] for i in range(0, len(x_u8), batch)
    ]
    return np.concatenate(outs) if outs else np.zeros((0, arch["classes"]), dtype=dtype)


def loss_and_grad_sum(
    weights: np.ndarray, x_u8: np.ndarray, y: np.ndarray, arch: Arch
) -> tuple[float, int, np.ndarray]:
    """(sum of per-sample cross-entropy, #correct, gradient of that sum) for a batch slice."""
    dtype = weights.dtype.type
    p = unflatten(weights, arch)
    x = _prep(x_u8, dtype)
    y = np.asarray(y).astype(np.int64)
    n = x.shape[0]
    logits, c = forward(p, x, arch, keep=True)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    probs = e / s
    idx = np.arange(n)
    loss_sum = float((np.log(s[:, 0]) + m[:, 0] - logits[idx, y]).astype(np.float64).sum())
    correct = int((logits.argmax(axis=1) == y).sum())

    dz = probs.copy()
    dz[idx, y] -= 1.0  # gradient of the SUM of losses (no 1/n)
    dwo = c["ad"].T @ dz
    dbo = dz.sum(axis=0)
    dad = dz @ p["Wo"].T
    dzd = dad * (c["zd"] > 0)
    dwd = c["flat"].T @ dzd
    dbd = dzd.sum(axis=0)
    dflat = dzd @ p["Wd"].T
    dq2 = dflat.reshape(c["q2_shape"])
    da2 = _pool_bwd(dq2, c["pc2"])
    dz2 = da2 * (c["z2"] > 0)
    dq1, dw2, db2 = _conv_bwd(dz2, c["cols2"], p["W2"], c["q1_shape"], need_dx=True)
    assert dq1 is not None
    da1 = _pool_bwd(dq1, c["pc1"])
    dz1 = da1 * (c["z1"] > 0)
    _, dw1, db1 = _conv_bwd(dz1, c["cols1"], p["W1"], c["x_shape"], need_dx=False)
    grads = {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2, "Wd": dwd, "bd": dbd, "Wo": dwo, "bo": dbo}
    return loss_sum, correct, flatten(grads, arch, dtype)


# ------------------------------------------------------------------ worker map / merge (JSON-safe)
def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def decode_vector(s: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s.encode("ascii")), dtype=dtype)


def map(  # noqa: A001
    x_u8: np.ndarray, y: np.ndarray, weights: np.ndarray, arch: Arch
) -> dict[str, Any]:
    """Partial result for one mini-batch slice (JSON-serializable; gradient as base64 float32)."""
    loss_sum, correct, grad = loss_and_grad_sum(
        np.asarray(weights, dtype=np.float32), x_u8, y, arch
    )
    return {
        "n": int(x_u8.shape[0]),
        "loss_sum": loss_sum,
        "correct": correct,
        "n_params": int(grad.size),
        "grad_b64": _b64(grad, "<f4"),
    }


def merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum slices in the given (chunk-index) order. Gradient sums are accumulated in float64."""
    if not partials:
        raise ValueError("nothing to merge")
    total = np.zeros(int(partials[0]["n_params"]), dtype=np.float64)
    n, loss_sum, correct = 0, 0.0, 0
    for p in partials:
        total += decode_vector(p["grad_b64"], "<f4").astype(np.float64)
        n += int(p["n"])
        loss_sum += float(p["loss_sum"])
        correct += int(p["correct"])
    return {
        "n": n,
        "loss_sum": loss_sum,
        "correct": correct,
        "n_params": int(total.size),
        "grad_sum_b64": _b64(total, "<f8"),
    }
