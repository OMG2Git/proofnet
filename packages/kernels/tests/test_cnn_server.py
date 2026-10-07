"""CNN backend kernel: image ingestion (incl. hostile zips), batching, optimizer, and the full
distributed-training replay that must equal centralized training."""

import io
import json
import zipfile
from typing import Any

import numpy as np
import pytest
from PIL import Image

from proofnet_kernels.core import cnn as core
from proofnet_kernels.server import cnn as srv
from proofnet_kernels.server import images
from proofnet_kernels.server.common import KernelValidationError


def synth_zip(per_class: int = 40, side: int = 28, color: bool = False, seed: int = 0) -> bytes:
    """Three easy classes: horizontal bar, vertical bar, filled square (+ noise)."""
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in ("hbar", "vbar", "square"):
            for i in range(per_class):
                a = rng.integers(0, 40, size=(side, side)).astype(np.uint8)
                o = int(rng.integers(3, 8))
                if name == "hbar":
                    a[o : o + 4, 4 : side - 4] = 230
                elif name == "vbar":
                    a[4 : side - 4, o : o + 4] = 230
                else:
                    a[o : o + 12, o : o + 12] = 230
                img = Image.fromarray(a).convert("RGB" if color else "L")
                b = io.BytesIO()
                img.save(b, format="PNG")
                z.writestr(f"train/{name}/{i:03d}.png", b.getvalue())
    return buf.getvalue()


# ---------- ingestion ----------
def test_from_zip_grayscale_and_rgb() -> None:
    ds = images.from_zip(synth_zip(40))
    assert ds.x.shape == (120, 28, 28, 1) and ds.x.dtype == np.uint8
    assert ds.class_names == ["hbar", "square", "vbar"]
    assert np.bincount(ds.y).tolist() == [40, 40, 40]
    rgb = images.from_zip(synth_zip(40, color=True))
    assert rgb.x.shape[-1] == 1  # colour-encoded but actually gray -> one channel
    prof = images.profile_images(ds)
    assert prof["n_images"] == 120 and prof["shape"] == [28, 28, 1] and len(prof["samples"]) == 24
    assert prof["class_counts"] == [40, 40, 40]


def test_from_zip_resizes_and_detects_colour() -> None:
    buf = io.BytesIO()
    rng = np.random.default_rng(1)
    with zipfile.ZipFile(buf, "w") as z:
        for c in ("a", "b"):
            for i in range(60):
                arr = rng.integers(0, 255, size=(40, 50, 3)).astype(np.uint8)  # colour, odd size
                b = io.BytesIO()
                Image.fromarray(arr).save(b, format="JPEG")
                z.writestr(f"{c}/{i}.jpg", b.getvalue())
    ds = images.from_zip(buf.getvalue(), side=24)
    assert ds.x.shape == (120, 24, 24, 3)


def _empty_zip() -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w"):
        pass
    return b.getvalue()


@pytest.mark.parametrize("data", [b"not a zip", _empty_zip()])
def test_from_zip_rejects_garbage_and_empty(data: bytes) -> None:
    with pytest.raises(KernelValidationError):
        images.from_zip(data)


def test_from_zip_rejects_bad_structure() -> None:
    def zipped(files: dict[str, bytes]) -> bytes:
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            for n, d in files.items():
                z.writestr(n, d)
        return b.getvalue()

    png = io.BytesIO()
    Image.fromarray(np.zeros((28, 28), np.uint8)).save(png, format="PNG")
    p = png.getvalue()
    with pytest.raises(KernelValidationError, match="class folders"):
        images.from_zip(zipped({"loose.png": p}))  # no class folder
    with pytest.raises(KernelValidationError, match="decode"):
        images.from_zip(zipped({"a/x.png": b"definitely not a png"}))
    with pytest.raises(KernelValidationError, match="classes"):
        images.from_zip(zipped({f"only/{i}.png": p for i in range(150)}))  # one class
    with pytest.raises(KernelValidationError, match="fewer than"):
        images.from_zip(zipped({**{f"a/{i}.png": p for i in range(150)}, "b/0.png": p}))
    # executable / traversal names are ignored, never written anywhere
    noisy = {f"a/{i}.png": p for i in range(60)} | {f"b/{i}.png": p for i in range(60)}
    noisy["../../evil.sh"] = b"rm -rf /"
    noisy["a/payload.py"] = b"import os"
    ds = images.from_zip(zipped(noisy))
    assert ds.x.shape[0] == 120


def test_from_zip_decompression_bomb_guard() -> None:
    big = io.BytesIO()
    Image.fromarray(np.zeros((4000, 4000), np.uint8)).save(big, format="PNG")  # 16 MP > cap
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for c in ("a", "b"):
            for i in range(60):
                z.writestr(f"{c}/{i}.png", big.getvalue() if i == 0 and c == "a" else b"")
    with pytest.raises(KernelValidationError):
        images.from_zip(b.getvalue())


def test_from_pixel_csv_kaggle_style() -> None:
    rng = np.random.default_rng(0)
    rows = 120
    px = rng.integers(0, 256, size=(rows, 16 * 16))
    lines = ["label," + ",".join(f"pixel{i + 1}" for i in range(256))]
    for r in range(rows):
        lines.append(f"{r % 3}," + ",".join(map(str, px[r])))
    ds = images.from_pixel_csv("\n".join(lines).encode())
    assert ds.x.shape == (120, 16, 16, 1) and ds.class_names == ["0", "1", "2"]
    with pytest.raises(KernelValidationError):
        images.from_pixel_csv(b"label,a,b,c\n1,2,3,4\n")  # not a square image


# ---------- params / validation ----------
def test_params_and_validate() -> None:
    p = srv.CnnParams()
    prof = {"n_images": 3000, "shape": [28, 28, 1], "classes": [str(i) for i in range(10)]}
    rep = srv.validate(prof, p)
    assert rep.ok and rep.summary["n_params"] == 27_562 and rep.summary["epochs"] > 5
    tiny = srv.validate({**prof, "n_images": 120}, srv.CnnParams(global_batch_size=128))
    assert not tiny.ok and any("global_batch_size" in e for e in tiny.errors)
    small_img = srv.validate({**prof, "shape": [6, 6, 1]}, p)
    assert not small_img.ok
    with pytest.raises(ValueError):
        srv.CnnParams(learning_rate=0)
    with pytest.raises(ValueError):
        srv.CnnParams(unknown=1)  # type: ignore[call-arg]


def test_prepare_split_is_deterministic_and_disjoint() -> None:
    ds = images.from_zip(synth_zip(60))
    a = srv.prepare(ds, srv.CnnParams(global_batch_size=32))
    b = srv.prepare(ds, srv.CnnParams(global_batch_size=32))
    assert a.sha256() == b.sha256()
    assert a.n_train + len(a.y_test) == 180 and len(a.y_test) == round(180 * 0.15)
    back = srv.load_image_prepared(a.to_npz_bytes(), a.class_names)
    assert np.array_equal(back.x_train, a.x_train)


def test_batch_indices_cover_each_epoch_exactly_and_are_deterministic() -> None:
    n, b = 100, 30
    seen: list[int] = []
    for step in range(10):  # 300 samples = exactly 3 epochs
        idx = srv.batch_indices(n, b, step, seed=5)
        assert len(idx) == b and idx.min() >= 0 and idx.max() < n
        assert np.array_equal(idx, srv.batch_indices(n, b, step, seed=5))
        seen.extend(idx.tolist())
    for e in range(3):
        epoch = seen[e * n : (e + 1) * n]
        assert sorted(epoch) == list(range(n))  # every sample once per epoch, even across batches
    assert not np.array_equal(
        srv.batch_indices(n, b, 0, seed=5), srv.batch_indices(n, b, 0, seed=6)
    )


def test_verify_round_set() -> None:
    assert srv.verify_round_set(200, 0) == []
    assert srv.verify_round_set(200, 1) == [0]
    assert srv.verify_round_set(200, 3) == [0, 100, 199]
    assert srv.verify_round_set(5, 10) == [0, 1, 2, 3, 4]


# ---------- partial validation ----------
def _partial(n: int = 8) -> dict[str, Any]:
    arch = {"input": [14, 14, 1], "conv1": 2, "conv2": 3, "dense": 5, "classes": 3}
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(n, 14, 14, 1)).astype(np.uint8)
    y = rng.integers(0, 3, size=n)
    return core.map(x, y, core.init_weights(arch, 0), arch)


def test_validate_partial() -> None:
    p = _partial()
    npar = p["n_params"]
    assert srv.validate_partial(json.loads(json.dumps(p)), 8, npar) == []
    assert srv.validate_partial(p, 9, npar)  # wrong row count
    assert srv.validate_partial(p, 8, npar + 1)  # wrong model size
    assert srv.validate_partial({**p, "loss_sum": -1.0}, 8, npar)
    assert srv.validate_partial({**p, "loss_sum": float("nan")}, 8, npar)
    assert srv.validate_partial({**p, "correct": 99}, 8, npar)
    assert srv.validate_partial({**p, "grad_b64": "AAAA"}, 8, npar)  # wrong size
    assert srv.validate_partial({**p, "grad_b64": "!!!not base64!!!"}, 8, npar)
    bad = np.full(npar, np.nan, dtype="<f4")
    import base64

    assert srv.validate_partial(
        {**p, "grad_b64": base64.b64encode(bad.tobytes()).decode()}, 8, npar
    )
    assert srv.validate_partial({k: v for k, v in p.items() if k != "n"}, 8, npar)


# ---------- THE correctness property: distributed training == centralized training ----------
def _train(
    x: np.ndarray,
    y: np.ndarray,
    arch: dict[str, Any],
    params: srv.CnnParams,
    splitter: Any,
) -> tuple[np.ndarray, list[float]]:
    w = core.init_weights(arch, params.init_seed)
    v = np.zeros_like(w)
    losses: list[float] = []
    for step in range(params.steps):
        idx = srv.batch_indices(len(x), params.global_batch_size, step, params.split_seed)
        parts = [
            json.loads(json.dumps(core.map(x[idx[a:b]], y[idx[a:b]], w, arch), allow_nan=False))
            for a, b in splitter(params.global_batch_size, step)
        ]
        merged = core.merge(parts)
        grad_sum = core.decode_vector(merged["grad_sum_b64"], "<f8")
        w, v = srv.sgd_step(
            w, v, grad_sum, params.global_batch_size, params.learning_rate, params.momentum
        )
        losses.append(merged["loss_sum"] / merged["n"])
    return w, losses


def test_full_training_distributed_equals_centralized() -> None:
    ds = images.from_zip(synth_zip(60, seed=3))
    params = srv.CnnParams(
        steps=30, global_batch_size=48, learning_rate=0.05, conv1_filters=4, conv2_filters=4,
        dense_units=16, verify_rounds=0,
    )  # fmt: skip
    prep = srv.prepare(ds, params)
    arch = srv.build_arch(params, prep.shape, len(prep.class_names))

    def whole(b: int, _: int) -> list[tuple[int, int]]:
        return [(0, b)]

    def uneven(b: int, step: int) -> list[tuple[int, int]]:
        # 2-4 devices with changing, very uneven slices (devices come and go between rounds)
        cuts = {0: [0.73], 1: [0.2, 0.55], 2: [0.1, 0.3, 0.9]}[step % 3]
        edges = [0, *[max(1, int(c * b)) for c in cuts], b]
        return list(zip(edges[:-1], edges[1:], strict=True))

    w_c, loss_c = _train(prep.x_train, prep.y_train, arch, params, whole)
    w_d, loss_d = _train(prep.x_train, prep.y_train, arch, params, uneven)
    rel = np.abs(w_c - w_d).max() / np.abs(w_c).max()
    assert rel < 1e-3, rel  # float32 summation-order noise only
    assert max(abs(a - b) for a, b in zip(loss_c, loss_d, strict=True)) < 1e-3
    m_c, pred_c = srv.evaluate(w_c, prep, arch)
    m_d, pred_d = srv.evaluate(w_d, prep, arch)
    assert (pred_c == pred_d).mean() >= 0.97
    assert abs(m_c["accuracy"] - m_d["accuracy"]) <= 0.03
    assert m_d["accuracy"] > 0.9 and loss_d[-1] < 0.5 * loss_d[0]  # and it really learns


def test_gradient_reference_check_catches_tampering() -> None:
    ds = images.from_zip(synth_zip(40))
    arch = {"input": [28, 28, 1], "conv1": 4, "conv2": 4, "dense": 16, "classes": 3}
    w = core.init_weights(arch, 0)
    x, y = ds.x[:48], ds.y[:48]
    parts = [core.map(x[:20], y[:20], w, arch), core.map(x[20:], y[20:], w, arch)]
    merged = core.decode_vector(core.merge(parts)["grad_sum_b64"], "<f8")
    _, _, ref = srv.reference_gradient(w, x, y, arch)
    assert srv.compare_gradients(merged, ref)["within_tolerance"]
    cheat = merged.copy()
    cheat[: cheat.size // 2] *= 1.01  # a device scaling its gradient
    assert not srv.compare_gradients(cheat, ref)["within_tolerance"]


def test_sgd_step_matches_formula() -> None:
    w = np.ones(5, np.float32)
    v = np.zeros(5, np.float32)
    g = np.full(5, 8.0)
    w1, v1 = srv.sgd_step(w, v, g, batch=4, lr=0.1, momentum=0.9)  # mean grad 2 -> v=-0.2
    assert np.allclose(v1, -0.2) and np.allclose(w1, 0.8)
    w2, v2 = srv.sgd_step(w1, v1, g, batch=4, lr=0.1, momentum=0.9)  # v = 0.9*-0.2 - 0.2 = -0.38
    assert np.allclose(v2, -0.38) and np.allclose(w2, 0.42, atol=1e-6)


def test_model_artifacts_roundtrip_and_inference_script() -> None:
    arch = {"input": [28, 28, 1], "conv1": 4, "conv2": 4, "dense": 16, "classes": 3}
    w = core.init_weights(arch, 1)
    arts = srv.model_artifacts(w, arch, ["hbar", "square", "vbar"])
    assert set(arts) == {"model.npz", "model.json", "inference.py"}
    z = np.load(io.BytesIO(arts["model.npz"]), allow_pickle=False)
    assert np.array_equal(z["weights"], w)
    assert json.loads(bytes(z["arch_json"]).decode()) == arch
    mj = json.loads(arts["model.json"])
    assert mj["n_params"] == w.size and mj["class_names"] == ["hbar", "square", "vbar"]
    # the generated inference script is runnable and agrees with predict_logits
    ns: dict[str, Any] = {"__name__": "inference"}
    exec(compile(arts["inference.py"].decode(), "inference.py", "exec"), ns)  # noqa: S102
    x = np.random.default_rng(0).integers(0, 256, size=(3, 28, 28, 1)).astype(np.uint8)
    assert np.allclose(ns["predict_logits"](w, x, arch), core.predict_logits(w, x, arch))
    assert (
        srv.model_artifacts(w, arch, ["a", "b", "c"])["model.npz"]
        == srv.model_artifacts(w, arch, ["a", "b", "c"])["model.npz"]
    )  # byte-reproducible
