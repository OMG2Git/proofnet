"""CNN image-classification kernel, backend side (ARCHITECTURE 4.5).

Iterative data-parallel training: every round the global mini-batch is split across devices by
measured benchmark; each device returns the SUM of per-sample gradients for its slice; the backend
adds them, divides by the global batch size and applies SGD with momentum. That is exactly the
gradient of the same batch on one machine, which the backend re-checks on sampled rounds.
Worker side (map/merge) lives in core/cnn.py.
"""

import base64
import inspect
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import confusion_matrix, f1_score

from ..core import cnn as core
from ..core.serialize import all_finite, sha256_hex
from .common import KernelValidationError, ValidationReport, deterministic_npz
from .images import ImageSet

TASK_TYPE = core.KERNEL
GRADIENT_TOLERANCE = 1e-4  # normwise relative, float32 gradient sums (summation order differs)
AUDIT_FLOOR = 1e-5  # smallest tolerance used by the trust system (float32 phone vs backend)
MIN_ROWS_PER_DEVICE = 8


class CnnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=200, ge=5, le=2000, description="training rounds (SGD steps)")
    global_batch_size: int = Field(default=128, ge=16, le=512)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    momentum: float = Field(default=0.9, ge=0, le=0.99)
    conv1_filters: int = Field(default=8, ge=4, le=16)
    conv2_filters: int = Field(default=16, ge=4, le=32)
    dense_units: int = Field(default=64, ge=16, le=128)
    test_fraction: float = Field(default=0.15, ge=0.05, le=0.5)
    split_seed: int = 42
    init_seed: int = 0
    verify_rounds: int = Field(
        default=3, ge=0, le=10, description="rounds re-checked against a centralized gradient"
    )


PARAMS_MODEL = CnnParams


def build_arch(params: CnnParams, shape: list[int], n_classes: int) -> dict[str, Any]:
    return {
        "input": [int(v) for v in shape],
        "conv1": params.conv1_filters,
        "conv2": params.conv2_filters,
        "dense": params.dense_units,
        "classes": int(n_classes),
    }


def validate(profile: dict[str, Any], params: CnnParams) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    n = int(profile["n_images"])
    n_train = n - max(1, round(n * params.test_fraction))
    if n_train < params.global_batch_size:
        errors.append(
            f"{n_train} training images but global_batch_size is {params.global_batch_size}"
        )
    try:
        arch = build_arch(params, profile["shape"], len(profile["classes"]))
        core.arch_dims(arch)
    except ValueError as e:
        errors.append(str(e))
        arch = None
    epochs = params.steps * params.global_batch_size / max(n_train, 1)
    if epochs < 1:
        warnings.append(f"only {epochs:.2f} epochs of training; accuracy will be limited")
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        summary={
            "n_images": n,
            "n_train_estimate": n_train,
            "epochs": round(epochs, 2),
            "n_params": core.n_params(arch) if arch else None,
        },
    )


@dataclass
class ImagePrepared:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def shape(self) -> list[int]:
        return [int(v) for v in self.x_train.shape[1:]]

    def to_npz_bytes(self) -> bytes:
        return deterministic_npz(
            {
                "X_train": self.x_train,
                "y_train": self.y_train,
                "X_test": self.x_test,
                "y_test": self.y_test,
            }
        )

    def sha256(self) -> str:
        return sha256_hex(self.to_npz_bytes())


def prepare(ds: ImageSet, params: CnnParams) -> ImagePrepared:
    """Seeded shuffle + holdout split (the holdout never leaves the backend)."""
    n = len(ds.x)
    perm = np.random.default_rng(params.split_seed).permutation(n)
    n_test = max(1, round(n * params.test_fraction))
    te, tr = perm[:n_test], perm[n_test:]
    if len(tr) < params.global_batch_size:
        raise KernelValidationError(
            ValidationReport(False, ["fewer training images than global_batch_size"])
        )
    return ImagePrepared(ds.x[tr], ds.y[tr], ds.x[te], ds.y[te], list(ds.class_names))


def load_image_prepared(data: bytes, class_names: list[str]) -> ImagePrepared:
    import io

    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return ImagePrepared(z["X_train"], z["y_train"], z["X_test"], z["y_test"], class_names)


def batch_indices(n_train: int, batch: int, step: int, seed: int) -> np.ndarray:
    """Training-set indices of the global batch for `step`: sequential walk over per-epoch
    permutations (deterministic; a batch may span two epochs)."""
    out: list[np.ndarray] = []
    pos = step * batch
    need = batch
    while need > 0:
        epoch, off = divmod(pos, n_train)
        perm = np.random.default_rng(seed + 1_000_003 * epoch).permutation(n_train)
        take = min(need, n_train - off)
        out.append(perm[off : off + take])
        pos += take
        need -= take
    return np.concatenate(out)


def verify_round_set(steps: int, k: int) -> list[int]:
    """Rounds re-checked against a centralized gradient: spread evenly, always including 0."""
    if k <= 0:
        return []
    if k == 1:
        return [0]
    return sorted({int(round(i * (steps - 1) / (k - 1))) for i in range(k)})


def sgd_step(
    w: np.ndarray, v: np.ndarray, grad_sum: np.ndarray, batch: int, lr: float, momentum: float
) -> tuple[np.ndarray, np.ndarray]:
    g = (grad_sum / batch).astype(np.float32)
    v2 = (momentum * v - lr * g).astype(np.float32)
    return (w + v2).astype(np.float32), v2


def validate_partial(partial: dict[str, Any], n_rows: int, n_params: int) -> list[str]:
    try:
        if partial["n"] != n_rows or not isinstance(partial["n"], int):
            return [f"n does not match chunk rows ({n_rows})"]
        if partial["n_params"] != n_params:
            return ["n_params does not match the model"]
        if not all_finite([partial["loss_sum"]]) or partial["loss_sum"] < 0:
            return ["loss_sum must be a finite non-negative number"]
        c = partial["correct"]
        if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= n_rows:
            return ["correct must be an integer in [0, n]"]
        raw = base64.b64decode(partial["grad_b64"].encode("ascii"), validate=True)
        if len(raw) != 4 * n_params:
            return ["gradient has the wrong size"]
        g = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(g).all():
            return ["gradient contains non-finite values"]
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return [f"malformed payload: {e}"]
    return []


def reference_gradient(
    weights: np.ndarray, x: np.ndarray, y: np.ndarray, arch: dict[str, Any]
) -> tuple[float, int, np.ndarray]:
    """Centralized recomputation of one global batch (the check that distributed == centralized)."""
    return core.loss_and_grad_sum(np.asarray(weights, dtype=np.float32), x, y, arch)


def compare_gradients(merged_sum: np.ndarray, reference_sum: np.ndarray) -> dict[str, Any]:
    a = np.asarray(merged_sum, dtype=np.float64)
    b = np.asarray(reference_sum, dtype=np.float64)
    scale = float(np.abs(b).max())
    diff = float(np.abs(a - b).max())
    rel = diff / scale if scale > 0 else diff
    return {
        "max_rel_diff": rel,
        "tolerance": GRADIENT_TOLERANCE,
        "within_tolerance": rel <= GRADIENT_TOLERANCE,
    }


def evaluate(
    weights: np.ndarray, prepared: ImagePrepared, arch: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    logits = core.predict_logits(weights, prepared.x_test, arch)
    pred = logits.argmax(axis=1)
    y = prepared.y_test
    m = logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(logits - m).sum(axis=1)) + m[:, 0]
    loss = float((lse - logits[np.arange(len(y)), y]).mean())
    k = len(prepared.class_names)
    cm = confusion_matrix(y, pred, labels=np.arange(k))
    per_class = [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(k)]
    metrics = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "loss": loss,
        "n_test": int(len(y)),
        "per_class_accuracy": dict(zip(prepared.class_names, per_class, strict=True)),
        "confusion_matrix": cm.tolist(),
        "class_names": prepared.class_names,
    }
    return metrics, pred


def model_artifacts(
    weights: np.ndarray, arch: dict[str, Any], class_names: list[str]
) -> dict[str, bytes]:
    """model.npz (weights, no pickle), model.json (portable) and a runnable inference.py."""
    w32 = np.asarray(weights, dtype=np.float32)
    npz = deterministic_npz(
        {
            "weights": w32,
            "arch_json": np.frombuffer(json.dumps(arch, sort_keys=True).encode(), dtype=np.uint8),
            "classes_json": np.frombuffer(json.dumps(class_names).encode(), dtype=np.uint8),
        }
    )
    model_json = json.dumps(
        {
            "kind": "cnn_image",
            "architecture": arch,
            "class_names": class_names,
            "preprocessing": "pixel/255, channels-last, resized to the input side",
            "n_params": int(w32.size),
            "weights_base64_float32_le": base64.b64encode(w32.astype("<f4").tobytes()).decode(),
        },
        indent=2,
    ).encode()
    return {"model.npz": npz, "model.json": model_json, "inference.py": inference_script().encode()}


def inference_script() -> str:
    """Self-contained script: the CNN forward code plus a tiny command-line wrapper."""
    src = inspect.getsource(core)
    main = """

if __name__ == "__main__":
    import json
    import sys

    from PIL import Image  # pip install pillow

    z = np.load(sys.argv[1], allow_pickle=False)  # model.npz
    arch = json.loads(bytes(z["arch_json"]).decode())
    names = json.loads(bytes(z["classes_json"]).decode())
    h, w, c = arch["input"]
    img = Image.open(sys.argv[2]).convert("L" if c == 1 else "RGB").resize((w, h))
    x = np.asarray(img, dtype=np.uint8).reshape(1, h, w, c)
    logits = predict_logits(z["weights"], x, arch)[0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    for i in np.argsort(-p)[:3]:
        print(f"{names[i]}: {p[i]:.3f}")
"""
    return '"""Run a ProofNet CNN: python inference.py model.npz image.png"""\n' + src + main
