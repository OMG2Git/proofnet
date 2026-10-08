"""Attack modes for the CLI worker (Part 2 test harness; never run on a real contributor).

Every attack returns a *structurally valid* result: right shapes, finite numbers, integer counts
that add up to the chunk size, non-negative second moments. The backend's structural checks
therefore pass, and only the verification layer (audit by recomputation) can catch them. That is
the point: these model a rational adversary, not a buggy client.

  subtle       relative error of 1e-4 on every statistic        (hard to see, still wrong)
  scale        every statistic x 1.05                            (inflated contribution)
  bias         every statistic shifted by 5 % of its magnitude   (systematic poisoning)
  noise        Gaussian noise of 1 % of each statistic           (sloppy / lossy device)
  sign_flip    means / gradients negated                         (classic gradient attack)
  zero         all statistics zero                               (free-rider: claims work, does none)
  random       random values of the right shape                  (garbage)
  replay       the device's previous result with the new counts (cheap: no computation)
  lazy         computes on half of the rows and scales up        (cuts compute cost by 50 %)

Schedules: `attack_after` honest results first (a sleeper that builds trust, then cheats) and
`attack_prob` (cheats only some of the time). Ground truth is kept in `AttackLog` on the worker
side only; the backend never sees it.
"""

import base64
import copy
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODES = (
    "none",
    "subtle",
    "scale",
    "bias",
    "noise",
    "sign_flip",
    "zero",
    "random",
    "replay",
    "lazy",
)
COUNT_KEYS = {"n", "n_params", "correct"}
NONNEG_KEYS = {"M2", "Sxx", "Syy"}  # must stay >= 0 to pass the structural check


@dataclass
class AttackLog:
    """Ground truth of what this worker really did (for tests and the simulator's scorecard)."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, assignment_id: str, attacked: bool, mode: str) -> None:
        self.entries.append({"assignment_id": assignment_id, "attacked": attacked, "mode": mode})

    def attacked_ids(self) -> set[str]:
        return {e["assignment_id"] for e in self.entries if e["attacked"]}


def _decode(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s.encode("ascii")), dtype="<f4").astype(np.float64)


def _encode(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _transform(arr: np.ndarray, mode: str, rng: random.Random, key: str) -> np.ndarray:
    scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = scale if scale > 0 else 1.0
    nprng = np.random.default_rng(rng.getrandbits(32))
    if mode == "subtle":
        return arr * (1.0 + 1e-4)
    if mode == "scale":
        return arr * 1.05
    if mode == "bias":
        return arr + 0.05 * scale
    if mode == "noise":
        out = arr + nprng.normal(0.0, 0.01 * scale, arr.shape)
    elif mode == "sign_flip":
        return arr if key in NONNEG_KEYS else -arr
    elif mode == "zero":
        return np.zeros_like(arr)
    elif mode == "random":
        out = nprng.normal(0.0, scale, arr.shape)
    else:
        return arr
    return np.abs(out) if key in NONNEG_KEYS else out


def _walk(obj: Any, mode: str, rng: random.Random, key: str = "") -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in COUNT_KEYS:
                out[k] = v
            elif k == "grad_b64":
                out[k] = _encode(_transform(_decode(v), mode, rng, k))
            elif k == "loss_sum":
                # a loss is a positive quantity: keep the forged value plausible
                out[k] = abs(float(_transform(np.array([v]), mode, rng, "loss_sum")[0]))
            else:
                out[k] = _walk(v, mode, rng, k)
        return out
    if isinstance(obj, list):
        arr = np.asarray(obj)
        if arr.dtype.kind in "iu":  # integer lists (class counts) are never touched
            return obj
        return _transform(arr.astype(np.float64), mode, rng, key).tolist()
    if isinstance(obj, float):
        return float(_transform(np.array([obj]), mode, rng, key)[0])
    return obj


def _replay(honest: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """Float fields from the previous result, integer counts from the current chunk."""
    if previous is None:
        return honest

    def merge(cur: Any, old: Any) -> Any:
        if isinstance(cur, dict) and isinstance(old, dict) and cur.keys() == old.keys():
            return {k: (cur[k] if k in COUNT_KEYS else merge(cur[k], old[k])) for k in cur}
        if isinstance(cur, list) and isinstance(old, list):
            if np.asarray(cur).dtype.kind in "iu" or np.shape(cur) != np.shape(old):
                return cur
            return old
        if isinstance(cur, float) and isinstance(old, float):
            return old
        if isinstance(cur, str) and isinstance(old, str):
            return old if len(cur) == len(old) else cur
        return cur

    return merge(honest, copy.deepcopy(previous))  # type: ignore[no-any-return]


def lazy_subsample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows a lazy device actually computes on (every second one)."""
    return x[::2], y[::2]


def lazy_fix(kernel: str, payload: dict[str, Any], n_rows: int) -> dict[str, Any]:
    """Make a half-data result look like a full one: counts scaled up to n_rows, sums doubled."""
    n_half = payload["n"]
    f = n_rows / max(n_half, 1)
    out = copy.deepcopy(payload)
    if kernel == "gaussian_nb_train":
        out["n"] = n_rows
        counts = out["classes"]["n"]
        scaled = [int(round(c * f)) for c in counts]
        scaled[int(np.argmax(scaled))] += n_rows - sum(scaled)
        out["classes"]["n"] = scaled
        out["M2"] = (np.asarray(out["M2"]) * f).tolist()
        out["classes"]["M2"] = (np.asarray(out["classes"]["M2"]) * f).tolist()
    elif kernel == "linear_ridge_train":
        out["n"] = n_rows
        out["Sxx"] = (np.asarray(out["Sxx"]) * f).tolist()
        out["Sxy"] = (np.asarray(out["Sxy"]) * f).tolist()
        out["Syy"] = float(out["Syy"]) * f
    else:  # cnn gradient sums
        out["n"] = n_rows
        out["loss_sum"] = float(out["loss_sum"]) * f
        out["correct"] = int(round(out["correct"] * f))
        out["grad_b64"] = _encode(_decode(out["grad_b64"]) * f)
    return out


def apply_attack(
    mode: str,
    honest: dict[str, Any],
    rng: random.Random,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transform an honestly computed payload (not used for `lazy`, which changes the compute)."""
    if mode in ("none", "lazy"):
        return honest
    if mode == "replay":
        return _replay(honest, previous)
    return _walk(copy.deepcopy(honest), mode, rng)  # type: ignore[no-any-return]
