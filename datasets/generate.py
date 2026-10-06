"""Deterministic demo dataset generator (stdlib + NumPy only).

Usage:  uv run python datasets/generate.py
Writes datasets/generated/*.csv (git-ignored) and prints SHA-256 digests. Same seed => same bytes.
Tiny committed fixtures live in datasets/fixtures/ (see --fixtures, --real).
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def classification(n: int, d: int, classes: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Gaussian blobs: class-specific means, per-feature scales. Last column = label."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 2.0, size=(classes, d))
    scales = rng.uniform(0.5, 1.5, size=(classes, d))
    y = rng.integers(0, classes, size=n)
    x = means[y] + rng.normal(size=(n, d)) * scales[y]
    cols = [f"f{i}" for i in range(d)] + ["label"]
    return cols, np.column_stack([x, y])


def regression(n: int, d: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Linear signal plus noise. Last column = target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3.0, size=d) + rng.normal(size=d)
    beta = rng.normal(size=d)
    y = x @ beta + 3.0 + rng.normal(scale=0.5, size=n)
    cols = [f"f{i}" for i in range(d)] + ["target"]
    return cols, np.column_stack([x, y])


def write_csv(path: Path, cols: list[str], data: np.ndarray, int_last: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for row in data:
            feats = ",".join(f"{v:.6f}" for v in row[:-1])
            last = str(int(row[-1])) if int_last else f"{row[-1]:.6f}"
            fh.write(f"{feats},{last}\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_datasets() -> list[tuple[Path, list[str], np.ndarray, bool]]:
    """Small real public datasets bundled with scikit-learn (no download). Dev-time only."""
    from sklearn.datasets import load_diabetes, load_wine

    out = ROOT / "fixtures"
    w = load_wine()
    d = load_diabetes()
    return [
        (out / "wine.csv", [*w.feature_names, "label"], np.column_stack([w.data, w.target]), True),
        (
            out / "diabetes.csv",
            [*d.feature_names, "target"],
            np.column_stack([d.data, d.target]),
            False,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true", help="write tiny fixtures only")
    ap.add_argument(
        "--real", action="store_true", help="export real public datasets (wine, diabetes)"
    )
    args = ap.parse_args()
    jobs: list[tuple[Path, list[str], np.ndarray, bool]] = []
    if args.real:
        jobs.extend(real_datasets())
    elif args.fixtures:
        c, a = classification(300, 4, 3, seed=1)
        jobs.append((ROOT / "fixtures" / "clf_small.csv", c, a, True))
        c, a = regression(300, 4, seed=2)
        jobs.append((ROOT / "fixtures" / "reg_small.csv", c, a, False))
    else:
        out = ROOT / "generated"
        c, a = classification(100_000, 16, 3, seed=42)
        jobs.append((out / "clf_100k_16.csv", c, a, True))
        c, a = regression(100_000, 16, seed=43)
        jobs.append((out / "reg_100k_16.csv", c, a, False))
    for path, cols, data, int_last in jobs:
        digest = write_csv(path, cols, data, int_last)
        print(f"{path.name}  {path.stat().st_size / 1e6:.2f} MB  sha256={digest}")


if __name__ == "__main__":
    main()
