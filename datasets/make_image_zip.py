"""Build a ProofNet image dataset zip (class folders of PNGs) from Fashion-MNIST.

Fashion-MNIST (Zalando Research) is the dataset hosted on Kaggle as "zalando-research/fashionmnist".
Downloading from Kaggle needs a login, so this script reads the identical official release
(https://github.com/zalandoresearch/fashion-mnist) or a Kaggle CSV you downloaded yourself:

    uv run python datasets/make_image_zip.py                          # official release, 10k images
    uv run python datasets/make_image_zip.py --per-class 300          # smaller, faster demo
    uv run python datasets/make_image_zip.py --kaggle-csv fashion-mnist_train.csv

Output: datasets/generated/fashion_mnist_<N>.zip, to upload in ProofNet (Images -> New image task).
Deterministic: the same arguments always produce the same zip.
"""

import argparse
import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "generated" / "_fashion_cache"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
NAMES = [
    "tshirt_top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot",
]  # fmt: skip


def load_official() -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (CACHE / f).exists():
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(BASE + f, CACHE / f)  # noqa: S310 (fixed https URL)
    with gzip.open(CACHE / FILES[0]) as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(CACHE / FILES[1]) as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return x, y


def load_kaggle_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    y = df["label"].to_numpy().astype(np.uint8)
    x = df.drop(columns=["label"]).to_numpy().astype(np.uint8).reshape(-1, 28, 28)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000, help="images per class")
    ap.add_argument("--kaggle-csv", type=Path, default=None, help="Kaggle fashion-mnist CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    x, y = load_kaggle_csv(args.kaggle_csv) if args.kaggle_csv else load_official()
    rng = np.random.default_rng(args.seed)
    picked: list[int] = []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        picked.extend(rng.choice(idx, size=min(args.per_class, len(idx)), replace=False).tolist())
    out = args.out or ROOT / "generated" / f"fashion_mnist_{len(picked) // 1000}k.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for n, i in enumerate(sorted(picked)):
            buf = io.BytesIO()
            Image.fromarray(x[i]).save(buf, format="PNG", optimize=True)
            info = zipfile.ZipInfo(
                f"{NAMES[int(y[i])]}/{n:05d}.png", date_time=(1980, 1, 1, 0, 0, 0)
            )
            z.writestr(info, buf.getvalue())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(picked)} images, 10 classes)")


if __name__ == "__main__":
    main()
