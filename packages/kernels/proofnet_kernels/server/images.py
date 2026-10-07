"""Image dataset ingestion (backend only): a zip of class folders, or a pixel CSV (Kaggle
"MNIST-style": label, pixel1..pixelN).

Hostile-input rules: nothing is extracted to disk, only png/jpg members are decoded, member and total
sizes are capped before decoding, pixel counts are capped (decompression-bomb guard), class names
come from the folder name only (no path is ever used to write anything).
"""

import base64
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .common import KernelValidationError, ValidationReport, deterministic_npz

MAX_IMAGES = 30_000
MIN_IMAGES = 100
MIN_CLASSES, MAX_CLASSES = 2, 20
MIN_PER_CLASS = 5
MAX_MEMBER_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 400 * 1024 * 1024
MAX_SOURCE_PIXELS = 4_000_000  # per image, before decoding
TARGET_SIDES = (16, 20, 24, 28, 32)
ALLOWED_EXT = (".png", ".jpg", ".jpeg")
_SAFE = re.compile(r"[^A-Za-z0-9_.\- ]")

Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


@dataclass
class ImageSet:
    x: np.ndarray  # (N, H, W, C) uint8
    y: np.ndarray  # (N,) int64
    class_names: list[str]


def _fail(*errors: str) -> KernelValidationError:
    return KernelValidationError(ValidationReport(False, list(errors)))


def _to_array(img: Image.Image, side: int, channels: int) -> np.ndarray:
    img = img.convert("L" if channels == 1 else "RGB")
    if img.size != (side, side):
        img = img.resize((side, side), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr.reshape(side, side, channels)


def _finish(items: list[tuple[str, np.ndarray]]) -> ImageSet:
    names = sorted({c for c, _ in items})
    counts = {n: sum(1 for c, _ in items if c == n) for n in names}
    if not MIN_CLASSES <= len(names) <= MAX_CLASSES:
        raise _fail(f"found {len(names)} classes; {MIN_CLASSES}-{MAX_CLASSES} required")
    small = [n for n, k in counts.items() if k < MIN_PER_CLASS]
    if small:
        raise _fail(f"classes with fewer than {MIN_PER_CLASS} images: {small}")
    if len(items) < MIN_IMAGES:
        raise _fail(f"{len(items)} images; at least {MIN_IMAGES} required")
    index = {n: i for i, n in enumerate(names)}
    x = np.stack([a for _, a in items])
    y = np.array([index[c] for c, _ in items], dtype=np.int64)
    # deterministic order: by class then insertion order, shuffled later by the split seed
    return ImageSet(x, y, names)


def from_zip(data: bytes, side: int = 28) -> ImageSet:
    """Zip of images in class folders: `<class>/<file>.png` (any depth; class = parent folder)."""
    if side not in TARGET_SIDES:
        raise _fail(f"image side must be one of {TARGET_SIDES}")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise _fail("not a valid zip file") from None
    members = [
        m
        for m in zf.infolist()
        if not m.is_dir()
        and m.filename.lower().endswith(ALLOWED_EXT)
        and "__MACOSX" not in m.filename
        and not m.filename.rsplit("/", 1)[-1].startswith(".")
        and "/" in m.filename.replace("\\", "/")  # needs a class folder
    ]
    if not members:
        raise _fail("no png/jpg images inside class folders found in the zip")
    if len(members) > MAX_IMAGES:
        raise _fail(f"{len(members)} images; at most {MAX_IMAGES} allowed")
    if sum(m.file_size for m in members) > MAX_TOTAL_BYTES or any(
        m.file_size > MAX_MEMBER_BYTES for m in members
    ):
        raise _fail("zip contents too large (member or total size limit)")
    rgb_items: list[tuple[str, np.ndarray]] = []
    all_gray = True
    for m in sorted(members, key=lambda m: m.filename):
        cls = _SAFE.sub("_", m.filename.replace("\\", "/").split("/")[-2])[:60]
        try:
            img = Image.open(io.BytesIO(zf.read(m)))
            img.load()
        except Exception:
            raise _fail(f"cannot decode image '{m.filename.split('/')[-1]}'") from None
        a = _to_array(img, side, 3)
        if all_gray and not ((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()):
            all_gray = False
        rgb_items.append((cls, a))
    # one channel when every image is gray (the usual case for MNIST-like datasets), else RGB
    items = [(c, a[..., :1].copy() if all_gray else a) for c, a in rgb_items]
    return _finish(items)


def from_pixel_csv(data: bytes, side: int | None = None) -> ImageSet:
    """Kaggle MNIST-style CSV: a `label` column and pixel columns (square, 0-255)."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        raise _fail("could not parse the CSV") from None
    label_col = next((c for c in df.columns if str(c).lower() == "label"), None)
    if label_col is None:
        raise _fail("pixel CSV needs a 'label' column")
    px = df.drop(columns=[label_col])
    n_px = px.shape[1]
    s = int(round(n_px**0.5))
    if s * s != n_px or s < 8:
        raise _fail(f"{n_px} pixel columns is not a square image")
    arr = px.to_numpy()
    if arr.min() < 0 or arr.max() > 255:
        raise _fail("pixel values must be 0-255")
    x = arr.astype(np.uint8).reshape(-1, s, s, 1)
    labels = df[label_col].astype(str).to_numpy()
    items = [(str(c), x[i]) for i, c in enumerate(labels)]
    out = _finish(items)
    if side is not None and side != s:
        resized = np.stack(
            [
                np.asarray(
                    Image.fromarray(a[..., 0]).resize((side, side), Image.Resampling.BILINEAR)
                )
                for a in out.x
            ]
        ).reshape(-1, side, side, 1)
        out = ImageSet(resized, out.y, out.class_names)
    return out


def profile_images(ds: ImageSet, n_samples: int = 24) -> dict[str, Any]:
    """Dataset summary shown in the UI, including a few small sample thumbnails (real images)."""
    counts = np.bincount(ds.y, minlength=len(ds.class_names))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds.x), size=min(n_samples, len(ds.x)), replace=False)
    samples = []
    for i in sorted(pick.tolist()):
        a = ds.x[i]
        img = Image.fromarray(a[..., 0] if a.shape[2] == 1 else a)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        samples.append(
            {
                "class": ds.class_names[int(ds.y[i])],
                "png_b64": base64.b64encode(buf.getvalue()).decode(),
            }
        )
    return {
        "n_images": int(len(ds.x)),
        "shape": [int(v) for v in ds.x.shape[1:]],
        "classes": ds.class_names,
        "class_counts": [int(c) for c in counts],
        "samples": samples,
    }


def to_npz(ds: ImageSet) -> bytes:
    return deterministic_npz({"X": ds.x, "y": ds.y})
