"""Kernel bundle served to workers: deterministic zip of proofnet_kernels/core (ARCH 7.2)."""

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import proofnet_kernels.core as core_pkg

PYODIDE_VERSION = "314.0.7"
NUMPY_VERSION = "2.4.6"
KERNEL_BUNDLE_VERSION = "1"

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class KernelBundle:
    version: str
    data: bytes
    sha256: str


def build_bundle() -> KernelBundle:
    """Deterministic: sorted paths, fixed timestamps, so the SHA-256 is stable across builds."""
    core_dir = Path(core_pkg.__file__ or "").parent
    files = {"proofnet_kernels/__init__.py": b""}
    for f in sorted(core_dir.glob("*.py")):
        files[f"proofnet_kernels/core/{f.name}"] = f.read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    data = buf.getvalue()
    return KernelBundle(KERNEL_BUNDLE_VERSION, data, hashlib.sha256(data).hexdigest())
