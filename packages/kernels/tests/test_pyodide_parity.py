"""Pyodide parity: core map kernels run in Pyodide (Node) must match CPython (P1 gate)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from proofnet_kernels.core import gaussian_nb, linear_ridge
from proofnet_kernels.server import gaussian_nb as srv_gnb
from proofnet_kernels.server import linear_ridge as srv_ridge

PARITY_DIR = Path(__file__).resolve().parents[1] / "pyodide-parity"
LOG = PARITY_DIR / "last_discrepancy.json"


@pytest.mark.skipif(
    shutil.which("node") is None or not (PARITY_DIR / "node_modules").exists(),
    reason="needs node and `npm ci` in packages/kernels/pyodide-parity",
)
def test_pyodide_matches_cpython(tmp_path: Path) -> None:
    rng = np.random.default_rng(2024)
    n, d, c = 20_000, 12, 4
    y_cls = rng.integers(0, c, size=n)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 4, size=d) + y_cls[:, None] * 0.5 + 1e3
    y_reg = x @ rng.normal(size=d) + rng.normal(scale=0.2, size=n)
    np.savez(tmp_path / "chunks.npz", X=x, y_cls=y_cls, y_reg=y_reg, n_classes=c)

    subprocess.run(
        ["node", "run.mjs", str(tmp_path)], cwd=PARITY_DIR, check=True, timeout=600, text=True
    )
    out = json.loads((tmp_path / "pyodide_out.json").read_text())

    cmp_gnb = srv_gnb.compare(out["gnb"], gaussian_nb.map(x, y_cls, c))
    cmp_ridge = srv_ridge.compare(out["ridge"], linear_ridge.map(x, y_reg))
    assert cmp_gnb["within_tolerance"], cmp_gnb["max_rel_diff"]
    assert cmp_ridge["within_tolerance"], cmp_ridge["max_rel_diff"]
    # Integer counts must match exactly.
    assert out["gnb"]["classes"]["n"] == gaussian_nb.map(x, y_cls, c)["classes"]["n"]

    LOG.write_text(
        json.dumps(
            {
                "gaussian_nb_max_rel_diff": cmp_gnb["max_rel_diff"],
                "linear_ridge_max_rel_diff": cmp_ridge["max_rel_diff"],
                "pyodide_runtime": out["runtime"],
                "pyodide_bench_cells_per_sec": out["bench"]["score_cells_per_sec"],
            },
            indent=2,
        )
        + "\n"
    )
