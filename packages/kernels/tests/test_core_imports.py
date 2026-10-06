"""core/ may import only NumPy + stdlib (and its own siblings): it ships to Pyodide workers."""

import ast
import sys
from pathlib import Path

import proofnet_kernels.core as core_pkg

ALLOWED = set(sys.stdlib_module_names) | {"numpy"}


def test_core_imports_only_numpy_and_stdlib() -> None:
    core_dir = Path(core_pkg.__file__).parent
    files = sorted(core_dir.glob("*.py"))
    assert files
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:  # relative import inside core/
                    continue
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            for m in mods:
                assert m in ALLOWED, f"{f.name} imports forbidden module '{m}'"


def test_bench_runs() -> None:
    from proofnet_kernels.core import bench

    r = bench.run()
    assert r["score_cells_per_sec"] > 0 and r["bench_version"] == "bench_v1"
