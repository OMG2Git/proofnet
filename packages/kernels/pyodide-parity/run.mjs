// Runs proofnet_kernels.core inside Pyodide (under Node) on chunks written by the pytest parity test.
// usage: node run.mjs <workdir>   (reads <workdir>/chunks.npz; writes <workdir>/pyodide_out.json)
import { loadPyodide } from "pyodide";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const work = process.argv[2];
const coreDir = join(dirname(fileURLToPath(import.meta.url)), "..", "proofnet_kernels", "core");

const py = await loadPyodide();
await py.loadPackage("numpy");

py.FS.mkdirTree("/pk/proofnet_kernels/core");
py.FS.writeFile("/pk/proofnet_kernels/__init__.py", "");
for (const f of readdirSync(coreDir).filter((n) => n.endsWith(".py"))) {
  py.FS.writeFile(`/pk/proofnet_kernels/core/${f}`, readFileSync(join(coreDir, f)));
}
py.FS.writeFile("/chunks.npz", readFileSync(join(work, "chunks.npz")));

const out = py.runPython(`
import sys, json, platform
sys.path.insert(0, "/pk")
import numpy as np
from proofnet_kernels.core import gaussian_nb, linear_ridge, bench, cnn
z = np.load("/chunks.npz", allow_pickle=False)
res = {
  "gnb": gaussian_nb.map(z["X"], z["y_cls"], int(z["n_classes"])),
  "ridge": linear_ridge.map(z["X"], z["y_reg"]),
  "runtime": {"python": platform.python_version(), "numpy": np.__version__},
  "cnn": cnn.map(z["X_img"], z["y_img"], z["w_img"], json.loads(bytes(z["arch_json"]).decode())),
  "cnn_ms": None,
  "bench": bench.run(),
}
import time as _t
_t0 = _t.perf_counter()
cnn.map(z["X_img"], z["y_img"], z["w_img"], json.loads(bytes(z["arch_json"]).decode()))
res["cnn_ms"] = (_t.perf_counter() - _t0) * 1000.0
json.dumps(res, allow_nan=False)
`);
writeFileSync(join(work, "pyodide_out.json"), out);
console.log("ok");
