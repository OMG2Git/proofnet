/// <reference lib="webworker" />
/**
 * Compute plane (browser): loads pinned Pyodide + NumPy, downloads the ProofNet kernel bundle,
 * verifies its SHA-256, and runs ProofNet-authored kernels. Executes no user code.
 */
import type { FromWorker, ToWorker, RuntimeFingerprint } from "./protocol";

type PyodideLike = {
  FS: { writeFile: (path: string, data: Uint8Array) => void };
  globals: { set: (name: string, value: unknown) => void };
  loadPackage: (name: string) => Promise<void>;
  unpackArchive: (data: ArrayBuffer, format: string, opts?: { extractDir?: string }) => void;
  runPython: (code: string) => unknown;
  version: string;
};

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let pyodide: PyodideLike | null = null;
let fingerprint: RuntimeFingerprint | null = null;

function post(msg: FromWorker) {
  ctx.postMessage(msg);
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function init(m: Extract<ToWorker, { type: "init" }>) {
  post({ type: "progress", stage: "runtime", detail: `Loading Pyodide ${m.pyodideVersion}` });
  const base = `https://cdn.jsdelivr.net/pyodide/v${m.pyodideVersion}/full/`;
  // Module worker (compiled by esbuild to public/compute.worker.js, not by the Next bundler):
  // a plain dynamic import of the pinned Pyodide ES module from the CDN.
  const mod = (await import(`${base}pyodide.mjs`)) as {
    loadPyodide: (o: { indexURL: string }) => Promise<PyodideLike>;
  };
  const py = await mod.loadPyodide({ indexURL: base });
  post({ type: "progress", stage: "numpy", detail: "Loading NumPy" });
  await py.loadPackage("numpy");

  post({ type: "progress", stage: "kernels", detail: "Downloading kernel bundle" });
  const res = await fetch(`${m.apiBase}/runtime/kernels/${m.kernelBundleVersion}`, {
    headers: { Authorization: `Bearer ${m.deviceToken}` },
  });
  if (!res.ok) throw new Error(`kernel bundle download failed (${res.status})`);
  const bundle = await res.arrayBuffer();
  const digest = await sha256Hex(bundle);
  if (digest !== m.kernelBundleSha256) {
    throw new Error("kernel bundle SHA-256 mismatch; refusing to run");
  }
  post({ type: "progress", stage: "kernels", detail: "Kernel bundle verified (SHA-256 ok)" });
  py.unpackArchive(bundle, "zip", { extractDir: "/pk" });
  py.runPython("import sys\nif '/pk' not in sys.path: sys.path.insert(0, '/pk')");
  const info = JSON.parse(
    py.runPython(
      "import json, platform, numpy\nfrom proofnet_kernels.core import gaussian_nb, bench\n" +
        "json.dumps({'python': platform.python_version(), 'numpy': numpy.__version__})",
    ) as string,
  ) as { python: string; numpy: string };
  pyodide = py;
  fingerprint = {
    kind: "pyodide",
    python: info.python,
    pyodide: py.version,
    numpy: info.numpy,
    bundle: m.kernelBundleVersion,
  };
  post({ type: "ready", runtime: fingerprint });
}

function bench() {
  if (!pyodide) throw new Error("runtime not initialised");
  post({ type: "progress", stage: "benchmark", detail: "Running bench_v1 (real kernel code)" });
  const out = pyodide.runPython(
    "import json\nfrom proofnet_kernels.core import bench\njson.dumps(bench.run())",
  ) as string;
  const r = JSON.parse(out) as {
    bench_version: string;
    score_cells_per_sec: number;
    median_seconds: number;
  };
  post({ type: "bench", result: r });
}

// Fixed, ProofNet-authored program: loads the chunk, runs the selected core kernel's map.
// No user-supplied code is ever executed.
const RUN_PROGRAM = `
import json, time
import numpy as np
from proofnet_kernels.core import gaussian_nb, linear_ridge
from proofnet_kernels.core.serialize import canonical_json, payload_sha256
z = np.load("/input.npz", allow_pickle=False)
X, y = z["X"], z["y"]
t = time.perf_counter()
if _pn_kernel == "gaussian_nb_train":
    payload = gaussian_nb.map(X, y, int(_pn_n_classes))
elif _pn_kernel == "linear_ridge_train":
    payload = linear_ridge.map(X, y)
else:
    raise ValueError("unknown kernel " + str(_pn_kernel))
compute_ms = (time.perf_counter() - t) * 1000.0
json.dumps({"payload_json": canonical_json(payload), "sha": payload_sha256(payload), "compute_ms": compute_ms})
`;

function run(m: Extract<ToWorker, { type: "run" }>) {
  if (!pyodide) throw new Error("runtime not initialised");
  pyodide.FS.writeFile("/input.npz", new Uint8Array(m.input));
  pyodide.globals.set("_pn_kernel", m.kernel);
  pyodide.globals.set("_pn_n_classes", Number(m.params["n_classes"] ?? 0));
  const out = JSON.parse(pyodide.runPython(RUN_PROGRAM) as string) as {
    payload_json: string;
    sha: string;
    compute_ms: number;
  };
  post({
    type: "result",
    payloadJson: out.payload_json,
    payloadSha256: out.sha,
    computeMs: out.compute_ms,
  });
}

ctx.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  const task =
    m.type === "init"
      ? init(m)
      : m.type === "run"
        ? Promise.resolve().then(() => run(m))
        : Promise.resolve().then(bench);
  task.catch((e: unknown) =>
    post({ type: "error", message: e instanceof Error ? e.message : String(e) }),
  );
};
