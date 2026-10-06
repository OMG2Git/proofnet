// worker-runtime/compute.worker.ts
var ctx = self;
var pyodide = null;
var fingerprint = null;
function post(msg) {
  ctx.postMessage(msg);
}
async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function init(m) {
  post({ type: "progress", stage: "runtime", detail: `Loading Pyodide ${m.pyodideVersion}` });
  const base = `https://cdn.jsdelivr.net/pyodide/v${m.pyodideVersion}/full/`;
  const mod = await import(`${base}pyodide.mjs`);
  const py = await mod.loadPyodide({ indexURL: base });
  post({ type: "progress", stage: "numpy", detail: "Loading NumPy" });
  await py.loadPackage("numpy");
  post({ type: "progress", stage: "kernels", detail: "Downloading kernel bundle" });
  const res = await fetch(`${m.apiBase}/runtime/kernels/${m.kernelBundleVersion}`, {
    headers: { Authorization: `Bearer ${m.deviceToken}` }
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
      "import json, platform, numpy\nfrom proofnet_kernels.core import gaussian_nb, bench\njson.dumps({'python': platform.python_version(), 'numpy': numpy.__version__})"
    )
  );
  pyodide = py;
  fingerprint = {
    kind: "pyodide",
    python: info.python,
    pyodide: py.version,
    numpy: info.numpy,
    bundle: m.kernelBundleVersion
  };
  post({ type: "ready", runtime: fingerprint });
}
function bench() {
  if (!pyodide) throw new Error("runtime not initialised");
  post({ type: "progress", stage: "benchmark", detail: "Running bench_v1 (real kernel code)" });
  const out = pyodide.runPython(
    "import json\nfrom proofnet_kernels.core import bench\njson.dumps(bench.run())"
  );
  const r = JSON.parse(out);
  post({ type: "bench", result: r });
}
ctx.onmessage = (ev) => {
  const m = ev.data;
  const task = m.type === "init" ? init(m) : Promise.resolve().then(bench);
  task.catch(
    (e) => post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  );
};
