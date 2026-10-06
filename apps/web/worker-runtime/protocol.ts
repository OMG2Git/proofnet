/** Messages between the controller (main thread) and compute.worker.ts (Web Worker). */

export type WorkerInit = {
  type: "init";
  apiBase: string;
  deviceToken: string;
  pyodideVersion: string;
  kernelBundleVersion: string;
  kernelBundleSha256: string;
};

export type ToWorker = WorkerInit | { type: "bench" };

export type RuntimeFingerprint = {
  kind: "pyodide";
  python: string;
  pyodide: string;
  numpy: string;
  bundle: string;
};

export type BenchResult = {
  bench_version: string;
  score_cells_per_sec: number;
  median_seconds: number;
};

export type FromWorker =
  | { type: "progress"; stage: string; detail?: string }
  | { type: "ready"; runtime: RuntimeFingerprint }
  | { type: "bench"; result: BenchResult }
  | { type: "error"; message: string };
