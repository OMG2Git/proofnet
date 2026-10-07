/** Messages between the controller (main thread) and compute.worker.ts (Web Worker). */

export type WorkerInit = {
  type: "init";
  apiBase: string;
  deviceToken: string;
  pyodideVersion: string;
  kernelBundleVersion: string;
  kernelBundleSha256: string;
};

export type RunMessage = {
  type: "run";
  kernel: string;
  params: Record<string, unknown>;
  /** Chunk input .npz bytes (X, y); transferred, never pickled. */
  input: ArrayBuffer;
};

export type ToWorker = WorkerInit | { type: "bench" } | RunMessage;

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
  | { type: "result"; payloadJson: string; payloadSha256: string; computeMs: number }
  | { type: "error"; message: string };
