/**
 * Main-thread controller: owns the device token, session, heartbeat loop, wake lock and UI state.
 * It keeps heartbeating while Python computes because computation runs in a Web Worker.
 * Dispatch (run/cancel directives) is added in P4.
 */
import { ApiRequestError, deviceApi } from "@/lib/api/client";
import { collectCapabilities, readBattery } from "./capabilities";
import type { BenchResult, FromWorker, RuntimeFingerprint, ToWorker } from "./protocol";

export type ControllerState = "stopped" | "loading" | "benchmarking" | "idle" | "busy" | "error";

export type ControllerSnapshot = {
  state: ControllerState;
  stage: string;
  score: number | null;
  runtime: RuntimeFingerprint | null;
  sessionId: string | null;
  lastHeartbeatAt: number | null;
  heartbeats: number;
  wakeLock: "active" | "unsupported" | "released" | "denied";
  connection: "ok" | "retrying";
  timings: { runtimeLoadMs: number | null; benchMs: number | null };
};

type Listener = (snap: ControllerSnapshot, log: string | null) => void;

type WakeLockSentinelLike = { release: () => Promise<void>; addEventListener: (e: string, f: () => void) => void };

export class WorkerController {
  private worker: Worker | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private wake: WakeLockSentinelLike | null = null;
  private stopped = true;
  private backoffMs = 1000;
  private t0 = 0;
  private snap: ControllerSnapshot = {
    state: "stopped",
    stage: "",
    score: null,
    runtime: null,
    sessionId: null,
    lastHeartbeatAt: null,
    heartbeats: 0,
    wakeLock: "unsupported",
    connection: "ok",
    timings: { runtimeLoadMs: null, benchMs: null },
  };
  private readonly api: ReturnType<typeof deviceApi>;
  private onVisibility = () => {
    if (!this.stopped && document.visibilityState === "visible") void this.requestWakeLock();
  };

  constructor(
    private readonly apiBase: string,
    private readonly deviceToken: string,
    private readonly listener: Listener,
  ) {
    this.api = deviceApi(deviceToken);
  }

  private emit(patch: Partial<ControllerSnapshot> = {}, log: string | null = null) {
    this.snap = { ...this.snap, ...patch };
    this.listener(this.snap, log);
  }

  async start(): Promise<void> {
    if (!this.stopped) return;
    this.stopped = false;
    this.backoffMs = 1000;
    document.addEventListener("visibilitychange", this.onVisibility);
    await this.requestWakeLock();
    this.emit({ state: "loading", stage: "Fetching runtime manifest" }, "Starting worker");
    try {
      const manifest = await this.api.manifest();
      this.t0 = performance.now();
      this.worker = new Worker("/compute.worker.js", { type: "module" });
      this.worker.onmessage = (ev: MessageEvent<FromWorker>) => void this.onWorker(ev.data, manifest);
      this.worker.onerror = (ev) => this.fail(`Worker crashed: ${ev.message}`);
      const init: ToWorker = {
        type: "init",
        apiBase: this.apiBase,
        deviceToken: this.deviceToken,
        pyodideVersion: manifest.pyodide_version,
        kernelBundleVersion: manifest.kernel_bundle_version,
        kernelBundleSha256: manifest.kernel_bundle_sha256,
      };
      this.worker.postMessage(init);
    } catch (e) {
      this.fail(e instanceof Error ? e.message : String(e));
    }
  }

  private async onWorker(msg: FromWorker, manifest: { kernel_bundle_version: string }) {
    if (this.stopped) return;
    switch (msg.type) {
      case "progress":
        this.emit({ stage: msg.detail ?? msg.stage }, msg.detail ?? msg.stage);
        break;
      case "ready": {
        const ms = Math.round(performance.now() - this.t0);
        this.emit(
          {
            runtime: msg.runtime,
            state: "benchmarking",
            stage: "Benchmarking",
            timings: { ...this.snap.timings, runtimeLoadMs: ms },
          },
          `Runtime ready in ${ms} ms (Pyodide ${msg.runtime.pyodide}, NumPy ${msg.runtime.numpy})`,
        );
        const t = performance.now();
        this.benchStart = t;
        this.worker?.postMessage({ type: "bench" } satisfies ToWorker);
        break;
      }
      case "bench":
        await this.onBench(msg.result, manifest.kernel_bundle_version);
        break;
      case "error":
        this.fail(msg.message);
        break;
    }
  }

  private benchStart = 0;

  private async onBench(result: BenchResult, bundle: string) {
    const benchMs = Math.round(performance.now() - this.benchStart);
    this.emit(
      { score: result.score_cells_per_sec, timings: { ...this.snap.timings, benchMs } },
      `Benchmark: ${(result.score_cells_per_sec / 1e6).toFixed(2)} M cells/s`,
    );
    const runtime = this.snap.runtime;
    if (!runtime) return this.fail("runtime fingerprint missing");
    try {
      const caps = await collectCapabilities();
      const s = await this.api.session({
        runtime: { ...runtime, bundle },
        benchmark: {
          score_cells_per_sec: result.score_cells_per_sec,
          bench_version: result.bench_version,
        },
        capabilities: caps,
      });
      this.emit({ sessionId: s.session_id, state: "idle", stage: "Ready" }, `Session ${s.session_id}`);
      this.scheduleHeartbeat(0);
    } catch (e) {
      this.fail(e instanceof Error ? e.message : String(e));
    }
  }

  private scheduleHeartbeat(delayMs: number) {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => void this.heartbeat(), delayMs);
  }

  private async heartbeat() {
    if (this.stopped || !this.snap.sessionId) return;
    try {
      const battery = await readBattery();
      const res = await this.api.heartbeat({
        session_id: this.snap.sessionId,
        state: this.snap.state === "busy" ? "busy" : "idle",
        battery: battery?.level,
      });
      this.backoffMs = 1000;
      this.emit({
        lastHeartbeatAt: Date.now(),
        heartbeats: this.snap.heartbeats + 1,
        connection: "ok",
      });
      for (const d of res.directives) {
        if (d.type === "reregister") {
          this.emit({}, "Server asked to start a new session");
          this.emit({ sessionId: null, state: "benchmarking" });
          this.worker?.postMessage({ type: "bench" } satisfies ToWorker);
          return;
        }
        this.emit({}, `Directive "${d.type}" ignored (dispatch arrives in P4)`);
      }
      this.scheduleHeartbeat(res.next_heartbeat_ms);
    } catch (e) {
      const wait = this.backoffMs;
      this.backoffMs = Math.min(this.backoffMs * 2, 30000);
      this.emit(
        { connection: "retrying" },
        `Heartbeat failed (${e instanceof ApiRequestError ? e.message : "network"}); retry in ${wait} ms`,
      );
      this.scheduleHeartbeat(wait);
    }
  }

  private async requestWakeLock() {
    const nav = navigator as Navigator & {
      wakeLock?: { request: (t: "screen") => Promise<WakeLockSentinelLike> };
    };
    if (!nav.wakeLock) {
      this.emit({ wakeLock: "unsupported" }, "Screen wake lock unsupported: keep the screen on manually");
      return;
    }
    try {
      this.wake = await nav.wakeLock.request("screen");
      this.wake.addEventListener("release", () => {
        if (!this.stopped) this.emit({ wakeLock: "released" });
      });
      this.emit({ wakeLock: "active" });
    } catch {
      this.emit({ wakeLock: "denied" }, "Wake lock denied: keep the screen on manually");
    }
  }

  private fail(message: string) {
    this.emit({ state: "error", stage: message }, `Error: ${message}`);
    this.teardown();
  }

  private teardown() {
    this.worker?.terminate();
    this.worker = null;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    void this.wake?.release().catch(() => undefined);
    this.wake = null;
    document.removeEventListener("visibilitychange", this.onVisibility);
  }

  stop(): void {
    this.stopped = true;
    this.teardown();
    this.emit({ state: "stopped", stage: "", sessionId: null, wakeLock: "released" }, "Stopped");
  }
}
