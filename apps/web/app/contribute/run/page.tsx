"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { getApiBase, useStoredDevice } from "@/lib/api/client";
import { WorkerController, type ControllerSnapshot } from "@/worker-runtime/controller";

const INITIAL: ControllerSnapshot = {
  state: "stopped",
  stage: "",
  score: null,
  runtime: null,
  sessionId: null,
  lastHeartbeatAt: null,
  heartbeats: 0,
  currentAssignmentId: null,
  completed: 0,
  wakeLock: "unsupported",
  connection: "ok",
  timings: { runtimeLoadMs: null, benchMs: null },
};

function Console() {
  const device = useStoredDevice();
  const [snap, setSnap] = useState<ControllerSnapshot>(INITIAL);
  const [log, setLog] = useState<string[]>([]);
  const [now, setNow] = useState(0);
  const ctl = useRef<WorkerController | null>(null);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => {
      clearInterval(t);
      ctl.current?.stop();
    };
  }, []);

  function start() {
    if (!device) return;
    setLog([]);
    ctl.current = new WorkerController(getApiBase(), device.deviceToken, (s, line) => {
      setSnap(s);
      if (line) setLog((l) => [...l.slice(-199), `${new Date().toLocaleTimeString()}  ${line}`]);
    });
    void ctl.current.start();
  }

  function stop() {
    ctl.current?.stop();
  }

  if (!device) {
    return (
      <section>
        <h1>Worker console</h1>
        <p>
          This browser is not registered yet. <Link href="/contribute">Register this device</Link> first.
        </p>
      </section>
    );
  }
  const running = snap.state !== "stopped" && snap.state !== "error";
  const hbAge =
    snap.lastHeartbeatAt && now ? Math.max(0, Math.round((now - snap.lastHeartbeatAt) / 1000)) : null;
  return (
    <section>
      <h1>Worker console</h1>
      <div className="card">
        <p>
          <strong>{device.name}</strong> <code>{device.deviceId}</code>
        </p>
        <p className="warn">
          Keep this tab in the foreground with the screen on (and the phone charging). A backgrounded
          tab is treated as offline.
        </p>
        <div className="row">
          <button data-testid="start" onClick={start} disabled={running}>
            Start contributing
          </button>
          <button data-testid="stop" onClick={stop} disabled={!running}>
            Stop
          </button>
        </div>
        <table>
          <tbody>
            <tr>
              <th>State</th>
              <td data-testid="state">
                <span className={`badge ${snap.state}`}>{snap.state}</span> {snap.stage}
              </td>
            </tr>
            <tr>
              <th>Runtime</th>
              <td>
                {snap.runtime
                  ? `Pyodide ${snap.runtime.pyodide} · Python ${snap.runtime.python} · NumPy ${snap.runtime.numpy}`
                  : "—"}
                {snap.timings.runtimeLoadMs !== null && ` (loaded in ${snap.timings.runtimeLoadMs} ms)`}
              </td>
            </tr>
            <tr>
              <th>Benchmark</th>
              <td data-testid="score">
                {snap.score !== null
                  ? `${(snap.score / 1e6).toFixed(2)} M cells/s (${snap.timings.benchMs} ms for bench_v1)`
                  : "—"}
              </td>
            </tr>
            <tr>
              <th>Session</th>
              <td>{snap.sessionId ?? "—"}</td>
            </tr>
            <tr>
              <th>Heartbeat</th>
              <td data-testid="heartbeat">
                {snap.heartbeats} sent{hbAge !== null && `, last ${hbAge}s ago`} · connection {snap.connection}
              </td>
            </tr>
            <tr>
              <th>Work</th>
              <td data-testid="work">
                {snap.currentAssignmentId ? `running ${snap.currentAssignmentId}` : "none"} · {snap.completed} completed
              </td>
            </tr>
            <tr>
              <th>Screen wake lock</th>
              <td>{snap.wakeLock}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h1>Log</h1>
      <pre className="log" data-testid="log">
        {log.join("\n") || "—"}
      </pre>
    </section>
  );
}

export default function RunPage() {
  return (
    <AuthGate>
      <Console />
    </AuthGate>
  );
}
