"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import AuthGate from "@/components/AuthGate";
import {
  api,
  getApiBase,
  useStoredDevice,
  type DeviceTrust,
  type NetworkDevice,
  type RewardsOut,
  type VerificationRecord,
} from "@/lib/api/client";
import { PHONE } from "@/lib/network/sprite-data";
import { formatAge } from "@/lib/network/logic";
import { WorkerController, type ControllerSnapshot, type ControllerState } from "@/worker-runtime/controller";

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

const STATE_COLOR: Record<ControllerState, string> = {
  stopped: "#66748c",
  loading: "#22d3ee",
  benchmarking: "#22d3ee",
  idle: "#34d399",
  busy: "#a78bfa",
  error: "#f87171",
};
const STATE_WORD: Record<ControllerState, string> = {
  stopped: "Stopped",
  loading: "Starting…",
  benchmarking: "Benchmarking…",
  idle: "Ready",
  busy: "Working",
  error: "Error",
};

/** The phone sprite from the network view, drawn as crisp SVG in the current state colour. */
function PixelPhone({ color }: { color: string }) {
  const pal: Record<string, string> = { k: "#05080e", g: "#2c3956", G: "#41527a", h: "#6b7fab", w: "#dbe4f3", S: color, s: color, a: color };
  return (
    <svg className="px" viewBox={`0 0 ${PHONE[0]!.length} ${PHONE.length}`} shapeRendering="crispEdges" role="img" aria-label="Device status icon">
      {PHONE.flatMap((row, y) =>
        [...row].map((ch, x) => (pal[ch] ? <rect key={`${x}-${y}`} x={x} y={y} width="1" height="1" fill={pal[ch]} opacity={ch === "s" ? 0.6 : 1} /> : null)),
      )}
    </svg>
  );
}

function subscribeOnline(cb: () => void): () => void {
  window.addEventListener("online", cb);
  window.addEventListener("offline", cb);
  return () => {
    window.removeEventListener("online", cb);
    window.removeEventListener("offline", cb);
  };
}

function Console() {
  const device = useStoredDevice();
  const [snap, setSnap] = useState<ControllerSnapshot>(INITIAL);
  const [log, setLog] = useState<string[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [attack, setAttack] = useState("none");
  const [demoOpen, setDemoOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const online = useSyncExternalStore(subscribeOnline, () => navigator.onLine, () => true);
  const [hiddenFor, setHiddenFor] = useState<number | null>(null);
  const [mine, setMine] = useState<NetworkDevice | null>(null);
  const [trust, setTrust] = useState<DeviceTrust | null>(null);
  const [rewards, setRewards] = useState<RewardsOut | null>(null);
  const [records, setRecords] = useState<VerificationRecord[]>([]);
  const ctl = useRef<WorkerController | null>(null);
  const hiddenAt = useRef<number | null>(null);

  const running = snap.state !== "stopped" && snap.state !== "error";

  // 1 s clock, only while contributing (a phone's battery matters more than a smooth counter).
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  useEffect(() => {
    const vis = () => {
      if (document.hidden) hiddenAt.current = Date.now();
      else if (hiddenAt.current) {
        setHiddenFor(Math.round((Date.now() - hiddenAt.current) / 1000));
        hiddenAt.current = null;
      }
    };
    document.addEventListener("visibilitychange", vis);
    return () => {
      document.removeEventListener("visibilitychange", vis);
      ctl.current?.stop();
    };
  }, []);

  // What the backend says about this device (assignment, audit outcomes, rewards): slow polling,
  // paused while the tab is hidden.
  const deviceId = device?.deviceId ?? null;
  const refresh = useCallback(async () => {
    if (!deviceId || document.hidden) return;
    const [net, tr, rw, rec] = await Promise.allSettled([
      api.networkSummary(),
      api.trustDevice(deviceId),
      api.myRewards(),
      api.trustRecentRecords(100),
    ]);
    if (net.status === "fulfilled") setMine(net.value.devices.find((d) => d.id === deviceId) ?? null);
    if (tr.status === "fulfilled") setTrust(tr.value);
    if (rw.status === "fulfilled") setRewards(rw.value);
    if (rec.status === "fulfilled") setRecords(rec.value.filter((r) => r.device_id === deviceId).slice(0, 3));
  }, [deviceId]);
  useEffect(() => {
    if (!deviceId) return;
    const first = setTimeout(() => void refresh(), 0);
    const t = setInterval(() => void refresh(), running ? 4000 : 12000);
    return () => {
      clearTimeout(first);
      clearInterval(t);
    };
  }, [deviceId, running, refresh]);

  function start() {
    if (!device) return;
    setLog([]);
    setHiddenFor(null);
    ctl.current = new WorkerController(getApiBase(), device.deviceToken, (s, line) => {
      setSnap(s);
      if (line) setLog((l) => [...l.slice(-99), `${new Date().toLocaleTimeString()}  ${line}`]);
    });
    ctl.current.demoAttack = attack;
    void ctl.current.start();
  }
  function stop() {
    ctl.current?.stop();
  }
  function pickAttack(mode: string) {
    setAttack(mode);
    if (ctl.current) ctl.current.demoAttack = mode;
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

  const hbAge = snap.lastHeartbeatAt ? Math.max(0, Math.round((now - snap.lastHeartbeatAt) / 1000)) : null;
  const color = STATE_COLOR[snap.state];
  const bal = rewards?.per_device[device.deviceId];
  const steps = [
    { label: "Registered", done: true, now: false },
    { label: "Runtime", done: snap.runtime !== null, now: snap.state === "loading" },
    { label: "Benchmark", done: snap.score !== null, now: snap.state === "benchmarking" },
    { label: "Connected", done: snap.sessionId !== null, now: false },
    { label: "Working", done: snap.completed > 0, now: snap.state === "busy" },
  ];
  const assignment = snap.currentAssignmentId ?? mine?.current_assignment_id ?? null;

  return (
    <section style={{ maxWidth: 640 }}>
      <h1>Worker console</h1>

      <div className="phone-status">
        <div className="phone-hero" data-testid="hero" style={{ borderColor: `color-mix(in srgb, ${color} 45%, transparent)` }}>
          <PixelPhone color={color} />
          <div style={{ minWidth: 0 }}>
            <div className="big" style={{ color }} data-testid="state" role="status" aria-live="polite">
              {STATE_WORD[snap.state]}
            </div>
            <div className="muted" style={{ overflowWrap: "anywhere" }}>
              {snap.stage || (running ? "—" : "Tap Start to begin contributing")}
            </div>
            <div className="muted mono" style={{ fontSize: 12, overflowWrap: "anywhere" }}>
              {device.name} · {device.deviceId}
            </div>
          </div>
        </div>

        {attack !== "none" && (
          <p className="error-box" role="alert" data-testid="cheat-banner" style={{ marginTop: 0 }}>
            <strong>This device is cheating on purpose ({attack}).</strong> The server is not told: it audits, rejects and builds evidence. Watch
            the verdicts below and in the log.
          </p>
        )}
        {!online && (
          <p className="error-box" role="alert">
            This phone is offline. Work in progress may be lost; the server marks a device offline after 20 s without a heartbeat. It reconnects
            on its own when the network returns (same device, no duplicate registration).
          </p>
        )}
        {hiddenFor !== null && hiddenFor > 15 && running && (
          <p className="sim-banner" role="status">
            This tab was in the background for {hiddenFor} s. Browsers pause background tabs, so the server may have treated this device as offline.
            Keep it in the foreground.
          </p>
        )}
        {snap.connection === "retrying" && online && (
          <p className="warn" role="status">
            Cannot reach the server; retrying with back-off…
          </p>
        )}

        <div className="row" style={{ marginTop: 0 }}>
          {running ? (
            <button className="btn-xl ghost" data-testid="stop" onClick={stop}>
              Stop contributing
            </button>
          ) : (
            <button className="btn-xl" data-testid="start" onClick={start}>
              Start contributing
            </button>
          )}
        </div>

        <ol className="steps" aria-label="Startup progress">
          {steps.map((s) => (
            <li key={s.label} className={s.done ? "done" : s.now ? "now" : ""}>
              <b>{s.label}</b>
              {s.done ? "✓ done" : s.now ? "in progress…" : "—"}
            </li>
          ))}
        </ol>
        {(snap.state === "loading" || snap.state === "benchmarking") && (
          <div className="progress" role="progressbar" aria-label={snap.stage || "Starting"} aria-busy="true">
            <div style={{ width: snap.state === "loading" ? "35%" : "75%" }} />
          </div>
        )}

        <div className="card" style={{ marginTop: 0 }}>
          <h2>Benchmark</h2>
          <p data-testid="score" style={{ fontSize: 18, margin: "4px 0" }} className="mono">
            {snap.score !== null ? `${(snap.score / 1e6).toFixed(2)} M cells/s` : "not measured yet"}
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {snap.timings.benchMs !== null ? `bench_v1 took ${snap.timings.benchMs} ms. ` : ""}
            The server splits work in proportion to this measured speed and re-measures it each session.
            {snap.runtime && ` Runtime: Pyodide ${snap.runtime.pyodide}, NumPy ${snap.runtime.numpy}`}
            {snap.timings.runtimeLoadMs !== null && ` (loaded in ${(snap.timings.runtimeLoadMs / 1000).toFixed(1)} s).`}
          </p>
        </div>

        <div className="card" style={{ marginTop: 0 }}>
          <h2>Current work</h2>
          {assignment ? (
            <dl className="kv" data-testid="work">
              <dt>Task</dt>
              <dd>{mine?.current_task_name ?? "—"}</dd>
              <dt>Chunk</dt>
              <dd>
                {mine?.current_chunk_index ?? "—"}
                {mine?.current_rows ? ` · ${mine.current_rows.toLocaleString()} rows` : ""}
              </dd>
              <dt>Assignment</dt>
              <dd className="mono">{assignment}</dd>
            </dl>
          ) : (
            <p className="muted" data-testid="work" style={{ margin: 0 }}>
              No chunk right now. {running ? "Waiting for the server to assign one." : ""}
            </p>
          )}
          <p className="muted" style={{ marginBottom: 0 }}>
            {snap.completed} chunk{snap.completed === 1 ? "" : "s"} completed this session.
          </p>
        </div>

        <div className="card" style={{ marginTop: 0 }}>
          <h2>Connection</h2>
          <dl className="kv" data-testid="heartbeat">
            <dt>Last heartbeat</dt>
            <dd>{hbAge !== null ? `${hbAge} s ago` : "—"}</dd>
            <dt>Sent</dt>
            <dd>{snap.heartbeats}</dd>
            <dt>Link</dt>
            <dd>
              {!online ? "offline" : snap.connection === "ok" ? "OK (last request succeeded)" : "retrying"}
              <span className="muted"> · quality is not measured</span>
            </dd>
            <dt>Server view</dt>
            <dd>{mine ? `${mine.status}, seen ${formatAge(mine.last_seen_age_seconds ?? null)}` : "—"}</dd>
          </dl>
        </div>

        <div className="card" style={{ marginTop: 0 }}>
          <h2>Verification & rewards</h2>
          <dl className="kv">
            <dt>Trust</dt>
            <dd>
              {trust ? (
                <>
                  <span className={`badge ${trust.status}`}>{trust.status}</span> {trust.trust.toFixed(2)}
                </>
              ) : (
                "—"
              )}
            </dd>
            <dt>Audits</dt>
            <dd>{trust ? `${trust.audits} of ${trust.results_seen} results audited` : "—"}</dd>
            <dt>Latest</dt>
            <dd>
              {records.length === 0
                ? "no audit yet"
                : records.map((r) => (
                    <span key={r.id} className={`badge ${r.decision === "verified" ? "verified" : r.decision === "rejected" ? "rejected" : "pending"}`} style={{ marginRight: 6 }}>
                      {r.decision}
                    </span>
                  ))}
            </dd>
            <dt>Credits</dt>
            <dd className="mono">{bal ? `${bal.confirmed.toFixed(3)} confirmed · ${bal.pending.toFixed(3)} pending` : "0 (no entries yet)"}</dd>
          </dl>
          {trust?.quarantine && <p className="error">This device is quarantined and receives no work.</p>}
        </div>

        <div className="card" style={{ marginTop: 0 }}>
          <h2>Keep the screen awake</h2>
          <p style={{ margin: "4px 0" }}>
            Wake lock:{" "}
            <span className={`badge ${snap.wakeLock === "active" ? "ready" : snap.wakeLock === "unsupported" ? "disabled" : "pending"}`} data-testid="wakelock">
              {snap.wakeLock}
            </span>
          </p>
          <p className="muted" style={{ marginBottom: 0 }}>
            {snap.wakeLock === "unsupported"
              ? "This browser has no Screen Wake Lock API: turn the screen timeout up manually."
              : snap.wakeLock === "denied"
                ? "The browser refused the wake lock (battery saver?). Turn off battery saver or raise the screen timeout."
                : "ProofNet asks the browser to keep the screen on while you contribute."}{" "}
            Browsers cannot guarantee background execution on Android: if the tab is hidden or the screen locks, the device goes offline and its work is
            reassigned. Charging is recommended.
          </p>
        </div>
      </div>

      <details
        style={{ marginTop: 16 }}
        open={demoOpen || attack !== "none"}
        onToggle={(e) => setDemoOpen((e.currentTarget as HTMLDetailsElement).open)}
      >
        <summary className="muted">Demo: misbehave (verification demo only)</summary>
        <div className="card">
          <p className="muted">
            When switched on, this device deliberately returns wrong (but well-formed) results so the audit layer can be shown rejecting them. The
            backend is not told. Leave it off for honest contributing.
          </p>
          <label>
            Simulate a cheating device
            <select data-testid="attack" value={attack} onChange={(e) => pickAttack(e.target.value)}>
              <option value="none">Off (honest)</option>
              <option value="scale">Inflate every statistic by 5%</option>
              <option value="bias">Shift statistics by 5%</option>
              <option value="sign_flip">Flip the sign of means / gradients</option>
              <option value="zero">Free-rider: return zeros</option>
              <option value="noise">Add 1% noise</option>
              <option value="random">Return random numbers</option>
              <option value="subtle">Subtle: 0.01% error</option>
            </select>
          </label>
          {attack !== "none" && <p className="error">This device is cheating on purpose ({attack}).</p>}
        </div>
      </details>

      <details
        style={{ marginTop: 12 }}
        open={logOpen || attack !== "none"}
        onToggle={(e) => setLogOpen((e.currentTarget as HTMLDetailsElement).open)}
      >
        <summary className="muted">Log ({log.length})</summary>
        <pre className="log" data-testid="log">
          {log.join("\n") || "—"}
        </pre>
      </details>
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
