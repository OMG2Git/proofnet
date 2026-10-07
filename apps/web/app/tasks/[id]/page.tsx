"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type ArtifactOut, type EventOut, type TaskStatus } from "@/lib/api/client";

const STATES = ["queued", "running", "aggregating", "completed"] as const;

type Share = {
  device_id: string;
  device_name: string | null;
  score_cells_per_sec: number;
  weight: number;
  rows: number;
  n_chunks: number;
  est_seconds: number;
};
type PlanDoc = { explanation?: string; shares?: Share[] };

function fmtSec(v: number | undefined | null): string {
  if (v === undefined || v === null) return "—";
  return v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`;
}

/** Horizontal bars per device over the task's real time window (from assignment timestamps). */
function Lanes({ st }: { st: TaskStatus }) {
  const asgs = st.assignments.filter((a) => a.assigned_at);
  if (asgs.length === 0) return null;
  const chunkIndex = new Map(st.chunks.map((c) => [c.id, c.index]));
  const serverNow = new Date(st.server_time).getTime();
  const start = Math.min(...asgs.map((a) => new Date(a.assigned_at).getTime()));
  const end = Math.max(...asgs.map((a) => (a.finished_at ? new Date(a.finished_at).getTime() : serverNow)));
  const span = Math.max(end - start, 1);
  const lanes = new Map<string, TaskStatus["assignments"]>();
  for (const a of asgs) lanes.set(a.device_name ?? a.device_id, [...(lanes.get(a.device_name ?? a.device_id) ?? []), a]);
  return (
    <>
      <h1>Device lanes</h1>
      <div className="lanes" data-testid="lanes">
        {[...lanes.entries()].map(([name, list]) => (
          <div className="lane" key={name}>
            <div className="lane-name">{name}</div>
            <div className="lane-track">
              {list.map((a) => {
                const s0 = new Date(a.started_at ?? a.assigned_at).getTime();
                const e0 = a.finished_at ? new Date(a.finished_at).getTime() : serverNow;
                return (
                  <div
                    key={a.id}
                    className={`bar ${a.status}`}
                    style={{ left: `${((s0 - start) / span) * 100}%`, width: `${Math.max(((e0 - s0) / span) * 100, 1.5)}%` }}
                    title={`chunk ${chunkIndex.get(a.chunk_id)} · ${a.status} · ${Math.round(e0 - s0)} ms`}
                  >
                    c{chunkIndex.get(a.chunk_id)}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
        <div className="muted">window: {fmtSec(span / 1000)} (assignment start to finish, real timestamps)</div>
      </div>
    </>
  );
}

function fmtMs(v: number | undefined | null): string {
  return v === undefined || v === null ? "—" : `${Math.round(v)} ms`;
}

function Timeline({ status, history }: { status: string; history: TaskStatus["task"]["status_history"] }) {
  const reached = new Map(history.map((h) => [String(h["status"]), String(h["at"])]));
  const failed = status === "failed" || status === "cancelled";
  return (
    <ol className="timeline">
      {STATES.map((s) => (
        <li key={s} className={reached.has(s) ? (s === status ? "current" : "done") : "todo"}>
          <strong>{s}</strong>
          <span className="muted">{reached.has(s) ? new Date(reached.get(s) ?? "").toLocaleTimeString() : ""}</span>
        </li>
      ))}
      {failed && (
        <li className="failed">
          <strong>{status}</strong>
          <span className="muted">{reached.has(status) ? new Date(reached.get(status) ?? "").toLocaleTimeString() : ""}</span>
        </li>
      )}
    </ol>
  );
}

async function save(api_: typeof api, a: ArtifactOut) {
  const blob = await api_.downloadArtifact(a);
  const url = URL.createObjectURL(blob);
  const el = document.createElement("a");
  el.href = url;
  el.download = a.filename;
  el.click();
  URL.revokeObjectURL(url);
}

function Monitor() {
  const { id } = useParams<{ id: string }>();
  const [st, setSt] = useState<TaskStatus | null>(null);
  const [events, setEvents] = useState<EventOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let done = false;
    const tick = async () => {
      try {
        const s = await api.taskStatus(id);
        const ev = await api.taskEvents(id);
        if (!alive) return;
        setSt(s);
        setEvents(ev);
        setError(null);
        done = ["completed", "failed", "cancelled"].includes(s.task.status);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "error");
      }
    };
    void tick();
    const t = setInterval(() => {
      if (!done) void tick();
    }, 1500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [id]);

  if (!st) return <p className={error ? "error" : "muted"}>{error ?? "Loading…"}</p>;
  const t = st.task;
  const result = t.result as {
    metrics?: Record<string, unknown>;
    reference_check?: { passed: boolean; max_relative_difference: number; tolerance: number };
  } | null;
  const asgByChunk = new Map<string, TaskStatus["assignments"]>();
  for (const a of st.assignments) asgByChunk.set(a.chunk_id, [...(asgByChunk.get(a.chunk_id) ?? []), a]);
  const metricRows = Object.entries(result?.metrics ?? {}).filter(([, v]) => typeof v === "number");

  return (
    <section>
      <p>
        <Link href="/tasks">← My tasks</Link>
      </p>
      <h1>
        {t.name} <span className={`badge ${t.status}`}>{t.status}</span>
      </h1>
      <p className="muted">
        {t.task_type} · {String(t.prepared["n_train"])} train / {String(t.prepared["n_test"])} test rows ·{" "}
        {String(t.prepared["n_features"])} features
      </p>
      {t.error && <p className="error">Failed: {t.error}</p>}

      <h1>State</h1>
      <Timeline status={t.status} history={t.status_history} />

      {(t.plan as PlanDoc | null)?.shares && (
        <>
          <h1>Plan vs actual</h1>
          <p className="muted">{(t.plan as PlanDoc).explanation}</p>
          <table data-testid="plan">
            <thead>
              <tr>
                <th>Device</th>
                <th>Benchmark</th>
                <th>Weight</th>
                <th>Rows</th>
                <th>Chunks</th>
                <th>Predicted compute</th>
                <th>Actual compute</th>
              </tr>
            </thead>
            <tbody>
              {((t.plan as PlanDoc).shares ?? []).map((sh) => {
                const mine = st.assignments.filter((a) => a.device_id === sh.device_id && a.status === "succeeded");
                const actual = mine.reduce((sum, a) => sum + (a.timings?.["compute_ms"] ?? 0), 0);
                return (
                  <tr key={sh.device_id}>
                    <td>{sh.device_name ?? sh.device_id}</td>
                    <td>{(sh.score_cells_per_sec / 1e6).toFixed(2)} M cells/s</td>
                    <td>{(sh.weight * 100).toFixed(1)}%</td>
                    <td>{sh.rows}</td>
                    <td>{sh.n_chunks}</td>
                    <td>{fmtSec(sh.est_seconds)}</td>
                    <td>{mine.length ? fmtSec(actual / 1000) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      <Lanes st={st} />

      <h1>Chunks and devices</h1>
      {st.chunks.length === 0 ? (
        <div>
          <p className="muted">
            {t.status === "queued"
              ? "Waiting for an eligible idle device (online, benchmarked, enough memory, battery ≥ 20% or charging)…"
              : "No chunks."}
          </p>
          {st.waiting_reasons.length > 0 && (
            <ul className="muted" data-testid="waiting">
              {st.waiting_reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Chunk</th>
              <th>Rows</th>
              <th>Device</th>
              <th>Status</th>
              <th>Attempt</th>
              <th>Download</th>
              <th>Compute</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {st.chunks.map((c) =>
              (asgByChunk.get(c.id) ?? [undefined]).map((a, i) => (
                <tr key={`${c.id}-${a?.id ?? i}`}>
                  <td>{c.index}</td>
                  <td>
                    [{c.row_start}, {c.row_end}) · {c.n_rows}
                  </td>
                  <td>{a?.device_name ?? "—"}</td>
                  <td>
                    <span className={`badge ${a?.status ?? c.status}`}>{a?.status ?? c.status}</span>
                  </td>
                  <td>
                    {a?.attempt_no ?? 0}/{c.max_attempts}
                  </td>
                  <td>{fmtMs(a?.timings?.["download_ms"])}</td>
                  <td>{fmtMs(a?.timings?.["compute_ms"])}</td>
                  <td>{fmtMs(a?.timings?.["total_ms"])}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      )}

      {result && (
        <>
          <h1>Result</h1>
          <div className="card">
            {result.reference_check && (
              <p data-testid="reference">
                Reference check (distributed == centralized):{" "}
                <strong className={result.reference_check.passed ? "ok" : "error"}>
                  {result.reference_check.passed ? "PASSED" : "FAILED"}
                </strong>{" "}
                <span className="muted">
                  max relative difference {result.reference_check.max_relative_difference.toExponential(2)} (tolerance{" "}
                  {result.reference_check.tolerance.toExponential(0)})
                </span>
              </p>
            )}
            <p className="muted">Holdout metrics (computed by the aggregator, not distributed):</p>
            <table>
              <tbody>
                {metricRows.map(([k, v]) => (
                  <tr key={k}>
                    <th>{k}</th>
                    <td>{typeof v === "number" ? Number(v.toPrecision(6)) : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="row">
              {st.artifacts.map((a) => (
                <button key={a.id} onClick={() => void save(api, a)} data-testid={`dl-${a.kind}`}>
                  Download {a.filename}
                </button>
              ))}
            </div>
            <p className="warn">
              model.joblib is a pickle: only load it from a trusted ProofNet instance. model.json is the safe
              alternative. Results are unverified (Part 2 adds verification).
            </p>
          </div>
        </>
      )}

      <h1>Events</h1>
      <pre className="log" data-testid="events">
        {events.map((e) => `${new Date(e.ts).toLocaleTimeString()}  ${e.message || e.type}`).join("\n") || "—"}
      </pre>
    </section>
  );
}

export default function TaskPage() {
  return (
    <AuthGate>
      <Monitor />
    </AuthGate>
  );
}
