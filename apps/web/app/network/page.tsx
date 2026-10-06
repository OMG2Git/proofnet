"use client";

import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type NetworkSummary } from "@/lib/api/client";

function Network() {
  const [data, setData] = useState<NetworkSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      api
        .networkSummary()
        .then((d) => {
          if (alive) {
            setData(d);
            setError(null);
          }
        })
        .catch((e: Error) => alive && setError(e.message));
    void tick();
    const t = setInterval(tick, 1500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (!data) return <p className={error ? "error" : "muted"}>{error ?? "Loading…"}</p>;
  return (
    <section className="wide">
      <h1>Network</h1>
      <p data-testid="counts">
        {Object.entries(data.counts).map(([k, v]) => (
          <span key={k} className={`badge ${k}`} style={{ marginRight: 8 }}>
            {k}: {v}
          </span>
        ))}
        <span className="muted"> tasks running: {data.tasks_running}</span>
      </p>
      {error && <p className="error">Connection problem: {error}</p>}
      <div className="devices">
        {data.devices.map((d) => (
          <div key={d.id} className={`device ${d.status}`} data-testid="device-card">
            <div>
              <strong>{d.name}</strong>
            </div>
            <div className="muted">
              {d.device_type} · {d.runtime_kind ?? "no runtime"}
            </div>
            <div>
              <span className={`badge ${d.status}`}>{d.status}</span>
            </div>
            <div>{d.score_cells_per_sec ? `${(d.score_cells_per_sec / 1e6).toFixed(2)} M cells/s` : "no benchmark"}</div>
            <div className="muted">
              last seen {d.last_seen_age_seconds === null || d.last_seen_age_seconds === undefined ? "never" : `${Math.round(d.last_seen_age_seconds)}s ago`}
            </div>
            <div className="muted">{d.current_assignment_id ? `chunk: ${d.current_assignment_id}` : "no chunk"}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function NetworkPage() {
  return (
    <AuthGate>
      <Network />
    </AuthGate>
  );
}
