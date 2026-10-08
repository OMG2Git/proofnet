"use client";

import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type SecurityOverview } from "@/lib/api/client";

const sevClass = (s: string) => (s === "critical" ? "rejected" : s === "warning" ? "busy" : "pending");

function describe(kind: string, data: Record<string, unknown>): string {
  switch (kind) {
    case "device_quarantined":
      return `Device quarantined: ${String(data["reason"] ?? "")}`;
    case "device_reinstated":
      return `Device reinstated by an administrator: ${String(data["reason"] ?? "")}`;
    case "rewards_revoked":
      return `Unverified rewards clawed back: ${String(data["entries"] ?? 0)} entries, ${String(data["amount"] ?? 0)} credits`;
    case "reference_check_failed":
      return `End-to-end check failed: culprit chunk(s) ${JSON.stringify(data["culprit_chunks"])} recomputed`;
    case "retroactive_audit":
      return `Earlier results re-audited: ${JSON.stringify(data)}`;
    case "login_lockout":
      return `Account locked after repeated failed sign-ins (${String(data["seconds"] ?? "")} s)`;
    case "rate_limited":
      return `Rate limit hit on ${String(data["scope"] ?? "")}`;
    case "foreign_assignment_access":
      return "A device tried to touch an assignment that is not its own";
    case "conflicting_resubmission":
      return "A device re-submitted a different result for an assignment that was already accepted";
    case "device_cap_reached":
      return "Device registration cap reached for an account";
    default:
      return JSON.stringify(data);
  }
}

function Security() {
  const [data, setData] = useState<SecurityOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [admin, setAdmin] = useState(false);

  useEffect(() => {
    api.adminMe().then((r) => setAdmin(r.admin)).catch(() => undefined);
    let alive = true;
    const tick = () =>
      api
        .securityOverview()
        .then((d) => alive && (setData(d), setError(null)))
        .catch((e: Error) => alive && setError(e.message));
    void tick();
    const t = setInterval(tick, 2500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (!data) return <p className={error ? "error" : "muted"}>{error ?? "Loading…"}</p>;
  return (
    <section className="wide">
      <h1>Security</h1>
      {error && <p className="error">Connection problem: {error}</p>}
      <div className="devices" data-testid="sec-counts">
        {Object.keys(data.counts).length === 0 ? (
          <p className="muted">No security events in the last 24 hours.</p>
        ) : (
          Object.entries(data.counts).map(([k, v]) => (
            <div key={k} className="device">
              <div className="muted">{k.replaceAll("_", " ")} (24 h)</div>
              <div style={{ fontSize: 24 }}>{v}</div>
            </div>
          ))
        )}
      </div>

      <h1>Quarantined devices</h1>
      {data.quarantined_devices.length === 0 ? (
        <p className="muted">None.</p>
      ) : (
        <table data-testid="quarantined">
          <thead>
            <tr>
              <th>Device</th>
              <th>Why</th>
              <th>Evidence</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.quarantined_devices.map((d) => (
              <tr key={d.device_id}>
                <td>{d.device_name}</td>
                <td>
                  <span className="muted">{String(d.quarantine?.["source"] ?? "")}</span>
                  <div>{String(d.quarantine?.["reason"] ?? "")}</div>
                </td>
                <td>
                  {d.audits} audits · {d.exceedances} wrong · suspicion {d.suspicion.toFixed(2)}
                </td>
                <td>
                  {admin ? (
                    <button
                      onClick={() => {
                        const reason = prompt("Reason for reinstating (the device restarts on probation):");
                        if (reason)
                          void api.reinstateDevice(d.device_id, reason).catch((e: Error) => alert(e.message));
                      }}
                    >
                      Reinstate
                    </button>
                  ) : (
                    <span className="muted">admin only</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h1>Event log</h1>
      <table data-testid="sec-events">
        <thead>
          <tr>
            <th>Time</th>
            <th>Severity</th>
            <th>Event</th>
          </tr>
        </thead>
        <tbody>
          {data.events.map((e) => (
            <tr key={e.id}>
              <td>{new Date(e.ts).toLocaleString()}</td>
              <td>
                <span className={`badge ${sevClass(e.severity)}`}>{e.severity}</span>
              </td>
              <td>
                <strong>{e.kind.replaceAll("_", " ")}</strong>
                <div className="muted">{describe(e.kind, e.data)}</div>
              </td>
            </tr>
          ))}
          {data.events.length === 0 && (
            <tr>
              <td colSpan={3} className="muted">
                Nothing yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <h1>What protects the system</h1>
      <div className="devices">
        {data.controls.map((c) => (
          <div key={c.name} className="device idle">
            <strong>{c.name}</strong>
            <div className="muted">{c.detail}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function SecurityPage() {
  return (
    <AuthGate>
      <Security />
    </AuthGate>
  );
}
