"use client";

import { Fragment, useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type DeviceTrust, type TrustOverview } from "@/lib/api/client";

const fmtP = (x: number) => `${(x * 100).toFixed(1)}%`;
const sci = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : x < 1e-290 ? "0" : x.toExponential(2);

function StatusBadge({ s }: { s: string }) {
  const cls = s === "trusted" ? "succeeded" : s === "quarantined" ? "rejected" : s === "watch" ? "busy" : "pending";
  return <span className={`badge ${cls}`}>{s}</span>;
}

function Bar({ value, color = "var(--ok)", label }: { value: number; color?: string; label?: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div title={label} style={{ background: "var(--line)", borderRadius: 4, height: 10, minWidth: 80 }}>
      <div style={{ width: `${v * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
  );
}

/** How close the betting evidence is to the accusation threshold, on a log scale. */
function evidenceFill(d: DeviceTrust): number {
  // log10_ratio = log10(evidence / threshold) is <= 0 until accusation; show the last 9 decades.
  return Math.max(0, Math.min(1, 1 + d.log10_ratio / 9));
}

function Trust() {
  const [data, setData] = useState<TrustOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [admin, setAdmin] = useState(false);

  useEffect(() => {
    api.adminMe().then((r) => setAdmin(r.admin)).catch(() => undefined);
    let alive = true;
    const tick = () =>
      api
        .trustOverview()
        .then((d) => alive && (setData(d), setError(null)))
        .catch((e: Error) => alive && setError(e.message));
    void tick();
    const t = setInterval(tick, 2000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (!data) return <p className={error ? "error" : "muted"}>{error ?? "Loading…"}</p>;
  const P = data.parameters;
  return (
    <section className="wide">
      <h1>Trust &amp; verification</h1>
      <p className="muted">
        Every result is checked by the backend with an adaptive probability. Evidence from those audits builds (or
        destroys) a device&apos;s trust, which sets how often it is audited and how much it earns.
      </p>
      <div className="devices" data-testid="trust-summary">
        {(
          [
            ["Devices", data.devices],
            ["Probation", data.probation],
            ["Trusted", data.trusted],
            ["Watch", data.watch],
            ["Quarantined", data.quarantined],
            ["Results", data.results_seen],
            ["Audits", data.audits],
            ["Rejected", data.exceedances],
            ["Audit rate", fmtP(data.audit_rate)],
          ] as [string, number | string][]
        ).map(([k, v]) => (
          <div key={k} className="device">
            <div className="muted">{k}</div>
            <div style={{ fontSize: 24 }}>{v}</div>
          </div>
        ))}
      </div>
      {error && <p className="error">Connection problem: {error}</p>}

      <h1>Devices</h1>
      <table data-testid="trust-table">
        <thead>
          <tr>
            <th>Device</th>
            <th>Status</th>
            <th>Results / audits</th>
            <th>Trust</th>
            <th>Next audit probability</th>
            <th title="Mean Shiryaev–Roberts evidence relative to the accusation threshold (log scale)">Evidence → threshold</th>
            <th>Suspicion (memory)</th>
            <th>Reward ×</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.devices_detail.map((d) => (
            <Fragment key={d.device_id}>
              <tr>
                <td>
                  <strong>{d.device_name}</strong>
                  <div className="muted">{d.runtime_kind ?? "—"}</div>
                </td>
                <td>
                  <StatusBadge s={d.status} />
                </td>
                <td>
                  {d.results_seen} / {d.audits}
                  <div className="muted">{d.n_clean} clean · {d.exceedances} wrong</div>
                </td>
                <td style={{ minWidth: 110 }}>
                  <Bar value={d.trust} label={`trust ${d.trust.toFixed(2)}`} />
                  <span className="muted">{d.trust.toFixed(2)}</span>
                </td>
                <td style={{ minWidth: 110 }}>
                  <Bar value={d.audit_probability} color="var(--accent)" />
                  <span className="muted">{fmtP(d.audit_probability)}</span>
                </td>
                <td style={{ minWidth: 150 }}>
                  <Bar value={evidenceFill(d)} color={d.log10_ratio > -3 ? "var(--err)" : "var(--warn)"} />
                  <span className="muted">
                    {sci(d.evidence)} / {sci(d.threshold)}
                  </span>
                </td>
                <td style={{ minWidth: 110 }}>
                  <Bar value={Math.max(d.suspicion, d.memory)} color="var(--err)" />
                  <span className="muted">
                    {d.suspicion.toFixed(2)} ({d.memory.toFixed(2)})
                  </span>
                </td>
                <td>×{d.reward_multiplier.toFixed(2)}</td>
                <td>
                  <button onClick={() => setOpen(open === d.device_id ? null : d.device_id)}>
                    {open === d.device_id ? "Hide" : "History"}
                  </button>{" "}
                  {admin && d.status !== "quarantined" && (
                    <button
                      onClick={() => {
                        if (confirm(`Quarantine ${d.device_name}?`))
                          void api.quarantineDevice(d.device_id, "manual (admin demo)").catch((e: Error) => alert(e.message));
                      }}
                    >
                      Quarantine
                    </button>
                  )}
                </td>
              </tr>
              {open === d.device_id && (
                <tr>
                  <td colSpan={9}>
                    {d.quarantine && (
                      <p className="error">
                        Quarantined ({String(d.quarantine["source"] ?? "")}): {String(d.quarantine["reason"] ?? "")}
                      </p>
                    )}
                    {d.history.length === 0 ? (
                      <p className="muted">No history.</p>
                    ) : (
                      <ul className="muted">
                        {d.history.map((h, i) => (
                          <li key={i}>
                            {new Date(h.at).toLocaleTimeString()} — {h.event.replaceAll("_", " ")}
                            {h.discrepancy !== null && h.discrepancy !== undefined &&
                              ` (discrepancy ${sci(h.discrepancy)}, tolerance ${sci(h.tolerance)}, suspicion ${(h.suspicion ?? 0).toFixed(2)})`}
                            {h.reason ? ` — ${h.reason}` : ""}
                          </li>
                        ))}
                      </ul>
                    )}
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
          {data.devices_detail.length === 0 && (
            <tr>
              <td colSpan={9} className="muted">
                No audited devices yet — run a task.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <h1>How a device earns trust</h1>
      <div className="card">
        <ol>
          <li>
            <strong>Probation.</strong> The first {P["probation_results"]} results of every device are always audited.
          </li>
          <li>
            <strong>Audit probability</strong> falls with clean audits from {fmtP(P["audit_initial"] ?? 0)} towards a floor of{" "}
            {fmtP(P["audit_floor"] ?? 0)} (never below), and is pushed back towards 100% by suspicion, which fades slowly
            (memory ×{P["memory_decay"]} per audit) so a cheater stays watched.
          </li>
          <li>
            <strong>Exceedance.</strong> An audit compares the device&apos;s result with the backend&apos;s recomputation. The
            tolerance is {P["margin"]}× the ({fmtP(P["p"] ?? 0)}, {fmtP(P["gamma"] ?? 0)}) order-statistic tolerance limit of
            honest discrepancies of the same kernel and runtime (distribution-free), never above the kernel&apos;s hard
            numerical bound.
          </li>
          <li>
            <strong>Evidence.</strong> A betting e-detector (Shiryaev–Roberts, restarts at every audit so sleepers cannot hide
            behind history) multiplies up with exceedances. It accuses only at a threshold chosen so that an honest device
            is wrongly accused with <em>lifetime</em> probability ≤ {P["alpha"]}.
          </li>
          <li>
            <strong>Reward multiplier</strong> = {P["reward_base"]} + {1 - (P["reward_base"] ?? 0.5)} × trust.
          </li>
        </ol>
      </div>

      <h1>Calibration (honest numerical noise per class)</h1>
      <table data-testid="calibration">
        <thead>
          <tr>
            <th>Class (kernel@version | runtime)</th>
            <th>Honest samples</th>
            <th>Needed</th>
            <th>Tolerance limit L</th>
            <th>Tolerance used</th>
            <th>Source</th>
            <th>Sample min / median / max</th>
          </tr>
        </thead>
        <tbody>
          {data.calibration.map((c) => (
            <tr key={c.class_key}>
              <td>{c.class_key}</td>
              <td>{c.n}</td>
              <td>{c.min_samples}</td>
              <td>{sci(c.limit)}</td>
              <td>{sci(c.tolerance)}</td>
              <td>
                <span className={`badge ${c.source === "calibrated" ? "succeeded" : "pending"}`}>{c.source.replace("_", " ")}</span>
              </td>
              <td className="muted">
                {sci(c.sample_min)} / {sci(c.sample_median)} / {sci(c.sample_max)}
              </td>
            </tr>
          ))}
          {data.calibration.length === 0 && (
            <tr>
              <td colSpan={7} className="muted">
                Calibration starts with the first audits.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <h1>Recent verification records</h1>
      <table data-testid="records">
        <thead>
          <tr>
            <th>Time</th>
            <th>Device</th>
            <th>Mode</th>
            <th>Audit p</th>
            <th>Audited</th>
            <th>Discrepancy</th>
            <th>Tolerance</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {data.recent_records.map((r) => (
            <tr key={r.id}>
              <td>{new Date(r.at).toLocaleTimeString()}</td>
              <td>{r.device_name ?? r.device_id}</td>
              <td>{r.mode}</td>
              <td>{r.audit_probability === null || r.audit_probability === undefined ? "—" : fmtP(r.audit_probability)}</td>
              <td>{r.audited ? "yes" : "no"}</td>
              <td>{sci(r.discrepancy)}</td>
              <td>{sci(r.tolerance)}</td>
              <td>
                <span className={`badge ${r.decision === "verified" ? "succeeded" : r.decision === "accepted_unverified" ? "pending" : "rejected"}`}>
                  {r.decision.replaceAll("_", " ")}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default function TrustPage() {
  return (
    <AuthGate>
      <Trust />
    </AuthGate>
  );
}
