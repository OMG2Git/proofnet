"use client";

import type { EventOut } from "@/lib/api/client";
import {
  eventLabel,
  eventTone,
  formatAge,
  formatClock,
  formatScore,
  VISUAL_LABEL,
  type Kpis,
  type WorldDevice,
} from "@/lib/network/logic";

/* ------------------------------------------------------------------ KPIs */

export type AuditTotals = { audits: number; results: number } | null;

export function KpiStrip({ kpis, audit }: { kpis: Kpis | null; audit: AuditTotals }) {
  const tiles: { label: string; value: string; note: string; color: string; id: string }[] = [
    { id: "registered", label: "Registered devices", value: kpis ? String(kpis.registered) : "—", note: "All devices in the database", color: "var(--accent)" },
    { id: "online", label: "Online devices", value: kpis ? String(kpis.online) : "—", note: "Status idle or busy (heartbeat < 20 s)", color: "var(--ok)" },
    { id: "tasks", label: "Active tasks", value: kpis ? String(kpis.activeTasks) : "—", note: "Running or aggregating now", color: "var(--compute)" },
    { id: "assignments", label: "Active assignments", value: kpis ? String(kpis.activeAssignments) : "—", note: "Devices holding a chunk now", color: "var(--accent)" },
    {
      id: "audited",
      label: "Results audited",
      value: audit ? `${audit.audits}/${audit.results}` : "—",
      note: "Lifetime audits ÷ results (audits run on intake, so nothing queues)",
      color: "var(--warn)",
    },
    { id: "quarantined", label: "Quarantined", value: kpis ? String(kpis.quarantined) : "—", note: "Isolated by the trust system", color: "var(--err)" },
  ];
  return (
    <div className="kpis" role="list" aria-label="Network key figures">
      {tiles.map((t) => (
        <div className="kpi" role="listitem" key={t.id} style={{ ["--kc" as string]: t.color }} data-testid={`kpi-${t.id}`}>
          <div className="k-label">{t.label}</div>
          <div className="k-value">{t.value}</div>
          <div className="k-note">{t.note}</div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------- assignment table */

export type DeviceExtras = {
  assignedAt: Map<string, string>; // assignment id -> chunk_assigned event ts
  rewards: Map<string, { confirmed: number; pending: number }>; // own devices only
};

const shortId = (id: string): string => (id.length > 12 ? `${id.slice(0, 12)}…` : id);

export function DeviceTable({
  devices,
  selectedId,
  onSelect,
  extras,
}: {
  devices: WorldDevice[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  extras: DeviceExtras;
}) {
  if (devices.length === 0) return <p className="muted" style={{ padding: "14px" }}>No devices match the current filter.</p>;
  return (
    <div className="table-wrap">
      <table data-testid="device-table">
        <caption className="sr-only">Devices, their assignments, verification and trust</caption>
        <thead>
          <tr>
            <th scope="col">Device</th>
            <th scope="col">Status</th>
            <th scope="col" className="num">
              Benchmark
            </th>
            <th scope="col">Assigned work</th>
            <th scope="col">Verification</th>
            <th scope="col">Trust</th>
            <th scope="col">Rewards</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((d) => {
            const at = d.assignmentId ? extras.assignedAt.get(d.assignmentId) : undefined;
            const rw = extras.rewards.get(d.id);
            return (
              <tr key={d.id} className={selectedId === d.id ? "selected" : ""} data-testid="device-card">
                <td>
                  <button className="ghost small" style={{ fontWeight: 600 }} onClick={() => onSelect(d.id)} aria-label={`Open details for ${d.name}`}>
                    {d.name}
                  </button>
                  <div className="muted mono" style={{ fontSize: 11.5 }} title={d.id}>
                    {shortId(d.id)} · {d.deviceType.replace("_", " ")}
                  </div>
                </td>
                <td>
                  <span className={`badge ${d.visual === "available" ? "idle" : d.visual}`}>{VISUAL_LABEL[d.visual]}</span>
                  <div className="muted" style={{ fontSize: 12 }}>
                    seen {formatAge(d.lastSeenAgeSeconds)}
                  </div>
                </td>
                <td className="num">{formatScore(d.scoreCellsPerSec)}</td>
                <td data-testid="holding">
                  {d.assignmentId ? (
                    <>
                      <div>
                        chunk <strong>{d.chunkIndex ?? "?"}</strong>
                        {d.rows !== null && ` · ${d.rows.toLocaleString()} rows`}
                      </div>
                      <div className="muted" style={{ fontSize: 12 }} title={d.taskId ?? undefined}>
                        {d.taskName ?? d.taskId ?? "task"} · <span className="mono">{shortId(d.assignmentId)}</span>
                      </div>
                      <div className="muted" style={{ fontSize: 12 }}>
                        {at ? `assigned ${formatClock(at)}` : "assigned (time not in feed)"} · progress not reported
                      </div>
                    </>
                  ) : (
                    <span className="muted">none</span>
                  )}
                </td>
                <td>
                  {d.flash ? (
                    <span className={`badge ${d.flash.kind === "verified" ? "verified" : "rejected"}`}>{d.flash.kind}</span>
                  ) : d.recentlyAudited ? (
                    <span className="muted">audited recently</span>
                  ) : (
                    <span className="muted">no recent audit</span>
                  )}
                  {d.recentFailure && <div className="warn" style={{ fontSize: 12, margin: 0 }}>recent failure</div>}
                </td>
                <td>
                  {d.trustStatus ? (
                    <span
                      className={`badge ${d.trustStatus}`}
                      title={d.trust !== null ? `trust ${d.trust.toFixed(2)}` : undefined}
                      data-testid="trust-badge"
                    >
                      {d.trustStatus}
                    </span>
                  ) : (
                    <span className="muted">unknown</span>
                  )}
                  <div className="muted mono" style={{ fontSize: 12 }}>
                    {d.trust !== null ? d.trust.toFixed(2) : "—"}
                  </div>
                </td>
                <td className="mono">
                  {rw ? (
                    <>
                      {rw.confirmed.toFixed(2)}
                      <div className="muted" style={{ fontSize: 11.5 }}>
                        +{rw.pending.toFixed(2)} pending
                      </div>
                    </>
                  ) : (
                    <span className="muted" title="Reward balances are visible to the owning account only">
                      owner only
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* -------------------------------------------------------------- event feed */

export function ActivityFeed({
  events,
  isAdmin,
  error,
  onSelectDevice,
}: {
  events: EventOut[];
  isAdmin: boolean | null;
  error: string | null;
  onSelectDevice: (id: string) => void;
}) {
  if (isAdmin === false)
    return (
      <p className="muted" style={{ padding: 14 }}>
        The live event feed spans every user&apos;s work, so it is visible to administrators only. Device states above still
        update live.
      </p>
    );
  const rows = [...events].reverse().slice(0, 120);
  return (
    <>
      {error && <p className="error" style={{ padding: "0 14px" }}>Event feed problem: {error}</p>}
      {rows.length === 0 ? (
        <p className="muted" style={{ padding: 14 }}>
          {isAdmin === null ? "Loading events…" : "No events recorded yet."}
        </p>
      ) : (
        <ol className="feed" data-testid="feed" aria-label="Recent network events, newest first">
          {rows.map((e) => (
            <li key={e.id} className={eventTone(e.type, e.data)}>
              <time dateTime={e.ts} title={`Event time ${e.ts}`}>
                {formatClock(e.ts)}
              </time>
              <span className="ev-type">{eventLabel(e.type)}</span>
              <span>
                {e.message}
                {e.device_id && (
                  <>
                    {" "}
                    <button className="ghost small" style={{ minHeight: 0, padding: "0 6px", fontSize: 12 }} onClick={() => onSelectDevice(e.device_id!)}>
                      device
                    </button>
                  </>
                )}
              </span>
            </li>
          ))}
        </ol>
      )}
      <p className="muted" style={{ fontSize: 11.5, padding: "6px 14px 10px", margin: 0 }}>
        Times are the backend&apos;s event timestamps shown in your local time zone. The feed polls every 2 s.
      </p>
    </>
  );
}
