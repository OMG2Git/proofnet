"use client";

import { useEffect, useRef, useState } from "react";
import {
  api,
  ApiRequestError,
  type DeviceTrust,
  type EventOut,
  type RewardsOut,
  type VerificationRecord,
} from "@/lib/api/client";
import { eventLabel, eventTone, formatAge, formatClock, formatScore, VISUAL_LABEL, type WorldDevice } from "@/lib/network/logic";

type Loaded = {
  trust: DeviceTrust | null;
  trustError: string | null;
  records: VerificationRecord[];
  rewards: RewardsOut | null;
};

/** Worker detail drawer. Everything shown is fetched from existing authenticated endpoints. */
export default function WorkerDrawer({
  device,
  events,
  isAdmin,
  onClose,
  onChanged,
}: {
  device: WorldDevice;
  events: EventOut[];
  isAdmin: boolean;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [data, setData] = useState<Loaded | null>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const id = device.id;

  useEffect(() => {
    let alive = true;
    const load = async () => {
      const [trust, records, rewards] = await Promise.allSettled([
        api.trustDevice(id),
        api.trustRecentRecords(300),
        api.myRewards(),
      ]);
      if (!alive) return;
      setData({
        trust: trust.status === "fulfilled" ? trust.value : null,
        trustError:
          trust.status === "rejected"
            ? trust.reason instanceof ApiRequestError && trust.reason.status === 404
              ? "Trust details are visible to the device owner and administrators."
              : "Could not load trust details."
            : null,
        records: records.status === "fulfilled" ? records.value.filter((r) => r.device_id === id).slice(0, 12) : [],
        rewards: rewards.status === "fulfilled" ? rewards.value : null,
      });
    };
    void load();
    const t = setInterval(() => void load(), 8000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [id]);

  // `onClose` changes identity on every parent render (the dashboard re-renders each second), so keep
  // it in a ref: focus must move to the drawer once on open and never be pulled back while typing.
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
  }, []);

  const mine = events.filter((e) => e.device_id === id).slice(-15).reverse();
  const bal = data?.rewards?.per_device[id];
  const ownDevice = data?.rewards ? id in data.rewards.device_names : false;
  const t = data?.trust;

  async function act(kind: "quarantine" | "reinstate") {
    setBusy(true);
    setActionMsg(null);
    try {
      if (kind === "quarantine") await api.quarantineDevice(id, reason.trim());
      else await api.reinstateDevice(id, reason.trim());
      setActionMsg(kind === "quarantine" ? "Device quarantined." : "Device reinstated (back on probation).");
      setReason("");
      onChanged();
    } catch (e) {
      setActionMsg(e instanceof ApiRequestError ? e.message : "Action failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} aria-hidden="true" />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Details for ${device.name}`} data-testid="drawer">
        <header>
          <h2>{device.name}</h2>
          <button ref={closeRef} className="ghost small" onClick={onClose} aria-label="Close details">
            Close ✕
          </button>
        </header>

        <h3>Identity & state</h3>
        <dl className="kv">
          <dt>Device ID</dt>
          <dd className="mono">{device.id}</dd>
          <dt>Type</dt>
          <dd>{device.deviceType.replace("_", " ")}</dd>
          <dt>State</dt>
          <dd>
            <span className={`badge ${device.visual === "available" ? "idle" : device.visual}`}>{VISUAL_LABEL[device.visual]}</span>{" "}
            <span className="muted">(backend status: {device.status})</span>
          </dd>
          <dt>Last heartbeat</dt>
          <dd>{formatAge(device.lastSeenAgeSeconds)}</dd>
          <dt>Benchmark</dt>
          <dd>{formatScore(device.scoreCellsPerSec)}</dd>
        </dl>

        <h3>Current assignment</h3>
        {device.assignmentId ? (
          <dl className="kv">
            <dt>Task</dt>
            <dd>{device.taskName ?? device.taskId}</dd>
            <dt>Chunk</dt>
            <dd>
              {device.chunkIndex ?? "?"} · {device.rows?.toLocaleString() ?? "?"} rows
            </dd>
            <dt>Assignment</dt>
            <dd className="mono">{device.assignmentId}</dd>
          </dl>
        ) : (
          <p className="muted">No chunk right now.</p>
        )}

        <h3>Trust</h3>
        {t ? (
          <>
            <dl className="kv">
              <dt>Status</dt>
              <dd>
                <span className={`badge ${t.status}`}>{t.status}</span>
              </dd>
              <dt>Trust score</dt>
              <dd>
                {t.trust.toFixed(2)}
                <div className="meter" role="img" aria-label={`Trust ${t.trust.toFixed(2)} of 1`}>
                  <i style={{ width: `${Math.round(t.trust * 100)}%`, ["--c" as string]: "var(--ok)" }} />
                </div>
              </dd>
              <dt>Suspicion</dt>
              <dd>
                {t.suspicion.toFixed(2)} (memory {t.memory.toFixed(2)})
                <div className="meter" role="img" aria-label={`Suspicion ${t.suspicion.toFixed(2)} of 1`}>
                  <i style={{ width: `${Math.round(t.suspicion * 100)}%`, ["--c" as string]: "var(--warn)" }} />
                </div>
              </dd>
              <dt>Audits</dt>
              <dd>
                {t.audits} of {t.results_seen} results · {t.exceedances} over tolerance
              </dd>
              <dt>Next audit</dt>
              <dd>{(t.audit_probability * 100).toFixed(0)}% chance</dd>
              <dt>Reward ×</dt>
              <dd>{t.reward_multiplier.toFixed(2)}</dd>
            </dl>
            {t.quarantine && (
              <div className="error-box" data-testid="quarantine-info">
                <strong>Quarantined</strong>
                <div>{String((t.quarantine as Record<string, unknown>)["reason"] ?? "no reason recorded")}</div>
                <div className="muted mono" style={{ fontSize: 12 }}>
                  {String((t.quarantine as Record<string, unknown>)["at"] ?? "")}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="muted">{data ? (data.trustError ?? "No trust profile yet.") : "Loading…"}</p>
        )}

        <h3>Audit outcomes</h3>
        {data && data.records.length === 0 && <p className="muted">No verification records for this device.</p>}
        {data && data.records.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Decision</th>
                  <th className="num">Discrepancy</th>
                </tr>
              </thead>
              <tbody>
                {data.records.map((r) => (
                  <tr key={r.id}>
                    <td className="mono">{formatClock(r.at)}</td>
                    <td>
                      <span className={`badge ${r.decision === "verified" ? "verified" : r.decision === "rejected" ? "rejected" : "pending"}`}>
                        {r.decision}
                      </span>
                      {!r.audited && <span className="muted"> not audited</span>}
                    </td>
                    <td className="num">{r.discrepancy !== null && r.discrepancy !== undefined ? r.discrepancy.toExponential(1) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <h3>Rewards</h3>
        {ownDevice && bal ? (
          <dl className="kv">
            <dt>Confirmed</dt>
            <dd className="mono">{bal.confirmed.toFixed(3)} credits</dd>
            <dt>Pending</dt>
            <dd className="mono">{bal.pending.toFixed(3)}</dd>
            <dt>Revoked</dt>
            <dd className="mono">{bal.revoked.toFixed(3)}</dd>
          </dl>
        ) : ownDevice ? (
          <p className="muted">No reward entries yet.</p>
        ) : (
          <p className="muted">Reward ledgers are visible to the owning account only.</p>
        )}

        <h3>Recent activity</h3>
        {mine.length === 0 ? (
          <p className="muted">{isAdmin ? "No events in the recent window." : "The event feed is admin-only."}</p>
        ) : (
          <ol className="feed" style={{ maxHeight: 260 }}>
            {mine.map((e) => (
              <li key={e.id} className={eventTone(e.type, e.data)} style={{ padding: "6px 0" }}>
                <time dateTime={e.ts}>{formatClock(e.ts)}</time>
                <span className="ev-type">{eventLabel(e.type)}</span>
                <span>{e.message}</span>
              </li>
            ))}
          </ol>
        )}

        {isAdmin && (
          <>
            <h3>Administrator actions</h3>
            <label>
              Reason (recorded in the security log)
              <input value={reason} onChange={(e) => setReason(e.target.value)} maxLength={300} placeholder="why?" />
            </label>
            <div className="row" style={{ marginTop: 6 }}>
              {device.visual === "quarantined" ? (
                <button className="ghost" disabled={busy || !reason.trim()} onClick={() => act("reinstate")}>
                  Reinstate
                </button>
              ) : (
                <button className="danger" disabled={busy || !reason.trim()} onClick={() => act("quarantine")}>
                  Quarantine
                </button>
              )}
            </div>
            {actionMsg && (
              <p role="status" className={actionMsg.endsWith(".") ? "ok" : "error"}>
                {actionMsg}
              </p>
            )}
          </>
        )}
      </aside>
    </>
  );
}
