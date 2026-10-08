"use client";

import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import { api, type LedgerCheck, type NetworkRewardRow, type RewardsOut } from "@/lib/api/client";

const cr = (x: number) => x.toFixed(4);

function Rewards() {
  const [me, setMe] = useState<RewardsOut | null>(null);
  const [net, setNet] = useState<NetworkRewardRow[]>([]);
  const [check, setCheck] = useState<LedgerCheck | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      Promise.all([api.myRewards(), api.networkRewards()])
        .then(([m, n]) => alive && (setMe(m), setNet(n), setError(null)))
        .catch((e: Error) => alive && setError(e.message));
    void tick();
    const t = setInterval(tick, 2500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (!me) return <p className={error ? "error" : "muted"}>{error ?? "Loading…"}</p>;
  const b = me.balance;
  return (
    <section className="wide">
      <h1>Rewards</h1>
      <p className="muted">
        Credits are an internal accounting unit (no money). Work earns {me.rate_credits_per_million_units} credit per
        million work units at full trust; an unproven device earns {(me.base_multiplier * 100).toFixed(0)}% of that, and
        the rate rises to 100% as audits confirm the device is honest. Work that was audited and correct is
        <strong> confirmed</strong> immediately; unaudited work stays <strong>pending</strong> until its task finishes with a
        passing end-to-end check, and is <strong>revoked</strong> if its device is later quarantined.
      </p>
      <div className="devices" data-testid="balances">
        <div className="device idle">
          <div className="muted">Confirmed</div>
          <div style={{ fontSize: 28 }} data-testid="bal-confirmed">{cr(b.confirmed ?? 0)}</div>
        </div>
        <div className="device busy">
          <div className="muted">Pending</div>
          <div style={{ fontSize: 28 }} data-testid="bal-pending">{cr(b.pending ?? 0)}</div>
        </div>
        <div className="device offline">
          <div className="muted">Revoked (clawback)</div>
          <div style={{ fontSize: 28 }} data-testid="bal-revoked">{cr(b.revoked ?? 0)}</div>
        </div>
      </div>
      {error && <p className="error">Connection problem: {error}</p>}

      <h1>By device</h1>
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>Confirmed</th>
            <th>Pending</th>
            <th>Revoked</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(me.per_device).map(([id, v]) => (
            <tr key={id}>
              <td>{me.device_names[id] ?? id}</td>
              <td>{cr(v.confirmed ?? 0)}</td>
              <td>{cr(v.pending ?? 0)}</td>
              <td>{cr(v.revoked ?? 0)}</td>
            </tr>
          ))}
          {Object.keys(me.per_device).length === 0 && (
            <tr>
              <td colSpan={4} className="muted">
                No rewards yet — contribute a device to a task.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <h1>Ledger entries</h1>
      <table data-testid="entries">
        <thead>
          <tr>
            <th>Time</th>
            <th>Device</th>
            <th>Work units</th>
            <th>Trust</th>
            <th>Multiplier</th>
            <th>Amount</th>
            <th>Verification</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {me.entries.map((e) => (
            <tr key={e.id}>
              <td>{new Date(e.created_at).toLocaleTimeString()}</td>
              <td>{e.device_name ?? e.device_id}</td>
              <td>{e.work_units.toLocaleString()}</td>
              <td>{e.trust.toFixed(2)}</td>
              <td>×{e.multiplier.toFixed(2)}</td>
              <td>{cr(e.amount)}</td>
              <td>{e.acceptance.replaceAll("_", " ")}</td>
              <td>
                <span className={`badge ${e.status === "confirmed" ? "succeeded" : e.status === "revoked" ? "rejected" : "pending"}`} title={e.reason ?? ""}>
                  {e.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h1>Network leaderboard</h1>
      <table data-testid="leaderboard">
        <thead>
          <tr>
            <th>Contributor</th>
            <th>Devices</th>
            <th>Confirmed</th>
            <th>Pending</th>
            <th>Revoked</th>
          </tr>
        </thead>
        <tbody>
          {net.map((r, i) => (
            <tr key={i} style={r.mine ? { fontWeight: 600 } : undefined}>
              <td>
                {r.display_name} {r.mine && <span className="badge">you</span>}
              </td>
              <td>{r.devices}</td>
              <td>{cr(r.balance.confirmed ?? 0)}</td>
              <td>{cr(r.balance.pending ?? 0)}</td>
              <td>{cr(r.balance.revoked ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h1>Ledger integrity</h1>
      <div className="card">
        <p className="muted">
          Every credit, confirmation and clawback is an append-only event. This check replays the event log from scratch
          and compares it with the current balances.
        </p>
        <button data-testid="ledger-check" onClick={() => void api.ledgerCheck().then(setCheck).catch((e: Error) => setError(e.message))}>
          Replay ledger
        </button>
        {check && (
          <p data-testid="ledger-result" className={check.consistent ? "ok" : "error"}>
            {check.consistent ? "CONSISTENT" : "MISMATCH"} — {check.events} events, {check.entries} entries, {check.users_checked} account(s)
            {check.mismatches.length > 0 && `: ${check.mismatches.join("; ")}`}
          </p>
        )}
      </div>
    </section>
  );
}

export default function RewardsPage() {
  return (
    <AuthGate>
      <Rewards />
    </AuthGate>
  );
}
