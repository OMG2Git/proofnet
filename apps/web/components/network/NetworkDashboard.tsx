"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api/client";
import { FILTERS, filterDevices, type DeviceFilter } from "@/lib/network/logic";
import { useNetworkData } from "@/lib/network/use-network-data";
import { ActivityFeed, DeviceTable, KpiStrip, type AuditTotals, type DeviceExtras } from "./panels";
import WorkerDrawer from "./WorkerDrawer";

// The canvas stack (PixiJS) is the heavy part of this page: load it on demand, client-side only.
const PixelNetworkWorld = dynamic(() => import("./PixelNetworkWorld"), {
  ssr: false,
  loading: () => (
    <div className="world-wrap" role="status">
      <div className="world-overlay">
        <span className="live ok">
          <i /> Loading the pixel world…
        </span>
      </div>
    </div>
  ),
});

function LiveBadge({ connection, age }: { connection: string; age: number | null }) {
  const cls = connection === "live" ? "ok" : connection === "down" ? "down" : "stale";
  const text =
    connection === "live"
      ? "LIVE"
      : connection === "down"
        ? "DISCONNECTED"
        : connection === "stale"
          ? "STALE"
          : "CONNECTING";
  return (
    <span className={`live ${cls}`} role="status" data-testid="live-badge" data-connection={connection}>
      <i aria-hidden="true" />
      {text}
      {age !== null && connection !== "loading" && <span>· updated {Math.round(age)} s ago</span>}
    </span>
  );
}

export default function NetworkDashboard() {
  const net = useNetworkData();
  const { devices, events, isAdmin } = net;
  const [filter, setFilter] = useState<DeviceFilter>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [big, setBig] = useState(false);
  const [audit, setAudit] = useState<AuditTotals>(null);
  const [rewards, setRewards] = useState<DeviceExtras["rewards"]>(new Map());

  // Slow-moving figures (lifetime audit totals; the viewer's own reward balances).
  useEffect(() => {
    let alive = true;
    const load = () => {
      api
        .trustOverview()
        .then((t) => alive && setAudit({ audits: t.audits, results: t.results_seen }))
        .catch(() => undefined);
      api
        .myRewards()
        .then((r) => {
          if (!alive) return;
          const m = new Map<string, { confirmed: number; pending: number }>();
          for (const [id, b] of Object.entries(r.per_device)) m.set(id, { confirmed: b.confirmed, pending: b.pending });
          for (const id of Object.keys(r.device_names)) if (!m.has(id)) m.set(id, { confirmed: 0, pending: 0 });
          setRewards(m);
        })
        .catch(() => undefined);
    };
    load();
    const t = setInterval(load, 10_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const visible = useMemo(() => filterDevices(devices, filter, query), [devices, filter, query]);
  const highlight = useMemo(
    () => (filter === "all" && !query.trim() ? null : new Set(visible.map((d) => d.id))),
    [visible, filter, query],
  );
  const extras = useMemo<DeviceExtras>(() => {
    const assignedAt = new Map<string, string>();
    for (const e of events) if (e.type === "chunk_assigned" && e.assignment_id) assignedAt.set(e.assignment_id, e.ts);
    return { assignedAt, rewards };
  }, [events, rewards]);
  const selected = selectedId ? (devices.find((d) => d.id === selectedId) ?? null) : null;

  return (
    <section className={big ? "wide bigscreen" : "wide"}>
      <h1>
        Network
        <LiveBadge connection={net.connection} age={net.ageSeconds} />
        <span className="spacer" />
        <button className="ghost small" data-testid="bigtoggle" onClick={() => setBig((b) => !b)} aria-pressed={big}>
          {big ? "Normal view" : "Big screen"}
        </button>
      </h1>

      <KpiStrip kpis={net.kpis} audit={audit} />

      <div className="card flush" style={{ marginTop: 0 }}>
        <div className="toolbar">
          <div className="chips" role="group" aria-label="Filter devices">
            {FILTERS.map((f) => (
              <button key={f.id} className="chip" aria-pressed={filter === f.id} title={f.hint} onClick={() => setFilter(f.id)}>
                {f.label}
                {filter === f.id && ` (${visible.length})`}
              </button>
            ))}
          </div>
          <span className="spacer" />
          <label className="sr-only" htmlFor="device-search">
            Search devices
          </label>
          <input
            id="device-search"
            type="search"
            placeholder="Search name, id, task…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <PixelNetworkWorld
          devices={devices}
          connection={net.connection}
          error={net.error}
          ageSeconds={net.ageSeconds}
          selectedId={selectedId}
          highlightIds={highlight}
          onSelect={setSelectedId}
          subscribeCues={net.subscribeCues}
          onRetry={net.refresh}
        />
      </div>

      <div className={isAdmin === false ? "dash" : "dash two"} style={{ marginTop: 16 }}>
        <div className="card flush" style={{ marginTop: 0 }}>
          <div className="panel-title">
            Device assignments
            <span className="spacer" />
            <span className="muted" style={{ textTransform: "none", letterSpacing: 0 }}>
              {visible.length} of {devices.length}
            </span>
          </div>
          <DeviceTable devices={visible} selectedId={selectedId} onSelect={setSelectedId} extras={extras} />
        </div>
        <div className="card flush" style={{ marginTop: 0 }}>
          <div className="panel-title">Task activity</div>
          <ActivityFeed events={events} isAdmin={isAdmin} error={net.eventsError} onSelectDevice={setSelectedId} />
        </div>
      </div>

      {selected && (
        <WorkerDrawer
          device={selected}
          events={events}
          isAdmin={isAdmin === true}
          onClose={() => setSelectedId(null)}
          onChanged={net.refresh}
        />
      )}
    </section>
  );
}
