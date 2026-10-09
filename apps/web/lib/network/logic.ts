/**
 * Pure mapping from real ProofNet backend data to the view model used by the network dashboard
 * and the pixel world. No React, no I/O, no clocks: every function takes its inputs (including
 * "now") explicitly, so it is unit-testable and cannot invent state.
 *
 * Honesty rules encoded here:
 *  - A device's visual state comes from GET /network/summary (status + trust_status).
 *  - "computing" is shown only after an `assignment_started` event for that device's *current*
 *    assignment; without it a busy device is "assigned" (events seen) or "busy" (no event info).
 *  - There is no persistent "auditing" backend state: audits run synchronously when a result is
 *    accepted, so audit outcomes are brief flashes derived from `result_audited` events.
 *  - Nothing here fabricates events, devices or positions.
 */
import type { EventOut, NetworkDevice, NetworkSummary } from "../api/client";

export type DeviceVisual =
  | "initializing"
  | "available"
  | "busy"
  | "assigned"
  | "computing"
  | "offline"
  | "disabled"
  | "quarantined";

export type Flash = { kind: "verified" | "rejected"; at: number };

export type WorldDevice = {
  id: string;
  name: string;
  deviceType: string;
  /** Raw backend device status: initializing | idle | busy | offline | disabled. */
  status: string;
  online: boolean;
  visual: DeviceVisual;
  trustStatus: string | null;
  trust: number | null;
  scoreCellsPerSec: number | null;
  lastSeenAgeSeconds: number | null;
  assignmentId: string | null;
  taskId: string | null;
  taskName: string | null;
  chunkIndex: number | null;
  rows: number | null;
  /** Latest audit outcome within FLASH_MS (for a short celebratory / warning animation). */
  flash: Flash | null;
  /** An assignment failed / was rejected / expired within RECENT_MS. */
  recentFailure: boolean;
  /** A result of this device was audited within RECENT_MS. */
  recentlyAudited: boolean;
};

export const FLASH_MS = 5_000;
export const RECENT_MS = 10 * 60_000;
export const FRESH_MS = 20_000;
export const POLL_STALE_MS = 6_000;
export const MAX_EVENTS = 300;

const FAIL_TYPES = new Set(["assignment_failed", "assignment_rejected", "assignment_expired"]);

export const ts = (iso: string): number => {
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : NaN;
};

/* ------------------------------------------------------------------ events */

/** Merge new events into the known list: dedupe by id, order by (ts, id), cap the length. */
export function mergeEvents(known: EventOut[], incoming: EventOut[], cap = MAX_EVENTS): EventOut[] {
  if (incoming.length === 0) return known;
  const byId = new Map<string, EventOut>();
  for (const e of known) byId.set(e.id, e);
  let added = 0;
  for (const e of incoming) {
    if (!byId.has(e.id)) added += 1;
    byId.set(e.id, e);
  }
  if (added === 0) return known;
  const all = [...byId.values()].sort((a, b) => {
    const d = ts(a.ts) - ts(b.ts);
    if (d !== 0 && !Number.isNaN(d)) return d;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return all.length > cap ? all.slice(all.length - cap) : all;
}

export type CueKind =
  | "assign" // task packet coordinator -> device
  | "start" // device began computing
  | "return" // result packet device -> coordinator (result accepted by the backend)
  | "audit_pass"
  | "audit_fail"
  | "fail" // assignment failed / expired
  | "quarantine"
  | "online"
  | "offline"
  | "registered"
  | "requeue";

export type Cue = { id: string; kind: CueKind; deviceId: string | null; at: number; label: string };

/** Map a real backend event to an animation cue; unknown event types animate nothing. */
export function cueFromEvent(e: EventOut): Cue | null {
  const at = ts(e.ts);
  const base = { id: e.id, deviceId: e.device_id ?? null, at, label: e.message || e.type };
  switch (e.type) {
    case "chunk_assigned":
      return { ...base, kind: "assign" };
    case "assignment_started":
      return { ...base, kind: "start" };
    case "result_accepted":
      return { ...base, kind: "return" };
    case "result_audited":
      return { ...base, kind: e.data["passed"] === false ? "audit_fail" : "audit_pass" };
    case "assignment_rejected":
      return { ...base, kind: "audit_fail" };
    case "assignment_failed":
    case "assignment_expired":
      return { ...base, kind: "fail" };
    case "device_quarantined":
      return { ...base, kind: "quarantine" };
    case "device_online":
      return { ...base, kind: "online" };
    case "device_offline":
      return { ...base, kind: "offline" };
    case "device_registered":
      return { ...base, kind: "registered" };
    case "chunk_requeued":
    case "chunk_reopened":
      return { ...base, kind: "requeue" };
    default:
      return null;
  }
}

/** Only events that happened shortly before the server's "now" are animated; older ones are history. */
export const isFresh = (eventTs: number, serverNow: number): boolean =>
  Number.isFinite(eventTs) && Math.abs(serverNow - eventTs) <= FRESH_MS;

/* ------------------------------------------------------------ device state */

type Phase = { assignmentId: string; phase: "assigned" | "computing" };

/** Per-device assignment phase reconstructed from the event log (oldest -> newest). */
function phases(events: EventOut[]): Map<string, Phase> {
  const out = new Map<string, Phase>();
  for (const e of events) {
    const dev = e.device_id;
    if (!dev || !e.assignment_id) continue;
    if (e.type === "chunk_assigned") out.set(dev, { assignmentId: e.assignment_id, phase: "assigned" });
    else if (e.type === "assignment_started") out.set(dev, { assignmentId: e.assignment_id, phase: "computing" });
    else if (e.type === "result_accepted" || FAIL_TYPES.has(e.type) || e.type === "assignment_cancelled") {
      if (out.get(dev)?.assignmentId === e.assignment_id) out.delete(dev);
    }
  }
  return out;
}

function visualOf(d: NetworkDevice, phase: Phase | undefined): DeviceVisual {
  if (d.trust_status === "quarantined") return "quarantined";
  if (d.status === "offline") return "offline";
  if (d.status === "disabled") return "disabled";
  if (d.status === "initializing") return "initializing";
  if (d.status === "busy") {
    // Trust the event-derived phase only if it refers to the assignment the backend says is current.
    if (phase && phase.assignmentId === d.current_assignment_id) return phase.phase;
    return "busy";
  }
  return "available";
}

export function deriveDevices(summary: NetworkSummary, events: EventOut[], serverNow: number): WorldDevice[] {
  const ph = phases(events);
  const flashes = new Map<string, Flash>();
  const failed = new Set<string>();
  const audited = new Set<string>();
  for (const e of events) {
    const dev = e.device_id;
    const at = ts(e.ts);
    if (!dev || Number.isNaN(at)) continue;
    const age = serverNow - at;
    if (e.type === "result_audited" && age <= RECENT_MS) {
      audited.add(dev);
      if (age <= FLASH_MS) flashes.set(dev, { kind: e.data["passed"] === false ? "rejected" : "verified", at });
    }
    if (FAIL_TYPES.has(e.type) && age <= RECENT_MS) failed.add(dev);
    if (e.type === "assignment_rejected" && age <= FLASH_MS) flashes.set(dev, { kind: "rejected", at });
  }
  return summary.devices.map((d) => ({
    id: d.id,
    name: d.name,
    deviceType: d.device_type,
    status: d.status,
    online: d.status === "idle" || d.status === "busy",
    visual: visualOf(d, ph.get(d.id)),
    trustStatus: d.trust_status ?? null,
    trust: d.trust ?? null,
    scoreCellsPerSec: d.score_cells_per_sec ?? null,
    lastSeenAgeSeconds: d.last_seen_age_seconds ?? null,
    assignmentId: d.current_assignment_id ?? null,
    taskId: d.current_task_id ?? null,
    taskName: d.current_task_name ?? null,
    chunkIndex: d.current_chunk_index ?? null,
    rows: d.current_rows ?? null,
    flash: flashes.get(d.id) ?? null,
    recentFailure: failed.has(d.id),
    recentlyAudited: audited.has(d.id),
  }));
}

/* --------------------------------------------------------- filters / KPIs */

export type DeviceFilter = "all" | "active" | "assigned" | "audited" | "failed" | "quarantined";

export const FILTERS: { id: DeviceFilter; label: string; hint: string }[] = [
  { id: "all", label: "All", hint: "Every registered device" },
  { id: "active", label: "Active", hint: "Online: idle or busy" },
  { id: "assigned", label: "Assigned", hint: "Holding a chunk right now" },
  { id: "audited", label: "Audited", hint: "A result was audited in the last 10 minutes" },
  { id: "failed", label: "Failed", hint: "A failed, rejected or expired assignment in the last 10 minutes" },
  { id: "quarantined", label: "Quarantined", hint: "Isolated by the trust system" },
];

export function matchesFilter(d: WorldDevice, f: DeviceFilter): boolean {
  switch (f) {
    case "all":
      return true;
    case "active":
      return d.online;
    case "assigned":
      return d.assignmentId !== null;
    case "audited":
      return d.recentlyAudited;
    case "failed":
      return d.recentFailure;
    case "quarantined":
      return d.visual === "quarantined";
  }
}

export function filterDevices(list: WorldDevice[], f: DeviceFilter, query: string): WorldDevice[] {
  const q = query.trim().toLowerCase();
  return list.filter((d) => {
    if (!matchesFilter(d, f)) return false;
    if (!q) return true;
    return [d.id, d.name, d.deviceType, d.taskName ?? "", d.taskId ?? "", d.assignmentId ?? ""].some((s) =>
      s.toLowerCase().includes(q),
    );
  });
}

export type Kpis = {
  registered: number;
  online: number;
  activeTasks: number;
  activeAssignments: number;
  quarantined: number;
};

export function computeKpis(summary: NetworkSummary, devices: WorldDevice[]): Kpis {
  return {
    registered: devices.length,
    online: devices.filter((d) => d.online).length,
    activeTasks: summary.tasks_running,
    activeAssignments: devices.filter((d) => d.assignmentId !== null).length,
    quarantined: devices.filter((d) => d.visual === "quarantined").length,
  };
}

/* ----------------------------------------------------------- connection */

export type Connection = "loading" | "live" | "stale" | "down";

/**
 * live  : the last successful update is recent
 * stale : data exists but no update for POLL_STALE_MS (a slow network, a sleeping tab)
 * down  : 3+ consecutive failures
 * `lastOkMs`/`nowMs` are client times; staleness never mixes them with server timestamps.
 */
export function connectionStatus(s: {
  hasData: boolean;
  lastOkMs: number | null;
  nowMs: number;
  failures: number;
}): Connection {
  if (s.failures >= 3) return "down";
  if (!s.hasData || s.lastOkMs === null) return s.failures > 0 ? "stale" : "loading";
  return s.nowMs - s.lastOkMs > POLL_STALE_MS ? "stale" : "live";
}

/* ------------------------------------------------------------- formatting */

export function formatScore(cellsPerSec: number | null): string {
  if (cellsPerSec === null) return "not benchmarked";
  return `${(cellsPerSec / 1e6).toFixed(2)} M cells/s`;
}

export function formatAge(seconds: number | null): string {
  if (seconds === null) return "never";
  if (seconds < 90) return `${Math.round(seconds)} s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)} h ago`;
}

export const VISUAL_LABEL: Record<DeviceVisual, string> = {
  initializing: "Initializing",
  available: "Available (idle)",
  busy: "Busy",
  assigned: "Chunk assigned",
  computing: "Computing",
  offline: "Offline",
  disabled: "Disabled",
  quarantined: "Quarantined",
};

/** Plain-text summary of the world for assistive technology and the canvas fallback. */
export function describeWorld(devices: WorldDevice[]): string {
  if (devices.length === 0) return "No devices are registered.";
  const counts = new Map<DeviceVisual, number>();
  for (const d of devices) counts.set(d.visual, (counts.get(d.visual) ?? 0) + 1);
  const parts = [...counts.entries()].map(([v, n]) => `${n} ${VISUAL_LABEL[v].toLowerCase()}`);
  return `${devices.length} devices: ${parts.join(", ")}.`;
}

/* ------------------------------------------------------------- event feed */

export type Tone = "good" | "bad" | "run" | "warnl" | "";

export function eventTone(type: string, data: Record<string, unknown> = {}): Tone {
  if (type === "result_audited") return data["passed"] === false ? "bad" : "good";
  if (["assignment_rejected", "assignment_failed", "assignment_expired", "device_quarantined", "task_failed"].includes(type))
    return "bad";
  if (["result_accepted", "task_completed", "device_online", "device_registered"].includes(type)) return "good";
  if (["assignment_started", "task_started", "task_aggregating", "training_progress"].includes(type)) return "run";
  if (["chunk_requeued", "chunk_reopened", "device_offline", "late_result", "demo_reset"].includes(type)) return "warnl";
  return "";
}

export const EVENT_LABEL: Record<string, string> = {
  device_registered: "Device registered",
  device_online: "Device online",
  device_offline: "Device offline",
  device_enabled: "Device enabled",
  device_disabled: "Device disabled",
  device_quarantined: "Quarantined",
  device_session_started: "Session started",
  chunk_assigned: "Chunk assigned",
  assignment_started: "Computation started",
  result_accepted: "Result submitted",
  result_audited: "Audit",
  assignment_rejected: "Result rejected",
  assignment_failed: "Assignment failed",
  assignment_expired: "Assignment expired",
  assignment_cancelled: "Assignment cancelled",
  chunk_requeued: "Chunk retried",
  chunk_reopened: "Chunk reopened",
  late_result: "Late result ignored",
  task_created: "Task created",
  task_started: "Plan created",
  task_aggregating: "Merging results",
  task_completed: "Task completed",
  task_failed: "Task failed",
  task_cancelled: "Task cancelled",
  training_progress: "Training round",
};

export const eventLabel = (type: string): string => EVENT_LABEL[type] ?? type.replace(/_/g, " ");

/** Wall-clock time of the *event* (backend timestamp) in the viewer's locale. */
export function formatClock(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "—";
  return new Date(t).toLocaleTimeString([], { hour12: false });
}
