import { test } from "node:test";
import assert from "node:assert/strict";
import { loadTs } from "./helpers/load-ts.mjs";

const L = await loadTs("../../lib/network/logic.ts");
const T0 = Date.parse("2026-10-09T10:00:00Z");
const iso = (ms) => new Date(ms).toISOString();

const dev = (over = {}) => ({
  id: "dev_a",
  name: "Phone A",
  device_type: "android_phone",
  status: "idle",
  score_cells_per_sec: 3.2e7,
  runtime_kind: "pyodide",
  trust_status: "trusted",
  trust: 0.9,
  last_seen_age_seconds: 1,
  current_assignment_id: null,
  current_task_id: null,
  current_task_name: null,
  current_chunk_index: null,
  current_rows: null,
  ...over,
});
const summary = (devices, over = {}) => ({
  server_time: iso(T0),
  counts: {},
  devices,
  tasks_running: 0,
  ...over,
});
let n = 0;
const ev = (type, over = {}) => ({
  id: `evt_${String(++n).padStart(4, "0")}`,
  ts: iso(T0 - 1000),
  type,
  task_id: "task_1",
  device_id: "dev_a",
  chunk_id: "chk_1",
  assignment_id: "asg_1",
  data: {},
  message: type,
  ...over,
});

test("idle device is available, never computing", () => {
  const [d] = L.deriveDevices(summary([dev()]), [], T0);
  assert.equal(d.visual, "available");
  assert.equal(d.online, true);
});

test("offline and disabled devices never look online", () => {
  const [a, b] = L.deriveDevices(
    summary([dev({ id: "o", status: "offline" }), dev({ id: "x", status: "disabled" })]),
    [ev("assignment_started", { device_id: "o" })],
    T0,
  );
  assert.equal(a.visual, "offline");
  assert.equal(a.online, false);
  assert.equal(b.visual, "disabled");
  assert.equal(b.online, false);
});

test("busy without event info is just busy (no claim about computing)", () => {
  const busy = dev({ status: "busy", current_assignment_id: "asg_1" });
  assert.equal(L.deriveDevices(summary([busy]), [], T0)[0].visual, "busy");
});

test("busy + chunk_assigned => assigned; + assignment_started => computing", () => {
  const busy = dev({ status: "busy", current_assignment_id: "asg_1" });
  const assigned = ev("chunk_assigned");
  assert.equal(L.deriveDevices(summary([busy]), [assigned], T0)[0].visual, "assigned");
  const started = ev("assignment_started");
  assert.equal(L.deriveDevices(summary([busy]), [assigned, started], T0)[0].visual, "computing");
});

test("an event about a different assignment never marks the current one as computing", () => {
  const busy = dev({ status: "busy", current_assignment_id: "asg_2" });
  const old = [ev("chunk_assigned"), ev("assignment_started")]; // asg_1
  assert.equal(L.deriveDevices(summary([busy]), old, T0)[0].visual, "busy");
});

test("a finished assignment clears the computing phase", () => {
  const busy = dev({ status: "busy", current_assignment_id: "asg_1" });
  const log = [ev("chunk_assigned"), ev("assignment_started"), ev("result_accepted")];
  assert.equal(L.deriveDevices(summary([busy]), log, T0)[0].visual, "busy");
});

test("quarantine wins over everything but keeps online honest", () => {
  const q = dev({ status: "offline", trust_status: "quarantined" });
  const [d] = L.deriveDevices(summary([q]), [], T0);
  assert.equal(d.visual, "quarantined");
  assert.equal(d.online, false);
});

test("unknown benchmark and trust stay null, not zero", () => {
  const [d] = L.deriveDevices(
    summary([dev({ score_cells_per_sec: null, trust: null, trust_status: null })]),
    [],
    T0,
  );
  assert.equal(d.scoreCellsPerSec, null);
  assert.equal(d.trust, null);
  assert.equal(L.formatScore(null), "not benchmarked");
  assert.equal(L.formatAge(null), "never");
});

test("audit outcomes flash briefly and then stop flashing", () => {
  const pass = ev("result_audited", { ts: iso(T0 - 1000), data: { passed: true } });
  const fail = ev("result_audited", { ts: iso(T0 - 500), data: { passed: false } });
  assert.equal(L.deriveDevices(summary([dev()]), [pass], T0)[0].flash.kind, "verified");
  assert.equal(L.deriveDevices(summary([dev()]), [pass, fail], T0)[0].flash.kind, "rejected");
  const later = T0 + L.FLASH_MS + 1000;
  const d = L.deriveDevices(summary([dev()]), [pass], later)[0];
  assert.equal(d.flash, null);
  assert.equal(d.recentlyAudited, true);
});

test("recent failures are tracked and expire", () => {
  const f = ev("assignment_expired", { ts: iso(T0 - 1000) });
  assert.equal(L.deriveDevices(summary([dev()]), [f], T0)[0].recentFailure, true);
  assert.equal(L.deriveDevices(summary([dev()]), [f], T0 + L.RECENT_MS + 5000)[0].recentFailure, false);
});

test("mergeEvents dedupes by id, sorts out-of-order input and caps the list", () => {
  const a = ev("chunk_assigned", { ts: iso(T0 - 3000) });
  const b = ev("assignment_started", { ts: iso(T0 - 2000) });
  const c = ev("result_accepted", { ts: iso(T0 - 1000) });
  const merged = L.mergeEvents([c], [b, a, c]);
  assert.deepEqual(merged.map((e) => e.id), [a.id, b.id, c.id]);
  assert.equal(L.mergeEvents(merged, [a, b]), merged, "nothing new => same reference (no re-render)");
  assert.equal(L.mergeEvents(merged, [], 10), merged);
  assert.equal(L.mergeEvents([], [a, b, c], 2).length, 2);
  assert.deepEqual(L.mergeEvents([], [a, b, c], 2).map((e) => e.id), [b.id, c.id]);
});

test("cues map real event types and ignore unknown ones", () => {
  assert.equal(L.cueFromEvent(ev("chunk_assigned")).kind, "assign");
  assert.equal(L.cueFromEvent(ev("result_accepted")).kind, "return");
  assert.equal(L.cueFromEvent(ev("result_audited", { data: { passed: true } })).kind, "audit_pass");
  assert.equal(L.cueFromEvent(ev("result_audited", { data: { passed: false } })).kind, "audit_fail");
  assert.equal(L.cueFromEvent(ev("assignment_rejected")).kind, "audit_fail");
  assert.equal(L.cueFromEvent(ev("device_quarantined")).kind, "quarantine");
  assert.equal(L.cueFromEvent(ev("chunk_requeued")).kind, "requeue");
  assert.equal(L.cueFromEvent(ev("dataset_uploaded")), null);
  assert.equal(L.cueFromEvent(ev("training_progress")), null);
});

test("only recent events are animated; history is not replayed", () => {
  assert.equal(L.isFresh(T0 - 2000, T0), true);
  assert.equal(L.isFresh(T0 - 5 * 60_000, T0), false);
  assert.equal(L.isFresh(NaN, T0), false);
});

test("connection status: loading, live, stale, down, and recovery", () => {
  const base = { hasData: true, lastOkMs: 10_000, nowMs: 11_000, failures: 0 };
  assert.equal(L.connectionStatus({ hasData: false, lastOkMs: null, nowMs: 1, failures: 0 }), "loading");
  assert.equal(L.connectionStatus(base), "live");
  assert.equal(L.connectionStatus({ ...base, nowMs: 10_000 + L.POLL_STALE_MS + 1 }), "stale");
  assert.equal(L.connectionStatus({ ...base, failures: 1 }), "live");
  assert.equal(L.connectionStatus({ ...base, nowMs: 30_000, failures: 3 }), "down");
  assert.equal(L.connectionStatus({ hasData: false, lastOkMs: null, nowMs: 5, failures: 1 }), "stale");
  assert.equal(
    L.connectionStatus({ ...base, failures: 0, lastOkMs: 29_500, nowMs: 30_000 }),
    "live",
    "reconnect",
  );
});

test("filters and search work on real fields", () => {
  const list = L.deriveDevices(
    summary(
      [
        dev({ id: "dev_a", name: "Alpha" }),
        dev({ id: "dev_b", name: "Bravo", status: "busy", current_assignment_id: "asg_9", current_task_name: "iris" }),
        dev({ id: "dev_c", name: "Charlie", status: "offline" }),
        dev({ id: "dev_d", name: "Delta", trust_status: "quarantined" }),
      ],
      { tasks_running: 1 },
    ),
    [],
    T0,
  );
  assert.deepEqual(L.filterDevices(list, "active", "").map((d) => d.id), ["dev_a", "dev_b", "dev_d"]);
  assert.deepEqual(L.filterDevices(list, "assigned", "").map((d) => d.id), ["dev_b"]);
  assert.deepEqual(L.filterDevices(list, "quarantined", "").map((d) => d.id), ["dev_d"]);
  assert.deepEqual(L.filterDevices(list, "all", "brav").map((d) => d.id), ["dev_b"]);
  assert.deepEqual(L.filterDevices(list, "all", "IRIS").map((d) => d.id), ["dev_b"]);
  assert.deepEqual(L.filterDevices(list, "all", "dev_c").map((d) => d.id), ["dev_c"]);
  const k = L.computeKpis(summary([], { tasks_running: 1 }), list);
  assert.deepEqual(k, { registered: 4, online: 3, activeTasks: 1, activeAssignments: 1, quarantined: 1 });
});

test("empty network has a text equivalent", () => {
  assert.equal(L.describeWorld([]), "No devices are registered.");
  const list = L.deriveDevices(summary([dev(), dev({ id: "z", status: "offline" })]), [], T0);
  assert.match(L.describeWorld(list), /2 devices: 1 available \(idle\), 1 offline\./);
});
