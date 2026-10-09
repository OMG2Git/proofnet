"use client";

/**
 * Live data source for the network dashboard and pixel world.
 *
 * The backend has no WebSocket/SSE; it exposes cheap REST snapshots (cached ~1 s server-side), so
 * this polls the existing authenticated endpoints:
 *   GET /network/summary          every 1.5 s  (5 s while the tab is hidden)
 *   GET /network/events?since=…   every 2 s    (admin only; deduplicated by event id)
 * Requests are cancelled on unmount/reconnect, stale or out-of-order responses are dropped, and
 * nothing is ever simulated: if the backend cannot be reached the state says so.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiRequestError, type EventOut, type NetworkSummary } from "../api/client";
import {
  computeKpis,
  connectionStatus,
  cueFromEvent,
  deriveDevices,
  isFresh,
  mergeEvents,
  ts,
  type Connection,
  type Cue,
  type Kpis,
  type WorldDevice,
} from "./logic";

const SUMMARY_MS = 1500;
const EVENTS_MS = 2000;
const HIDDEN_MS = 5000;
const OVERLAP_MS = 2000; // re-request a little history so same-millisecond events are never missed

export type CueListener = (cue: Cue) => void;

export type NetworkData = {
  summary: NetworkSummary | null;
  devices: WorldDevice[];
  events: EventOut[];
  kpis: Kpis | null;
  connection: Connection;
  /** Client time (ms) of the last successful summary, or null. */
  lastOkMs: number | null;
  /** Seconds since the last successful summary (re-evaluated every second), or null. */
  ageSeconds: number | null;
  error: string | null;
  /** null = still checking; false = signed-in user is not an admin (no global event feed). */
  isAdmin: boolean | null;
  eventsError: string | null;
  /** Subscribe to animation cues for events that arrive live (not for history on first load). */
  subscribeCues: (l: CueListener) => () => void;
  refresh: () => void;
};

export function useNetworkData(): NetworkData {
  const [summary, setSummary] = useState<NetworkSummary | null>(null);
  const [events, setEvents] = useState<EventOut[]>([]);
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const [failures, setFailures] = useState(0);
  const [lastOkMs, setLastOkMs] = useState<number | null>(null);
  // client/server clock pair of the latest snapshot (render-safe copy of the refs below)
  const [timing, setTiming] = useState<{ clientAt: number; serverAt: number } | null>(null);
  const [clock, setClock] = useState(() => Date.now());
  const fetchedAt = useRef(0); // client time at which `summary` was received
  const serverAtFetch = useRef(0); // server time carried by that summary
  const nudge = useRef<() => void>(() => undefined);
  const listeners = useRef(new Set<CueListener>());

  const subscribeCues = useCallback((l: CueListener) => {
    listeners.current.add(l);
    return () => {
      listeners.current.delete(l);
    };
  }, []);

  // 1 s clock so flashes expire and staleness is noticed even if polling stalls.
  useEffect(() => {
    const t = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .adminMe()
      .then((r) => alive && setIsAdmin(r.admin))
      .catch(() => alive && setIsAdmin(false));
    return () => {
      alive = false;
    };
  }, []);

  // Summary polling.
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let ctl: AbortController | null = null;
    let seq = 0;
    let applied = 0;
    let lastServer = 0;

    const tick = async () => {
      const mine = ++seq;
      ctl?.abort();
      ctl = new AbortController();
      try {
        const s = await api.networkSummary({ signal: ctl.signal });
        if (!alive || mine < applied) return;
        applied = mine;
        const st = ts(s.server_time);
        if (st >= lastServer) {
          lastServer = st;
          fetchedAt.current = Date.now();
          serverAtFetch.current = st;
          setSummary(s);
          setTiming({ clientAt: fetchedAt.current, serverAt: st });
        }
        setLastOkMs(Date.now());
        setFailures(0);
        setError(null);
      } catch (e) {
        if (!alive || (e instanceof DOMException && e.name === "AbortError")) return;
        setFailures((n) => n + 1);
        setError(e instanceof ApiRequestError ? e.message : "Cannot reach the ProofNet backend");
      } finally {
        if (alive) {
          const hidden = typeof document !== "undefined" && document.hidden;
          timer = setTimeout(() => void tick(), hidden ? HIDDEN_MS : SUMMARY_MS);
        }
      }
    };
    nudge.current = () => {
      clearTimeout(timer);
      void tick();
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
      ctl?.abort();
    };
  }, []);

  // Global event feed (admin only).
  useEffect(() => {
    if (isAdmin !== true) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let ctl: AbortController | null = null;
    let last: number | null = null; // newest event ts seen
    let first = true;
    const seen = new Set<string>();

    const tick = async () => {
      ctl?.abort();
      ctl = new AbortController();
      try {
        const since = last === null ? undefined : new Date(last - OVERLAP_MS).toISOString();
        const incoming = await api.networkEvents({ since, limit: first ? 150 : 100, signal: ctl.signal });
        if (!alive) return;
        setEventsError(null);
        const fresh = incoming.filter((e) => !seen.has(e.id));
        for (const e of fresh) seen.add(e.id);
        setEvents((known) => mergeEvents(known, incoming));
        for (const e of incoming) {
          const t = ts(e.ts);
          if (!Number.isNaN(t) && (last === null || t > last)) last = t;
        }
        // History on first load is shown in the feed but never animated as if it were live.
        if (!first) {
          const serverNow = fetchedAt.current
            ? serverAtFetch.current + (Date.now() - fetchedAt.current)
            : Date.now();
          const cues = fresh.map(cueFromEvent).filter((c): c is Cue => c !== null && isFresh(c.at, serverNow));
          for (const c of cues) for (const l of listeners.current) l(c);
        }
        first = false;
      } catch (e) {
        if (!alive || (e instanceof DOMException && e.name === "AbortError")) return;
        setEventsError(e instanceof ApiRequestError ? e.message : "Cannot reach the ProofNet backend");
      } finally {
        if (alive) {
          const hidden = typeof document !== "undefined" && document.hidden;
          timer = setTimeout(() => void tick(), hidden ? HIDDEN_MS : EVENTS_MS);
        }
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
      ctl?.abort();
    };
  }, [isAdmin]);

  // Server "now" = server time at the last snapshot + time elapsed on this client since.
  const serverNow = timing ? timing.serverAt + Math.max(0, clock - timing.clientAt) : clock;
  const serverSecond = Math.floor(serverNow / 1000); // flashes expire as this ticks
  const devices = useMemo(
    () => (summary ? deriveDevices(summary, events, serverSecond * 1000) : []),
    [summary, events, serverSecond],
  );
  const kpis = useMemo(() => (summary ? computeKpis(summary, devices) : null), [summary, devices]);
  const connection = connectionStatus({ hasData: summary !== null, lastOkMs, nowMs: clock, failures });

  return {
    summary,
    devices,
    events,
    kpis,
    connection,
    lastOkMs,
    ageSeconds: lastOkMs === null ? null : Math.max(0, (clock - lastOkMs) / 1000),
    error,
    isAdmin,
    eventsError,
    subscribeCues,
    refresh: () => nudge.current(),
  };
}
