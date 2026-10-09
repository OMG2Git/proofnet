"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  describeWorld,
  formatScore,
  VISUAL_LABEL,
  type Connection,
  type Cue,
  type WorldDevice,
} from "@/lib/network/logic";
import { hex, LEGEND, STATE_STYLE } from "@/lib/network/visuals";
import type { PixelWorld } from "./pixel-world";

type Props = {
  devices: WorldDevice[];
  connection: Connection;
  error: string | null;
  /** Seconds since the last successful update (for the stale banner); null if never. */
  ageSeconds: number | null;
  selectedId: string | null;
  /** Ids that match the active filter/search; others are dimmed. null = no filtering. */
  highlightIds: Set<string> | null;
  onSelect: (id: string | null) => void;
  subscribeCues: (l: (c: Cue) => void) => () => void;
  onRetry?: () => void;
};

/** Static, accessible rendering used when canvas/WebGL is unavailable (and as a text equivalent). */
export function NetworkFallback({ devices, onSelect }: { devices: WorldDevice[]; onSelect: (id: string) => void }) {
  return (
    <div className="devices" style={{ padding: 14 }} data-testid="world-fallback">
      {devices.map((d) => (
        <button
          key={d.id}
          className={`device ${d.status} ${d.visual === "quarantined" ? "quarantined" : ""}`}
          style={{ textAlign: "left", background: "var(--card)", color: "var(--text)", fontWeight: 400 }}
          onClick={() => onSelect(d.id)}
        >
          <strong>{d.name}</strong>
          <div>
            <span className={`badge ${d.visual === "available" ? "idle" : d.visual}`}>{VISUAL_LABEL[d.visual]}</span>
          </div>
          <div className="muted">{formatScore(d.scoreCellsPerSec)}</div>
        </button>
      ))}
    </div>
  );
}

export default function PixelNetworkWorld(props: Props) {
  const { devices, connection, error, ageSeconds, selectedId, highlightIds, onSelect, subscribeCues, onRetry } = props;
  const hostRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const worldRef = useRef<PixelWorld | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  // Handlers/data the long-lived engine needs, kept current without recreating it.
  const live = useRef({ onSelect, devices });
  useEffect(() => {
    live.current = { onSelect, devices };
  });

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let cancelled = false;
    let world: PixelWorld | null = null;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onMq = () => world?.setReducedMotion(mq.matches);
    (async () => {
      try {
        const { PixelWorld } = await import("./pixel-world"); // heavy: loaded only when needed
        if (cancelled) return;
        world = new PixelWorld(host, {
          reducedMotion: mq.matches,
          onSelect: (id) => live.current.onSelect(id),
          onHover: (id, x, y) => {
            const tip = tipRef.current;
            if (!tip) return;
            const d = id ? live.current.devices.find((v) => v.id === id) : undefined;
            if (!d) {
              tip.hidden = true;
              return;
            }
            tip.hidden = false;
            tip.style.transform = `translate(${Math.round(x + 14)}px, ${Math.round(y + 14)}px)`;
            tip.textContent = `${d.name} · ${VISUAL_LABEL[d.visual]}${
              d.assignmentId ? ` · chunk ${d.chunkIndex ?? "?"}${d.taskName ? ` of ${d.taskName}` : ""}` : ""
            } · ${formatScore(d.scoreCellsPerSec)}`;
          },
        });
        worldRef.current = world;
        await world.init();
        if (cancelled) return;
        mq.addEventListener("change", onMq);
        setReady(true);
      } catch (e) {
        if (!cancelled) setFailed(e instanceof Error ? e.message : "Canvas rendering is unavailable");
      }
    })();
    return () => {
      cancelled = true;
      mq.removeEventListener("change", onMq);
      worldRef.current = null;
      world?.destroy();
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    worldRef.current?.setDevices(devices);
    worldRef.current?.setLabel(describeWorld(devices));
  }, [ready, devices]);
  useEffect(() => {
    if (ready) worldRef.current?.setSelected(selectedId);
  }, [ready, selectedId]);
  useEffect(() => {
    if (ready) worldRef.current?.setHighlight(highlightIds);
  }, [ready, highlightIds]);
  useEffect(() => {
    if (!ready) return;
    return subscribeCues((c) => worldRef.current?.pushCue(c));
  }, [ready, subscribeCues]);

  const empty = connection !== "loading" && devices.length === 0 && connection !== "down";
  const noData = devices.length === 0;

  return (
    <div>
      <div className="world-wrap" ref={hostRef} data-testid="pixel-world" data-ready={ready ? "1" : "0"}>
        {failed ? (
          <div style={{ position: "absolute", inset: 0, overflow: "auto" }}>
            <p className="warn" style={{ padding: "10px 14px" }} role="alert">
              The pixel view could not start ({failed}). Showing the plain device list instead.
            </p>
            <NetworkFallback devices={devices} onSelect={onSelect} />
          </div>
        ) : (
          <>
            <div className="world-controls" role="group" aria-label="View controls">
              <button aria-label="Zoom in" title="Zoom in (+)" onClick={() => worldRef.current?.zoomBy(1.25)}>
                +
              </button>
              <button aria-label="Zoom out" title="Zoom out (-)" onClick={() => worldRef.current?.zoomBy(0.8)}>
                −
              </button>
              <button aria-label="Fit all devices on screen" title="Fit to screen (0)" onClick={() => worldRef.current?.fit()}>
                ▣
              </button>
              <button
                aria-label="Reset view and clear selection"
                title="Reset view"
                onClick={() => {
                  onSelect(null);
                  worldRef.current?.fit();
                }}
              >
                ↺
              </button>
            </div>
            {connection === "loading" && noData && (
              <div className="world-overlay" role="status">
                <span className="live ok">
                  <i /> Connecting to ProofNet…
                </span>
              </div>
            )}
            {empty && (
              <div className="world-overlay" role="status">
                <strong>No devices registered yet</strong>
                <span className="muted">Open the contributor page on a phone or laptop to add the first node.</span>
                <Link href="/contribute" className="btn">
                  Register a device
                </Link>
              </div>
            )}
            {connection === "down" && noData && (
              <div className="world-overlay down" role="alert">
                <strong className="error">Cannot reach the ProofNet backend</strong>
                <span className="muted">{error ?? "The connection was lost."} Retrying automatically.</span>
                {onRetry && (
                  <button onClick={onRetry} className="ghost">
                    Retry now
                  </button>
                )}
              </div>
            )}
            {(connection === "stale" || (connection === "down" && !noData)) && !noData && (
              <div
                role="status"
                className={connection === "down" ? "error-box" : "sim-banner"}
                style={{ position: "absolute", left: 10, top: 10, margin: 0, padding: "4px 10px", zIndex: 4, fontSize: 12.5 }}
              >
                {connection === "down" ? "Connection lost" : "Data may be stale"}
                {ageSeconds !== null && ` · last update ${Math.round(ageSeconds)} s ago`}. Showing the last known state.
              </div>
            )}
            <div
              ref={tipRef}
              hidden
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                zIndex: 5,
                pointerEvents: "none",
                background: "rgba(8,12,20,.94)",
                border: "1px solid var(--line-2)",
                padding: "4px 8px",
                font: "12px var(--font-mono)",
                color: "var(--text)",
                maxWidth: 320,
              }}
            />
          </>
        )}
      </div>
      <p className="sr-only" role="status" aria-live="polite">
        {describeWorld(devices)}
      </p>
      <div className="world-legend" aria-label="Legend">
        {LEGEND.map((l) => (
          <span key={l.key} title={l.meaning}>
            <i style={{ ["--c" as string]: hex(l.color) }} aria-hidden="true" />
            {l.label}
            {STATE_STYLE[l.key as keyof typeof STATE_STYLE]?.glyph === "cross" && " ✕"}
          </span>
        ))}
        <span title="Cyan square: a chunk travelling to a device. Amber: a result travelling back.">
          <i style={{ ["--c" as string]: "#fbbf24" }} aria-hidden="true" />
          Packets are real assign / result events
        </span>
      </div>
    </div>
  );
}
