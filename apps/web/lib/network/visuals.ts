/**
 * Single source of truth for how each real device state looks. The canvas, the legend, the table
 * badges and the screen-reader text all read this, so they cannot drift apart.
 * State is never conveyed by colour alone: every state also has a glyph and a text label.
 */
import type { DeviceVisual } from "./logic";

export const COLORS = {
  cyan: 0x22d3ee, // assigned work / network
  violet: 0xa78bfa, // active computation
  green: 0x34d399, // verified
  amber: 0xfbbf24, // pending audit / warning
  red: 0xf87171, // rejected / quarantined
  grey: 0x66748c, // offline
  text: 0xdbe4f3,
  muted: 0x8493ab,
  line: 0x1f2b40,
  bg: 0x070b12,
} as const;

export type Glyph = "arrow" | "play" | "dots" | "check" | "cross" | "none";

export type StateStyle = {
  color: number;
  glyph: Glyph;
  label: string;
  /** One-line explanation shown in the legend; says exactly which backend fact it reflects. */
  meaning: string;
};

export const STATE_STYLE: Record<DeviceVisual, StateStyle> = {
  available: { color: COLORS.green, glyph: "none", label: "Available", meaning: "Online and idle (backend status idle)" },
  assigned: { color: COLORS.cyan, glyph: "arrow", label: "Assigned", meaning: "Holds a chunk; compute not yet confirmed started" },
  computing: { color: COLORS.violet, glyph: "play", label: "Computing", meaning: "Worker reported the assignment started" },
  busy: { color: COLORS.violet, glyph: "dots", label: "Busy", meaning: "Holds a chunk (start not observed)" },
  initializing: { color: COLORS.amber, glyph: "dots", label: "Initializing", meaning: "Registered, not yet contributing" },
  offline: { color: COLORS.grey, glyph: "none", label: "Offline", meaning: "No heartbeat for 20 s" },
  disabled: { color: COLORS.grey, glyph: "none", label: "Disabled", meaning: "Disabled by its owner" },
  quarantined: { color: COLORS.red, glyph: "cross", label: "Quarantined", meaning: "Isolated by the trust system" },
};

export const FLASH_STYLE = {
  verified: { color: COLORS.green, glyph: "check" as Glyph, label: "Verified" },
  rejected: { color: COLORS.red, glyph: "cross" as Glyph, label: "Rejected" },
};

export const PACKET_STYLE = {
  assign: { color: COLORS.cyan, label: "Task chunk sent to a device" },
  return: { color: COLORS.amber, label: "Result returned (audit pending/decided)" },
};

export const hex = (n: number): string => `#${n.toString(16).padStart(6, "0")}`;

/** Legend rows in display order (only states the backend can actually produce). */
export const LEGEND: { key: string; color: number; label: string; meaning: string }[] = [
  ...(["available", "assigned", "computing", "busy", "initializing", "offline", "quarantined"] as DeviceVisual[]).map(
    (k) => ({ key: k, color: STATE_STYLE[k].color, label: STATE_STYLE[k].label, meaning: STATE_STYLE[k].meaning }),
  ),
  { key: "verified", color: COLORS.green, label: "Verified", meaning: "Recent audit passed (flash, 5 s)" },
  { key: "rejected", color: COLORS.red, label: "Rejected", meaning: "Recent audit failed / result rejected (flash, 5 s)" },
];
