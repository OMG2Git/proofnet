/**
 * Presentation-only placement of devices around the coordinator. Positions are NOT facts about
 * where devices are: they are slots on concentric rings, assigned by the order the backend lists
 * devices (creation order), so a device keeps its slot as others join. Unit: world pixels.
 */
export type Slot = { x: number; y: number; ring: number };

const RING_CAP = [6, 12, 18, 24];
const BASE_R = 280;
const RING_STEP = 170;
const SQUASH = 0.62; // isometric-style vertical squash

export function ringOf(index: number): { ring: number; pos: number; count: number } {
  let rest = index;
  for (let r = 0; ; r++) {
    const cap = RING_CAP[r] ?? RING_CAP[RING_CAP.length - 1]! + 6 * (r - RING_CAP.length + 1);
    if (rest < cap) return { ring: r, pos: rest, count: cap };
    rest -= cap;
  }
}

export function slotFor(index: number): Slot {
  const { ring, pos, count } = ringOf(index);
  const angle = -Math.PI / 2 + (pos / count) * Math.PI * 2 + (ring % 2 ? Math.PI / count : 0);
  const r = BASE_R + ring * RING_STEP;
  return { x: Math.round(Math.cos(angle) * r), y: Math.round(Math.sin(angle) * r * SQUASH), ring };
}

/** Slots for the first n devices. The first k slots never depend on n. */
export function layout(n: number): Slot[] {
  return Array.from({ length: n }, (_, i) => slotFor(i));
}

export function ringRadii(n: number): { rx: number; ry: number }[] {
  if (n === 0) return [];
  const rings = ringOf(n - 1).ring + 1;
  return Array.from({ length: rings }, (_, r) => ({
    rx: BASE_R + r * RING_STEP,
    ry: (BASE_R + r * RING_STEP) * SQUASH,
  }));
}

export function bounds(slots: Slot[], pad = 90): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = -pad;
  let maxX = pad;
  let minY = -pad;
  let maxY = pad;
  for (const s of slots) {
    minX = Math.min(minX, s.x - pad);
    maxX = Math.max(maxX, s.x + pad);
    minY = Math.min(minY, s.y - pad);
    maxY = Math.max(maxY, s.y + pad);
  }
  return { minX, minY, maxX, maxY };
}
