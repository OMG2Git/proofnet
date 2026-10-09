/**
 * PixelWorld: the PixiJS 8 scene behind <PixelNetworkWorld>. Framework-free so React only owns
 * its lifecycle. Everything drawn comes from real data handed in via setDevices()/pushCue():
 *  - a device exists in the scene iff the backend lists it;
 *  - its look comes from WorldDevice.visual (see lib/network/logic.ts);
 *  - packets/pulses are spawned only by cues derived from real backend events.
 * Positions are presentation-only slots (lib/network/layout.ts), not geography.
 */
import { Application, Container, Graphics, Rectangle, Sprite, Text, Texture } from "pixi.js";
import type { Cue, WorldDevice } from "../../lib/network/logic";
import { bounds, layout, ringRadii } from "../../lib/network/layout";
import { DEVICE_SPRITES, GLYPHS, SERVER, spriteSize, type Sprite as SpriteData } from "../../lib/network/sprite-data";
import { COLORS, FLASH_STYLE, STATE_STYLE, type Glyph } from "../../lib/network/visuals";

const SCALE = 4; // device sprite pixel size
const SERVER_SCALE = 4;
const FONT = "ui-monospace, 'IBM Plex Mono', Menlo, Consolas, monospace";
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3;
const PACKET_MS = 850;
const MAX_EFFECTS = 80;

export type WorldOptions = {
  reducedMotion: boolean;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null, screenX: number, screenY: number) => void;
};

const rgb = (n: number): [number, number, number] => [(n >> 16) & 255, (n >> 8) & 255, n & 255];
const css = (n: number, k = 1): string => {
  const [r, g, b] = rgb(n);
  return `rgb(${Math.round(r * k)},${Math.round(g * k)},${Math.round(b * k)})`;
};

type Effect = {
  g: Graphics | Text;
  start: number;
  dur: number;
  step: (t: number) => void; // t in [0,1]
  done?: () => void;
};

class DeviceNode {
  root = new Container();
  private platform = new Graphics();
  private sprite = new Sprite();
  private glyph = new Sprite();
  private bits = new Graphics();
  private frame = new Graphics();
  private chunk = new Container();
  private chunkBox = new Graphics();
  private chunkText: Text;
  private name: Text;
  private state: Text;
  data!: WorldDevice;
  sig = "";

  constructor(
    readonly id: string,
    readonly x: number,
    readonly y: number,
    private tex: (kind: string, color: number, dim: boolean) => Texture,
    private glyphTex: (g: Glyph, color: number) => Texture | null,
  ) {
    this.root.position.set(x, y);
    this.root.zIndex = y; // nearer (lower on screen) devices draw on top
    this.root.eventMode = "static";
    this.root.cursor = "pointer";
    this.root.hitArea = new Rectangle(-46, -100, 92, 146);
    this.root.label = id;
    this.sprite.anchor.set(0.5, 1);
    this.sprite.scale.set(SCALE);
    this.glyph.anchor.set(0.5, 0.5);
    this.glyph.scale.set(3);
    this.glyph.position.set(36, -74);
    this.name = new Text({ text: "", style: { fontFamily: FONT, fontSize: 11, fill: COLORS.text } });
    this.name.anchor.set(0.5, 0);
    this.name.position.set(0, 26);
    this.state = new Text({ text: "", style: { fontFamily: FONT, fontSize: 9, fill: COLORS.muted, letterSpacing: 1 } });
    this.state.anchor.set(0.5, 0);
    this.state.position.set(0, 41);
    this.chunkText = new Text({ text: "", style: { fontFamily: FONT, fontSize: 9, fill: COLORS.cyan } });
    this.chunkText.anchor.set(0, 0.5);
    this.chunkText.position.set(14, 0);
    this.chunk.position.set(-66, -74);
    this.chunk.addChild(this.chunkBox, this.chunkText);
    this.root.addChild(this.platform, this.frame, this.sprite, this.bits, this.glyph, this.chunk, this.name, this.state);
  }

  update(d: WorldDevice, selected: boolean, dimmed: boolean): boolean {
    const style = STATE_STYLE[d.visual];
    const flash = d.flash ? FLASH_STYLE[d.flash.kind] : null;
    const sig = [d.visual, d.name, d.deviceType, d.online, flash?.label ?? "", d.chunkIndex, d.assignmentId, selected, dimmed].join("|");
    this.data = d;
    if (sig === this.sig) return false;
    this.sig = sig;
    const color = flash ? flash.color : style.color;
    const dim = !d.online || d.visual === "disabled";
    this.sprite.texture = this.tex(d.deviceType, d.online || d.visual === "quarantined" ? color : COLORS.grey, dim);
    this.sprite.alpha = dim ? 0.55 : 1;
    this.root.alpha = dimmed ? 0.2 : 1;

    // platform (isometric diamond) in the state colour
    this.platform.clear();
    this.platform
      .poly([0, -10, 46, 8, 0, 26, -46, 8])
      .fill({ color: 0x0b1220, alpha: 0.95 })
      .stroke({ width: 2, color: dim ? COLORS.grey : color, alpha: dim ? 0.5 : 0.9 });

    // selection brackets / quarantine isolation box
    this.frame.clear();
    if (d.visual === "quarantined") {
      dashRect(this.frame, -54, -106, 108, 150, COLORS.red);
    }
    if (selected) {
      const c = COLORS.cyan;
      const L = 12;
      const x0 = -58;
      const y0 = -112;
      const x1 = 58;
      const y1 = 60;
      this.frame
        .moveTo(x0, y0 + L).lineTo(x0, y0).lineTo(x0 + L, y0)
        .moveTo(x1 - L, y0).lineTo(x1, y0).lineTo(x1, y0 + L)
        .moveTo(x0, y1 - L).lineTo(x0, y1).lineTo(x0 + L, y1)
        .moveTo(x1 - L, y1).lineTo(x1, y1).lineTo(x1, y1 - L)
        .stroke({ width: 2, color: c });
    }

    // state glyph: shape, not just colour
    const g: Glyph = flash ? flash.glyph : style.glyph;
    const gt = this.glyphTex(g, color);
    this.glyph.visible = gt !== null;
    if (gt) this.glyph.texture = gt;

    // chunk indicator = the real assignment this device holds
    this.chunk.visible = d.assignmentId !== null;
    if (d.assignmentId !== null) {
      this.chunkBox.clear().rect(0, -6, 10, 12).fill({ color: COLORS.cyan, alpha: 0.2 }).stroke({ width: 1, color: COLORS.cyan });
      this.chunkText.text = d.chunkIndex !== null ? `c${d.chunkIndex}` : "chunk";
    }

    this.name.text = d.name.length > 16 ? `${d.name.slice(0, 15)}…` : d.name;
    this.state.text = (flash ? flash.label : style.label).toUpperCase();
    this.state.style.fill = color;
    return true;
  }

  /** Rising bits while (and only while) the device is computing. */
  animateBits(now: number, on: boolean): void {
    this.bits.clear();
    if (!on) return;
    for (let i = 0; i < 4; i++) {
      const p = ((now / 1100 + i / 4) % 1);
      this.bits
        .rect(-22 + i * 15 + ((i * 7) % 5), -58 - p * 40, 4, 4)
        .fill({ color: COLORS.violet, alpha: 1 - p });
    }
  }
  staticBits(on: boolean): void {
    this.bits.clear();
    if (on) for (let i = 0; i < 3; i++) this.bits.rect(-15 + i * 15, -66, 4, 4).fill({ color: COLORS.violet });
  }
}

function dashRect(g: Graphics, x: number, y: number, w: number, h: number, color: number): void {
  const D = 6;
  const seg = (x1: number, y1: number, x2: number, y2: number) => {
    const len = Math.hypot(x2 - x1, y2 - y1);
    for (let t = 0; t < len; t += D * 2) {
      const a = t / len;
      const b = Math.min(t + D, len) / len;
      g.moveTo(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a).lineTo(x1 + (x2 - x1) * b, y1 + (y2 - y1) * b);
    }
  };
  seg(x, y, x + w, y);
  seg(x + w, y, x + w, y + h);
  seg(x + w, y + h, x, y + h);
  seg(x, y + h, x, y);
  g.stroke({ width: 2, color, alpha: 0.9 });
}

export class PixelWorld {
  private app = new Application();
  private world = new Container();
  private floor = new Graphics();
  private links = new Graphics();
  private nodesLayer = new Container();
  private fxLayer = new Container();
  private coordinator = new Sprite();
  private coordLabel: Text;
  private nodes = new Map<string, DeviceNode>();
  private textures = new Map<string, Texture>();
  private effects: Effect[] = [];
  private order: string[] = [];
  private selected: string | null = null;
  private highlight: Set<string> | null = null;
  private devices: WorldDevice[] = [];
  private ready = false;
  private destroyed = false;
  private userMoved = false;
  private ro: ResizeObserver | null = null;
  private pointers = new Map<number, { x: number; y: number }>();
  private pinchDist = 0;
  private dragMoved = false;
  private raf = 0;
  private ringKey = "";
  private canvasEl: HTMLCanvasElement | null = null;
  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const r = this.app.canvas.getBoundingClientRect();
    this.zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  };
  private onKey = (e: KeyboardEvent) => {
    const step = 48;
    let handled = true;
    if (e.key === "ArrowLeft") this.panBy(step, 0);
    else if (e.key === "ArrowRight") this.panBy(-step, 0);
    else if (e.key === "ArrowUp") this.panBy(0, step);
    else if (e.key === "ArrowDown") this.panBy(0, -step);
    else if (e.key === "+" || e.key === "=") this.zoomBy(1.2);
    else if (e.key === "-" || e.key === "_") this.zoomBy(1 / 1.2);
    else if (e.key === "0") this.fit();
    else if (e.key === "Escape") this.opts.onSelect(null);
    else handled = false;
    if (handled) e.preventDefault();
  };

  constructor(
    private host: HTMLElement,
    private opts: WorldOptions,
  ) {
    this.coordLabel = new Text({ text: "PROOFNET COORDINATOR", style: { fontFamily: FONT, fontSize: 10, fill: COLORS.cyan, letterSpacing: 1 } });
    this.coordLabel.anchor.set(0.5, 0);
  }

  /** Throws if WebGL/canvas is unavailable; the caller shows the static fallback. */
  async init(): Promise<void> {
    const w = Math.max(this.host.clientWidth, 200);
    const h = Math.max(this.host.clientHeight, 200);
    await this.app.init({
      width: w,
      height: h,
      backgroundAlpha: 0,
      antialias: false,
      autoDensity: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      roundPixels: true,
      autoStart: false,
      powerPreference: "low-power",
    });
    if (this.destroyed) {
      this.app.destroy({ removeView: true }, { children: true, texture: true, textureSource: true });
      return;
    }
    const canvas = this.app.canvas;
    this.canvasEl = canvas;
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img"); // text equivalent: setLabel(), plus the device table in the DOM
    canvas.setAttribute("aria-label", "ProofNet network visualization");
    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    canvas.addEventListener("keydown", this.onKey);
    this.host.appendChild(canvas);
    this.app.ticker.maxFPS = 30; // retro look, half the battery
    this.app.stage.addChild(this.world);
    this.world.addChild(this.floor, this.links, this.nodesLayer, this.fxLayer);
    this.coordinator.texture = this.spriteTexture("server", COLORS.cyan, false);
    this.coordinator.anchor.set(0.5, 1);
    this.coordinator.scale.set(SERVER_SCALE);
    this.coordinator.position.set(0, 24);
    this.coordLabel.position.set(0, 98);
    this.coordinator.zIndex = 24;
    this.coordLabel.zIndex = 98;
    this.nodesLayer.sortableChildren = true;
    this.nodesLayer.addChild(this.coordinator, this.coordLabel);

    const stage = this.app.stage;
    stage.eventMode = "static";
    stage.hitArea = this.app.screen;
    stage.on("pointerdown", (e) => {
      this.pointers.set(e.pointerId, { x: e.global.x, y: e.global.y });
      this.dragMoved = false;
      if (this.pointers.size === 2) this.pinchDist = this.pinchSpan();
    });
    stage.on("pointermove", (e) => {
      const p = this.pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.global.x - p.x;
      const dy = e.global.y - p.y;
      p.x = e.global.x;
      p.y = e.global.y;
      if (this.pointers.size === 1) {
        if (Math.abs(dx) + Math.abs(dy) > 0) {
          if (!this.dragMoved && Math.hypot(dx, dy) < 2) return;
          this.dragMoved = true;
          this.panBy(dx, dy);
        }
      } else if (this.pointers.size === 2) {
        const span = this.pinchSpan();
        if (this.pinchDist > 0) {
          const mid = this.pinchMid();
          this.zoomAt(span / this.pinchDist, mid.x, mid.y);
        }
        this.pinchDist = span;
        this.dragMoved = true;
      }
    });
    const up = (e: { pointerId: number }) => {
      this.pointers.delete(e.pointerId);
      this.pinchDist = 0;
      if (this.pointers.size === 0) setTimeout(() => (this.dragMoved = false), 0);
    };
    stage.on("pointerup", up);
    stage.on("pointerupoutside", up);
    stage.on("pointercancel", up);
    stage.on("pointertap", (e) => {
      if (!this.dragMoved && e.target === stage) this.opts.onSelect(null);
    });

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.host);
    this.ready = true;
    this.rebuild();
    this.fit();
    this.invalidate();
  }

  /* ------------------------------------------------------------ textures */

  private spriteTexture(kind: string, color: number, dim: boolean): Texture {
    const key = `${kind}|${color}|${dim}`;
    const hit = this.textures.get(key);
    if (hit) return hit;
    const data: SpriteData = kind === "server" ? SERVER : (DEVICE_SPRITES[kind] ?? DEVICE_SPRITES["other"]!);
    const pal: Record<string, string> = {
      k: "#05080e",
      g: dim ? "#232c3f" : "#2c3956",
      G: dim ? "#2e3850" : "#41527a",
      h: "#6b7fab",
      w: "#dbe4f3",
      S: css(color, dim ? 0.55 : 1),
      s: css(color, 0.55),
      a: css(color, 1.15),
      c: css(color),
    };
    const tex = this.paint(data, pal);
    this.textures.set(key, tex);
    return tex;
  }

  private glyphTexture(g: Glyph, color: number): Texture | null {
    if (g === "none") return null;
    const key = `glyph|${g}|${color}`;
    const hit = this.textures.get(key);
    if (hit) return hit;
    const tex = this.paint(GLYPHS[g], { c: css(color) });
    this.textures.set(key, tex);
    return tex;
  }

  private paint(data: SpriteData, pal: Record<string, string>): Texture {
    const { w, h } = spriteSize(data);
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext("2d");
    if (!ctx) throw new Error("2D canvas unavailable");
    data.forEach((row, y) => {
      for (let x = 0; x < row.length; x++) {
        const ch = row[x]!;
        const fill = pal[ch];
        if (ch === "." || !fill) continue;
        ctx.fillStyle = fill;
        ctx.fillRect(x, y, 1, 1);
      }
    });
    const tex = Texture.from(cv);
    tex.source.scaleMode = "nearest";
    return tex;
  }

  /* --------------------------------------------------------------- scene */

  setDevices(list: WorldDevice[]): void {
    this.devices = list;
    if (!this.ready) return;
    const ids = list.map((d) => d.id);
    if (ids.length !== this.order.length || ids.some((id, i) => id !== this.order[i])) this.rebuild();
    else this.refresh();
  }

  setSelected(id: string | null): void {
    if (this.selected === id) return;
    this.selected = id;
    if (this.ready) this.refresh();
  }

  setHighlight(ids: Set<string> | null): void {
    this.highlight = ids;
    if (this.ready) this.refresh();
  }

  setReducedMotion(v: boolean): void {
    this.opts.reducedMotion = v;
    this.refresh();
  }

  private rebuild(): void {
    for (const n of this.nodes.values()) n.root.destroy({ children: true });
    this.nodes.clear();
    this.order = this.devices.map((d) => d.id);
    const slots = layout(this.order.length);
    this.order.forEach((id, i) => {
      const s = slots[i]!;
      const node = new DeviceNode(
        id,
        s.x,
        s.y,
        (k, c, d) => this.spriteTexture(k, c, d),
        (g, c) => this.glyphTexture(g, c),
      );
      node.root.on("pointertap", () => {
        if (!this.dragMoved) this.opts.onSelect(id);
      });
      node.root.on("pointerover", (e) => this.opts.onHover(id, e.global.x, e.global.y));
      node.root.on("pointermove", (e) => this.opts.onHover(id, e.global.x, e.global.y));
      node.root.on("pointerout", () => this.opts.onHover(null, 0, 0));
      this.nodes.set(id, node);
      this.nodesLayer.addChild(node.root);
    });
    this.drawFloor();
    this.refresh(true);
    if (!this.userMoved) this.fit();
  }

  private refresh(force = false): void {
    let changed = force;
    for (const d of this.devices) {
      const n = this.nodes.get(d.id);
      if (!n) continue;
      const dimmed = this.highlight !== null && !this.highlight.has(d.id);
      if (n.update(d, this.selected === d.id, dimmed)) changed = true;
    }
    if (changed) this.drawLinks();
    const computing = this.devices.some((d) => d.visual === "computing");
    for (const n of this.nodes.values()) n.staticBits(this.opts.reducedMotion && n.data.visual === "computing");
    if (computing && !this.opts.reducedMotion) this.startTicker();
    this.invalidate();
  }

  private drawFloor(): void {
    this.floor.clear();
    const key = String(this.order.length);
    if (key === this.ringKey) return;
    this.ringKey = key;
    const N = 16;
    for (let gx = -N; gx <= N; gx++) {
      for (let gy = -N; gy <= N; gy++) {
        const x = (gx - gy) * 44;
        const y = (gx + gy) * 27;
        this.floor.rect(x - 1, y - 1, 2, 2).fill({ color: COLORS.cyan, alpha: 0.12 });
      }
    }
    for (const { rx, ry } of ringRadii(this.order.length)) {
      this.floor.ellipse(0, 0, rx, ry).stroke({ width: 1, color: COLORS.cyan, alpha: 0.1 });
    }
    this.floor.poly([0, 18, 74, 54, 0, 90, -74, 54]).fill({ color: 0x0b1220 }).stroke({ width: 2, color: COLORS.cyan, alpha: 0.6 });
  }

  private drawLinks(): void {
    const g = this.links;
    g.clear();
    for (const d of this.devices) {
      const n = this.nodes.get(d.id);
      if (!n) continue;
      const x1 = 0;
      const y1 = 36;
      const x2 = n.x;
      const y2 = n.y + 6;
      const v = d.visual;
      if (v === "quarantined") {
        // isolated: the link is cut in the middle
        segLine(g, x1, y1, x1 + (x2 - x1) * 0.42, y1 + (y2 - y1) * 0.42, 6, 6, COLORS.red, 0.5, 2);
        segLine(g, x1 + (x2 - x1) * 0.58, y1 + (y2 - y1) * 0.58, x2, y2, 6, 6, COLORS.red, 0.25, 2);
        const mx = x1 + (x2 - x1) * 0.5;
        const my = y1 + (y2 - y1) * 0.5;
        g.moveTo(mx - 5, my - 5).lineTo(mx + 5, my + 5).moveTo(mx + 5, my - 5).lineTo(mx - 5, my + 5).stroke({ width: 2, color: COLORS.red });
      } else if (v === "computing" || v === "busy") {
        segLine(g, x1, y1, x2, y2, 8, 0, COLORS.violet, 0.85, 2);
      } else if (v === "assigned") {
        segLine(g, x1, y1, x2, y2, 8, 0, COLORS.cyan, 0.85, 2);
      } else if (v === "available") {
        segLine(g, x1, y1, x2, y2, 4, 8, COLORS.cyan, 0.35, 1);
      } else if (v === "initializing") {
        segLine(g, x1, y1, x2, y2, 3, 9, COLORS.amber, 0.35, 1);
      } else {
        segLine(g, x1, y1, x2, y2, 2, 14, COLORS.grey, 0.18, 1);
      }
    }
  }

  /* ------------------------------------------------------------- effects */

  pushCue(cue: Cue): void {
    if (!this.ready || !cue.deviceId) return;
    const n = this.nodes.get(cue.deviceId);
    if (!n) return;
    const reduced = this.opts.reducedMotion;
    switch (cue.kind) {
      case "assign":
        this.packet(n, true, COLORS.cyan, () => this.pulse(n.x, n.y - 24, COLORS.cyan));
        break;
      case "start":
        this.pulse(n.x, n.y - 24, COLORS.violet);
        break;
      case "return":
        this.packet(n, false, COLORS.amber, () => this.pulse(0, 0, COLORS.amber));
        break;
      case "audit_pass":
        this.pulse(n.x, n.y - 24, COLORS.green);
        this.popup(n.x, n.y - 124, "✓ VERIFIED", COLORS.green);
        break;
      case "audit_fail":
        this.pulse(n.x, n.y - 24, COLORS.red);
        this.popup(n.x, n.y - 124, "✕ REJECTED", COLORS.red);
        break;
      case "fail":
        this.pulse(n.x, n.y - 24, COLORS.red);
        this.popup(n.x, n.y - 124, "✕ FAILED", COLORS.red);
        break;
      case "quarantine":
        this.pulse(n.x, n.y - 24, COLORS.red);
        if (!reduced) this.pulse(n.x, n.y - 24, COLORS.red, 220);
        this.popup(n.x, n.y - 124, "QUARANTINED", COLORS.red);
        break;
      case "online":
        this.pulse(n.x, n.y - 24, COLORS.green);
        break;
      case "offline":
        this.pulse(n.x, n.y - 24, COLORS.grey);
        break;
      case "registered":
        this.pulse(n.x, n.y - 24, COLORS.cyan);
        this.popup(n.x, n.y - 124, "NEW DEVICE", COLORS.cyan);
        break;
      default:
        break;
    }
  }

  private addEffect(e: Effect): void {
    if (this.effects.length >= MAX_EFFECTS) {
      const old = this.effects.shift();
      old?.g.destroy();
    }
    this.effects.push(e);
    this.fxLayer.addChild(e.g);
    this.startTicker();
  }

  private packet(n: DeviceNode, outbound: boolean, color: number, arrive: () => void): void {
    const reduced = this.opts.reducedMotion;
    const a = { x: 0, y: 36 };
    const b = { x: n.x, y: n.y + 6 };
    const from = outbound ? a : b;
    const to = outbound ? b : a;
    const g = new Graphics();
    g.rect(-4, -4, 8, 8).fill({ color: 0x05080e }).rect(-3, -3, 6, 6).fill({ color });
    g.position.set(reduced ? to.x : from.x, reduced ? to.y : from.y);
    this.addEffect({
      g,
      start: performance.now(),
      dur: reduced ? 600 : PACKET_MS,
      step: (t) => {
        if (reduced) return;
        const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // ease in-out
        g.position.set(Math.round(from.x + (to.x - from.x) * e), Math.round(from.y + (to.y - from.y) * e));
      },
      done: arrive,
    });
  }

  private pulse(x: number, y: number, color: number, delay = 0): void {
    const reduced = this.opts.reducedMotion;
    const g = new Graphics();
    g.position.set(x, y);
    const start = performance.now() + delay;
    if (reduced) g.rect(-26, -26, 52, 52).stroke({ width: 2, color });
    g.visible = reduced && delay === 0;
    this.addEffect({
      g,
      start,
      dur: reduced ? 900 : 700,
      step: (t) => {
        if (reduced) return;
        g.visible = true;
        const r = 14 + t * 40;
        g.clear().rect(-r, -r * 0.8, r * 2, r * 1.6).stroke({ width: 2, color, alpha: 1 - t });
      },
    });
  }

  private popup(x: number, y: number, text: string, color: number): void {
    const reduced = this.opts.reducedMotion;
    const t = new Text({ text, style: { fontFamily: FONT, fontSize: 11, fill: color, fontWeight: "bold", letterSpacing: 1 } });
    t.anchor.set(0.5, 1);
    t.position.set(x, y);
    this.addEffect({
      g: t,
      start: performance.now(),
      dur: reduced ? 1600 : 1500,
      step: (p) => {
        if (reduced) return;
        t.position.set(x, y - p * 18);
        t.alpha = p < 0.7 ? 1 : 1 - (p - 0.7) / 0.3;
      },
    });
  }

  /* -------------------------------------------------------------- ticker */

  private tickFn = () => {
    const now = performance.now();
    const keep: Effect[] = [];
    for (const e of this.effects) {
      const t = (now - e.start) / e.dur;
      if (t < 0) {
        keep.push(e);
        continue;
      }
      if (t >= 1) {
        e.g.destroy();
        e.done?.();
        continue;
      }
      e.step(t);
      keep.push(e);
    }
    this.effects = keep;
    const computing = !this.opts.reducedMotion;
    for (const n of this.nodes.values()) n.animateBits(now, computing && n.data.visual === "computing");
    const anyComputing = computing && this.devices.some((d) => d.visual === "computing");
    if (this.effects.length === 0 && !anyComputing) {
      // nothing left to animate: draw this last frame, then sleep (saves CPU/battery)
      this.app.ticker.stop();
    }
  };

  private startTicker(): void {
    if (!this.ready || this.destroyed) return;
    if (!this.app.ticker.started) {
      this.app.ticker.remove(this.tickFn);
      this.app.ticker.add(this.tickFn);
      this.app.ticker.start();
    }
  }

  private invalidate(): void {
    if (!this.ready || this.destroyed || this.app.ticker.started || this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (!this.destroyed) this.app.render();
    });
  }

  /* --------------------------------------------------------- camera/zoom */

  private pinchSpan(): number {
    const [a, b] = [...this.pointers.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }
  private pinchMid(): { x: number; y: number } {
    const [a, b] = [...this.pointers.values()];
    return a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } : { x: 0, y: 0 };
  }

  private panBy(dx: number, dy: number): void {
    this.userMoved = true;
    this.world.position.set(this.world.x + dx, this.world.y + dy);
    this.invalidate();
  }

  private zoomAt(factor: number, px: number, py: number): void {
    const s = this.world.scale.x;
    const ns = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, s * factor));
    if (ns === s) return;
    this.userMoved = true;
    this.world.position.set(px - ((px - this.world.x) * ns) / s, py - ((py - this.world.y) * ns) / s);
    this.world.scale.set(ns);
    this.invalidate();
  }

  zoomBy(f: number): void {
    this.zoomAt(f, this.app.screen.width / 2, this.app.screen.height / 2);
  }

  /** Fit all devices on screen and forget manual pan/zoom. */
  fit(): void {
    if (!this.ready) return;
    this.userMoved = false;
    const b = bounds(layout(this.order.length), 110);
    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const bw = b.maxX - b.minX;
    const bh = b.maxY - b.minY + 80;
    const s = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.min(w / bw, h / bh, 1.6)));
    this.world.scale.set(s);
    this.world.position.set(Math.round(w / 2 - ((b.minX + b.maxX) / 2) * s), Math.round(h / 2 - ((b.minY + b.maxY) / 2 + 20) * s));
    this.invalidate();
  }

  private resize(): void {
    if (!this.ready || this.destroyed) return;
    const w = Math.max(this.host.clientWidth, 100);
    const h = Math.max(this.host.clientHeight, 100);
    this.app.renderer.resize(w, h);
    this.app.stage.hitArea = this.app.screen;
    if (!this.userMoved) this.fit();
    else this.invalidate();
  }

  setLabel(text: string): void {
    this.canvasEl?.setAttribute("aria-label", text);
  }

  focus(): void {
    this.canvasEl?.focus();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    cancelAnimationFrame(this.raf);
    this.ro?.disconnect();
    if (this.ready) {
      this.app.ticker.remove(this.tickFn);
      this.app.ticker.stop();
      this.canvasEl?.removeEventListener("wheel", this.onWheel);
      this.canvasEl?.removeEventListener("keydown", this.onKey);
      this.app.stage.removeAllListeners();
      for (const t of this.textures.values()) t.destroy(true);
      this.textures.clear();
      this.effects = [];
      this.app.destroy({ removeView: true }, { children: true, texture: true, textureSource: true });
    }
    // if init() is still pending it destroys the app itself when it resumes
  }
}

function segLine(
  g: Graphics,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  dash: number,
  gap: number,
  color: number,
  alpha: number,
  width: number,
): void {
  const len = Math.hypot(x2 - x1, y2 - y1);
  if (len === 0) return;
  if (gap === 0) {
    g.moveTo(x1, y1).lineTo(x2, y2).stroke({ width, color, alpha });
    return;
  }
  const ux = (x2 - x1) / len;
  const uy = (y2 - y1) / len;
  for (let t = 0; t < len; t += dash + gap) {
    const e = Math.min(t + dash, len);
    g.moveTo(Math.round(x1 + ux * t), Math.round(y1 + uy * t)).lineTo(Math.round(x1 + ux * e), Math.round(y1 + uy * e));
  }
  g.stroke({ width, color, alpha });
}
