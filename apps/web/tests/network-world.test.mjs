import { test } from "node:test";
import assert from "node:assert/strict";
import { loadTs } from "./helpers/load-ts.mjs";

const S = await loadTs("../../lib/network/sprite-data.ts");
const Y = await loadTs("../../lib/network/layout.ts");
const V = await loadTs("../../lib/network/visuals.ts");
const L = await loadTs("../../lib/network/logic.ts");

test("every sprite and glyph is a clean rectangle using known palette characters", () => {
  const allowed = new Set(".kgGhwSsac".split(""));
  const all = { SERVER: S.SERVER, ...S.DEVICE_SPRITES, ...S.GLYPHS };
  for (const [name, sprite] of Object.entries(all)) {
    assert.ok(S.isRect(sprite), `${name} rows differ in width`);
    for (const row of sprite) for (const ch of row) assert.ok(allowed.has(ch), `${name}: bad char ${ch}`);
  }
  assert.equal(S.spriteSize(S.SERVER).w, 20);
});

test("layout is deterministic, collision-free, and stable as devices join", () => {
  const a = Y.layout(10);
  const b = Y.layout(40);
  assert.deepEqual(b.slice(0, 10), a, "existing devices keep their slot when more join");
  const seen = new Set(b.map((s) => `${s.x},${s.y}`));
  assert.equal(seen.size, 40, "no two devices share a slot");
  assert.equal(Y.layout(0).length, 0);
  const bounds = Y.bounds(b);
  assert.ok(bounds.maxX > bounds.minX && bounds.maxY > bounds.minY);
});

test("legend covers every state the logic can produce, with a glyph or text for each", () => {
  const visuals = Object.keys(L.VISUAL_LABEL);
  for (const v of visuals) {
    const st = V.STATE_STYLE[v];
    assert.ok(st, `no style for ${v}`);
    assert.ok(st.label.length > 0 && st.meaning.length > 0);
  }
  // colour alone must never separate two states that share a colour: they differ by glyph or label
  const keys = new Set(visuals.map((v) => `${V.STATE_STYLE[v].color}|${V.STATE_STYLE[v].glyph}`));
  const labels = new Set(visuals.map((v) => V.STATE_STYLE[v].label));
  assert.equal(labels.size, visuals.length);
  assert.ok(keys.size >= visuals.length - 1);
  assert.match(V.hex(0x22d3ee), /^#22d3ee$/);
});
