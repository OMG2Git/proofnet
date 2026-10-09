/**
 * Pixel-art sprite definitions (pure data, no rendering). Each sprite is a list of equal-width
 * strings; characters map to a palette at render time:
 *   .  transparent        k  outline          g  body            G  body (light)
 *   h  highlight          w  white detail     S  screen          s  screen shade
 *   a  accent light       c  glyph colour
 * `S`, `s`, `a` and `c` take the colour of the device state, so one sprite serves every state.
 */

export type Sprite = readonly string[];

const rep = (ch: string, n: number): string => ch.repeat(n);

function serverRack(): Sprite {
  const W = 20;
  const inner = W - 4;
  const frame = (mid: string) => `.k${mid}k.`;
  const edge = `..${rep("k", inner)}..`;
  const sep = frame(rep("g", inner));
  const plain = frame(`g${rep("G", inner - 2)}g`);
  const unit = frame(`gGa${rep("G", inner - 8)}hhGGg`); // status light on the left, vent on the right
  const rows: string[] = [edge, sep];
  for (let i = 0; i < 4; i++) rows.push(plain, unit, plain, sep);
  rows.push(edge, `...kk${rep(".", W - 10)}kk...`, `..kkkk${rep(".", W - 12)}kkkk..`);
  return rows;
}

export const SERVER: Sprite = serverRack();

export const PHONE: Sprite = [
  "..kkkkkk..",
  ".kggggggk.",
  ".kgSSSSgk.",
  ".kgSSSSgk.",
  ".kgSsSSgk.",
  ".kgSSSSgk.",
  ".kgSSSsgk.",
  ".kgSSSSgk.",
  ".kgSSSSgk.",
  ".kgSSSSgk.",
  ".kggggggk.",
  ".kggwwggk.",
  ".kggggggk.",
  "..kkkkkk..",
];

export const LAPTOP: Sprite = [
  "..kkkkkkkkkkkk..",
  "..kggggggggggk..",
  "..kgSSSSSSSSgk..",
  "..kgSSSSSSSSgk..",
  "..kgSsSSSSSSgk..",
  "..kgSSSSSSSSgk..",
  "..kgSSSSSsSSgk..",
  "..kggggggggggk..",
  ".kkkkkkkkkkkkkk.",
  "kGGGGGGGGGGGGGGk",
  "kGhhhhhhhhhhhhGk",
  ".kkkkkkkkkkkkkk.",
];

export const DESKTOP: Sprite = [
  ".kkkkkkkkkkkkkk.",
  ".kggggggggggggk.",
  ".kgSSSSSSSSSSgk.",
  ".kgSSSSSSSSSSgk.",
  ".kgSSsSSSSSSSgk.",
  ".kgSSSSSSSSSSgk.",
  ".kgSSSSSSsSSSgk.",
  ".kgSSSSSSSSSSgk.",
  ".kgggggggwggggk.",
  ".kkkkkkkkkkkkkk.",
  ".......kk.......",
  "......kGGk......",
  ".....kGGGGk.....",
  "...kkkkkkkkkk...",
];

export const CHIP: Sprite = [
  "..k..k..k..k..",
  ".kkkkkkkkkkkk.",
  "kkggggggggggkk",
  ".kgSSSSSSSSgk.",
  "kkgSSSSSSSSgkk",
  ".kgSSsSSSSSgk.",
  "kkgSSSSSSSSgkk",
  ".kgSSSSsSSSgk.",
  "kkgSSSSSSSSgkk",
  ".kgSSSSSSSSgk.",
  "kkggggggggggkk",
  ".kkkkkkkkkkkk.",
  "..k..k..k..k..",
];

export const DEVICE_SPRITES: Record<string, Sprite> = {
  android_phone: PHONE,
  laptop: LAPTOP,
  desktop: DESKTOP,
  other: CHIP,
};

export const GLYPHS: Record<"arrow" | "play" | "dots" | "check" | "cross", Sprite> = {
  arrow: [".......", "...c...", "....c..", "ccccccc", "....c..", "...c...", "......."],
  play: [".c.....", ".cc....", ".ccc...", ".cccc..", ".ccc...", ".cc....", ".c....."],
  dots: [".......", ".......", ".c.c.c.", ".......", ".......", ".......", "......."],
  check: [".......", ".....cc", "....cc.", "cc.cc..", ".ccc...", "..c....", "......."],
  cross: [".......", ".cc.cc.", "..ccc..", "...c...", "..ccc..", ".cc.cc.", "......."],
};

export function spriteSize(s: Sprite): { w: number; h: number } {
  return { w: s[0]?.length ?? 0, h: s.length };
}

export function isRect(s: Sprite): boolean {
  const w = s[0]?.length ?? 0;
  return s.length > 0 && s.every((r) => r.length === w);
}
