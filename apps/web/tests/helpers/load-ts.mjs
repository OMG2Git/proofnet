// Bundle a TypeScript module with esbuild and import it, so node:test can exercise pure TS logic
// without adding a test framework. Type-only imports are erased by esbuild.
import { build } from "esbuild";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export async function loadTs(entry) {
  const out = join(mkdtempSync(join(tmpdir(), "proofnet-test-")), "bundle.mjs");
  await build({
    entryPoints: [fileURLToPath(new URL(entry, import.meta.url))],
    bundle: true,
    format: "esm",
    platform: "node",
    outfile: out,
    logLevel: "silent",
    external: ["react", "pixi.js"],
  });
  return import(pathToFileURL(out).href);
}
