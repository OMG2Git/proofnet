import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("generated API types contain the contract models", () => {
  const dts = readFileSync(new URL("../lib/api/schema.d.ts", import.meta.url), "utf8");
  for (const name of ["TaskManifest", "HeartbeatResponse", "PartialResultEnvelope", "ApiError"]) {
    assert.ok(dts.includes(name), `${name} missing`);
  }
});
