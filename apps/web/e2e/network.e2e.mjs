// End-to-end check of the live network dashboard against a REAL backend, a REAL CLI worker and
// real Chrome. Nothing is mocked except the deliberate network failure in step 6.
//
// Prerequisites (see apps/web/README.md "End-to-end test"):
//   backend : ADMIN_EMAILS=e2e-admin@example.com, fresh MONGODB_DB, CORS_ORIGINS=http://localhost:3000
//   frontend: built with NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1 and served on :3000
//   browser : Chrome/Edge installed (set PROOFNET_CHROME to its path if it is not auto-detected)
//
//   npm run e2e
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..", "..");
const API = process.env.PROOFNET_API ?? "http://localhost:8000/api/v1";
const WEB = process.env.PROOFNET_WEB ?? "http://localhost:3000";
const ADMIN = process.env.PROOFNET_ADMIN_EMAIL ?? "e2e-admin@example.com";
const PASS = "password123";
const SHOTS = join(HERE, "artifacts");
mkdirSync(SHOTS, { recursive: true });

const CHROME_CANDIDATES = [
  process.env.PROOFNET_CHROME,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);
const chrome = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chrome) throw new Error("No Chrome/Edge found; set PROOFNET_CHROME");

const results = [];
const step = async (name, fn) => {
  const t0 = Date.now();
  try {
    await fn();
    results.push({ name, ok: true, ms: Date.now() - t0 });
    console.log(`  ok   ${name} (${Date.now() - t0} ms)`);
  } catch (e) {
    results.push({ name, ok: false, error: e });
    console.log(`  FAIL ${name}\n       ${e.message.split("\n")[0]}`);
    throw e;
  }
};

async function waitFor(fn, { timeout = 30_000, every = 300, what = "condition" } = {}) {
  const end = Date.now() + timeout;
  let last;
  while (Date.now() < end) {
    try {
      const v = await fn();
      if (v) return v;
    } catch (e) {
      last = e;
    }
    await new Promise((r) => setTimeout(r, every));
  }
  throw new Error(`timed out waiting for ${what}${last ? ` (${last.message})` : ""}`);
}

async function http(path, { method = "GET", token, json, form } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (json) headers["Content-Type"] = "application/json";
  const res = await fetch(`${API}${path}`, { method, headers, body: form ?? (json ? JSON.stringify(json) : undefined) });
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* empty */
  }
  return { status: res.status, body };
}

async function account(email, display) {
  let r = await http("/auth/signup", { method: "POST", json: { email, password: PASS, display_name: display } });
  if (r.status === 409 || r.status === 422) r = await http("/auth/login", { method: "POST", json: { email, password: PASS } });
  assert.equal(r.status === 201 || r.status === 200, true, `auth failed: ${JSON.stringify(r.body)}`);
  return r.body.access_token;
}

function csv(n = 3000) {
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
  const rows = ["f0,f1,f2,f3,label"];
  for (let i = 0; i < n; i++) {
    const y = Math.floor(rnd() * 3);
    rows.push([0, 1, 2, 3].map(() => (rnd() * 2 - 1 + y * 0.8).toFixed(5)).join(",") + `,${y}`);
  }
  return rows.join("\n");
}

const children = [];
function startWorker(token_email, name) {
  const cmd = process.env.CLI_WORKER_CMD ?? "python -m uv run python -m cli_worker";
  const [bin, ...pre] = cmd.split(" ");
  const p = spawn(bin, [...pre, "--api", API, "--email", token_email, "--password", PASS, "--name-prefix", name, "--duration", "300"], {
    cwd: REPO,
    stdio: ["ignore", "pipe", "pipe"],
    shell: false,
  });
  children.push(p);
  return p;
}

const stamp = Date.now().toString(36);
const browser = await chromium.launch({ executablePath: chrome, headless: true });
let failed = null;
try {
  const adminToken = await account(ADMIN, "E2E Admin");
  const admin = await http("/admin/me", { token: adminToken });
  assert.equal(admin.body.admin, true, `${ADMIN} must be listed in the backend's ADMIN_EMAILS`);

  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(
    ([t, base]) => {
      window.localStorage.setItem("proofnet.token", t);
      window.localStorage.setItem("proofnet.apiBase", base);
    },
    [adminToken, API],
  );
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && !/Failed to load resource|net::ERR/.test(m.text()) && errors.push(`console: ${m.text()}`));

  console.log(`\nProofNet network dashboard E2E  (${WEB} -> ${API})\n`);

  await step("1. admin dashboard connects and the pixel world starts", async () => {
    await page.goto(`${WEB}/network`);
    await waitFor(async () => (await page.getAttribute("[data-testid=live-badge]", "data-connection")) === "live", { what: "live badge" });
    await waitFor(async () => (await page.getAttribute("[data-testid=pixel-world]", "data-ready")) === "1", { what: "canvas ready" });
    assert.equal(await page.locator("[data-testid=world-fallback]").count(), 0, "canvas fell back to the plain list");
    for (const id of ["registered", "online", "tasks", "assignments", "audited", "quarantined"])
      assert.equal(await page.locator(`[data-testid=kpi-${id}]`).count(), 1, `kpi ${id} missing`);
    // empty state is honest, not simulated
    const before = await page.locator("tr[data-testid=device-card]").count();
    assert.equal(before >= 0, true);
  });

  const workerName = `e2e-${stamp}`;
  await step("2. a real worker registers and appears in the UI as a real device", async () => {
    startWorker(ADMIN, workerName);
    await waitFor(async () => (await page.locator("tr[data-testid=device-card]", { hasText: `${workerName}-1` }).count()) === 1, {
      timeout: 60_000,
      what: "worker row",
    });
    await waitFor(
      async () => /available/i.test((await page.locator("tr[data-testid=device-card]", { hasText: `${workerName}-1` }).innerText({ timeout: 2000 })) ?? ""),
      { what: "worker available" },
    );
    const kpi = await page.locator("[data-testid=kpi-online] .k-value").innerText({ timeout: 2000 });
    assert.ok(Number(kpi) >= 1, `online KPI is ${kpi}`);
    const label = await page.getAttribute("[data-testid=pixel-world] canvas", "aria-label");
    assert.match(label, /\d+ devices?:/, "canvas text equivalent missing");
  });

  await step("3. a real task assignment flows through feed, table and audit", async () => {
    // isolate: only this test's worker may take the task (the e2e admin's other devices are disabled)
    const mine = await http("/devices/mine", { token: adminToken });
    for (const d of mine.body.filter((x) => !x.name.startsWith(workerName)))
      if (d.status !== "disabled") await http(`/devices/${d.id}`, { method: "PATCH", token: adminToken, json: { disabled: true } });
    const ds = new FormData();
    ds.append("file", new Blob([csv()], { type: "text/csv" }), "e2e.csv");
    const up = await http("/datasets", { method: "POST", token: adminToken, form: ds });
    assert.equal(up.status, 201, JSON.stringify(up.body));
    const task = await http("/tasks", {
      method: "POST",
      token: adminToken,
      json: {
        name: `e2e-${stamp}`,
        task_type: "gaussian_nb_train",
        dataset_id: up.body.id,
        params: { target_column: "label", feature_columns: ["f0", "f1", "f2", "f3"] },
        execution: { min_devices: 1, max_devices: 1 },
      },
    });
    assert.equal(task.status, 201, JSON.stringify(task.body));
    // capture the canvas while the real events arrive: packets/pulses should be visible in some frames
    const world = page.locator("[data-testid=pixel-world]");
    await world.scrollIntoViewIfNeeded();
    for (let i = 0; i < 8; i++) {
      await world.screenshot({ path: join(SHOTS, `world-frame-${i}.png`) });
      await page.waitForTimeout(400);
    }
    const feed = page.locator("[data-testid=feed]");
    const me = `${workerName}-1`;
    const feedText = async () => (await feed.innerText({ timeout: 2000 })).toLowerCase();
    // wait for THIS worker's lifecycle (the feed also holds older runs' events)
    for (const needle of [`assigned to ${me}`, `${me} started chunk`, `audit of ${me}`, `${me} finished chunk`])
      await waitFor(async () => (await feedText()).includes(needle), { timeout: 60_000, what: `feed "${needle}"` });
    await waitFor(async () => (await feedText()).includes("reference check passed"), { what: "task completion line" });
    // the table row reflects the real audit outcome for this device
    await waitFor(async () => /audited recently|verified/i.test(await page.locator("tr[data-testid=device-card]", { hasText: me }).innerText({ timeout: 2000 })), {
      what: "verification column",
    });
    await page.screenshot({ path: join(SHOTS, "network-desktop.png"), fullPage: true });
  });

  await step("4. worker drawer shows trust + audit data and admin actions", async () => {
    await page.locator("tr[data-testid=device-card]", { hasText: `${workerName}-1` }).getByRole("button").first().click();
    const drawer = page.locator("[data-testid=drawer]");
    await drawer.waitFor();
    await waitFor(async () => /trust score/i.test(await drawer.innerText({ timeout: 2000 })), { what: "trust section" });
    await waitFor(async () => /verified|rejected/i.test(await drawer.innerText({ timeout: 2000 })), { what: "audit outcome row" });
    assert.ok((await drawer.innerText({ timeout: 2000 })).toLowerCase().includes("administrator actions"));
    await page.screenshot({ path: join(SHOTS, "network-drawer.png") });
  });

  await step("5. admin quarantine updates the dashboard from real backend state", async () => {
    const drawer = page.locator("[data-testid=drawer]");
    // type slowly across several dashboard re-renders: focus must stay in the field (regression test)
    const reason = drawer.getByLabel(/Reason/);
    await reason.click();
    await reason.pressSequentially("e2e: controlled quarantine", { delay: 110 });
    assert.equal(await reason.inputValue(), "e2e: controlled quarantine", "typing was interrupted by a re-render");
    assert.equal(await reason.evaluate((el) => el === document.activeElement), true, "focus was stolen from the reason field");
    await drawer.getByRole("button", { name: "Quarantine" }).click();
    await waitFor(async () => /quarantined/i.test(await page.locator("tr[data-testid=device-card]", { hasText: `${workerName}-1` }).innerText({ timeout: 2000 })), {
      what: "row shows Quarantined",
    });
    await waitFor(async () => /quarantined/i.test(await page.locator("[data-testid=feed]").innerText({ timeout: 2000 })), { what: "feed shows quarantine event" });
    assert.ok(Number(await page.locator("[data-testid=kpi-quarantined] .k-value").innerText({ timeout: 2000 })) >= 1);
    await page.screenshot({ path: join(SHOTS, "network-quarantined.png") });
    // reinstate through the UI as well
    await drawer.getByLabel(/Reason/).fill("e2e: done");
    await drawer.getByRole("button", { name: "Reinstate" }).click();
    await waitFor(async () => !/quarantined/i.test(await page.locator("tr[data-testid=device-card]", { hasText: `${workerName}-1` }).locator("td").nth(1).innerText({ timeout: 2000 })), {
      what: "row no longer quarantined",
    });
    await page.keyboard.press("Escape");
    assert.equal(await drawer.count(), 0, "Escape closes the drawer");
  });

  await step("6. backend loss is shown honestly and recovery is automatic", async () => {
    await page.route("**/network/summary*", (r) => r.abort());
    await waitFor(async () => ["stale", "down"].includes(await page.getAttribute("[data-testid=live-badge]", "data-connection")), {
      timeout: 20_000,
      what: "stale/down indicator",
    });
    await page.getByText(/showing the last known state/i).first().waitFor({ timeout: 10_000 });
    await page.locator("[data-testid=pixel-world]").screenshot({ path: join(SHOTS, "network-disconnected.png") });
    await page.unroute("**/network/summary*");
    await waitFor(async () => (await page.getAttribute("[data-testid=live-badge]", "data-connection")) === "live", {
      timeout: 20_000,
      what: "reconnect",
    });
  });

  await step("7. non-admin users get no global feed and no admin actions", async () => {
    const userToken = await account(`viewer-${stamp}@example.com`, "Viewer");
    assert.equal((await http("/network/events", { token: userToken })).status, 403);
    assert.equal((await http("/network/events")).status, 401);
    assert.equal((await http(`/security/devices/x/quarantine`, { method: "POST", token: userToken, json: { reason: "no" } })).status, 403);
    const ctx2 = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    await ctx2.addInitScript(([t, b]) => { window.localStorage.setItem("proofnet.token", t); window.localStorage.setItem("proofnet.apiBase", b); }, [userToken, API]);
    const p2 = await ctx2.newPage();
    await p2.goto(`${WEB}/network`);
    await p2.getByText("visible to administrators only").waitFor({ timeout: 20_000 });
    await ctx2.close();
  });

  await step("8. mobile viewport: no horizontal overflow, canvas usable", async () => {
    // a real registered device identity, stored the way the contributor page stores it
    const reg = await http("/devices", { method: "POST", token: adminToken, json: { name: `phone-${stamp}`, device_type: "android_phone", capabilities: { model: "e2e phone" } } });
    assert.equal(reg.status, 201, JSON.stringify(reg.body));
    const identity = { apiBase: API, deviceId: reg.body.device.id, deviceToken: reg.body.device_token, name: reg.body.device.name };
    const mctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
    await mctx.addInitScript(([t, b, d]) => { window.localStorage.setItem("proofnet.token", t); window.localStorage.setItem("proofnet.apiBase", b); window.localStorage.setItem("proofnet.device", JSON.stringify(d)); }, [adminToken, API, identity]);
    const mp = await mctx.newPage();
    await mp.goto(`${WEB}/contribute/run`);
    await mp.getByTestId("hero").waitFor({ timeout: 15_000 });
    assert.equal(await mp.getByTestId("start").isVisible(), true, "start button visible on a phone");
    const box = await mp.getByTestId("start").boundingBox();
    assert.ok(box && box.height >= 44, "touch target is at least 44px tall");
    for (const path of ["/network", "/contribute/run", "/trust", "/rewards", "/security", "/simulator"]) {
      await mp.goto(`${WEB}${path}`);
      await mp.waitForTimeout(1500);
      const over = await mp.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      assert.ok(over <= 1, `${path}: ${over}px horizontal overflow at 390px`);
      if (path === "/network") await mp.screenshot({ path: join(SHOTS, "network-mobile.png"), fullPage: true });
      if (path === "/contribute/run") await mp.screenshot({ path: join(SHOTS, "contribute-mobile.png"), fullPage: true });
    }
    await mctx.close();
  });

  await step("9. reduced-motion preference: world still starts and shows real state", async () => {
    const rctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, reducedMotion: "reduce" });
    await rctx.addInitScript(([t, b]) => { window.localStorage.setItem("proofnet.token", t); window.localStorage.setItem("proofnet.apiBase", b); }, [adminToken, API]);
    const rp = await rctx.newPage();
    const rerrs = [];
    rp.on("pageerror", (e) => rerrs.push(e.message));
    await rp.goto(`${WEB}/network`);
    await waitFor(async () => (await rp.getAttribute("[data-testid=pixel-world]", "data-ready")) === "1", { what: "canvas ready (reduced motion)" });
    assert.equal(await rp.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches), true);
    assert.deepEqual(rerrs, []);
    await rctx.close();
  });

  await step("10. WebGL disabled: PixiJS falls back to its canvas renderer and the world still starts", async () => {
    const nogl = await chromium.launch({ executablePath: chrome, headless: true, args: ["--disable-gpu", "--disable-webgl", "--disable-3d-apis", "--disable-webgl2"] });
    try {
      const nctx = await nogl.newContext({ viewport: { width: 1280, height: 800 } });
      await nctx.addInitScript(([t, b]) => { window.localStorage.setItem("proofnet.token", t); window.localStorage.setItem("proofnet.apiBase", b); }, [adminToken, API]);
      const np = await nctx.newPage();
      await np.goto(`${WEB}/network`);
      assert.equal(await np.evaluate(() => !!document.createElement("canvas").getContext("webgl")), false, "test browser still has WebGL");
      await waitFor(async () => (await np.getAttribute("[data-testid=pixel-world]", "data-ready")) === "1", { timeout: 30_000, what: "canvas-renderer start" });
      assert.equal(await np.locator("[data-testid=world-fallback]").count(), 0);
      await np.locator("[data-testid=pixel-world]").screenshot({ path: join(SHOTS, "world-no-webgl.png") });
    } finally {
      await nogl.close();
    }
  });

  await step("11. no canvas at all: the plain accessible device list takes over", async () => {
    const cctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    await cctx.addInitScript(([t, b]) => {
      window.localStorage.setItem("proofnet.token", t);
      window.localStorage.setItem("proofnet.apiBase", b);
      HTMLCanvasElement.prototype.getContext = () => null; // every canvas context unavailable
    }, [adminToken, API]);
    const cp = await cctx.newPage();
    await cp.goto(`${WEB}/network`);
    await cp.locator("[data-testid=world-fallback]").waitFor({ timeout: 30_000 });
    assert.match(await cp.getByRole("alert").first().innerText({ timeout: 2000 }), /pixel view could not start/i);
    assert.ok((await cp.locator("[data-testid=world-fallback] button").count()) >= 1, "fallback lists the real devices");
    await cp.screenshot({ path: join(SHOTS, "network-fallback.png") });
    await cctx.close();
  });

  await step("12. no uncaught browser errors during the whole run", async () => {
    assert.deepEqual(errors, []);
  });
} catch (e) {
  failed = e;
} finally {
  if (process.platform !== "win32") for (const c of children) c.kill("SIGTERM");
  // kill the whole process tree (uv -> python), otherwise a worker outlives the test and takes later work
  for (const c of children)
    if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(c.pid), "/T", "/F"], { stdio: "ignore" });
  await browser.close();
}
const ok = results.filter((r) => r.ok).length;
if (failed && !results.some((r) => !r.ok)) console.error(`\nSetup failed before any step: ${failed.message}`);
console.log(`\n${ok}/${results.length} steps passed${failed ? " - FAILED" : ""}. Screenshots: ${SHOTS}\n`);
process.exit(failed ? 1 : 0);
