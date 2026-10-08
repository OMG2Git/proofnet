"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, getApiBase, getToken, setApiBase } from "@/lib/api/client";

/**
 * Backend settings. Needs no login: when the cloud backend is down this is how you point the
 * frontend at the fallback (laptop backend + HTTPS tunnel), ARCHITECTURE 16.2. Also pre-warms the
 * free-tier backend (it sleeps after 15 minutes idle and takes about a minute to wake).
 */
export default function SettingsPage() {
  const [current, setCurrent] = useState("");
  const [input, setInput] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [resetMsg, setResetMsg] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      const base = getApiBase();
      setCurrent(base);
      setInput(base);
      if (getToken()) {
        api
          .adminMe()
          .then((r) => setIsAdmin(r.admin))
          .catch(() => undefined);
      }
    }, 0);
    return () => clearTimeout(t);
  }, []);

  async function test(url?: string) {
    setBusy(true);
    setResult("Contacting the backend (a sleeping free-tier backend needs up to a minute)…");
    const t0 = performance.now();
    try {
      const target = (url ?? getApiBase()).replace(/\/$/, "");
      const res = await fetch(`${target}/health`);
      const body = (await res.json()) as { status?: string; db?: string };
      const ms = Math.round(performance.now() - t0);
      setResult(
        res.ok
          ? `OK: backend status "${body.status}", database "${body.db}", answered in ${ms} ms`
          : `Backend answered with HTTP ${res.status}`,
      );
    } catch {
      setResult("Cannot reach that backend (check the URL, that it is running and uses HTTPS).");
    } finally {
      setBusy(false);
    }
  }

  function save() {
    const url = input.trim().replace(/\/$/, "");
    if (!/^https?:\/\//.test(url)) {
      setResult("The URL must start with https:// (or http://localhost for local development).");
      return;
    }
    setApiBase(url);
    setCurrent(url);
    void test(url);
  }

  return (
    <main>
      <h1>Backend settings</h1>
      <div className="card">
        <p>
          The frontend talks to this API (stored in this browser only):
          <br />
          <code data-testid="current-api">{current}</code>
        </p>
        <label>
          API base URL (ends with /api/v1)
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="https://…/api/v1" />
        </label>
        <div className="row">
          <button onClick={save} disabled={busy}>
            Save &amp; test
          </button>
          <button
            onClick={() => {
              setApiBase(null);
              const base = getApiBase();
              setCurrent(base);
              setInput(base);
              void test(base);
            }}
            disabled={busy}
          >
            Use the default
          </button>
          <button onClick={() => void test()} disabled={busy} data-testid="warm">
            Test / pre-warm
          </button>
        </div>
        {result && <p data-testid="result">{result}</p>}
        <p className="muted">
          Fallback during a demo: run the backend on a laptop, expose it with a free HTTPS tunnel, paste the
          tunnel URL here (with <code>/api/v1</code>), and sign in again. Devices registered against another
          backend must be registered again. See DEMO_RUNBOOK.md.
        </p>
      </div>

      {isAdmin && (
        <div className="card">
          <h1>Admin</h1>
          <p className="muted">
            Clears every task, dataset, result and uploaded file. Accounts and registered devices are kept.
          </p>
          <button
            data-testid="reset"
            onClick={() => {
              if (!window.confirm("Delete ALL tasks, datasets and results? This cannot be undone.")) return;
              api
                .resetDemo()
                .then((r) => setResetMsg(`Reset done: ${JSON.stringify(r.deleted)}`))
                .catch((e: Error) => setResetMsg(e.message));
            }}
          >
            Reset demo data
          </button>
          {resetMsg && <p className="muted">{resetMsg}</p>}
        </div>
      )}
      <p>
        <Link href="/">← Home</Link>
      </p>
    </main>
  );
}
