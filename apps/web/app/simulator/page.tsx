"use client";

import { useState } from "react";
import AuthGate from "@/components/AuthGate";
import LineChart from "@/components/LineChart";
import { api, type SimulateRequest, type SimulationOut } from "@/lib/api/client";

type Summary = Record<string, number | string | null>;

const DEFAULTS: SimulateRequest = {
  honest: 30,
  attackers: 6,
  rounds: 400,
  policy: "adaptive",
  alpha: 0.001,
  q0: 0.03,
  audit_floor: 0.05,
  attack_strength: 1,
  cheat_rate: 1,
  sleeper_after: 0,
  fixed_rate: 0.3,
  seed: 1,
};

const pct = (x: unknown) => (typeof x === "number" ? `${(x * 100).toFixed(1)}%` : "—");

function Simulator() {
  const [req, setReq] = useState<SimulateRequest>(DEFAULTS);
  const [out, setOut] = useState<SimulationOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const num = (k: keyof SimulateRequest, label: string, step: number | "any" = 1, min = 0, max?: number) => (
    <label>
      {label}
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={Number(req[k])}
        onChange={(e) => setReq({ ...req, [k]: Number(e.target.value) })}
      />
    </label>
  );

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setOut(await api.simulate(req));
    } catch (e) {
      setError(e instanceof Error ? e.message : "error");
    } finally {
      setBusy(false);
    }
  }

  const cmp = (out?.summary["comparison"] ?? null) as Record<string, Summary> | null;
  const s = out?.series;
  return (
    <section className="wide">
      <h1>Audit-policy simulator</h1>
      <p className="muted">
        A population model that runs the <em>same</em> PWAV decision code as the live system (audit probability, evidence
        e-detector, suspicion memory, accusation) over synthetic devices, so the policy can be studied at a scale no demo can
        reach. Device behaviour is simulated; the policy is not. Live results are on the Trust page.
      </p>
      <div className="card">
        <div className="grid">
          {num("honest", "Honest devices", 1, 1, 200)}
          {num("attackers", "Attackers", 1, 0, 100)}
          {num("rounds", "Rounds", 10, 10, 2000)}
          {num("cheat_rate", "Attacker cheat rate", 0.05, 0.05, 1)}
          {num("attack_strength", "Corruption detectable", 0.05, 0, 1)}
          {num("sleeper_after", "Sleeper: honest results first", 10, 0, 5000)}
          {num("alpha", "Lifetime false-accusation α", "any", 1e-9, 0.5)}
          {num("audit_floor", "Audit floor", 0.01, 0, 1)}
          {num("fixed_rate", "Fixed-rate baseline", 0.05, 0, 1)}
          {num("seed", "Seed", 1, 0)}
          <label>
            Policy shown
            <select value={req.policy} onChange={(e) => setReq({ ...req, policy: e.target.value })}>
              <option value="adaptive">adaptive (PWAV)</option>
              <option value="fixed">fixed rate</option>
              <option value="none">no audits</option>
            </select>
          </label>
        </div>
        <div className="row">
          <button onClick={() => void run()} disabled={busy} data-testid="sim-run">
            {busy ? "Simulating…" : "Run simulation"}
          </button>
          <button onClick={() => setReq({ ...DEFAULTS, sleeper_after: 300, rounds: 700 })}>Sleeper preset</button>
          <button onClick={() => setReq({ ...DEFAULTS, cheat_rate: 0.25, rounds: 900 })}>Intermittent preset</button>
          <button onClick={() => setReq(DEFAULTS)}>Reset</button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>

      {out && s && cmp && (
        <>
          <h1>Policies compared on the same population</h1>
          <table data-testid="sim-compare">
            <thead>
              <tr>
                <th>Policy</th>
                <th>Audit cost (share of results re-computed)</th>
                <th>Attackers caught</th>
                <th>False accusations</th>
                <th>Corrupt results merged</th>
                <th>Median / max results to catch</th>
              </tr>
            </thead>
            <tbody>
              {(["none", "fixed", "adaptive"] as const).map((p) => {
                const c = cmp[p] as Summary;
                return (
                  <tr key={p} style={p === req.policy ? { fontWeight: 600 } : undefined}>
                    <td>{p === "adaptive" ? "adaptive (PWAV)" : p === "fixed" ? `fixed ${pct(req.fixed_rate)}` : "no audits"}</td>
                    <td>{pct(c["audit_cost_fraction"])}</td>
                    <td>
                      {String(c["attackers_caught"])} / {String(c["attackers"])}
                    </td>
                    <td>{String(c["false_accusations"])}</td>
                    <td>{pct(c["corrupt_merged_fraction"])}</td>
                    <td>
                      {c["median_detection_delay_results"] ?? "—"} / {c["max_detection_delay_results"] ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="muted">
            Corrupt merged = share of cheating results that reached the final model before the cheater was caught. The
            live system adds an end-to-end reference check, forensics and retroactive audits on top, which this model does not
            include.
          </p>

          <h1>Shown policy over time: {out.summary["policy"] as string}</h1>
          <div className="legend">
            <span><i style={{ background: "#b3261e" }} />corrupt results merged (cumulative)</span>
            <span><i style={{ background: "#1c7c4a" }} />corrupt results rejected (cumulative)</span>
          </div>
          <LineChart
            series={[
              { name: "merged", color: "#b3261e", values: s["corrupt_accepted"] ?? [] },
              { name: "rejected", color: "#1c7c4a", values: s["corrupt_rejected"] ?? [] },
            ]}
            yFormat={(v) => String(Math.round(v))}
          />
          <div className="legend">
            <span><i style={{ background: "#2b59d9" }} />attackers quarantined</span>
            <span><i style={{ background: "#9a6a00" }} />honest devices quarantined (false accusations)</span>
          </div>
          <LineChart
            series={[
              { name: "attackers", color: "#2b59d9", values: s["attackers_quarantined"] ?? [] },
              { name: "honest", color: "#9a6a00", values: s["honest_quarantined"] ?? [] },
            ]}
            yFormat={(v) => String(Math.round(v))}
          />
          <div className="legend">
            <span><i style={{ background: "#1c7c4a" }} />honest devices: audit probability</span>
            <span><i style={{ background: "#b3261e" }} />attackers: audit probability</span>
          </div>
          <LineChart
            series={[
              { name: "honest", color: "#1c7c4a", values: s["honest_audit_probability"] ?? [] },
              { name: "attackers", color: "#b3261e", values: s["attacker_audit_probability"] ?? [] },
            ]}
            yMin={0}
            yMax={1}
            yFormat={(v) => `${Math.round(v * 100)}%`}
          />

          <h1>Attackers</h1>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Quarantined in round</th>
                <th>Results before that</th>
                <th>Corrupt sent</th>
                <th>Corrupt merged</th>
              </tr>
            </thead>
            <tbody>
              {out.detections.map((d) => (
                <tr key={String(d["device"])}>
                  <td>{String(d["device"])}</td>
                  <td>{d["quarantined_round"] === null ? "never" : String(d["quarantined_round"])}</td>
                  <td>{String(d["results_before"])}</td>
                  <td>{String(d["corrupt_sent"])}</td>
                  <td>{String(d["corrupt_accepted"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

export default function SimulatorPage() {
  return (
    <AuthGate>
      <Simulator />
    </AuthGate>
  );
}
