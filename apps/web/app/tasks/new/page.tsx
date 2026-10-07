"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import {
  api,
  ApiRequestError,
  type DatasetOut,
  type TaskManifest,
  type TaskTypeInfo,
  type ValidationResult,
} from "@/lib/api/client";

type Column = { name: string; dtype: string };

type PreviewShare = {
  device_name: string | null;
  score_cells_per_sec: number;
  weight: number;
  rows: number;
  n_chunks: number;
  est_seconds: number;
};
type Preview = {
  n_train: number;
  eligible_devices: number;
  min_devices: number;
  ready_to_start: boolean;
  message?: string;
  explanation?: string;
  shares: PreviewShare[];
  not_eligible: string[];
};

function PlanPreview({ pv }: { pv: Preview }) {
  return (
    <div data-testid="preview">
      <p>
        <strong>Plan preview</strong>{" "}
        <span className="muted">
          (estimate from the devices online now; ~{pv.n_train} training rows; {pv.eligible_devices} eligible device(s))
        </span>
      </p>
      {!pv.ready_to_start && <p className="warn">{pv.message}</p>}
      {pv.shares.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Device</th>
              <th>Benchmark</th>
              <th>Weight</th>
              <th>Rows</th>
              <th>Chunks</th>
              <th>Predicted compute</th>
            </tr>
          </thead>
          <tbody>
            {pv.shares.map((s, i) => (
              <tr key={i}>
                <td>{s.device_name}</td>
                <td>{(s.score_cells_per_sec / 1e6).toFixed(2)} M cells/s</td>
                <td>{(s.weight * 100).toFixed(1)}%</td>
                <td>{s.rows}</td>
                <td>{s.n_chunks}</td>
                <td>{s.est_seconds < 1 ? `${Math.round(s.est_seconds * 1000)} ms` : `${s.est_seconds.toFixed(2)} s`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {pv.explanation && <p className="muted">{pv.explanation}</p>}
      {pv.not_eligible.length > 0 && (
        <details>
          <summary className="muted">Devices not eligible right now ({pv.not_eligible.length})</summary>
          <ul className="muted">
            {pv.not_eligible.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function Wizard() {
  const router = useRouter();
  const [types, setTypes] = useState<TaskTypeInfo[]>([]);
  const [taskType, setTaskType] = useState<string>("gaussian_nb_train");
  const [dataset, setDataset] = useState<DatasetOut | null>(null);
  const [name, setName] = useState("My task");
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [testFraction, setTestFraction] = useState(0.2);
  const [splitSeed, setSplitSeed] = useState(42);
  const [special, setSpecial] = useState<number>(1e-9); // var_smoothing | alpha
  const [dropMissing, setDropMissing] = useState(true);
  const [minDevices, setMinDevices] = useState(1);
  const [maxDevices, setMaxDevices] = useState(4);
  const [report, setReport] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .taskTypes()
      .then(setTypes)
      .catch((e: Error) => setError(e.message));
  }, []);

  const isGnb = taskType === "gaussian_nb_train";
  const columns = ((dataset?.profile["columns"] ?? []) as Column[]) || [];

  function onTypeChange(t: string) {
    setTaskType(t);
    setSpecial(t === "gaussian_nb_train" ? 1e-9 : 1.0);
    setReport(null);
  }

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const ds = await api.uploadDataset(file);
      setDataset(ds);
      const cols = ds.profile["columns"] as Column[];
      const last = cols[cols.length - 1];
      setTarget(last?.name ?? "");
      setFeatures(cols.filter((c) => c.name !== last?.name && c.dtype === "numeric").map((c) => c.name));
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  function manifest(): TaskManifest | null {
    if (!dataset) return null;
    const common = {
      target_column: target,
      feature_columns: features,
      missing_values: dropMissing ? ("drop_rows" as const) : ("reject" as const),
      test_fraction: testFraction,
      split_seed: splitSeed,
    };
    return {
      name,
      task_type: taskType as TaskManifest["task_type"],
      kernel_version: "1",
      dataset_id: dataset.id,
      params: isGnb ? { ...common, var_smoothing: special } : { ...common, alpha: special },
      execution: {
        min_devices: minDevices,
        max_devices: maxDevices,
        start_policy: "wait_for_min_devices",
      },
    };
  }

  async function run(kind: "validate" | "submit") {
    const m = manifest();
    if (!m) return;
    setBusy(true);
    setError(null);
    try {
      if (kind === "validate") setReport(await api.validateTask(m));
      else {
        const t = await api.createTask(m);
        router.push(`/tasks/${t.id}`);
      }
    } catch (e) {
      if (e instanceof ApiRequestError) {
        setError([e.message, ...e.details.map(String)].join(" — "));
      } else setError("Unexpected error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card">
      <h1>New task</h1>
      <label>
        Task type
        <select value={taskType} onChange={(e) => onTypeChange(e.target.value)}>
          {types.map((t) => (
            <option key={t.task_type} value={t.task_type}>
              {t.task_type} — {t.description}
            </option>
          ))}
        </select>
      </label>
      <label>
        CSV file (UTF-8, header row, ≤ 25 MB)
        <input type="file" accept=".csv,text/csv" onChange={(e) => onFile(e.target.files?.[0])} />
      </label>

      {dataset && (
        <>
          <p className="muted">
            {dataset.filename}: {String(dataset.profile["n_rows"])} rows, {columns.length} columns
          </p>
          <label>
            Task name
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} />
          </label>
          <label>
            Target column
            <select
              value={target}
              onChange={(e) => {
                setTarget(e.target.value);
                setFeatures((f) => f.filter((x) => x !== e.target.value));
                setReport(null);
              }}
            >
              {columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.dtype})
                </option>
              ))}
            </select>
          </label>
          <fieldset>
            <legend>Feature columns (numeric)</legend>
            {columns
              .filter((c) => c.name !== target)
              .map((c) => (
                <label key={c.name} className="inline">
                  <input
                    type="checkbox"
                    checked={features.includes(c.name)}
                    disabled={c.dtype !== "numeric"}
                    onChange={(e) => {
                      setFeatures((f) =>
                        e.target.checked ? [...f, c.name] : f.filter((x) => x !== c.name),
                      );
                      setReport(null);
                    }}
                  />
                  {c.name}
                </label>
              ))}
          </fieldset>
          <div className="grid">
            <label>
              Test fraction
              <input
                type="number"
                step="0.05"
                min={0.05}
                max={0.5}
                value={testFraction}
                onChange={(e) => setTestFraction(Number(e.target.value))}
              />
            </label>
            <label>
              Split seed
              <input type="number" value={splitSeed} onChange={(e) => setSplitSeed(Number(e.target.value))} />
            </label>
            <label>
              {isGnb ? "var_smoothing" : "alpha (ridge)"}
              <input
                type="number"
                step="any"
                min={0}
                value={special}
                onChange={(e) => setSpecial(Number(e.target.value))}
              />
            </label>
            <label title="The task waits (queued) until this many eligible devices are online">
              Min devices
              <input type="number" min={1} max={8} value={minDevices} onChange={(e) => setMinDevices(Number(e.target.value))} />
            </label>
            <label>
              Max devices
              <input type="number" min={1} max={8} value={maxDevices} onChange={(e) => setMaxDevices(Number(e.target.value))} />
            </label>
          </div>
          <label className="inline">
            <input type="checkbox" checked={dropMissing} onChange={(e) => setDropMissing(e.target.checked)} />
            Drop rows with missing values (otherwise reject)
          </label>

          <div className="row">
            <button onClick={() => run("validate")} disabled={busy || features.length === 0}>
              Validate
            </button>
            <button
              onClick={() => run("submit")}
              disabled={busy || !report?.ok}
              title={report?.ok ? "" : "Validate first"}
            >
              Submit task
            </button>
          </div>
        </>
      )}

      {error && <p className="error">{error}</p>}
      {report && (
        <div className={report.ok ? "ok-box" : "error-box"}>
          <strong>{report.ok ? "Valid" : "Not valid"}</strong>
          {report.errors.map((e) => (
            <p key={e} className="error">
              {e}
            </p>
          ))}
          {report.warnings.map((w) => (
            <p key={w} className="warn">
              {w}
            </p>
          ))}
          {report.plan_preview && <PlanPreview pv={report.plan_preview as Preview} />}
        </div>
      )}
    </section>
  );
}

export default function NewTaskPage() {
  return (
    <AuthGate>
      <Wizard />
    </AuthGate>
  );
}
