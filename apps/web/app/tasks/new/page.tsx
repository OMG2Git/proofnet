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
            <label>
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
          <p className="muted">Plan preview (rows per device) is available from P5.</p>
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
