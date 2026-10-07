"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import AuthGate from "@/components/AuthGate";
import PlanPreview, { type Preview } from "@/components/PlanPreview";
import {
  api,
  ApiRequestError,
  type ImageDatasetOut,
  type ImageTaskManifest,
  type ValidationResult,
} from "@/lib/api/client";

type Profile = {
  n_images: number;
  shape: number[];
  classes: string[];
  class_counts: number[];
  samples: { class: string; png_b64: string }[];
};

function Wizard() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<ImageDatasetOut[]>([]);
  const [dataset, setDataset] = useState<ImageDatasetOut | null>(null);
  const [side, setSide] = useState(28);
  const [name, setName] = useState("Fashion-MNIST CNN");
  const [steps, setSteps] = useState(200);
  const [batch, setBatch] = useState(128);
  const [lr, setLr] = useState(0.05);
  const [momentum, setMomentum] = useState(0.9);
  const [c1, setC1] = useState(8);
  const [c2, setC2] = useState(16);
  const [dense, setDense] = useState(64);
  const [verify, setVerify] = useState(3);
  const [minDevices, setMinDevices] = useState(1);
  const [maxDevices, setMaxDevices] = useState(4);
  const [report, setReport] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .imageDatasets()
      .then((d) => {
        setDatasets(d);
        if (d[0]) setDataset(d[0]);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const ds = await api.uploadImageDataset(file, side);
      setDatasets((d) => [ds, ...d]);
      setDataset(ds);
    } catch (e) {
      setError(e instanceof ApiRequestError ? [e.message, ...e.details.map(String)].join(" — ") : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  function manifest(): ImageTaskManifest | null {
    if (!dataset) return null;
    return {
      name,
      task_type: "cnn_image_train",
      kernel_version: "1",
      dataset_id: dataset.id,
      params: {
        steps,
        global_batch_size: batch,
        learning_rate: lr,
        momentum,
        conv1_filters: c1,
        conv2_filters: c2,
        dense_units: dense,
        verify_rounds: verify,
        test_fraction: 0.15,
        split_seed: 42,
        init_seed: 0,
      },
      execution: { min_devices: minDevices, max_devices: maxDevices, start_policy: "wait_for_min_devices" },
    };
  }

  async function run(kind: "validate" | "submit") {
    const m = manifest();
    if (!m) return;
    setBusy(true);
    setError(null);
    try {
      if (kind === "validate") setReport(await api.validateImageTask(m));
      else {
        const t = await api.createImageTask(m);
        router.push(`/tasks/${t.id}`);
      }
    } catch (e) {
      setError(e instanceof ApiRequestError ? [e.message, ...e.details.map(String)].join(" — ") : "Unexpected error");
    } finally {
      setBusy(false);
    }
  }

  const prof = dataset?.profile as unknown as Profile | undefined;
  return (
    <section className="card">
      <h1>Image training (CNN)</h1>
      <p className="muted">
        A small CNN is trained across contributor devices. The dataset stays on the server: each round every
        device receives only its slice of one mini-batch (plus the current weights), computes gradients and
        returns them. Gradients are added, so the result equals centralized training; sampled rounds are
        re-checked against a centralized computation.
      </p>
      <label>
        Image dataset: a .zip of class folders (png/jpg) or a Kaggle MNIST-style pixel .csv (≤ 25 MB)
        <input type="file" accept=".zip,.csv" onChange={(e) => onFile(e.target.files?.[0])} />
      </label>
      <label>
        Resize uploaded images to
        <select value={side} onChange={(e) => setSide(Number(e.target.value))}>
          {[16, 20, 24, 28, 32].map((v) => (
            <option key={v} value={v}>
              {v} × {v}
            </option>
          ))}
        </select>
      </label>
      {datasets.length > 1 && (
        <label>
          Or use an uploaded dataset
          <select
            value={dataset?.id}
            onChange={(e) => {
              setDataset(datasets.find((d) => d.id === e.target.value) ?? null);
              setReport(null);
            }}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename} ({String((d.profile as { n_images?: number }).n_images)} images)
              </option>
            ))}
          </select>
        </label>
      )}

      {dataset && prof && (
        <>
          <p>
            <strong>{dataset.filename}</strong>: {prof.n_images} images, {prof.shape.join("×")}, {prof.classes.length} classes
          </p>
          <div className="thumbs" data-testid="thumbs">
            {prof.samples.map((s, i) => (
              <figure key={i}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`data:image/png;base64,${s.png_b64}`} alt={s.class} />
                <figcaption>{s.class}</figcaption>
              </figure>
            ))}
          </div>
          <label>
            Task name
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} />
          </label>
          <div className="grid">
            <label>
              Training rounds (steps)
              <input type="number" min={5} max={2000} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
            </label>
            <label>
              Global batch size
              <input type="number" min={16} max={512} value={batch} onChange={(e) => setBatch(Number(e.target.value))} />
            </label>
            <label>
              Learning rate
              <input type="number" step="0.01" min={0.001} max={1} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
            </label>
            <label>
              Momentum
              <input type="number" step="0.05" min={0} max={0.99} value={momentum} onChange={(e) => setMomentum(Number(e.target.value))} />
            </label>
            <label>
              Conv1 filters
              <input type="number" min={4} max={16} value={c1} onChange={(e) => setC1(Number(e.target.value))} />
            </label>
            <label>
              Conv2 filters
              <input type="number" min={4} max={32} value={c2} onChange={(e) => setC2(Number(e.target.value))} />
            </label>
            <label>
              Dense units
              <input type="number" min={16} max={128} value={dense} onChange={(e) => setDense(Number(e.target.value))} />
            </label>
            <label title="Rounds re-checked against a centralized gradient">
              Verified rounds
              <input type="number" min={0} max={10} value={verify} onChange={(e) => setVerify(Number(e.target.value))} />
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
          <div className="row">
            <button onClick={() => run("validate")} disabled={busy}>
              Validate
            </button>
            <button onClick={() => run("submit")} disabled={busy || !report?.ok} title={report?.ok ? "" : "Validate first"}>
              Start training
            </button>
          </div>
        </>
      )}

      {error && <p className="error">{error}</p>}
      {report && (
        <div className={report.ok ? "ok-box" : "error-box"}>
          <strong>{report.ok ? "Valid" : "Not valid"}</strong>
          {report.ok && (
            <p className="muted">
              {String(report.summary["n_params"])} parameters · ~{String(report.summary["epochs"])} epochs of the
              training split
            </p>
          )}
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

export default function NewImageTaskPage() {
  return (
    <AuthGate>
      <Wizard />
    </AuthGate>
  );
}
