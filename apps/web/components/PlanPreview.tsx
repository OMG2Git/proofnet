"use client";

export type PreviewShare = {
  device_name: string | null;
  score_cells_per_sec: number;
  weight: number;
  rows: number;
  n_chunks: number;
  est_seconds: number | null;
};

export type Preview = {
  n_train: number;
  eligible_devices: number;
  min_devices: number;
  ready_to_start: boolean;
  kind?: string;
  steps?: number;
  global_batch_size?: number;
  message?: string | null;
  explanation?: string;
  shares: PreviewShare[];
  not_eligible: string[];
};

/** Plan preview from the devices online right now (CSV chunk plan, or per-round batch split). */
export default function PlanPreview({ pv }: { pv: Preview }) {
  const iterative = pv.kind === "iterative";
  return (
    <div data-testid="preview">
      <p>
        <strong>Plan preview</strong>{" "}
        <span className="muted">
          (estimate from the devices online now; ~{pv.n_train} training {iterative ? "images" : "rows"};{" "}
          {pv.eligible_devices} eligible device(s))
        </span>
      </p>
      {iterative && (
        <p className="muted">
          {pv.steps} rounds; each round the global batch of {pv.global_batch_size} images is split as below.
        </p>
      )}
      {!pv.ready_to_start && <p className="warn">{pv.message}</p>}
      {pv.shares.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Device</th>
              <th>Benchmark</th>
              <th>Weight</th>
              <th>{iterative ? "Images per round" : "Rows"}</th>
              {!iterative && <th>Chunks</th>}
              {!iterative && <th>Predicted compute</th>}
            </tr>
          </thead>
          <tbody>
            {pv.shares.map((s, i) => (
              <tr key={i}>
                <td>{s.device_name}</td>
                <td>{(s.score_cells_per_sec / 1e6).toFixed(2)} M cells/s</td>
                <td>{(s.weight * 100).toFixed(1)}%</td>
                <td>{s.rows}</td>
                {!iterative && <td>{s.n_chunks}</td>}
                {!iterative && (
                  <td>
                    {s.est_seconds === null
                      ? "—"
                      : s.est_seconds < 1
                        ? `${Math.round(s.est_seconds * 1000)} ms`
                        : `${s.est_seconds.toFixed(2)} s`}
                  </td>
                )}
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
