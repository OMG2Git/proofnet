"use client";

export type Series = { name: string; color: string; values: number[] };

/** Dependency-free SVG line chart for the live training curve (real values only). */
export default function LineChart({
  series,
  marks = [],
  height = 180,
  yMin,
  yMax,
  yFormat = (v: number) => v.toFixed(2),
}: {
  series: Series[];
  marks?: number[]; // x positions (round numbers) to highlight, e.g. centrally verified rounds
  height?: number;
  yMin?: number;
  yMax?: number;
  yFormat?: (v: number) => string;
}) {
  const W = 640;
  const P = { l: 42, r: 10, t: 10, b: 22 };
  const all = series.flatMap((s) => s.values);
  const n = Math.max(...series.map((s) => s.values.length), 1);
  if (all.length === 0) return <p className="muted">No rounds completed yet.</p>;
  const lo = yMin ?? Math.min(...all);
  const hi = yMax ?? Math.max(...all);
  const span = hi - lo || 1;
  const x = (i: number) => P.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - P.l - P.r));
  const y = (v: number) => P.t + (1 - (v - lo) / span) * (height - P.t - P.b);
  const ticks = [0, 0.5, 1].map((f) => lo + f * span);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="training curve" className="chart">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} className="grid" />
          <text x={P.l - 6} y={y(t) + 4} textAnchor="end" className="tick">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {series.map((s) => (
        <polyline
          key={s.name}
          fill="none"
          stroke={s.color}
          strokeWidth={2}
          points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
        />
      ))}
      {marks
        .filter((m) => m < n && series[0])
        .map((m) => (
          <circle key={m} cx={x(m)} cy={y(series[0]?.values[m] ?? lo)} r={4} className="mark" />
        ))}
      <text x={P.l} y={height - 4} className="tick">
        round 0
      </text>
      <text x={W - P.r} y={height - 4} textAnchor="end" className="tick">
        round {n - 1}
      </text>
    </svg>
  );
}
