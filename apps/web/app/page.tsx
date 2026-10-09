import Link from "next/link";

const FLOW = [
  ["Register", "A phone or laptop joins and benchmarks itself with the real kernel code."],
  ["Submit", "A user uploads data and picks a catalogued task. No user code ever runs."],
  ["Split", "The planner divides the work in proportion to each device's measured speed."],
  ["Compute", "Each device computes its chunk in Python (Pyodide) in the browser."],
  ["Audit", "The backend recomputes a sampled share of results and checks them."],
  ["Merge", "Verified partial results are merged and compared with centralized training."],
];

export default function Home() {
  return (
    <>
      <nav className="nav" aria-label="Main">
        <span className="brand">
          <i aria-hidden="true" />
          ProofNet
        </span>
        <span className="spacer" />
        <Link href="/login">Sign in</Link>
        <Link href="/signup">Create account</Link>
      </nav>
      <main>
      <div className="hero">
        <div className="eyebrow">Decentralized AI compute · verified by audit</div>
        <h1>Distribute machine learning across devices you cannot fully trust.</h1>
        <p className="lead">
          ProofNet splits a training job across contributor phones and laptops, audits their work by recomputation, tracks each
          device&apos;s trustworthiness, and merges the verified result, with a reference check against centralized training.
        </p>
        <div className="row">
          <Link href="/login" className="btn">
            Open the live network
          </Link>
          <Link href="/login" className="btn" style={{ background: "transparent", color: "var(--accent)", border: "1px solid var(--accent)" }}>
            Contribute a device
          </Link>
        </div>
      </div>
      <div className="hero" style={{ paddingTop: 0 }}>
        <div className="flow" aria-label="How a task flows through ProofNet">
          {FLOW.map(([t, d]) => (
            <div key={t}>
              <b>{t}</b>
              {d}
            </div>
          ))}
        </div>
        <p className="muted">
          Research prototype. Results are audited probabilistically (floor 5%), not all verified; do not upload sensitive data. See the{" "}
          <Link href="/limitations">honest limitations</Link>.
        </p>
        <div className="row">
          <Link href="/tasks">My tasks</Link>
          <Link href="/trust">Trust</Link>
          <Link href="/rewards">Rewards</Link>
          <Link href="/security">Security</Link>
          <Link href="/simulator">Simulator</Link>
          <Link href="/settings">Backend settings</Link>
        </div>
      </div>
      </main>
    </>
  );
}
