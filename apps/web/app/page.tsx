import Link from "next/link";

export default function Home() {
  return (
    <main>
      <h1>ProofNet</h1>
      <p>Decentralized AI compute: submit a supported ML task and let contributor devices compute it.</p>
      <p className="muted">
        Research prototype. Results are audited by recomputation with an adaptive probability; do not upload sensitive data.
      </p>
      <div className="row">
        <Link href="/login">Sign in</Link>
        <Link href="/signup">Create account</Link>
        <Link href="/tasks">My tasks</Link>
        <Link href="/trust">Trust</Link>
        <Link href="/rewards">Rewards</Link>
        <Link href="/security">Security</Link>
        <Link href="/limitations">Limitations</Link>
        <Link href="/settings">Backend settings</Link>
      </div>
    </main>
  );
}
