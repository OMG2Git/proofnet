import Link from "next/link";

export const metadata = { title: "ProofNet limitations" };

export default function LimitationsPage() {
  return (
    <main>
      <h1>What ProofNet does and does not guarantee</h1>
      <p>
        ProofNet is a research prototype. It distributes a small set of ProofNet-written computations across
        contributor devices and checks that the combined result equals computing it on one machine. It is
        deliberately honest about what it does not do yet.
      </p>
      <div className="card">
        <h2>Real, verifiable</h2>
        <ul>
          <li>All computation shown is real: devices, timings, chunk assignments and results come from actual state.</li>
          <li>
            CSV workloads (Gaussian Naive Bayes, Linear/Ridge regression) are split by rows and merged exactly; the
            merged statistics are compared with a centralized recomputation.
          </li>
          <li>
            The image CNN is trained by synchronous data-parallel SGD; sampled rounds are re-checked against a
            centralized gradient.
          </li>
          <li>Failures (a device going offline, corrupt or late results, cancellation) are survived or reported.</li>
        </ul>
        <h2>Not guaranteed (Part 2 research)</h2>
        <ul>
          <li>
            <strong>Results are unverified.</strong> A malicious contributor could return well-formed but wrong
            numbers. Only the sampled CNN rounds and the final reference checks catch some of this. Detection,
            trust scores and rewards are future work.
          </li>
          <li>
            <strong>No confidentiality.</strong> Contributors see the raw rows (CSV) or training images (CNN) of the
            work they compute. Do not upload sensitive data.
          </li>
          <li>No protection against fake devices (Sybil) or collusion; device tokens live in the contributor&apos;s browser.</li>
          <li>
            Phones must keep the page in the foreground with the screen on; a locked or backgrounded phone is treated
            as offline and its work is reassigned.
          </li>
          <li>
            Only ProofNet-written computations run. Users cannot upload code (that would let strangers run code on
            other people&apos;s phones).
          </li>
          <li>
            The free hosting tiers are slow to wake (about a minute) and limited in storage (512 MB); old data is
            released automatically when space runs low.
          </li>
        </ul>
        <h2>Performance, honestly</h2>
        <p>
          For these workloads, moving data to and from the phone usually costs far more than the computation: a
          10.9 MB CSV chunk took 4 s to download on Wi-Fi and over 100 s on weak mobile data, while computing it
          took tens of milliseconds. The CNN is a small NumPy network (27k parameters, about 80% on Fashion-MNIST);
          it demonstrates correct distributed training, not state-of-the-art accuracy.
        </p>
      </div>
      <p>
        <Link href="/">← Home</Link>
      </p>
    </main>
  );
}
