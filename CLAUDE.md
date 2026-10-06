# CLAUDE.md — ProofNet

Operational source of truth for anyone (human or agent) implementing ProofNet.
Details live in `ARCHITECTURE.md` (how it works) and `PHASE_PLAN.md` (in what order). The three documents must never contradict each other; if you change a decision, update all three in the same change.

---

## 1. What ProofNet is

ProofNet is a decentralized AI compute platform. A **user** submits an ML workload; registered **contributor devices** (initially Android phones) each compute a portion; ProofNet merges the partial results into the correct final output and returns it.

Full lifecycle:

```
device registers → capabilities known → user submits task → task validated → workload partitioned
→ chunks assigned to suitable devices → devices execute → partial results collected → merged → final artifact to user
                                                        └─(Part 2) verification → trust/reputation → malicious-node handling → reward
```

- **Part 1 — Compute Fabric:** "Can ProofNet actually distribute and complete an AI computation?" ← **current focus (MVP)**
- **Part 2 — Trust, Verification & Security:** "Can ProofNet tell whether devices behaved correctly and reward trustworthy work?" Based on the team's research *Evidence-Adaptive Auditing of Untrusted AI Compute Nodes with Lifetime False-Accusation Control*, proposing **PWAV** (per-class order-statistic tolerance limits, betting/e-process evidence, adaptive audit probability, minimum audit floor, persistent suspicion memory).

Evolution: working compute → reliable multi-device compute → verification → attack testing → trust/reputation → PWAV → rewards → larger network.

---

## 2. The MVP in one paragraph

Two real Android phones open the ProofNet contributor page in Chrome, register, and benchmark themselves. A user uploads a CSV and submits a `gaussian_nb_train` (or `linear_ridge_train`) task. ProofNet validates it, splits the training rows in proportion to each phone's measured speed, each phone computes sufficient statistics for its rows in Python (Pyodide) and returns them, the backend merges them into a scikit-learn model, verifies it equals centralized training (reference check), and the user downloads `model.joblib` + `report.json`. Everything on screen reflects real state; nothing is simulated.

**First milestone is one device** (M1), then two devices (M2), then a reliable, repeatable demo (M3). See `PHASE_PLAN.md`.

---

## 3. Current priorities

1. P0–P1: repo + contracts; kernels with proven distributed == centralized equivalence (CPython **and** Pyodide).
2. P2–P3: control plane + Android browser worker.
3. **P4 ★ M1 — one device end-to-end.** Do not start multi-device work before this passes on a real phone.
4. P5 ★ M2 — two+ devices, device-aware split.
5. P6–P7 ★ M3 — failures, second kernel, deployment, five consecutive successful demos.
6. Part 2 only after M3 (offline research may run in parallel).

Current phase: **P2** — Level-1 work implemented and tested locally against real Atlas (auth, devices, worker session/heartbeat, runtime bundle, datasets, tasks → queued, reconciler marks devices offline, minimal frontend). **Remaining for the P2 gate: deploy backend to Render + frontend to Vercel and verify Level 1 deployed** (needs the user's Render/Vercel/GitHub accounts). P0, P1 completed 2026-10-07. Update this line as phases complete.

---

## 4. Technology & architecture (fixed decisions)

| Layer | Choice |
|---|---|
| Frontend | **Next.js + TypeScript** on **Vercel**. Pure API client; no workload logic, no Next API routes needed. |
| Backend / control plane | **Python + FastAPI**, **one long-running instance** on a free container host (not Vercel). Includes an idempotent **reconciler loop** (every 2 s). |
| Database | **MongoDB Atlas free tier** — metadata in collections, files in **GridFS**. Only persistent store. |
| Compute plane | Contributor devices. **Android = Chrome browser + Pyodide (Python/WASM) + NumPy in a Web Worker.** Laptops/CI = CPython CLI worker using the same kernel code. |
| Communication | **HTTPS pull/polling.** `POST /worker/heartbeat` (2 s idle / 5 s busy) returns directives (`run`, `cancel`). No inbound connections to devices. Offline after 20 s without heartbeat. |
| Workloads | Controlled **task catalog** of ProofNet-authored **kernels**: `gaussian_nb_train@1`, `linear_ridge_train@1` (exact sufficient-statistics merge). |
| Scheduling | Eligibility filter + **weighted proportional rows by measured benchmark (cells/s)**, memory caps, deterministic ordering. |
| API types | FastAPI OpenAPI → generated TypeScript types. |

Backend host: **Render free web service** (chosen in P0, 2026-10-07; see §12). Known: Render free web services sleep after 15 min idle, take ~1 min to wake, have ephemeral disk, a single instance, and 750 instance-hours/month. Atlas free tier: 512 MB storage, limited ops/s.

Demo fallback: FastAPI on a team laptop + free HTTPS tunnel + frontend runtime API-URL switch.

Repository layout: `apps/web`, `services/api`, `packages/kernels` (`core/` = NumPy+stdlib only, runs on workers; `server/` = backend-only), `workers/cli-worker`, `datasets`, `scripts`.

---

## 5. Core concepts (use these words consistently)

- **Task** — a validated manifest (task type + dataset + params + execution settings). States: `queued → running → aggregating → completed`, or `failed` / `cancelled`.
- **Dataset** — uploaded CSV (GridFS). **Prepared data** — validated numeric `.npz` (train/test split, seeded) derived from it.
- **Chunk** — a row range of the prepared training data. States: `pending → assigned → completed`, or `failed` / `cancelled`; back to `pending` on a failed attempt (max 3 attempts).
- **Assignment** — one attempt to run a chunk on one device (= "execution"). States: `assigned, running, succeeded, rejected, failed, expired, cancelled`. Has `purpose` (`primary` now; `replica`/`audit` in Part 2).
- **Partial result** — JSON statistics returned by a device for one assignment. `acceptance = accepted_unverified` in the MVP.
- **Aggregation** — `merge → finalize → metrics → reference check → artifacts`.
- **Artifact** — `model.joblib`, `model.json`, `report.json`, `predictions.csv`.
- **Device** states: `initializing, idle, busy, offline, disabled`.
- **Kernel** — `validate, prepare, map (worker), validate_partial, merge, finalize, reference, compare`.

---

## 6. Execution model & security boundary

**Rules that must not be broken:**

1. **No user-supplied code is executed anywhere** in the MVP. Users choose a task type; only ProofNet kernels run.
2. Worker-executed code lives only in `packages/kernels/proofnet_kernels/core` and imports only NumPy + stdlib.
3. **Never unpickle data from users or workers.** Chunk inputs are `.npz` loaded with `allow_pickle=False`; results are JSON; `.joblib` is only produced by the backend.
4. Workers can only access their own assignments (device token, hashed in DB).
5. Never execute the distributed workload in Vercel or in the backend (the backend only merges, finalizes and runs the small reference check).
6. Never fake progress, timings or device activity. If something is aggregator-side (holdout metrics), label it.

**Honest limitations (state them in UI/report):** results are unverified until Part 2; contributors see their partition's raw data; Sybil/collusion not handled; this is a research prototype, not production-grade secure remote execution.

---

## 7. Architectural principles

- Layers: presentation · control/API · task/workload · scheduling · worker/device · result/aggregation · persistence · (future) verification/trust. Modules, not microservices.
- **Control plane ≠ compute plane.** The backend decides; devices compute.
- Every state transition is a conditional atomic MongoDB update; everything is idempotent; all state survives a backend restart.
- Chunk ≠ assignment, so one chunk can later have several assignments (replication/audits).
- Small files, clear interfaces: `ChunkPlanner`, `AssignmentPolicy`, `DeviceEligibilityPolicy`, `VerificationHook`, kernel interface. Part 2 replaces implementations, not callers.
- Deterministic where possible (seeded splits, sorted device order, fixed merge order) so tests and audits are reproducible.
- Measured over assumed: scheduling uses the benchmark, not guessed specs.

---

## 8. Constraints

- **Free / free-tier only.** No GCP, AWS, Azure, paid DBs, queues, object storage, AI APIs or monitoring as dependencies. If something cannot fit, document it and choose the simplest free alternative — never add a hidden paid dependency.
- Vercel is frontend only. FastAPI is a separate service.
- MVP limits: CSV ≤ 25 MB, ≤ 300k rows, ≤ 64 numeric features, 2–50 classes, ≤ 8 devices per task.
- Android page must stay in the foreground with the screen on (wake lock); a backgrounded tab is treated as offline.
- Atlas storage is small: no chunk copies stored; demo reset script.

---

## 9. Scope

**In scope (MVP):** accounts; device registration, capabilities, benchmark, heartbeat; dataset upload, validation, preparation; two kernels; plan preview; device-aware scheduler; dispatch, execution, result intake; aggregation with reference check; artifacts and download; MVP failure handling (offline, timeout, invalid result, duplicates, cancel, retries); live dashboards; free deployment + fallback.

**Explicitly out of scope:** user Python code; arbitrary ML frameworks or deep learning; blockchain, tokens, cryptocurrency; consensus protocols; trust scores, reputation, PWAV, rewards (Part 2); Kubernetes, Docker Compose stacks, microservices, message queues; LLM-based task parsing; GPU workers; production-grade security; categorical feature encoding.

---

## 10. Future extension points (do not implement in Part 1, do not block)

`assignments.purpose`, `chunks.role` (`challenge`), `partial_results.acceptance`, `tasks.verification_policy`, `VerificationHook`, `AssignmentPolicy`, `DeviceEligibilityPolicy`, `kernel.compare()`, per-assignment `runtime_fingerprint` (defines PWAV "class"), `devices.stats`, append-only `events`, CLI-worker fault injection (→ attack harness). Future collections: `verification_records`, `challenges`, `trust_profiles`, `reputation_events`, `attack_events`, `reward_records`.

---

## 11. Development & testing principles

- Local dev = MongoDB (local or personal Atlas DB) + `uvicorn` + `next dev` + browser-tab or CLI workers. No Docker required.
- Test levels (gate each before the next): L1 backend works · L2 device registers · L3 one device executes · L4 one task end-to-end · L5 two devices execute separate chunks · L6 merged result correct · L7 failure/retry works · L8 full demo repeats.
- Every kernel ships **equivalence tests** (distributed == centralized for random partitionings) and a **Pyodide parity test**.
- API tests run against a real MongoDB. E2E tests use CLI workers; milestone gates use **real phones**.
- Python: typed (Pydantic, type hints), formatter + linter + type checker. TypeScript: strict.
- Keep the three planning documents consistent; record measured numbers (load times, benchmark ranges, host limits) in §12.

---

## 12. Measured facts & decisions log (fill in during implementation)

| Item | Value | Date |
|---|---|---|
| Backend host chosen | **Render free web service** (primary). Rejected: Hugging Face Spaces (Docker Spaces now require a paid plan to create — violates free-only rule); Koyeb (pricing page lists no free plan). Fallback: laptop + HTTPS tunnel. Vercel project: to be created by the user at first deploy (P2). | 2026-10-07 |
| Pinned versions | Python 3.12 (backend/dev; core must also run on Pyodide's Python 3.14.2); **Pyodide 314.0.7** (Python 3.14.2, **NumPy 2.4.6**, from its lock file); backend NumPy **2.4.6** (matches Pyodide); scikit-learn 1.9.1; pandas 3.0.6; FastAPI 0.142.2; Pydantic 2.13.5; PyMongo 4.18.2; Next.js 16.4.0; React 19.3.0; TypeScript 5.9.3; Node 22; openapi-typescript 7.13.0; ruff 0.16.10; mypy 2.4.0; pytest 9.1.1 | 2026-10-07 |
| Pyodide first load on phone (Wi-Fi / mobile data) | _TBD in P3_ | |
| Benchmark score range (phones) | _TBD in P3_ | |
| Max observed Pyodide vs CPython discrepancy per kernel | gaussian_nb: 0.0 (exact); linear_ridge: 7.0e-15 normwise relative (Pyodide 314.0.7 / NumPy 2.4.6 under Node vs CPython 3.12 / NumPy 2.4.6; tolerances 1e-8 / 1e-6). Pyodide-in-Node bench_v1 ≈ 1.85e7 cells/s (desktop reference, not a phone). | 2026-10-07 |
