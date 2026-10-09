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

- **Part 1 — Compute Fabric:** "Can ProofNet actually distribute and complete an AI computation?" (MVP; implemented, real-phone gates still pending)
- **Part 2 — Trust, Verification & Security:** "Can ProofNet tell whether devices behaved correctly and reward trustworthy work?" Based on the team's research *Evidence-Adaptive Auditing of Untrusted AI Compute Nodes with Lifetime False-Accusation Control*, proposing **PWAV** (per-class order-statistic tolerance limits, betting/e-process evidence, adaptive audit probability, minimum audit floor, persistent suspicion memory). **Implemented 2026-10-08 (P8–P11, `ARCHITECTURE.md` §17)** from this mechanism summary, not from the paper itself.

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
6. Part 2 only after M3 (offline research may run in parallel). **Overridden by the owner on 2026-10-08: Part 2 (P8–P11) was built and tested before M3. Part 1's real-phone gates (P6 scenarios, P6b CNN run, M3 five demos) remain open.**

Current phase: **P6 / P6b** — P0–P5 completed (M1, M2 passed 2026-10-07). P6 (reliability) implemented and verified automatically; its phone scenarios are pending. **P6b (image CNN workload, added 2026-10-08) implemented and verified automatically and in headless Chromium with two Pyodide workers (Fashion-MNIST subset: 40 rounds in 84 s, batch split 67.2%/32.8% by measured benchmark, centralized gradient check PASSED at 1.35e-6); real-phone gate pending.** **Part 2 (P8–P11) implemented and verified automatically 2026-10-08:** audit by backend recomputation, PWAV trust/quarantine, forensics, reward ledger with clawback and replay check, security hardening, attack harness + simulator, dashboards `/trust` `/rewards` `/security` `/simulator`. **UI layer (P12, 2026-10-09):** design system on all routes, `/network` live dashboard with PixiJS pixel world, mobile-first worker console, `GET /network/events` (admin); verified by `npm run e2e` against a real backend, a CLI worker and Chrome; real-phone check pending. Update this line as phases complete.

---

## 4. Technology & architecture (fixed decisions)

| Layer | Choice |
|---|---|
| Frontend | **Next.js + TypeScript** on **Vercel**. Pure API client; no workload logic, no Next API routes needed. Live dashboard `/network` uses **PixiJS 8** (pixel-art world) fed only by real backend state (ARCHITECTURE 3.2.1). |
| Backend / control plane | **Python + FastAPI**, **one long-running instance** on a free container host (not Vercel). Includes an idempotent **reconciler loop** (every 2 s). |
| Database | **MongoDB Atlas free tier** — metadata in collections, files in **GridFS**. Only persistent store. |
| Compute plane | Contributor devices. **Android = Chrome browser + Pyodide (Python/WASM) + NumPy in a Web Worker.** Laptops/CI = CPython CLI worker using the same kernel code. |
| Communication | **HTTPS pull/polling.** `POST /worker/heartbeat` (2 s idle / 5 s busy) returns directives (`run`, `cancel`). No inbound connections to devices. Offline after 20 s without heartbeat. |
| Workloads | Controlled **task catalog** of ProofNet-authored **kernels**: `gaussian_nb_train@1`, `linear_ridge_train@1` (exact sufficient-statistics merge). Plus one iterative, data-parallel workload for images: **`cnn_image_train@1`** (ProofNet-authored NumPy CNN; synchronous SGD over gradient sums; separate pipeline, ARCHITECTURE 4.5) |
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
- **Round** *(image CNN only)* — one training step: the global mini-batch is split across devices by measured benchmark; each device returns the **sum** of its per-sample gradients; the backend adds them, divides by the batch size and applies SGD. A task has `steps` rounds; round *r* uses model version *r*. Devices never hold the dataset: they receive only their slice of one mini-batch (+ the current weights, ~110 KB) per assignment.
- **Image dataset** — a zip of class folders (png/jpg) or a Kaggle MNIST-style pixel CSV, decoded server-side into a uint8 array (`image_datasets`, GridFS); separate from CSV datasets.
- **Artifact** — `model.joblib`, `model.json`, `report.json`, `predictions.csv` (CSV workloads); for image CNNs `model.npz`, `model.json`, `inference.py`, `training_curve.csv`, `predictions.csv`, `report.json`.
- **Device** states: `initializing, idle, busy, offline, disabled`.
- **Kernel** — `validate, prepare, map (worker), validate_partial, merge, finalize, reference, compare`.

---

## 6. Execution model & security boundary

**Rules that must not be broken:**

1. **No user-supplied code is executed anywhere** in the MVP. Users choose a task type; only ProofNet kernels run.
2. Image uploads are hostile input: decoded in memory only (never extracted to disk), only png/jpg members, size/pixel/count caps before decoding, class = folder name only.
2b. Worker-executed code lives only in `packages/kernels/proofnet_kernels/core` and imports only NumPy + stdlib.
3. **Never unpickle data from users or workers.** Chunk inputs are `.npz` loaded with `allow_pickle=False`; results are JSON; `.joblib` is only produced by the backend.
4. Workers can only access their own assignments (device token, hashed in DB).
5. Never execute the distributed workload in Vercel or in the backend (the backend only merges, finalizes and runs the small reference check).
6. Never fake progress, timings or device activity. If something is aggregator-side (holdout metrics), label it.

**Honest limitations (state them in UI/report):** results are audited probabilistically (floor 5 %), not all verified; corruptions below the tolerance are undetectable; unaudited cheats on merged CNN rounds cannot be undone (flagged); contributors see their partition's raw data; Sybil resistance is limited to inherited suspicion per account; this is a research prototype, not production-grade secure remote execution.

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
- Atlas storage is small (512 MB, writes blocked at the limit): no chunk copies stored; a storage guard releases old data above `STORAGE_BUDGET_MB` (ARCHITECTURE 16.1). The database name must not contain spaces (the project is named "Project 0"; the database is `Project0`).

---

## 9. Scope

**In scope (MVP):** accounts; device registration, capabilities, benchmark, heartbeat; dataset upload, validation, preparation; two kernels; plan preview; device-aware scheduler; dispatch, execution, result intake; aggregation with reference check; artifacts and download; MVP failure handling (offline, timeout, invalid result, duplicates, cancel, retries); live dashboards; free deployment + fallback.

**Also in scope (added in P6b, 2026-10-08):** one deep-learning workload, `cnn_image_train@1` — a small ProofNet-authored NumPy CNN trained by synchronous data-parallel SGD over image mini-batches, with its own upload/validate/monitor flow. The CSV workloads are unchanged.

**Also in scope (Part 2, 2026-10-08):** verification by backend recomputation, PWAV trust and quarantine, non-monetary reward credits with clawback, attack harness and population simulator, security hardening and their dashboards.

**Explicitly out of scope:** user Python code; arbitrary ML frameworks, or deep learning beyond the single catalog CNN kernel (no PyTorch/TensorFlow, no custom architectures from users); blockchain, tokens, cryptocurrency, real money; consensus protocols; Kubernetes, Docker Compose stacks, microservices, message queues; LLM-based task parsing; GPU workers; production-grade security; categorical feature encoding.

---

## 10. Extension points (Part 2 used them; future work may still use them)

`assignments.purpose`, `chunks.role` (`challenge`), `partial_results.acceptance`, `tasks.verification_policy`, `VerificationHook`, `AssignmentPolicy`, `DeviceEligibilityPolicy`, `kernel.compare()`, per-assignment `runtime_fingerprint` (defines PWAV "class"), `devices.stats`, append-only `events`, CLI-worker fault injection (→ attack harness). Part 2 collections (now present): `verification_records`, `device_trust`, `calibrations`, `reward_entries`, `reward_events`, `security_events`, `login_attempts`. Not built (not needed): `challenges`, `attack_events`, `reputation_events`; `assignments.purpose` `replica/audit` and `chunks.role` `challenge` remain reserved.

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
| Deployed URLs (P2) | Backend https://proofnet-api.onrender.com (Render free, Singapore, auto-deploys from `main`); frontend https://proofnet.vercel.app (Vercel, root `apps/web`); Atlas M0 cluster `proofnet-dev`, prod DB `proofnet_prod`. `CORS_ORIGINS` must list the Vercel origin without a trailing slash (backend now tolerates one). | 2026-10-07 |
| Render cold start after idle (first /health) | 52.5 s for the first request after >16 min idle; 0.2–0.3 s afterwards (Render free, Singapore) | 2026-10-07 |
| M1 run log (real phone, deployed) | Run 1 ✔ 2026-10-07: OPPO F31 Pro+ 5G, gaussian_nb_train, 80,000×16 chunk; download 31.6 s (10.88 MB, phone on 5G), compute 36 ms, total 32.4 s; reference check PASSED (1.8e-16); downloaded model.joblib predicts identically to centralized sklearn (20,000 holdout rows, parameter diff 0.0). Run 2 ✔ 2026-10-07: OPPO, linear_ridge_train, 80,000×16; download 110.7 s (weak 5G), compute 156 ms; reference PASSED (1.2e-14); R² 0.988856. Run 3 ✔ (Wi-Fi, GNB): download 4.36 s, compute 39 ms, total 5.1 s, reference PASSED; the phone was charging at 11% battery. **M1 gate passed.** Download time varies 4.4 s (Wi-Fi) → 31.6 s / 110.7 s (weak 5G) for the same 10.9 MB: the phone's network, not ProofNet. **Finding: transfer dominates — compute is 36 ms vs 31.6 s download.** | 2026-10-07 |
| M2 run (real phones, deployed) | 2026-10-07, gaussian_nb_train 80,000×16, min=max=2 devices: OPPO F31 Pro+ 5G (32.52 M cells/s) got rows [0, 58,655) — compute 28 ms, download 1.85 s; vivo Y22 (11.83 M cells/s) got [58,655, 80,000) — compute 75 ms, download 4.32 s. Predicted compute 29 ms each; the vivo ran 2.6× slower than predicted (benchmark varies per session) and transfer dominates compute on both. Reference check PASSED (1.2e-14). Idea for later: weight by transfer speed as well as compute. | 2026-10-07 |
| Pyodide first load on phone (Wi-Fi / mobile data) | Real phones, Chrome, Wi-Fi, cold cache: OPPO F31 Pro+ 5G runtime ready in 11.4 s; vivo Y22 in 26.2 s. Mobile-data and cached-reload numbers still to record. Desktop Chromium reference: ≈ 3.4 s. | 2026-10-07 |
| Benchmark score range (phones) | Re-benchmarks vary between sessions (OPPO 25.7 → 33.8 M cells/s, vivo 12.1 → 12.7), as the plan expects (scores re-measured every session). First run: OPPO F31 Pro+ 5G: 25.72 M cells/s (bench_v1 330 ms); vivo Y22: 12.07 M cells/s (577 ms). Desktop references: CPython ≈ 3.9–5.4e7; Pyodide in Node ≈ 1.9e7; Pyodide in desktop Chromium ≈ 6.5–7.8e7. Screen wake lock `active` on both phones. | 2026-10-07 |
| Max observed Pyodide vs CPython discrepancy per kernel | gaussian_nb: 0.0 (exact); linear_ridge: 7.0e-15 normwise relative (Pyodide 314.0.7 / NumPy 2.4.6 under Node vs CPython 3.12 / NumPy 2.4.6; tolerances 1e-8 / 1e-6). Pyodide-in-Node bench_v1 ≈ 1.85e7 cells/s (desktop reference, not a phone). | 2026-10-07 |
| Part 2 results (P8–P11) | 2026-10-08, automated: every attack mode (subtle 1e-4, scale, bias, noise, sign_flip, zero, random, lazy, replay) rejected by recomputation and the task still completes with a passing reference check (GNB/Ridge and CNN); persistent cheater quarantined after ≈ 6 audits at α = 0.2 (≈ 12 at α = 1e-3); unaudited cheat found by the reference check, culprit quarantined, chunk re-run, final model correct; simulator (30 honest + 6 always-cheating, 400 rounds): adaptive audit cost 13.7 % vs 30.7 % for a fixed 30 % rate, 6/6 attackers caught, 0 false accusations, 2.3 % of corrupt results merged vs 71 % for fixed and 100 % for no audits. Real-phone Part 2 demo not yet run. | 2026-10-08 |
