# ProofNet — Technical Architecture

> Status: planning baseline for the MVP (Part 1 — Compute Fabric) with defined insertion points for Part 2 (Trust, Verification, Security).
> Companion documents: `CLAUDE.md` (operational source of truth) and `PHASE_PLAN.md` (implementation roadmap). If these documents ever disagree, fix the disagreement — do not pick one silently.

---

## 0. How to read this document

| Section | What it answers |
|---|---|
| 1 | What the system is and the key decisions behind it |
| 2 | The two planes: control plane vs compute plane |
| 3 | Components (frontend, backend, worker, kernels, database, storage) |
| 4 | Workload model — what we compute and why it is correct to split |
| 5 | Task submission, validation and preparation |
| 6 | Chunking and device-aware scheduling |
| 7 | Contributor devices: registration, capabilities, the Android worker |
| 8 | Worker ↔ backend communication protocol |
| 9 | Execution, result collection and aggregation |
| 10 | State machines (task, chunk, assignment, device) |
| 11 | Failure handling |
| 12 | Data model (MongoDB collections) |
| 13 | API surface |
| 14 | Security boundary |
| 15 | Observability for the demo |
| 16 | Deployment (Vercel, backend host, MongoDB Atlas) and local development |
| 17 | Part 2 insertion points (verification → trust → PWAV → reward) |
| 18 | Future scalability paths |
| 19 | Decision log |

---

## 1. System summary and key decisions

ProofNet lets a **user** submit an ML workload and have it executed by **contributor devices** (initially Android phones). The backend validates the task, splits the dataset into partitions sized for each device, dispatches the partitions, collects partial results, merges them with a mathematically valid aggregation, and returns a final artifact (trained model + report).

The MVP is deliberately narrow so that it is **real**:

| # | Decision | Short reason |
|---|---|---|
| D1 | **Android worker = browser worker.** The contributor opens a ProofNet page in Chrome on Android. Python runs inside the page via **Pyodide** (CPython compiled to WebAssembly) in a **Web Worker**. | Zero install, works on any modern Android phone, WASM sandbox protects the phone, NumPy is available in Pyodide. |
| D2 | **Pull-based HTTPS polling.** Workers call `POST /worker/heartbeat`; the response carries any work directives. No inbound connections to phones. | Phones sit behind NAT/mobile networks; free hosts sleep and drop long-lived sockets; polling is the most debuggable option. |
| D3 | **FastAPI runs as a separate long-running service**, not inside Vercel. Vercel hosts only the Next.js frontend (including the worker page). | The backend needs a background reconciler loop, file streaming and stable state. Serverless functions are the wrong fit. |
| D4 | **MongoDB Atlas free tier is the only persistent store** — metadata in collections, files (datasets, prepared arrays, artifacts) in **GridFS**. | Free hosts have ephemeral disks; avoids any paid object storage; MVP files are small. |
| D5 | **Controlled task catalog, no user-supplied Python.** Users pick a task type and supply data + parameters. All executable code is ProofNet-authored **kernels**. | Removes the arbitrary-remote-code-execution problem from the MVP entirely. |
| D6 | **Initial workloads are sufficient-statistics ML training**: Gaussian Naive Bayes (classification) and Linear/Ridge regression. | Their distributed result is *provably identical* (up to floating-point tolerance) to centralized training. |
| D7 | **Scheduler = weighted proportional partitioning by measured benchmark score**, with eligibility filters and memory caps. | Simple, explainable, deterministic, device-aware, easy to replace. |
| D8 | **Single backend instance with an idempotent reconciler loop**, all state transitions as conditional atomic MongoDB updates. | No queue service needed; survives restarts because state is in the DB. |
| D9 | **Workers return JSON only; the backend never unpickles anything received from a worker or user.** `.joblib` artifacts are produced only by the backend from validated numbers. | Pickle is code execution. |
| D11 | **Image CNN = stateless synchronous data-parallel rounds** (added P6b). The dataset stays on the backend; each assignment carries one slice of a mini-batch + the current weights; devices return gradient *sums*; the backend adds, divides, applies SGD. Mathematically identical to centralized SGD on the same batch, and re-checked on sampled rounds. | Phones cannot hold or download a dataset; a stateless round also makes every existing failure rule (retry, exclusion, expiry, cancel) apply unchanged. |
| D10 | **Contract-first:** FastAPI's OpenAPI schema is the single source of API types; the TypeScript client types are generated from it. | Keeps frontend, worker and backend in sync for a student team. |

What the MVP explicitly does **not** do: arbitrary ML frameworks, user Python code, deep learning, blockchain/tokens, consensus, Kubernetes, microservices. (Trust scores, PWAV, verification and rewards were out of the MVP and are implemented in Part 2, §17.) See `CLAUDE.md` § Scope.

---

## 2. Control plane vs compute plane

This distinction is fundamental. Nothing in the control plane performs the distributed workload; nothing in the compute plane decides what to do.

```mermaid
flowchart LR
    subgraph Users["People"]
        U["User / task submitter<br/>(browser)"]
        C["Contributor<br/>(Android phone browser)"]
        P["Evaluator screen<br/>(Network dashboard)"]
    end

    subgraph Vercel["Vercel — static/frontend only"]
        WEB["Next.js app<br/>User UI · Contributor UI · Dashboard<br/>Worker page (JS controller)"]
    end

    subgraph CP["CONTROL PLANE — FastAPI (one long-running service)"]
        API["REST API /api/v1"]
        WG["Worker gateway<br/>heartbeat · dispatch · results"]
        TS["Task service<br/>catalog · validation · preparation"]
        SCH["Scheduler<br/>chunk planning · assignment"]
        AGG["Aggregator<br/>merge · finalize · reference check"]
        REC["Reconciler loop (every 2 s)<br/>offline · leases · retries · timeouts"]
        VH["Verification hook<br/>(MVP: accept-all)"]
    end

    subgraph DB["MongoDB Atlas (free tier)"]
        COL["Collections<br/>users · devices · datasets · tasks<br/>chunks · assignments · partial_results<br/>artifacts · events"]
        GFS["GridFS<br/>raw CSV · prepared .npz · artifacts"]
    end

    subgraph COMP["COMPUTE PLANE — contributor devices"]
        W1["Phone A<br/>Web Worker + Pyodide + NumPy<br/>proofnet_kernels"]
        W2["Phone B<br/>Web Worker + Pyodide + NumPy"]
        W3["Laptop (optional)<br/>CPython CLI worker"]
    end

    U --> WEB
    C --> WEB
    P --> WEB
    WEB -- "HTTPS JSON (CORS)" --> API
    W1 -- "HTTPS poll / upload" --> WG
    W2 -- "HTTPS poll / upload" --> WG
    W3 -- "HTTPS poll / upload" --> WG
    API --- TS
    API --- SCH
    WG --- SCH
    WG --- VH
    VH --- AGG
    REC --- SCH
    CP --- COL
    CP --- GFS
```

**Control plane** (FastAPI + MongoDB): identity, device registry, task catalog, validation, preparation of data, planning, assignment, leases, result validation, aggregation, artifacts, events. It does only *light* computation: parsing CSV, slicing arrays, merging small statistics, building the final model object, and (optionally) the centralized reference check on small data.

**Compute plane** (contributor devices): executes the map kernel on its assigned partition and returns a partial result. It holds no authoritative state.

**Vercel** serves the frontend bundle, including the JavaScript of the worker page. Vercel functions are not used for workload execution, polling targets, or file storage.

---

## 3. Components

### 3.1 Repository layout (monorepo)

```
proofnet/
├── CLAUDE.md · PHASE_PLAN.md · ARCHITECTURE.md
├── apps/web/                    Next.js + TypeScript (deployed to Vercel)
│   ├── app/                     routes (see 3.2)
│   ├── components/              UI components
│   ├── lib/api/                 generated OpenAPI types + thin fetch client
│   └── worker-runtime/          browser compute agent
│       ├── controller.ts        main-thread loop: heartbeat, dispatch, wake lock, UI state
│       └── compute.worker.ts    Web Worker: loads Pyodide, kernel bundle, runs map kernels
├── services/api/                FastAPI control plane
│   └── proofnet_api/
│       ├── main.py · config.py · db.py · ids.py
│       ├── auth/                users, JWT, device tokens
│       ├── devices/             registry, capabilities, benchmark records
│       ├── worker_gateway/      heartbeat, directives, input streaming, result intake
│       ├── datasets/            upload, profiling, GridFS
│       ├── tasks/               catalog, validation, preparation, task lifecycle
│       ├── scheduling/          planner (chunking) + assigner + policies
│       ├── aggregation/         merge, finalize, reference check, reports
│       ├── artifacts/           storage + download
│       ├── events/              append-only event log + status snapshots
│       ├── verification/        verifier (audit by recomputation), forensics, acceptance states — Part 2
│       ├── trust/               pwav math, trust store/calibration, population simulator, /trust API — Part 2
│       ├── rewards/             reward ledger (entries + events), /rewards API — Part 2
│       ├── security/            quarantine, rate limit/lockout, headers, security events, /security API — Part 2
│       ├── training/            iterative (round-based) tasks: rounds, model states — P6b
│       ├── image_tasks/         image dataset upload, CNN task validate/create, training curve — P6b
│       └── reconciler.py        periodic idempotent maintenance loop
├── packages/kernels/            proofnet_kernels — SHARED Python package
│   └── proofnet_kernels/
│       ├── core/                NumPy + stdlib ONLY (runs in Pyodide and CPython)
│       │   ├── gaussian_nb.py   map + merge
│       │   ├── linear_ridge.py  map + merge
│       │   ├── moments.py       Chan/parallel moment merge utilities
│       │   ├── bench.py         benchmark workload
│       │   └── serialize.py     canonical JSON, digests, array encoding
│       └── server/              backend-only (scikit-learn, pandas, Pillow): finalize, reference, compare
│           ├── cnn.py · images.py   CNN params/optimizer/evaluation; image dataset ingestion (zip/CSV) — P6b
├── workers/cli-worker/          CPython reference worker (laptops, CI, fault injection)
├── datasets/                    deterministic demo-dataset generator + tiny fixtures
└── scripts/                     dev helpers: seed users, reset demo, run N simulated workers
```

The split `core/` vs `server/` is the key code boundary: **anything a worker runs must live in `core/` and import only NumPy and the standard library.** The backend imports both.

### 3.2 Frontend (Next.js + TypeScript, Vercel)

Pure client of the FastAPI API. No business logic in Next.js API routes (none are needed in the MVP). Pages:

| Route | Audience | Purpose |
|---|---|---|
| `/` | everyone | Landing + live network summary (devices online, tasks running) |
| `/login`, `/signup` | everyone | Simple account auth |
| `/tasks/new` | user | Wizard: choose task type → upload CSV → map columns/params → **validate** → **plan preview** → submit |
| `/tasks` | user | My tasks |
| `/tasks/[id]` | user, evaluator | Live task monitor: state timeline, chunk table, device lanes, aggregation, correctness check, downloads |
| `/contribute` | contributor | My devices, register this device |
| `/contribute/run` | contributor (on the phone) | **Worker console** (mobile-first): startup steps, benchmark, current chunk (task, chunk, rows), heartbeat and offline/background warnings, audit outcomes and credits, wake-lock explanation; log and demo-misbehave switch collapsed |
| `/network` | evaluator / admin (projector) | **Live network dashboard**: KPI strip, pixel-art world (PixiJS), filters + search, device assignment table, task activity feed, worker detail drawer with admin quarantine/reinstate |

Data freshness: dashboards poll `GET /tasks/{id}/status`, `GET /network/summary` (every 1.5 s, 5 s in a hidden tab) and, for admins, `GET /network/events?since=` (every 2 s). (SSE is a later enhancement, §18.)

#### 3.2.1 UI layer (added 2026-10-09)

- **Design system**: one dark theme in `app/globals.css`: tokens (cyan = assigned work / network, violet = computation, green = verified, amber = pending / warning, red = rejected / quarantined), pixel display font (Silkscreen) + IBM Plex Mono via `next/font`, existing class names kept so every route restyles at once. State is never colour-only (badges carry a glyph and text; canvas states carry a glyph and a label). Honours `prefers-reduced-motion`.
- **Data adapter**: `lib/network/logic.ts` is pure (no React, no clocks): backend `NetworkSummary` + `EventOut[]` become a `WorldDevice[]` view model; events become animation `Cue`s; also filters, KPIs and connection status (`loading | live | stale | down`). `lib/network/use-network-data.ts` is the only I/O: polling with `AbortController`, out-of-order/stale responses dropped, events de-duplicated by id with a 2 s overlap on `since`, slower polling in hidden tabs.
- **Truthfulness rules**: a device exists in the scene iff the backend lists it; `computing` is shown only after an `assignment_started` event for its *current* assignment (otherwise `assigned` or plain `busy`); audits run synchronously on result intake, so there is no "auditing" state, only `result_audited` flashes; history on first load is never replayed as live animation; slot positions are presentation-only (ring order = backend creation order).
- **Pixel world**: `components/network/pixel-world.ts` (PixiJS 8, used directly; `@pixi/react` was evaluated and not needed). Sprites are drawn from string matrices in `lib/network/sprite-data.ts` (no image assets). The ticker sleeps when nothing animates (30 fps cap otherwise); WebGL failure falls back to a plain device list; the canvas has a text equivalent (`role="img"` + live region) and the same data is in the table. Reference repos (ClawBoard, NetViz, procedural-isometric, pixi-react) are MIT-licensed; their *ideas* were used, no code was copied. NetViz animates mock traffic; ProofNet animates only real events.

### 3.3 Backend (Python + FastAPI)

- Python 3.12 (pinned; versions in CLAUDE.md §12), FastAPI, Pydantic v2, PyMongo's async API (`AsyncMongoClient`), NumPy, pandas, scikit-learn (pinned versions), PyJWT, a password hashing library.
- One process, one instance. On startup it launches the **reconciler** as an asyncio background task.
- All state transitions use conditional updates (`find_one_and_update` with a status precondition), so a duplicate request or a reconciler re-run cannot apply a transition twice.
- Expected load is tiny (a few devices and dashboards). The design keeps backend CPU work small because free hosts provide very little CPU (§16).

### 3.4 Kernels (`proofnet_kernels`)

A **kernel** is the unit of supported computation. Each task type is implemented by one kernel with a fixed interface:

| Function | Runs on | Purpose |
|---|---|---|
| `Params` (Pydantic model) | backend | Typed parameters + defaults + limits |
| `validate(dataset_profile, params)` | backend | Can this dataset + params be executed? Returns a structured report |
| `prepare(dataframe, params)` | backend | Produce a numeric, validated `PreparedData` (train/test arrays, encodings) |
| `map(X, y, worker_params)` | **worker** (`core/`) | Compute the partial result for one partition → JSON-serializable dict |
| `validate_partial(partial, chunk)` | backend | Structural check: keys, shapes, dtypes, finiteness, counts match chunk |
| `merge(partials)` | backend (`core/`, so future hierarchical merge is possible) | Combine partials into one merged state |
| `finalize(merged, prepared, params)` | backend (`server/`) | Build final model object, metrics on holdout, artifacts |
| `reference(prepared, params)` | backend (`server/`) | Centralized computation of the same merged state (correctness check) |
| `compare(a, b)` | backend (`server/`) | Discrepancy metrics between two states/partials — reused by Part 2 verification |

Kernels are versioned (`gaussian_nb_train@1`). A task records the kernel version it was created with; workers report which kernel bundle they loaded; the scheduler only assigns chunks to workers with a matching bundle.

### 3.5 Database and storage summary

| Data | Where | Persistent? |
|---|---|---|
| Users, devices, tasks, chunks, assignments, partial results, artifact metadata, events | MongoDB collections | Yes |
| Uploaded raw CSV | GridFS (`fs.files`/`fs.chunks`) | Yes |
| Prepared dataset (`.npz`: `X_train, y_train, X_test, y_test`, no pickle) | GridFS | Yes |
| Chunk input | **Not stored** — sliced on demand from the prepared array by row range and streamed as `.npz` | Derived (reproducible) |
| Partial results | Inline JSON in `partial_results` documents (they are KB-sized) | Yes |
| Final artifacts (`model.joblib`, `model.json`, `report.json`, `predictions.csv`) | GridFS + `artifacts` metadata | Yes |
| Backend temp files | Backend local disk / memory | **Ephemeral** — never relied on |
| Worker data | Phone memory (WASM heap) during execution | **Ephemeral** — discarded after each assignment |
| Device token | Phone `localStorage` (raw) + DB (SHA-256 hash) | Yes |

Capacity: Atlas free tier provides 512 MB total storage, so MVP limits are a 25 MB CSV per upload, demo data cleanup via a reset script, and no storage of chunk copies. Atlas free tier also caps throughput (on the order of 100 operations/second), which is why status snapshots are cached for ~1 s in the backend and heartbeat intervals are a few seconds.

---

## 4. Workload model

### 4.1 Why not "any ML training"

Most training algorithms (gradient-boosted trees, neural nets, k-means, SVMs) cannot be split into independent partitions whose results merge into the exact centralized model. Pretending otherwise would produce a demo that "works" but computes a different, unexplained result. ProofNet therefore starts with workloads whose decomposition is **exact**:

```
dataset ──partition by rows──▶ P1 … Pk ──map on devices──▶ partial statistics ──merge──▶ global statistics ──finalize──▶ model
```

This is the classic **sufficient statistics** pattern: the model depends on the data only through quantities (counts, means, co-moment matrices) that combine associatively across partitions.

### 4.2 MVP task types

#### `gaussian_nb_train` — classification (first, must-work)

- **Map (per partition, per class c):** `n_c`, `mean_c` (d-vector), `M2_c` (d-vector of summed squared deviations). Also overall `n`, `mean`, `M2` per feature (needed for sklearn's variance smoothing).
- **Merge:** pairwise parallel moment combination (Chan et al.):
  `n = n_a + n_b`, `δ = mean_b − mean_a`, `mean = mean_a + δ·n_b/n`, `M2 = M2_a + M2_b + δ²·n_a·n_b/n`. Numerically stable; avoids `Σx² − (Σx)²` cancellation.
- **Finalize:** `theta_ = mean_c`, `var_ = M2_c/n_c + ε`, `ε = var_smoothing · max(global feature variance)`, `class_prior_ = n_c / n`. Construct a scikit-learn `GaussianNB` with these fitted attributes (pinned sklearn version), save `model.joblib` and a portable `model.json`, evaluate accuracy / macro-F1 / confusion matrix on the holdout set.
- **Correctness:** equals `GaussianNB().fit(X_train, y_train)` within tolerance (default `rtol = 1e-8`), for any partitioning.
- **Partial size:** `C · (1 + 2d)` numbers — a few KB.

#### `linear_ridge_train` — regression (second, required for the MVP freeze)

- **Map:** `n`, `mean_x` (d), `mean_y`, `Sxx = Σ(x−x̄)(x−x̄)ᵀ` (d×d), `Sxy = Σ(x−x̄)(y−ȳ)` (d), `Syy`.
- **Merge:** parallel co-moment update: `Sxx = Sxx_a + Sxx_b + (n_a n_b / n)·δx δxᵀ`, `Sxy = … + (n_a n_b / n)·δx δy`, `Syy = … + (n_a n_b / n)·δy²`, means as above.
- **Finalize:** solve `(Sxx + αI) β = Sxy`, `intercept = ȳ − x̄ᵀβ` (identical to sklearn `Ridge(fit_intercept=True)`; `α = 0` → ordinary least squares, with least-squares fallback if singular). Metrics: R², RMSE, MAE on holdout.
- **Correctness:** equals centralized sklearn `Ridge`/`LinearRegression` within tolerance (coefficients `rtol = 1e-6`, defined in the kernel).
- **Partial size:** `O(d²)` numbers; with `d ≤ 64` that is ≤ ~4,200 numbers.

### 4.3 Honest notes about the workload

- Per-chunk compute for these kernels is small (typically well under a few seconds on a phone even for tens of thousands of rows). Transfer and runtime start-up often dominate. The dashboard shows **real measured timings**; ProofNet must never add artificial delays or fake progress.
- Holdout evaluation (metrics) is computed by the aggregator in the MVP. This is labelled as aggregator-side work in the UI. Distributed evaluation (a second map stage summing confusion matrices / squared errors) is a planned enhancement (§18).
- The first visibly compute-heavy workload planned after the MVP is **partitioned exact kNN prediction**: the training set is partitioned across devices; each device returns the top-k nearest neighbours per query from its partition; the merge takes the global top-k. Exact, mergeable, and CPU-bound.

### 4.4 Task specification (manifest)

A submitted task is a validated JSON manifest — this is ProofNet's deterministic "task understanding" mechanism:

```json
{
  "name": "Iris-like 3-class demo",
  "task_type": "gaussian_nb_train",
  "kernel_version": "1",
  "dataset_id": "ds_01J…",
  "params": {
    "target_column": "label",
    "feature_columns": ["f0", "f1", "…"],
    "missing_values": "drop_rows",
    "test_fraction": 0.2,
    "split_seed": 42,
    "var_smoothing": 1e-9
  },
  "execution": {
    "min_devices": 2,
    "max_devices": 4,
    "start_policy": "wait_for_min_devices"
  }
}
```

The task type determines: what inputs are required, what the computation is, how it partitions (row ranges), and how partials merge. No guesswork, no LLM parsing.

---

### 4.5 Iterative workload: image CNN (`cnn_image_train@1`, added in P6b)

Everything above is single-pass map → merge. Training a neural network is **iterative**: many steps, each depending on the previous result. ProofNet supports exactly one such workload, designed so the existing chunk/assignment/failure machinery is reused unchanged.

**Model.** A small CNN written in NumPy by ProofNet (`core/cnn.py`, runs unchanged on CPython and Pyodide): `conv3×3 → ReLU → maxpool2 → conv3×3 → ReLU → maxpool2 → dense → ReLU → dense(softmax)`, float32, channels-last uint8 inputs scaled by 1/255. Filters/units are catalog parameters with bounds (not code). For 28×28×1 inputs with 8/16/64 it has 27,562 parameters (~110 KB).

**Why not PyTorch/TensorFlow.** Neither runs in Pyodide, and the security rule (D5) is that only ProofNet-authored code executes on devices. A hand-written NumPy CNN is slower but fully controlled, testable (finite-difference gradient checks) and identical on backend and phone.

**Data never lives on the phone.** The prepared dataset stays in GridFS on the backend (uint8 arrays; an input of a 10k-image Fashion-MNIST subset is 7.9 MB in total). A device receives per assignment only `{X: uint8 slice of one mini-batch, y, w: float32 weights}` (~190 KB for 94 images) and keeps nothing afterwards.

**One round (= one SGD step).**
```
round r (model version r):
  batch   = indices of the global mini-batch r              (deterministic: per-epoch seeded permutation)
  plan    = split B rows across eligible devices ∝ benchmark (same allocate-rows rule as 6.2; tiny shares dropped)
  chunks  = one per device (role work, round r, row range inside the batch); assigned like any chunk
  device  : loss_sum, #correct, grad_sum = Σ over its images of ∂loss/∂w          (core/cnn.py map)
  backend : once every chunk of round r is accepted:
              grad_sum_total = Σ device grad_sums (float64);  g = grad_sum_total / B
              v ← μ·v − η·g ;  w ← w + v           (SGD with momentum) → model version r+1
              on sampled rounds: recompute the gradient of the whole batch centrally and compare
              create the chunks of round r+1 from the devices eligible *now*
after the last round: evaluate on the aggregator-held holdout, write artifacts
```
**Correctness.** The loss is a sum over samples, so the gradient of the global batch equals the sum of the gradients of any partition of it. The distributed update therefore equals centralized SGD on the same batch up to float32 summation order. This is *checked*, not assumed: on `verify_rounds` sampled rounds (always including round 0) the backend recomputes the whole batch's gradient and compares it normwise (tolerance 1e-4; observed ≈ 1e-6), and the test suite replays a whole training run distributed-vs-centralized (final weights agree to < 1e-3). A device that returns a well-formed but scaled gradient fails the check and the report says `FAILED`.

**State and recovery.** The model lives in `model_states` (`<task>:v<n>`: float32 weights + momentum), only the newest two versions are kept. Round closing is a conditional transition `collecting → closing → collecting(r+1) | done` guarded by `training.round`; model states and next-round chunks are deterministic (`unique(task, round, index)`), so a crashed attempt is simply re-run by the reconciler (`resume_training`). Per-round metrics go to `training_rounds` (loss/accuracy from the devices' own batch sums, per-device rows and timings). Failures inside a round use the normal rules: a dead/slow/corrupt device's chunk is retried elsewhere (max 3 attempts); an exhausted chunk fails the task.

**Honest limitations.** Per round the work is small; transfer and round-trip latency dominate on phones (≈ 2 s/round on a fast link with Pyodide). Accuracy is modest (a 27k-parameter CNN reaches ≈ 80% on Fashion-MNIST after a few epochs). Contributors see the images of the mini-batches they compute on. Only sampled rounds are checked against a centralized gradient; others are `accepted_unverified` (Part 2 generalizes this).

## 5. Task submission, validation and preparation

```mermaid
sequenceDiagram
    autonumber
    actor U as User (browser)
    participant WEB as Next.js
    participant API as FastAPI
    participant DB as MongoDB / GridFS

    U->>WEB: Open /tasks/new, pick task type
    WEB->>API: GET /task-types
    API-->>WEB: catalog + parameter schemas
    U->>WEB: Upload CSV
    WEB->>API: POST /datasets (multipart)
    API->>DB: store raw CSV in GridFS
    API->>API: profile columns (dtype, missing, unique, min/max)
    API-->>WEB: dataset_id + column profile
    U->>WEB: choose target, features, params
    WEB->>API: POST /tasks/validate (manifest)
    API->>API: kernel.validate() + current device pool → plan preview
    API-->>WEB: errors/warnings + predicted chunk split per device
    U->>WEB: Submit
    WEB->>API: POST /tasks
    API->>API: re-validate, kernel.prepare(), seeded shuffle + train/test split
    API->>DB: store prepared .npz in GridFS, insert task (status=queued)
    API-->>WEB: task_id
    WEB->>API: poll GET /tasks/{id}/status
```

### 5.1 Validation rules (MVP)

| Check | Rule |
|---|---|
| File | UTF-8 CSV with header; ≤ 25 MB; ≤ 300,000 rows; ≥ 100 rows |
| Columns | Target exists; 1 ≤ features ≤ 64; feature columns numeric (categorical encoding is out of MVP scope — reject with a clear message) |
| Missing values | `drop_rows` (default, reports how many) or `reject` |
| Classification target | 2–50 distinct classes; each class has ≥ 2 training rows |
| Regression target | numeric, finite |
| Degenerate data | constant features → warning; all-constant → reject |
| Params | Pydantic bounds (e.g. `0.05 ≤ test_fraction ≤ 0.5`, `alpha ≥ 0`) |
| Execution | `1 ≤ min_devices ≤ max_devices ≤ 8` |

Validation output is a structured report (`ok`, `errors[]`, `warnings[]`, `summary`) stored on the task and displayed in the wizard. Invalid manifests never become tasks.

### 5.2 Preparation

`kernel.prepare` converts the CSV into `float64` features and target (class labels mapped to `0..C−1`, mapping stored), applies the missing-value policy, applies a seeded shuffle, and splits train/test. The result is stored once as a pickle-free `.npz` in GridFS with its SHA-256. Chunks reference **row ranges of `X_train`**; chunk inputs are derived from this file, so they are reproducible byte-for-byte (important for Part 2 re-execution).

---

## 6. Chunking and device-aware scheduling

### 6.1 Eligibility

A device is eligible for a new assignment when **all** hold:

1. `status = idle` and heartbeat seen within the offline threshold (20 s).
2. Runtime ready: kernel bundle version matches the task's kernel version.
3. Has a benchmark score (measured this session).
4. Battery ≥ 20 % or charging (if the browser exposes battery info; unknown = allowed).
5. Memory budget ≥ chunk memory estimate (§6.2).
6. Not owner-disabled, and not in the chunk's `excluded_device_ids` (devices that already failed this chunk).
7. *(Part 2 hook)* Passes `DeviceEligibilityPolicy` — MVP implementation always returns true.

### 6.2 Planning algorithm (deterministic heuristic)

Executed once when a queued task can start (enough eligible devices per `start_policy`):

```
inputs: N training rows, d features, eligible devices E
sort E by (benchmark_score desc, device_id asc)                  # deterministic
k = min(|E|, max_devices, floor(N / MIN_CHUNK_ROWS))  (k ≥ 1)    # MIN_CHUNK_ROWS default 1000
take top-k devices; weights w_i = score_i / Σ score
rows_i = floor(N · w_i); give remaining rows by largest fractional part (ties → device order)
for each device i: if mem_estimate(rows_i, d) > mem_budget_i:
        split its share into ⌈estimate / budget⌉ equal chunks   # extra chunks queue on the same preferred device
create chunks with contiguous row ranges [start, end), preferred_device_id = device i
estimated_seconds = rows · d / score_i                          # shown in plan preview
```

- `benchmark_score` is measured in **cells/second** (rows × features processed by the real kernel code, §7.4), so it directly predicts chunk time.
- `mem_estimate = rows × (d + 1) × 8 bytes × 3` (input + working copies). Browser workers: `mem_budget = min(256 MB, 10 % of reported device memory)`; CLI workers: configured.
- Example: Phone A score 3.0 M cells/s, Phone B 1.5 M cells/s, N = 80,000 → exact shares 53,333.33 and 26,666.67; floors 53,333 and 26,666 leave 1 row, which goes to the larger fractional part (B, .67) → A gets 53,333 rows, B gets 26,667 rows. (Corrected in P5: an earlier draft said 53,334 / 26,666, which contradicted the algorithm.) Phone C offline → not considered. Phone D busy → not eligible; a `wait_for_min_devices` task stays queued until it frees up.

The plan (device shares, predicted times) is stored on the task and shown to the evaluator — explainability is part of the demo.

### 6.3 Assignment

Run by the reconciler and immediately after events that free capacity (task created, result received, device became idle):

1. For each `pending` chunk (oldest task first, then chunk index):
   - if its preferred device is eligible → assign to it;
   - else, if the chunk has been pending longer than `PREFERRED_WAIT` (10 s) or the preferred device is offline → assign to the best eligible device (by score) not in `excluded_device_ids`.
2. Assignment = insert an `assignments` document (`status = assigned`, lease) + conditional update of chunk `pending → assigned` + device `idle → busy`. If any precondition fails, roll back and try the next candidate.
3. **One active assignment per device** in the MVP (Pyodide is single-threaded).

The planner and assigner are separate, small, pure-ish modules behind interfaces (`ChunkPlanner`, `AssignmentPolicy`) so that Part 2 (replication, audits) and future schedulers (dynamic work-stealing, trust-weighted) can replace them.

---

## 7. Contributor devices

### 7.1 Android worker approach — options considered

| Option | Verdict |
|---|---|
| Windows `.exe` copied to phone | Not possible on Android. Rejected. |
| Termux + CPython + pip packages | Works for enthusiasts, but install friction is high, scientific packages are fragile to install on-device, and it is hard to set up reliably on an evaluator's phone. Kept as an *optional* path for the CLI worker. |
| Native Android app with embedded Python (e.g. Chaquopy) | Best background behaviour (foreground service), but adds Android build tooling, signing and sideloading. Good **future** upgrade. |
| Kivy / BeeWare app | Heavy packaging; same drawbacks as above. |
| **Browser + Pyodide (chosen)** | Open a URL, log in, tap "Start contributing". No install. WASM sandbox isolates the phone. NumPy ships with Pyodide. Same kernel source as the backend. Limitation: page must stay in the foreground with the screen on. |

**Accepted limitation and mitigation:** Android/Chrome throttles or freezes background tabs and pauses work when the screen turns off. The worker page therefore requests a **Screen Wake Lock** (requires HTTPS), tells the contributor to keep the tab in front and the phone charging, and the backend treats a frozen tab as an offline device (heartbeats stop → lease expires → chunk reassigned). This is honest and recoverable. A native wrapper with a foreground service is the documented future fix.

### 7.2 Worker runtime structure (browser)

- **Main thread (`controller.ts`)**: owns the device token, heartbeat/poll timer, HTTP calls, wake lock, UI. It keeps heartbeating *while* Python computes, because computation runs elsewhere.
- **Web Worker (`compute.worker.ts`)**: loads Pyodide (pinned version, from the jsDelivr CDN), loads NumPy, downloads the kernel bundle from the backend (`GET /runtime/kernels/{version}`), verifies its SHA-256 against `GET /runtime/manifest`, unpacks it, imports `proofnet_kernels.core`. Receives `{kernel, params, npz bytes}` messages, returns `{partial, compute_ms}`.
- **Cancel** = `worker.terminate()` + spin up a fresh Web Worker (simple and reliable).
- Pyodide + NumPy is a download of several MB on first load and is cached by the browser afterwards. Demo checklist: preload on Wi-Fi before the presentation.

The **CLI worker** (`workers/cli-worker`) implements the same protocol in CPython using the same `proofnet_kernels.core`. It is used for development, CI, multi-worker simulation on one laptop, fault injection, and laptops as optional extra contributors. It is not the primary demo device.

### 7.3 Capability discovery (what is genuinely available)

| Field | Browser source | Used for |
|---|---|---|
| `device_type` | user-agent heuristic + user confirmation (`android_phone`, `laptop`, `desktop`, `other`) | display, future policy |
| `model`, `platform_version` | `navigator.userAgentData.getHighEntropyValues` (Chromium) | display |
| `logical_cores` | `navigator.hardwareConcurrency` | display only (Pyodide is single-threaded) |
| `memory_gb_reported` | `navigator.deviceMemory` (Chromium, coarse, capped at 8) | memory budget |
| `storage_quota_mb` | `navigator.storage.estimate()` | display |
| `battery` | Battery Status API (Chrome Android) | eligibility |
| `network_type` | `navigator.connection.effectiveType` | display, future policy |
| `runtime` | Pyodide version, NumPy version, kernel bundle version | matching, Part 2 fingerprint |
| `benchmark` | measured (§7.4) | **scheduling weight** |

The browser cannot reveal the CPU model or exact RAM. ProofNet does not pretend otherwise; the measured benchmark is the real scheduling signal.

### 7.4 Benchmark (`bench_v1`)

Runs the actual `gaussian_nb` map kernel on a fixed-seed synthetic matrix (50,000 × 16, 3 classes) three times inside the Web Worker; score = cells / median seconds. Re-run on every worker session start. Stored with `bench_version` and runtime fingerprint so scores are comparable.

### 7.5 Registration flow

```mermaid
sequenceDiagram
    autonumber
    actor C as Contributor
    participant PG as Worker page (/contribute/run)
    participant WW as Web Worker (Pyodide)
    participant API as FastAPI
    participant DB as MongoDB

    C->>PG: Open ProofNet on phone, log in
    alt no device token in localStorage
        C->>PG: "Register this device" (name, type confirmed)
        PG->>API: POST /devices (user JWT, static capabilities)
        API->>DB: insert device (status=initializing), store token hash
        API-->>PG: device_id + device_token (shown once)
        PG->>PG: save token in localStorage
    end
    PG->>API: GET /runtime/manifest
    PG->>WW: start: load Pyodide + NumPy
    WW->>API: GET /runtime/kernels/{version}
    WW->>WW: verify SHA-256, import kernels
    WW->>WW: run bench_v1
    WW-->>PG: score, runtime fingerprint
    PG->>API: POST /worker/session (device token, capabilities, runtime, benchmark)
    API->>DB: device status=idle, new session_id
    API-->>PG: session_id, heartbeat interval
    loop every 2 s idle / 5 s busy
        PG->>API: POST /worker/heartbeat
        API-->>PG: directives (none / run / cancel)
    end
    PG->>PG: request Screen Wake Lock, show "Ready"
```

Device identity is **server-issued** (`dev_…`), bound to the owning user. Re-opening the page reuses the stored token → same device, new session. Clearing site data = re-register (old record can be disabled by the owner).

---

## 8. Communication protocol

### 8.1 Why polling

| Option | Assessment |
|---|---|
| Push to device (inbound) | Impossible behind NAT / mobile carriers. |
| WebSocket | Works, but adds reconnection logic, behaves poorly across free-host sleeps and redeploys, and is harder to debug. Not needed at MVP scale. |
| SSE | Fine for dashboards later; not needed for workers. |
| **HTTPS polling (chosen)** | Stateless, survives backend restarts, trivial to inspect with browser dev tools, works with any host and through tunnels. Latency of 1–2 s is irrelevant to the demo. |

### 8.2 Heartbeat = poll

`POST /worker/heartbeat` (device token) carries `{session_id, state, current_assignment_id?, battery?}` and returns:

```json
{
  "server_time": "2026-…Z",
  "next_heartbeat_ms": 2000,
  "directives": [
    { "type": "run", "assignment": {
        "assignment_id": "asg_…", "task_id": "tsk_…", "chunk_id": "chk_…",
        "kernel": "gaussian_nb_train", "kernel_version": "1",
        "params": { "n_classes": 3 },
        "input_url": "/api/v1/worker/assignments/asg_…/input",
        "input_sha256": "…", "n_rows": 26666, "n_features": 16,
        "deadline_at": "2026-…Z" } }
  ]
}
```

Other directive types: `cancel {assignment_id}`, `refresh_runtime {kernel_version}`, `reregister`.

Intervals: 2 s when idle, 5 s when busy. Offline threshold: 20 s without heartbeat.

### 8.3 Assignment protocol

1. Worker receives `run` → `POST /worker/assignments/{id}/start` (assigned → running).
2. `GET /worker/assignments/{id}/input` → `.npz` bytes with `X`, `y` for the row range. Worker checks SHA-256.
3. Worker runs `map` in the Web Worker. `np.load(..., allow_pickle=False)`.
4. `POST /worker/assignments/{id}/result` with:
   ```json
   { "kernel": "gaussian_nb_train", "kernel_version": "1",
     "input_sha256": "…", "n_rows": 26666,
     "payload": { "…partial statistics…" },
     "payload_sha256": "…",
     "timings": { "download_ms": 410, "compute_ms": 180, "total_ms": 650 },
     "runtime": { "kind": "pyodide", "pyodide": "x.y.z", "numpy": "a.b.c", "bundle": "1" } }
   ```
   or `POST /worker/assignments/{id}/fail` with `{code, message}`.
5. Worker returns to idle and keeps heartbeating.

Every write endpoint is **idempotent** per assignment (§11).

---

## 9. Execution, result collection and aggregation

### 9.1 End-to-end multi-device execution

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI (scheduler + gateway)
    participant DB as MongoDB
    participant A as Phone A (score 3.0M)
    participant B as Phone B (score 1.5M)
    participant AGG as Aggregator

    Note over API: task queued, 2 eligible devices
    API->>DB: plan: chunk 0 rows [0,53333) → A, chunk 1 rows [53333,80000) → B
    API->>DB: assignments asg_A, asg_B (assigned, leases)
    A->>API: heartbeat
    API-->>A: run asg_A
    B->>API: heartbeat
    API-->>B: run asg_B
    A->>API: start, GET input (53,333 rows)
    B->>API: start, GET input (26,667 rows)
    par compute
        A->>A: map(X0, y0) in Pyodide
    and
        B->>B: map(X1, y1) in Pyodide
    end
    B->>API: POST result (partial 1)
    API->>API: validate_partial ✓ → verification hook (MVP accept)
    API->>DB: chunk 1 completed
    A->>API: POST result (partial 0)
    API->>API: validate_partial ✓ → accept
    API->>DB: chunk 0 completed → all chunks done → task aggregating (atomic)
    API->>AGG: aggregate(task)
    AGG->>AGG: merge → finalize → metrics → reference check
    AGG->>DB: artifacts in GridFS, task completed
```

### 9.2 Result intake pipeline

```
POST result
  → auth: token belongs to assignment's device; assignment in {running}
  → structural: kernel/version match, input_sha256 match, kernel.validate_partial (shapes, finite, counts == chunk rows)
  → store partial_results doc (acceptance = pending)
  → VerificationHook.on_partial(partial) → MVP: accept → acceptance = accepted_unverified
  → assignment succeeded; chunk completed (accepted_assignment_id); device idle; stats updated
  → if all chunks completed: conditional update task running → aggregating, schedule aggregation
```

### 9.3 Aggregation

```mermaid
flowchart TD
    P["Accepted partial results<br/>(one per chunk, acceptance ∈ policy)"] --> CK{"All chunks covered?<br/>row ranges exactly tile [0, N)"}
    CK -- no --> WAIT["Stay running<br/>(retries / reassignment)"]
    CK -- yes --> M["kernel.merge()<br/>pairwise in chunk-index order"]
    M --> F["kernel.finalize()<br/>sklearn model object · model.json"]
    F --> EV["Holdout metrics<br/>(aggregator-side)"]
    F --> REF{"Reference check enabled?<br/>(N ≤ 300k)"}
    REF -- yes --> R["kernel.reference() centralized fit<br/>kernel.compare(distributed, reference)"]
    REF -- no --> REP
    R --> REP["report.json<br/>per-device contribution · timings · plan vs actual · correctness"]
    EV --> REP
    REP --> ART["Artifacts → GridFS<br/>model.joblib · model.json · report.json · predictions.csv"]
    ART --> DONE["task completed"]
```

- Merge order is fixed (chunk index) so results are reproducible.
- The **reference check** — "distributed result equals centralized result (max relative difference 3.1e-14) ✓" — is the strongest proof for evaluators that the distributed computation is real and correct. It is cheap for these kernels.
- Aggregation is idempotent: guarded by the `running → aggregating` transition; if the backend restarts mid-aggregation, the reconciler re-runs aggregation for tasks stuck in `aggregating`.

### 9.4 Artifact management

| Artifact | Content |
|---|---|
| `model.joblib` | scikit-learn estimator built by the backend from merged statistics (pinned sklearn version recorded in report) |
| `model.json` | Portable parameters (classes, means, variances / coefficients, intercept, feature names) — safe to load anywhere |
| `report.json` | Task manifest, validation summary, plan, per-chunk/per-device timings, metrics, reference check, kernel and runtime versions, digests |
| `predictions.csv` | Holdout predictions vs actual |

Downloads via `GET /artifacts/{id}/download` (owner only). The UI warns that `.joblib` files are pickles and should only be loaded from a trusted ProofNet instance; `model.json` is the safe alternative.

---

## 10. State machines

### 10.1 Task

```mermaid
stateDiagram-v2
    [*] --> queued: POST /tasks (validated + prepared)
    queued --> running: plan created (enough eligible devices)
    queued --> cancelled: user cancels
    queued --> failed: queue timeout (no devices)
    running --> aggregating: all chunks completed
    running --> failed: a chunk exhausted retries / task timeout
    running --> cancelled: user cancels
    aggregating --> completed: artifacts stored
    aggregating --> failed: merge/finalize error
    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

Every transition appends to `status_history` and emits an event. Invalid submissions never enter this machine.

### 10.2 Chunk and assignment

```mermaid
stateDiagram-v2
    state Chunk {
        [*] --> pending
        pending --> assigned: assignment created
        assigned --> completed: accepted result
        assigned --> pending: attempt failed/expired/rejected (attempts left)
        assigned --> failed: attempts exhausted (max 3)
        pending --> cancelled
        assigned --> cancelled
    }
    state Assignment {
        [*] --> a_assigned
        a_assigned --> a_running: worker /start
        a_running --> a_succeeded: valid result accepted
        a_running --> a_rejected: invalid result
        a_running --> a_failed: worker reported failure
        a_assigned --> a_expired: lease/heartbeat lost
        a_running --> a_expired: lease/heartbeat lost
        a_assigned --> a_cancelled
        a_running --> a_cancelled
    }
```

(`a_` prefixes only disambiguate the diagram; stored values are `assigned, running, succeeded, rejected, failed, expired, cancelled`.)

A **chunk** is a logical piece of work. An **assignment** is one attempt to execute a chunk on one device (it plays the role of "Execution"). Keeping them separate is what later allows *several* assignments of the same chunk (replication, audits) without changing the model.

### 10.3 Device

```mermaid
stateDiagram-v2
    [*] --> initializing: registered / new session
    initializing --> idle: runtime loaded + benchmark posted
    idle --> busy: assignment created
    busy --> idle: assignment finished (any outcome)
    idle --> offline: no heartbeat 20 s
    busy --> offline: no heartbeat 20 s (assignment expires)
    offline --> initializing: page reopened / session restarted
    idle --> disabled: owner disables
    offline --> disabled
    disabled --> initializing: owner re-enables + new session
```

`online` is derived (`last_seen_at` within threshold) and also materialized by the reconciler as `offline`.

---

## 11. Failure handling (MVP level)

| Situation | Detection | Response |
|---|---|---|
| Device offline before task | stale heartbeat | Not eligible; task waits per `start_policy` |
| Device offline during chunk | no heartbeat 20 s (reconciler) | Assignment `expired`, device `offline`, chunk → `pending` with device in `excluded_device_ids`, reassigned |
| Chunk execution error | worker `POST …/fail` | Assignment `failed`; retry on a different device if possible |
| Timeout | `deadline_at = assigned_at + clamp(4 × estimated_seconds + input_bytes / MIN_BANDWIDTH, 60 s, 600 s)` with `MIN_BANDWIDTH = 50 KB/s` | Assignment `expired`; retry |
| Malformed / invalid output | structural validation fails | Assignment `rejected`, device `invalid_results += 1` (first Part 2 signal), retry elsewhere |
| Duplicate result (same assignment) | assignment already `succeeded` | `200` with no state change (idempotent) |
| Late result (expired/cancelled assignment) | assignment not `running` | `409`; stored as a `late_result` event with its payload digest (useful for Part 2), never merged |
| Retries exhausted | `attempt_count = 3` | Chunk `failed` → task `failed` with an explanation; no partial aggregation |
| User cancels | `POST /tasks/{id}/cancel` | Task `cancelled`, open assignments `cancelled`, next heartbeat returns `cancel` directive → worker terminates its Web Worker |
| No devices for too long | queued > `QUEUE_TIMEOUT` (10 min) | Task `failed` ("no eligible devices") |
| Task runs too long | running > `TASK_TIMEOUT` (15 min) | Task `failed` |
| Backend restart | — | All state in MongoDB; reconciler resumes: expires stale leases, re-runs stuck aggregation |
| Worker restarts mid-chunk (page reload) | `POST /worker/session` while an assignment is `running` | Assignment `expired` (`SESSION_RESTARTED`); chunk retried. An assignment that was only `assigned` is simply re-dispatched to the new session |
| Nobody else can take a retried chunk | chunk pending longer than `EXCLUSION_RELAX` (15 s) with no eligible non-excluded device | An excluded device may retry it (still bounded by `max_attempts`) |
| Backend sleeping (free host) | first request is slow | Workers retry with backoff; dashboard shows "connecting"; demo runbook pre-warms |

All retry/timeout constants live in one config module.

**Design notes (P6).** (1) The deadline includes a *transfer allowance* because on phones the chunk download, not the compute, dominates: a real run needed 110 s to download a 10.9 MB chunk on weak 5G, which a compute-only deadline (60 s floor) would have expired. (2) `excluded_device_ids` is a preference, not a hard rule: it is honoured while another eligible device exists, and relaxed after `EXCLUSION_RELAX` so that a single-device deployment still retries (the three-attempt bound still ends a task whose chunk always fails). (3) `cancel` directives are derived, not stored: when a worker's heartbeat reports a `current_assignment_id` whose assignment is `cancelled`/`expired`/`failed`/`rejected`, the response carries `cancel {assignment_id}`; the browser worker terminates its Web Worker and starts a fresh one.

---

## 12. Data model (MongoDB)

IDs are prefixed readable strings (`usr_`, `dev_`, `ds_`, `tsk_`, `chk_`, `asg_`, `pr_`, `art_`, `evt_`) stored in `_id`. All timestamps are server-side UTC.

```mermaid
erDiagram
    USER ||--o{ DEVICE : owns
    USER ||--o{ DATASET : uploads
    USER ||--o{ TASK : submits
    DATASET ||--o{ TASK : "input of"
    TASK ||--|{ CHUNK : "partitioned into"
    CHUNK ||--o{ ASSIGNMENT : "attempted by"
    DEVICE ||--o{ ASSIGNMENT : executes
    ASSIGNMENT ||--o| PARTIAL_RESULT : produces
    TASK ||--o{ ARTIFACT : outputs
    TASK ||--o{ EVENT : logs
    DEVICE ||--o{ EVENT : logs
```

| Collection | Key fields | Notes |
|---|---|---|
| `users` | `email`, `password_hash`, `display_name`, `roles[]` (`user`, `contributor`, `admin`), `created_at` | One account can submit and contribute |
| `devices` | `owner_user_id`, `name`, `device_type`, `capabilities{…}` (§7.3, embedded = "DeviceCapability"), `runtime{kind, versions, bundle}`, `benchmark{score_cells_per_sec, bench_version, measured_at}`, `status`, `session_id`, `last_seen_at`, `current_assignment_id`, `token_hash`, `stats{succeeded, failed, expired, invalid_results, cells_processed}` | `stats` is the raw material for future trust |
| `datasets` | `owner_user_id`, `filename`, `size_bytes`, `sha256`, `raw_file_id` (GridFS), `profile{n_rows, columns[{name, dtype, missing, unique, min, max}]}` | "TaskInput" = dataset + task.prepared |
| `tasks` | `owner_user_id`, `name`, `task_type`, `kernel_version`, `dataset_id`, `params`, `execution{min_devices, max_devices, start_policy}`, `validation{…}`, `prepared{file_id, sha256, n_train, n_test, n_features, class_labels[], feature_names[]}`, `plan{created_at, shares[{device_id, rows, est_seconds}]}`, `status`, `status_history[]`, `result{artifact_ids[], metrics, reference_check}`, `error`, `verification_policy{mode: "none"}` | `verification_policy` exists from day one; Part 2 adds modes |
| `chunks` | `task_id`, `round?` (image CNN: one chunk set per round; `unique(task, round, index)`), `index`, `role` (`work`; future `challenge`), `row_start`, `row_end`, `n_rows`, `work_units`, `input_sha256`, `preferred_device_id`, `status`, `attempt_count`, `max_attempts`, `excluded_device_ids[]`, `accepted_assignment_id` | Row ranges must tile `[0, n_train)` |
| `assignments` | `task_id`, `chunk_id`, `device_id`, `attempt_no`, `purpose` (`primary`; future `replica`, `audit`), `status`, `assigned_at`, `started_at`, `finished_at`, `deadline_at`, `timings{download_ms, compute_ms, upload_ms, total_ms}`, `runtime_fingerprint`, `error{code, message}`, `partial_result_id` | = "Assignment + Execution" |
| `partial_results` | `assignment_id`, `chunk_id`, `task_id`, `device_id`, `kernel`, `kernel_version`, `payload`, `payload_sha256`, `structural{ok, errors[]}`, `acceptance` (`accepted_unverified`, `rejected_structural`; future `verified`, `disputed`, `rejected_verification`), `received_at` | Never deleted during a task — Part 2 needs them |
| `artifacts` | `task_id`, `kind`, `filename`, `file_id`, `size_bytes`, `sha256`, `created_at` | "FinalResult" = task.result + artifacts |
| `image_datasets` | `owner_user_id`, `filename`, `size_bytes`, `sha256`, `npz_file_id` (GridFS uint8 arrays), `profile{n_images, shape, classes, class_counts, samples[]}` | P6b; zips are never stored, only the decoded arrays |
| `model_states` | `task_id`, `version`, `weights_b64` (float32), `velocity_b64` | P6b; newest two versions per task |
| `training_rounds` | `task_id`, `round`, `loss`, `accuracy`, `n`, `devices[{device_id, name, rows, compute_ms, download_ms}]`, `verification?` | P6b; the live training curve |
| `events` | `ts`, `type`, `task_id?`, `device_id?`, `chunk_id?`, `assignment_id?`, `data{}` | Append-only audit/timeline; feeds dashboards |

Indexes (minimum): `devices(status, last_seen_at)`, `chunks(task_id, status)`, `assignments(device_id, status)`, `assignments(chunk_id)`, `events(task_id, ts)`, `tasks(owner_user_id, created_at)`.

**Part 2 collections (§17):**

| Collection | Key fields |
|---|---|
| `verification_records` | `task_id`, `chunk_id`, `assignment_id`, `device_id`, `class_key`, `mode`, `audit_probability`, `draw`, `audited`, `decision`, `discrepancy`, `tolerance`, `tolerance_source`, `calibration_limit`, `evidence`, `threshold`, `suspicion`, `accused` — one per received result (plus `retroactive` / `forensic` records) |
| `device_trust` | `_id`=device id, `user_id`, `status` (`probation/trusted/watch/quarantined`), `results_seen`, `audits`, `n_clean`, `exceedances`, `e_state{n, exceed, R[]}`, `evidence`, `suspicion`, `memory`, `inherited_memory`, `quarantine{at, reason, source}`, `history[]` (≤ 60) |
| `calibrations` | `_id`=class key, `samples[]` (≤ 5000 honest discrepancies), `n`, `limit` |
| `reward_entries` | `_id`=assignment id, `user_id`, `device_id`, `task_id`, `work_units`, `trust`, `multiplier`, `amount`, `acceptance`, `status` (`pending/confirmed/revoked`) |
| `reward_events` | append-only `accrue/confirm/revoke` with `seq`; balances are replayable from this alone |
| `security_events` | `ts`, `kind`, `severity`, `user_id?`, `device_id?`, `task_id?`, `data{}` |
| `login_attempts` | `_id`=e-mail, `fails`, `locked_until` |

`devices` gains `quarantined`; `tasks.verification_policy.mode` is `adaptive|full|off`; `tasks.integrity_flags[]` marks finished tasks containing a result later proven wrong. `challenges`, `attack_events` and `reputation_events` from the original plan were not needed (ground truth lives in the attacker worker's own log; trust history lives in `device_trust.history`).

---

## 13. API surface (`/api/v1`)

Auth: users use `Authorization: Bearer <JWT>`; workers use `Authorization: Bearer <device_token>`. CORS allows the Vercel origin(s) and local dev origins.

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness + DB check (used to pre-warm) |
| `POST /auth/signup`, `POST /auth/login`, `GET /auth/me` | none / user | Accounts |
| `GET /task-types` | user | Catalog: task types, kernel versions, parameter JSON schemas |
| `POST /datasets` | user | Upload CSV (multipart) → profile |
| `GET /datasets/{id}` | owner | Dataset profile |
| `POST /tasks/validate` | user | Dry-run validation + plan preview against current devices |
| `POST /tasks` | user | Create task (validate + prepare) |
| `GET /tasks`, `GET /tasks/{id}` | owner | Task list / detail |
| `GET /tasks/{id}/status` | owner or demo-public flag | Compact live snapshot (task, chunks, assignments, devices) — cached ~1 s |
| `GET /tasks/{id}/events` | owner | Timeline |
| `POST /tasks/{id}/cancel` | owner | Cancel |
| `GET /tasks/{id}/artifacts`, `GET /artifacts/{id}/download` | owner | Results |
| `POST /devices` | user | Register a device → `device_id`, `device_token` |
| `GET /devices/mine`, `PATCH /devices/{id}` | owner | List, rename, disable/enable |
| `POST /image-datasets`, `GET /image-datasets[/{id}]` | user | Upload a zip of class folders or a pixel CSV (≤ 25 MB) → decoded, profiled (class counts + sample thumbnails). Separate from `/datasets` |
| `POST /image-tasks/validate`, `POST /image-tasks` | user | CNN task: validate + per-round batch-split preview / create (prepare split → `queued`) |
| `GET /tasks/{id}/training` | owner | Training curve: per-round loss/accuracy, per-device rows and timings, centrally verified rounds |
| `GET /network/summary` | user (or demo-public) | Online/idle/busy counts, device list (names, scores, states), active chunks |
| `GET /network/events?since=&limit=` | admin | Newest `limit` (<= 200) events across all tasks/devices, oldest first, with human-readable `message`; `since` = strictly newer (added 2026-10-09) |
| `GET /runtime/manifest` | device | Pyodide version, kernel bundle version + SHA-256, intervals |
| `GET /runtime/kernels/{version}` | device | Kernel bundle zip (`proofnet_kernels/core`) |
| `POST /worker/session` | device | Start session: capabilities, runtime, benchmark |
| `POST /worker/heartbeat` | device | Heartbeat + directives |
| `POST /worker/assignments/{id}/start` | device | assigned → running |
| `GET /worker/assignments/{id}/input` | device | Chunk input `.npz` |
| `POST /worker/assignments/{id}/result` | device | Submit partial result |
| `POST /worker/assignments/{id}/fail` | device | Report failure |
| `POST /admin/demo/reset` | admin | Clear demo tasks/datasets (keeps users/devices) |
| `GET /trust/overview` | user (admin: all) | Device trust profiles, calibration classes, recent verification records, PWAV parameters |
| `GET /trust/devices/{id}`, `GET /trust/records?task_id=` | owner / admin | One profile with history; verification records of a task or of my devices |
| `POST /trust/simulate` | user | Population simulation of the audit policy (pure model) |
| `GET /rewards/me`, `GET /rewards/network`, `GET /rewards/ledger/check` | user | Balances (pending/confirmed/revoked), entries, leaderboard, replay-vs-entries integrity check |
| `GET /security/overview` | user (admin: all) | Event log, quarantined devices, 24 h counters, list of controls |
| `POST /security/devices/{id}/quarantine`, `…/reinstate` | admin | Manual quarantine / reinstatement |

Error format: `{ "error": { "code": "VALIDATION_FAILED", "message": "…", "details": [...] } }`.

---

## 14. Security boundary (honest statement)

*Image uploads (P6b)* are decoded in memory only: never extracted to disk, only png/jpg members, member/total/pixel/count caps checked before decoding (decompression-bomb guard), class names from the folder name only, no member path is ever used for I/O. Weights and gradients travel as base64 float32 inside JSON/`.npz` with `allow_pickle=False`; gradient payloads are validated for size, finiteness and counts like any partial result.

**What the MVP protects:**

- **No user code executes anywhere.** Users choose from ProofNet-authored kernels. This is the primary RCE defence.
- **Contributor phones are sandboxed** by the browser + WebAssembly; the kernel bundle is integrity-checked by SHA-256.
- **No pickle across trust boundaries:** chunk inputs are `.npz` loaded with `allow_pickle=False`; worker results are JSON; uploaded CSVs are parsed by pandas with size/row limits; `.joblib` is only *produced* by the backend.
- **Authentication:** hashed passwords, JWT for users, random per-device tokens stored only as hashes, workers can only touch their own assignments.
- **Input limits:** upload size, row/feature caps, request body limits, simple per-token rate limiting.
- **Transport:** HTTPS in deployment (Vercel and the backend host both provide TLS).

**What the MVP does not protect (stated in the UI and report):**

- A contributor can return **well-formed but wrong** statistics. Part 1 accepted them unverified; Part 2 (§17) audits them probabilistically (never below a floor), repairs finished tasks via the reference check + forensics, and quarantines repeat offenders. Corruptions smaller than the tolerance and unaudited cheats on already-merged CNN rounds remain possible (§17.8).
- **Data confidentiality:** contributors receive raw rows of their partition. Do not upload sensitive data.
- Sybil devices, collusion, denial-of-service by many fake devices, and token theft from a contributor's own browser.
- It is a research prototype, not production-grade secure remote execution.

**Future path for custom code (not MVP):** team-reviewed, versioned kernel plugins first; later, possibly user-supplied *map* functions executed only inside the Pyodide sandbox on devices, while merge/finalize remain trusted backend code.

---

## 15. Observability for the demo

The `/network` and `/tasks/[id]` screens are designed for evaluators:

- Device cards: name, model, benchmark score, state (idle/busy/offline), current chunk, last heartbeat age.
- **Plan view**: rows per device and *why* (score share), predicted vs actual time.
- **Chunk table**: chunk → device → status → attempt → download/compute/upload ms.
- **Device lanes timeline**: horizontal bars per device showing when each chunk ran.
- Aggregation panel: merge done, metrics, **reference check result**, artifact downloads.
- Event feed: "Phone B finished chunk 1 in 180 ms", "Phone A went offline — chunk 0 reassigned to Laptop".
- Worker console on each phone shows its own log, so the audience can see the phone working.

No external monitoring stack. Backend logs to stdout (host log viewer) and to the `events` collection.

---

## 16. Deployment and local development

### 16.1 Deployed topology (all free tier)

```mermaid
flowchart LR
    PH["Android phones<br/>Chrome"] -- HTTPS --> V["Vercel<br/>Next.js frontend + worker page"]
    PH -- "HTTPS API (CORS)" --> BE["FastAPI container<br/>free-tier host"]
    BR["User / evaluator browsers"] -- HTTPS --> V
    BR -- "HTTPS API (CORS)" --> BE
    PH -- "Pyodide + NumPy (cached)" --> CDN["jsDelivr CDN"]
    BE -- "TLS" --> ATL["MongoDB Atlas free cluster<br/>collections + GridFS"]
```

| Component | Host | Notes |
|---|---|---|
| Frontend | **Vercel** (Hobby/free) | `NEXT_PUBLIC_API_BASE_URL` points at the backend |
| Backend | **Any free container host that runs a long-lived `uvicorn` process with HTTPS.** **Chosen (P0, 2026-10-07): Render free web service.** Considered and rejected: Hugging Face Spaces (Docker Spaces now require a paid plan to create), Koyeb (no free plan listed). Terms re-checked in P0; re-check before deploying. | Must be a single instance. Must not depend on local disk. |
| Database + files | **MongoDB Atlas free cluster** | Allowlist the backend host egress (or `0.0.0.0/0` with strong credentials for the prototype) |
| Python runtime for phones | Pyodide from jsDelivr | Version pinned |

Free-tier realities to design around (verified for Render as of mid-2026; re-check before deploying):

- Render free web services spin down after 15 minutes without inbound traffic and take about a minute to wake; they have an ephemeral filesystem and very small CPU (0.1 CPU, 512 MB). Heartbeating workers keep the service awake during a demo; the runbook pre-warms it via `/health`.
- Atlas free cluster: 512 MB storage and limited operations per second → caching of status snapshots, modest polling intervals, reset script for old demo data.
- Because backend CPU may be tiny, keep backend computation light (merge + small reference check). If the reference check becomes slow on the chosen host, cap its dataset size.

The backend is **hosting-agnostic**: configuration entirely via environment variables (`MONGODB_URI`, `MONGODB_DB`, `JWT_SECRET`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `PUBLIC_BASE_URL`). Container packaging files are created during implementation, not now.

**Storage budget (added after the free tier filled up, 2026-10-08).** Atlas M0 blocks *all writes* at 512 MB, which also stops the backend from starting (it creates indexes at startup). Defences: (1) a periodic storage guard (`STORAGE_BUDGET_MB`, default 380; checked every 2 min) releases old data above the budget, oldest first and never data of queued/running tasks: gradient payloads of finished image-training tasks, then prepared datasets of finished tasks, then raw uploaded datasets; artifacts and all metadata are kept; (2) uploads first make room and are refused with `507 STORAGE_FULL` if the database is still too full; a removed dataset gives `410 DATASET_REMOVED` ("upload it again"); (3) gradient payloads are dropped as soon as their round has been applied (only digests and metadata stay); (4) a failure to create indexes at startup is logged and the app still serves reads and logins. The biggest consumers are raw/prepared CSV copies (15 MB each for 100k rows) and CNN gradients (~150 KB per device per round).

### 16.2 Demo fallback: laptop backend + tunnel (still free)

If the cloud backend is slow or unavailable on demo day: run FastAPI on a team laptop (still using Atlas, or a local MongoDB), expose it with a free HTTPS tunnel (e.g. a Cloudflare quick tunnel), and point the frontend at it (a runtime "API URL" setting in the frontend, stored per browser, makes this a 30-second switch). Phones only need internet access. HTTPS is required because the Vercel page is HTTPS (mixed content) and Wake Lock needs a secure context.

### 16.3 Local development

| Process | Command (conceptual) | Port |
|---|---|---|
| MongoDB | local `mongod`, **or** a personal Atlas free database (`proofnet_dev`) | 27017 |
| Backend | `uvicorn proofnet_api.main:app --reload --host 0.0.0.0` | 8000 |
| Frontend | `next dev` | 3000 |
| Workers | `/contribute/run` in a desktop browser tab (several tabs = several devices), and/or `python -m cli_worker --count 3` | — |

Three processes plus the database — no Docker, queue or cache required. Testing a real phone against a laptop: use the tunnel from §16.2 (HTTPS), or accept that Wake Lock is unavailable over plain-HTTP LAN.

---

## 17. Part 2 — verification, trust, rewards, security (implemented in P8–P11)

> Status 2026-10-08: implemented and tested ahead of the M3 gate at the owner's explicit request (the "Part 2 only after M3" rule in `CLAUDE.md` was overridden by the owner; Part 1's real-phone checks remain open and are the owner's to run). The Part 1 compute fabric was extended through the insertion points below; nothing was rewritten.

### 17.1 Flow

```mermaid
flowchart LR
    RES["Result received<br/>POST /worker/assignments/{id}/result"] --> SV["Structural validation<br/>(shapes, finite, counts)"]
    SV --> DEC{"Audit?<br/>HMAC draw < p(device)"}
    DEC -- "no" --> UNV["accepted_unverified<br/>reward pending"]
    DEC -- "yes" --> REC["Backend recomputes the chunk<br/>discrepancy vs tolerance"]
    REC -- "within tolerance" --> VER["verified<br/>reward confirmed<br/>+ honest sample for calibration"]
    REC -- "exceeds" --> REJ["rejected_verification<br/>assignment rejected, chunk re-queued<br/>generic 422 to the worker"]
    VER --> EV["Evidence update (e-detector)<br/>suspicion, memory, trust"]
    REJ --> EV
    EV -- "evidence ≥ threshold" --> Q["Quarantine<br/>no new work, clawback,<br/>retroactive audit"]
    UNV --> AGG["Aggregation"]
    VER --> AGG
    AGG --> REF{"Reference check<br/>(merged vs centralized)"}
    REF -- "fails" --> FOR["Forensics: recompute every chunk,<br/>name culprit, quarantine, re-run chunk"]
    FOR --> AGG
    REF -- "passes" --> DONE["Task completed<br/>pending rewards confirmed"]
```

### 17.2 Verification (P8)

- **Audit by backend recomputation.** The MVP kernels are cheap enough to recompute any single chunk on the backend from the stored prepared data (`verification/recompute.py`: `expected_partial`, `discrepancy`). The oracle is the backend itself, so audits are immune to collusion between devices. (Replica assignments and hidden challenge chunks from the original plan were **not** built: recomputation gives exact ground truth for these workloads and a replica would only add a second untrusted opinion.)
- **Where it runs.** Inside result intake, after structural validation and **before** the assignment flips to `succeeded`. A failed audit releases the attempt exactly like a structural failure (`assignment: rejected`, `error.code VERIFICATION_FAILED`, chunk re-queued with the device excluded for that chunk); the worker only sees the generic `422 INVALID_RESULT`.
- **Modes** per task (`verification_policy.mode`): `adaptive` (default), `full` (audit every result), `off` (the Part 1 behaviour; legacy `none` is read as `off`).
- **The audit draw is reproducible and unpredictable:** `u = HMAC-SHA256(key, assignment_id) / 2^64` with a server-side key derived from the JWT secret; audited iff `u < p`. The draw, probability and decision are stored in `verification_records`, so any record can be re-derived.
- **Discrepancy.** GNB/Ridge: `kernel.server.compare` (normwise relative, max over fields); CNN: max of the relative gradient-sum difference, the loss-sum difference and the correct-count difference.
- **Tolerance per class.** A *class* is `kernel@version | runtime kind` (e.g. `gaussian_nb_train@1|pyodide`). `tolerance = min(hard, max(margin × L, floor))` where `L` is the class's (p, γ) tolerance limit (§17.3) once `calibrations` holds enough honest samples, `hard` is the kernel's fixed numerical bound (`TOLERANCE`/`GRADIENT_TOLERANCE`) and `floor` the smallest tolerance the trust system uses (`AUDIT_FLOOR`). Until the class has `ln(1−γ)/ln p` samples (59 for p = .95; 99 for p = .97) the tolerance is `hard`. Only discrepancies ≤ `hard` from devices not under suspicion are ever added to a calibration, and `hard` caps the result, so a poisoned calibration can never loosen the check.
- **Exceedance** `Z = 1` iff discrepancy > tolerance. An exceedance is also a rejection: the result is not merged.
- **Acceptance states** on `partial_results`: `pending` → `verified` | `accepted_unverified` | `rejected_verification` | `rejected_structural` | `rejected_forensic` | `rejected_retroactive`. Aggregation merges `verified` and `accepted_unverified` only.
- **Safety net 1 — reference check + forensics.** If the finished task's merged statistics differ from the centralized recomputation, `verification/forensics.py` recomputes every chunk, names the culprit chunks (discrepancy > `hard`: a deterministic proof), marks them `rejected_forensic`, revokes their rewards, quarantines the devices, re-queues the chunks (task `aggregating → running`) and aggregation runs again. The final model is correct even if an unaudited cheat got through.
- **Safety net 2 — retroactive audit.** On quarantine, all earlier `accepted_unverified` results of that device are recomputed: correct ones become `verified`; wrong ones are rejected, their rewards revoked and, in a still-running CSV task, the chunk is re-queued. Finished/iterative tasks get `integrity_flags` (a merged CNN round cannot be un-merged; gradient payloads are dropped after a round closes).
- **Collections:** `verification_records` (one per result: mode, probability, draw, audited, discrepancy, tolerance and its source, evidence, decision), `calibrations`, `device_trust`.

### 17.3 PWAV: Evidence-Adaptive Auditing (P10) — `trust/pwav.py`

> Honest note: PWAV is implemented here from the mechanism summary in `CLAUDE.md` (per-class order-statistic tolerance limits, betting e-process evidence, adaptive audit probability with a minimum floor, persistent suspicion memory), **not** from the original paper, which was not available. The guarantees below are proved for *this* implementation (see the module docstring and `tests/test_pwav.py`); they may differ in detail from the paper's.

1. **Distribution-free (p, γ) tolerance limit.** For `n` honest discrepancy samples the `j`-th smallest is an upper tolerance limit iff `P(Bin(n, p) ≤ j−1) ≥ γ`; with probability ≥ γ at least a fraction `p` of future honest discrepancies lie below it. Defaults: `p = 1 − q0 = 0.97`, `γ = 0.95`, margin ×10.
2. **Betting e-detector with lifetime false-accusation control.** For each audit `Z ∈ {0,1}`; under honesty `E[Z | past] ≤ q0`, so `f = 1 + λ(Z − q0)` (0 < λ < 1/q0) has conditional mean ≤ 1. Per bet λ ∈ {1,2,4,8,16,30}: Shiryaev–Roberts `R_t = (R_{t−1} + 1)·f_t`, averaged over λ (restarting at every audit, so a sleeper cannot hide behind a long honest history). `R_t` is a sum over start times of products that are non-negative supermartingales started at 1; if `R_t ≥ h` within `N` audits some term is ≥ h/N, Ville's inequality bounds that by `N/h` and a union bound over starts gives `P(false alarm within N audits) ≤ N²/h`. Accuse when `R_t ≥ h_m = N_m² / α_m` for audit block `m` (`N_m = 1000·2^m`, `α_m = α·6/(π²(m+1)²)`, Σα_m ≤ α): **the probability that an honest device is ever accused is ≤ α** (conditionally on the tolerance limit being valid). Defaults: `α = 1e-3`; a device that cheats on every audit is accused after ≈ 12 audits (`tests/test_pwav.py`).
3. **Suspicion** = log-scale position of the evidence between its all-clean baseline and the threshold, in [0, 1].
4. **Adaptive audit probability.** `p = 1` during probation (first 5 results); then `base(n_clean) = floor + (initial − floor)/(1 + n_clean/τ)` (floor 5 %, initial 30 %, τ = 10 clean audits), raised to `base + (1 − base)·memory`. Never below the floor, for any history.
5. **Persistent suspicion memory.** `memory ← max(0.98·memory, suspicion)`: evidence raises it at once, it fades slowly, so a device that cheated once stays heavily audited for a few hundred audits even if it behaves afterwards. It survives sessions because it lives in `device_trust`.
6. **Trust and reward multiplier.** `trust = (1 − max(suspicion, memory)) · n_clean/(n_clean + 10)`; `reward_multiplier = 0.5 + 0.5·trust`.

### 17.4 Trust profile, quarantine, reinstatement

- `device_trust` (one per device): `status` (`probation → trusted ⇄ watch → quarantined`), counters, e-detector state, evidence, suspicion, memory, inherited memory, bounded history, quarantine record.
- **Quarantine** (`security/quarantine.py`) is sticky and triggered by (a) the e-detector, (b) forensics, (c) an administrator. It sets `devices.quarantined` (the eligibility rule excludes the device: `"quarantined after failed verification"` appears in the task's waiting reasons), releases the device's active assignments, claws back never-verified rewards, runs the retroactive audit and writes a `critical` security event. The device keeps heartbeating and stays visible, it just receives no work.
- **Reinstatement** is explicit and admin-only: back to probation, `memory = 0.6`, evidence reset.
- **Sybil/whitewashing resistance:** a new device starts on probation and inherits half of the highest suspicion memory among its owner's other devices (1.0 for a quarantined one).

### 17.5 Rewards (P11) — `rewards/ledger.py`

Credits are an internal accounting unit; no money, no token. `amount = work_units/1e6 × reward_rate × reward_multiplier(trust)`. One `reward_entries` document per accepted assignment (`_id` = assignment id, so accrual is idempotent), one append-only `reward_events` document per transition (`accrue`, `confirm`, `revoke`, with a strictly increasing `seq`). `verified` work is `confirmed` at once; `accepted_unverified` work is `pending` and becomes `confirmed` when its task completes (the end-to-end check has passed); a quarantined device loses (`revoked`) every entry that was never individually verified; rejected results earn nothing. `GET /rewards/ledger/check` replays the event log and compares it with the entries (balances are reproducible from the events alone).

### 17.6 Security hardening

Login lockout (5 failures → 5 min, also for unknown e-mails, no enumeration), sliding-window rate limits on sign-up / sign-in / device registration (per address, last `X-Forwarded-For` entry), device cap per account, security headers on every response, tenant isolation (a device can only touch its own assignments; probing a foreign one is a 404 + a `foreign_assignment_access` event), conflicting re-submission detection, generic error to the worker on a failed audit, append-only `security_events` (shown on `/security`). Pre-existing: no user code, no pickle, hashed tokens/passwords, SHA-256 on inputs/kernels/results.

### 17.7 Attack harness and simulator (P9)

- **CLI attack modes** (`cli_worker/attacks.py`): `subtle`, `scale`, `bias`, `noise`, `sign_flip`, `zero`, `random`, `replay`, `lazy`; schedules `--attack-after N` (sleeper) and `--attack-prob q` (intermittent). Every attack is **structurally valid** (right shapes, finite, counts add up, non-negative second moments), so only verification can catch it. The worker keeps its own ground-truth `AttackLog`; the backend never sees it. The browser worker has an owner-controlled *Demo: misbehave* switch on the worker console (same perturbations) for the live presentation.
- **Population simulator** (`trust/simulator.py`, `POST /trust/simulate`, `/simulator`): runs the real PWAV functions over synthetic devices and compares `none` / fixed-rate / adaptive on the same population. It models device *behaviour* only and does not include the reference check, forensics or retroactive audits; results are labelled as simulation.

### 17.8 Limits (honest)

- A corruption smaller than the tolerance is not detectable (and by design harmless: ≤ 1e-8 relative for GNB, 1e-6 Ridge, 1e-4 CNN gradients).
- With a floor of 5 %, an *unaudited* cheat is merged until a later audit (or the reference check) exposes it. For CSV tasks the reference check + forensics repair the final result; for CNN training merged rounds cannot be undone (flagged in `integrity_flags`), and only sampled rounds get the centralized gradient check.
- The e-detector's false-accusation bound is conditional on the tolerance limit being valid (probability ≥ γ) and on honest exceedance probability ≤ q0; the 10× margin makes the real honest exceedance rate far lower.
- Rate limiting is in memory (per backend instance); a restart resets it. Lockout state is in MongoDB.
- Credits are not money; there is no payment, token or blockchain.
- The browser "misbehave" switch ships in the production worker for demonstration; a malicious owner could always modify his own device, which is exactly the threat model verification addresses.

### 17.9 Insertion points — as implemented

| Insertion point | Part 1 | Part 2 (implemented) |
|---|---|---|
| `VerificationHook` in result intake | `accepted_unverified` | `verify_result` (async): audit decision, recomputation, trust update, `rejected_verification` |
| `partial_results.acceptance` | `accepted_unverified`, `rejected_structural` | + `verified`, `rejected_verification`, `rejected_forensic`, `rejected_retroactive` |
| `kernel.compare` | reference check | discrepancy statistic for audits and forensics |
| `runtime` fingerprint / kind | recorded | defines the calibration class |
| `DeviceEligibilityPolicy` | allow all | `devices.quarantined` rule inside `ineligibility_reasons` |
| `devices.stats`, `events` | counters, log | + `device_trust`, `verification_records`, `calibrations`, `security_events` |
| CLI fault flags | reliability tests | `--attack*` modes + ground-truth log |
| `assignments.work_units` + acceptance | recorded | reward entries and events |
| `assignments.purpose` (`replica`/`audit`), `chunks.role` (`challenge`) | reserved | **unused** — audits are backend recomputations |

---

## 18. Future scalability paths (not MVP)

- More kernels: distributed evaluation stage, partitioned exact kNN, dataset profiling, mini-batch k-means (approximate, flagged), ensemble/bagging (explicitly *not* equal to centralized training).
- Task graphs (train → evaluate → predict) built from the same chunk/assignment model.
- Dynamic scheduling: over-partitioning + work-stealing, benchmark updated from observed throughput.
- Native Android wrapper with a foreground service; GPU/desktop workers; WebGPU later.
- SSE/WebSocket for dashboard push.
- Multiple backend instances with a Mongo lease lock for the reconciler; hierarchical aggregation on devices (merge is already in `core/`).
- Stronger auth (device attestation), encrypted chunk transfer, privacy-preserving workloads.

---

## 19. Decision log

| ID | Decision | Alternatives rejected | Revisit when |
|---|---|---|---|
| D1 | Browser + Pyodide worker on Android | Termux, Chaquopy app, Kivy, `.exe` | Background execution becomes a requirement |
| D2 | HTTPS polling | WebSocket, push | Hundreds of devices or sub-second dispatch needed |
| D3 | Separate FastAPI service | FastAPI on Vercel serverless | — |
| D4 | MongoDB + GridFS only | Object storage, local disk | Files exceed free-tier capacity |
| D5 | Controlled catalog, no user code | Sandboxed user Python | Part 2 and security review complete |
| D6 | Sufficient-statistics kernels (GNB, Ridge) | Generic sklearn split-and-merge | Each new kernel must prove exact/flagged merge |
| D7 | Weighted proportional benchmark scheduler | Optimizers, ML-based schedulers | Measured imbalance hurts demos |
| D8 | Single instance + reconciler | Celery/Redis/queues | Multi-instance needed |
| D9 | JSON results, no pickle intake | Pickled partials | Never |
| D10 | OpenAPI-generated TS types | Hand-written types | — |
| D11 | Stateless synchronous data-parallel rounds for the image CNN (gradient sums; datasets stay on the backend) | Shipping shards to phones; federated averaging of local weights (not equal to centralized); PyTorch/TF in the browser | Larger models need cached shards or hierarchical aggregation |
| D12 | Audit = backend recomputation; no replica/challenge assignments | Duplicate execution on a second device (collusion-prone, 2x cost); hidden known-answer chunks | Workloads too expensive to recompute on the backend |
| D13 | Shiryaev–Roberts e-detector with block-wise thresholds `N²/α_m` for lifetime false-accusation control | Plain product e-process (sleeper agents hide behind history); power-of-two restarts (failed the sleeper test) | A tighter provable bound is needed |
| D14 | Reward ledger = mutable entries + append-only events, conditional transitions, replay check | Single balance counter | Real payments are introduced |
| D15 | Part 2 integrated before the M3 gate at the owner's request | Wait for M3 (original rule) | — |
| D16 | Live dashboard by REST polling + one admin-only global event feed; PixiJS used directly; every animation derived from a real event or snapshot | WebSocket/SSE (no backend support, not needed at this scale); `@pixi/react`; mock traffic for visual effect | Hundreds of devices or sub-second visuals needed (then SSE, §18) |
