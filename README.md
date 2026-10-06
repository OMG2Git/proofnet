# ProofNet

Decentralized AI compute platform (Part 1: Compute Fabric). Specification: [CLAUDE.md](CLAUDE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PHASE_PLAN.md](PHASE_PLAN.md).

## Prerequisites (Windows / macOS / Linux)

- [uv](https://docs.astral.sh/uv/) (installs the pinned Python 3.12 automatically)
- Node.js 22+ and npm
- MongoDB: local `mongod`, or a personal free Atlas cluster (see `.env.example`)

## Setup

```
uv sync --all-packages                 # Python workspace (.venv)
cd apps/web && npm ci && cd ..         # frontend
cp .env.example .env                   # then edit; never commit secrets
```

## Checks (same as CI)

```
uv run ruff check . && uv run ruff format --check .
uv run mypy services/api packages/kernels workers/cli-worker
uv run pytest
cd apps/web && npm run typecheck && npm run lint && npm test && npm run build
```

## Contract workflow (OpenAPI is the source of truth)

```
uv run python scripts/export_openapi.py      # FastAPI -> apps/web/lib/api/openapi.json
cd apps/web && npm run gen:api               # -> lib/api/schema.d.ts
```

Commit both generated files; CI fails if they drift.

## Demo datasets

`uv run python datasets/generate.py` (see [datasets/README.md](datasets/README.md)).

## Conventions

- Branches: `main` is protected; work on `feat/<phase>-<topic>`; merge by PR with green CI.
- Definition of Done: PHASE_PLAN.md section 12 (tests + CI green, contracts regenerated, the three planning documents kept consistent, acceptance criteria demonstrated, nothing simulated presented as real).
- Worker-executed code lives only in `packages/kernels/proofnet_kernels/core` and imports only NumPy + stdlib.
