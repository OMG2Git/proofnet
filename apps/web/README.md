# apps/web

Next.js + TypeScript (strict) frontend for Vercel. Pure API client.

```
npm ci
npm run gen:api     # regenerate lib/api/schema.d.ts from lib/api/openapi.json
npm run typecheck && npm run lint && npm run build
```

Refresh `openapi.json` from the backend with `uv run python scripts/export_openapi.py` (repo root) before `gen:api`.

## Live network dashboard (`/network`)

Design: `app/globals.css` (tokens, components), `components/network/*` (dashboard, PixiJS world, drawer), `lib/network/*` (pure view-model `logic.ts`, polling hook `use-network-data.ts`, `layout.ts`, `sprite-data.ts`, `visuals.ts`). See ARCHITECTURE.md 3.2.1 for the rules that keep it truthful (nothing is animated that the backend did not do).

Data used: `GET /network/summary` (all users), `GET /network/events` (admin only; added for this view), `GET /trust/*`, `GET /rewards/me`, `POST /security/devices/{id}/quarantine|reinstate` (admin). Admin = account listed in the backend's `ADMIN_EMAILS`.

## Tests

```
npm test            # node:test: view-model mapping, state transitions, layout, sprites (no browser)
npm run e2e         # real backend + real CLI worker + real Chrome (see below)
```

### End-to-end test (`npm run e2e`)

1. Backend (repo root), on a **fresh database**, with an admin account:
   `MONGODB_DB=proofnet_e2e ADMIN_EMAILS=e2e-admin@example.com JWT_SECRET=<long> CORS_ORIGINS=http://localhost:3000 RATE_LIMIT_AUTH_PER_MINUTE=100000 uv run uvicorn proofnet_api.main:app --app-dir services/api --port 8000`
2. Frontend: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1 npm run build && npx next start -p 3000`
3. `npm run e2e` (override with `PROOFNET_WEB`, `PROOFNET_API`, `PROOFNET_ADMIN_EMAIL`, `PROOFNET_CHROME`, `CLI_WORKER_CMD`). It spawns a CLI worker, submits a real task, and checks the dashboard, drawer, quarantine/reinstate, backend loss/recovery, non-admin permissions and a 390 px viewport. Screenshots go to `e2e/artifacts/` (git-ignored). It disables the e2e admin's *other* devices so only its own worker takes work: use a dedicated account and database.
