# apps/web

Next.js + TypeScript (strict) frontend for Vercel. Pure API client.

```
npm ci
npm run gen:api     # regenerate lib/api/schema.d.ts from lib/api/openapi.json
npm run typecheck && npm run lint && npm run build
```

Refresh `openapi.json` from the backend with `uv run python scripts/export_openapi.py` (repo root) before `gen:api`.
