# ProofNet demo runbook

Everything here uses only free services. URLs: frontend https://proofnet.vercel.app, backend https://proofnet-api.onrender.com, database MongoDB Atlas (project "Project 0", database `Project0`).

## 1. The day before

- [ ] Both phones: Chrome updated, **battery saver off**, screen timeout long, charged. Open the site once on Wi-Fi so the Python runtime (Pyodide, several MB) is cached.
- [ ] Sign in on every device with the **same account** (it can see and run everything).
- [ ] On each phone: Contribute → register → **Open worker console → Start contributing** once; confirm `idle` and a benchmark score.
- [ ] Build the demo data on the laptop: `uv run python datasets/generate.py` (CSV, 100k rows) and `uv run python datasets/make_image_zip.py` (Fashion-MNIST, 10k images).
- [ ] Atlas: Network Access allows `0.0.0.0/0` (Render's address changes); check storage is well under 512 MB (the app also cleans up automatically).
- [ ] Run one full rehearsal of section 3 and note the timings.

## 2. 30 minutes before

1. Use your **own mobile hotspot** (venue Wi-Fi is the biggest risk). Phones and laptop on it.
2. Open **https://proofnet.vercel.app/settings → Test / pre-warm**. The free backend sleeps after 15 minutes idle and takes up to a minute to wake: wait for "OK … answered in … ms".
3. Phones: open `/contribute/run`, hard-refresh, **Start contributing**, wait for `idle`. Keep screens on, charging.
4. Laptop: open `/network` and press **Big screen** for the projector.
5. Optional clean slate: Settings → Admin → *Reset demo data* (keeps accounts and registered devices; only for accounts listed in `ADMIN_EMAILS`).

## 3. The demo, in order (≈ 10 min)

| # | Action | Say / point at |
|---|---|---|
| 1 | `/network`: two phones, different benchmark scores | "Each phone benchmarked itself with the real kernel code; the scores are measured, not assumed." |
| 2 | **New task** → `clf_100k_16.csv`, Min = Max = 2 devices → **Validate** | Plan preview: rows split in proportion to the scores. |
| 3 | **Submit** → task page | Live chunk table, device lanes, event feed; each phone's compute and download time. |
| 4 | Reference check **PASSED** | "The merged result equals centralized training (difference ≈ 1e-14)." Download `model.joblib`/`report.json`. |
| 5 | Failure: submit another task, **lock one phone** during the transfer | Within ~25 s: "went offline - chunk reassigned"; the task still completes and still passes. |
| 6 | **Image training** → `fashion_mnist_10k.zip`, steps 100, batch 64, Min = Max = 2 | Live loss/accuracy curve, each phone's share of every batch, "centrally verified rounds". Final accuracy and per-class bars. |
| 7 | Limitations page | Results are unverified (Part 2); contributors see their data; phones must stay in the foreground. |

## 4. If something goes wrong

| Symptom | Do this |
|---|---|
| Site/API does not answer | Backend is waking (≈ 50 s). Settings → Test / pre-warm. |
| Phone stuck on `initializing` | It only leaves that state after **Start contributing** in the worker console. |
| Phone shows `offline` | Screen locked / tab in background. Reopen the tab, tap Start contributing (same device comes back). |
| Task stays `queued` | The task page lists why each device cannot take it (battery < 20 % and not charging, offline, busy …). |
| Phone very slow to start | Weak mobile data: download dominates. Move to Wi-Fi/hotspot; compute itself is milliseconds. |
| Atlas "writes blocked" / API fails to start | Storage full. The app frees old data automatically above 380 MB; or Settings → Admin → Reset demo data. |
| Cloud backend unavailable | **Fallback** (below). |
| Nothing works | Play the recorded M2 video. |

### Fallback: laptop backend + free HTTPS tunnel (≈ 2 min)

1. Laptop: `uv run uvicorn proofnet_api.main:app --app-dir services/api --port 8000` (uses `.env` → Atlas).
2. Second terminal: `cloudflared tunnel --url http://localhost:8000` (free quick tunnel, no account). Copy the `https://….trycloudflare.com` address.
3. Backend `CORS_ORIGINS` must include `https://proofnet.vercel.app` (set it in `.env` before step 1).
4. In the browser on every device: https://proofnet.vercel.app/settings → paste `https://….trycloudflare.com/api/v1` → **Save & test** → sign in again, and re-register / Start contributing on each phone (identities are per backend).

## 5. After the demo

- Settings → Admin → Reset demo data if you want to clean up.
- Rotate the Atlas database password if it was shared.
- The honest limitations are on the site (`/limitations`) and in `LIMITATIONS` of the report.
