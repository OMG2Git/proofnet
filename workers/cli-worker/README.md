# cli-worker

CPython reference worker using `proofnet_kernels.core` and the same protocol as the browser worker.
Used for development, CI, multi-device simulation and (from P6) fault injection.

```
uv run python -m cli_worker --api http://localhost:8000/api/v1 --email me@example.com --password '...' --count 3
```

Each simulated device registers once (token cached in `~/.proofnet-cli/devices.json`), verifies the
kernel bundle SHA-256, runs the real `bench_v1`, opens a session and heartbeats. Scores are measured,
never faked. Credentials can come from `PROOFNET_EMAIL` / `PROOFNET_PASSWORD`.
