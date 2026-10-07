"""CPython reference worker: same protocol as the browser worker (ARCHITECTURE 7.5, 8).

Register (user JWT) -> fetch manifest + kernel bundle, verify SHA-256 -> run bench_v1 (the real
kernel code) -> POST /worker/session -> heartbeat loop (2 s idle / 5 s busy).
On a `run` directive it executes the assignment with the real core kernel (start -> download ->
verify -> map -> result). Fault injection flags arrive in P6.
"""

import contextlib
import hashlib
import json
import logging
import os
import platform
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from proofnet_kernels.core import bench, gaussian_nb, linear_ridge
from proofnet_kernels.core.serialize import payload_sha256

log = logging.getLogger("cli_worker")
BUNDLE_VERSION = "1"


class WorkerError(RuntimeError):
    pass


def _json(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise WorkerError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {msg}")
    return r.json()


def login_or_signup(
    client: httpx.Client, email: str, password: str, display_name: str = "CLI worker"
) -> str:
    """Returns a user JWT; creates the account if it does not exist yet."""
    r = client.post("/auth/login", json={"email": email, "password": password})
    if r.status_code == 401:
        r = client.post(
            "/auth/signup",
            json={"email": email, "password": password, "display_name": display_name},
        )
    return str(_json(r)["access_token"])


class StateFile:
    """Persists device tokens so a re-run reuses the same device identity (like localStorage)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if path and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict[str, str] | None:
        with self._lock:
            v = self._data.get(key)
            return v if isinstance(v, dict) else None

    def put(self, key: str, value: dict[str, str]) -> None:
        with self._lock:
            self._data[key] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


class Worker:
    def __init__(
        self,
        client: httpx.Client,
        user_token: str,
        name: str,
        state: StateFile | None = None,
        state_key: str | None = None,
        device_type: str = "laptop",
    ) -> None:
        self.http = client
        self.user_token = user_token
        self.name = name
        self.state = state or StateFile(None)
        self.state_key = state_key or name
        self.device_type = device_type
        self.device_id: str | None = None
        self.device_token: str | None = None
        self.session_id: str | None = None
        self.state_name = "initializing"
        self.heartbeat_ms = {"idle": 2000, "busy": 5000}
        self.score: float | None = None
        self.stop = threading.Event()

    # ---- auth helpers ----
    def _user(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.user_token}"}

    def _dev(self) -> dict[str, str]:
        assert self.device_token
        return {"Authorization": f"Bearer {self.device_token}"}

    # ---- steps ----
    def register(self) -> None:
        saved = self.state.get(self.state_key)
        if saved:
            self.device_id, self.device_token = saved["device_id"], saved["device_token"]
            log.info("[%s] reusing device %s", self.name, self.device_id)
            return
        caps = {
            "model": f"{platform.system()} {platform.machine()}",
            "platform_version": platform.version()[:80],
            "logical_cores": os.cpu_count(),
        }
        r = self.http.post(
            "/devices",
            headers=self._user(),
            json={"name": self.name, "device_type": self.device_type, "capabilities": caps},
        )
        body = _json(r)
        self.device_id = body["device"]["id"]
        self.device_token = body["device_token"]
        assert self.device_id and self.device_token
        self.state.put(
            self.state_key, {"device_id": self.device_id, "device_token": self.device_token}
        )
        log.info("[%s] registered device %s", self.name, self.device_id)

    def load_runtime(self) -> dict[str, Any]:
        """Fetch the manifest and the kernel bundle and verify its SHA-256 (as the browser does)."""
        manifest = _json(self.http.get("/runtime/manifest", headers=self._dev()))
        r = self.http.get(
            f"/runtime/kernels/{manifest['kernel_bundle_version']}", headers=self._dev()
        )
        if r.status_code != 200:
            raise WorkerError(f"kernel bundle download failed: {r.status_code}")
        digest = hashlib.sha256(r.content).hexdigest()
        if digest != manifest["kernel_bundle_sha256"]:
            raise WorkerError("kernel bundle SHA-256 mismatch; refusing to run")
        names = zipfile.ZipFile(BytesIO(r.content)).namelist()
        if "proofnet_kernels/core/gaussian_nb.py" not in names:
            raise WorkerError("kernel bundle is missing expected modules")
        self.heartbeat_ms = {
            "idle": manifest["heartbeat_idle_ms"],
            "busy": manifest["heartbeat_busy_ms"],
        }
        return dict(manifest)

    def start_session(self, manifest: dict[str, Any]) -> None:
        result = bench.run()  # real kernel workload; the score is the scheduling weight
        self.score = float(result["score_cells_per_sec"])
        r = self.http.post(
            "/worker/session",
            headers=self._dev(),
            json={
                "runtime": {
                    "kind": "cpython",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "bundle": manifest["kernel_bundle_version"],
                },
                "benchmark": {
                    "score_cells_per_sec": self.score,
                    "bench_version": result["bench_version"],
                },
                "capabilities": {"logical_cores": os.cpu_count()},
            },
        )
        body = _json(r)
        self.session_id = body["session_id"]
        self.heartbeat_ms = {"idle": body["heartbeat_idle_ms"], "busy": body["heartbeat_busy_ms"]}
        self.state_name = "idle"
        log.info("[%s] session %s, score %.2e cells/s", self.name, self.session_id, self.score)

    def heartbeat_once(self) -> dict[str, Any]:
        r = self.http.post(
            "/worker/heartbeat",
            headers=self._dev(),
            json={"session_id": self.session_id, "state": self.state_name},
        )
        body: dict[str, Any] = _json(r)
        for d in body["directives"]:
            if d["type"] == "reregister":
                log.warning("[%s] server asked to re-register; starting a new session", self.name)
                self.start_session(self.load_runtime())
            elif d["type"] == "run":
                self.execute(d["assignment"])
            else:
                log.info("[%s] directive %s not handled yet", self.name, d["type"])
        return body

    # ---- assignment execution (ARCHITECTURE 8.3) ----
    def execute(self, a: dict[str, Any]) -> None:
        asg_id = a["assignment_id"]
        base = f"/worker/assignments/{asg_id}"
        self.state_name = "busy"
        t_total = time.perf_counter()
        try:
            _json(self.http.post(f"{base}/start", headers=self._dev()))
            t0 = time.perf_counter()
            r = self.http.get(a["input_url"], headers=self._dev())
            if r.status_code != 200:
                raise WorkerError(f"input download failed: {r.status_code}")
            download_ms = (time.perf_counter() - t0) * 1000
            if hashlib.sha256(r.content).hexdigest() != a["input_sha256"]:
                raise WorkerError("input SHA-256 mismatch")
            with np.load(BytesIO(r.content), allow_pickle=False) as z:  # never pickle
                x, y = z["X"], z["y"]
            if x.shape != (a["n_rows"], a["n_features"]):
                raise WorkerError(f"unexpected input shape {x.shape}")
            t1 = time.perf_counter()
            if a["kernel"] == "gaussian_nb_train":
                payload = gaussian_nb.map(x, y, int(a["params"]["n_classes"]))
            elif a["kernel"] == "linear_ridge_train":
                payload = linear_ridge.map(x, y)
            else:
                raise WorkerError(f"unknown kernel {a['kernel']}")
            compute_ms = (time.perf_counter() - t1) * 1000
            body = {
                "kernel": a["kernel"],
                "kernel_version": a["kernel_version"],
                "input_sha256": a["input_sha256"],
                "n_rows": a["n_rows"],
                "payload": payload,
                "payload_sha256": payload_sha256(payload),
                "timings": {
                    "download_ms": download_ms,
                    "compute_ms": compute_ms,
                    "total_ms": (time.perf_counter() - t_total) * 1000,
                },
                "runtime": {
                    "kind": "cpython",
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "bundle": BUNDLE_VERSION,
                },
            }
            _json(self.http.post(f"{base}/result", headers=self._dev(), json=body))
            log.info(
                "[%s] assignment %s done (%d rows, %.0f ms)",
                self.name,
                asg_id,
                a["n_rows"],
                compute_ms,
            )
        except (WorkerError, httpx.TransportError, ValueError, KeyError) as e:
            log.error("[%s] assignment %s failed: %s", self.name, asg_id, e)
            with contextlib.suppress(httpx.TransportError):
                self.http.post(
                    f"{base}/fail",
                    headers=self._dev(),
                    json={"code": "EXECUTION_ERROR", "message": str(e)[:500]},
                )
        finally:
            self.state_name = "idle"

    # ---- main loop ----
    def run(self, max_seconds: float | None = None) -> None:
        deadline = time.monotonic() + max_seconds if max_seconds else None
        backoff = 1.0
        manifest: dict[str, Any] | None = None
        while not self.stop.is_set():
            try:
                if self.device_token is None:
                    self.register()
                if manifest is None:
                    manifest = self.load_runtime()
                if self.session_id is None:
                    self.start_session(manifest)
                self.heartbeat_once()
                backoff = 1.0
                wait = self.heartbeat_ms[self.state_name if self.state_name == "busy" else "idle"]
            except (httpx.TransportError, WorkerError) as e:
                log.warning("[%s] %s; retrying in %.0fs", self.name, e, backoff)
                wait = int(backoff * 1000)
                backoff = min(backoff * 2, 30.0)
            if deadline and time.monotonic() >= deadline:
                return
            self.stop.wait(wait / 1000)
