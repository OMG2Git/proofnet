"""python -m cli_worker --api URL --email E --password P --count N"""

import argparse
import logging
import os
import threading
from pathlib import Path

import httpx

from .worker import StateFile, Worker, login_or_signup


def main() -> None:
    ap = argparse.ArgumentParser(prog="cli_worker", description="ProofNet CPython worker")
    ap.add_argument("--api", default=os.environ.get("PROOFNET_API", "http://localhost:8000/api/v1"))
    ap.add_argument("--email", default=os.environ.get("PROOFNET_EMAIL"))
    ap.add_argument("--password", default=os.environ.get("PROOFNET_PASSWORD"))
    ap.add_argument("--count", type=int, default=1, help="number of simulated devices")
    ap.add_argument("--name-prefix", default="cli")
    ap.add_argument("--device-type", default="laptop", choices=["laptop", "desktop", "other"])
    ap.add_argument("--state-file", default=str(Path.home() / ".proofnet-cli" / "devices.json"))
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    args = ap.parse_args()
    if not args.email or not args.password:
        ap.error("--email and --password (or PROOFNET_EMAIL / PROOFNET_PASSWORD) are required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = StateFile(Path(args.state_file))
    client = httpx.Client(base_url=args.api.rstrip("/"), timeout=30)
    token = login_or_signup(client, args.email, args.password)
    workers = [
        Worker(
            client,
            token,
            f"{args.name_prefix}-{i + 1}",
            state=state,
            state_key=f"{args.api}|{args.email}|{args.name_prefix}-{i + 1}",
            device_type=args.device_type,
        )
        for i in range(args.count)
    ]
    threads = [
        threading.Thread(target=w.run, kwargs={"max_seconds": args.duration}, daemon=True)
        for w in workers
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        for w in workers:
            w.stop.set()


if __name__ == "__main__":
    main()
