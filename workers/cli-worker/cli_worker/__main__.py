"""python -m cli_worker --api URL --email E --password P --count N"""

import argparse
import logging
import os
import threading
from pathlib import Path

import httpx

from . import attacks
from .worker import Faults, StateFile, Worker, login_or_signup


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
    f = ap.add_argument_group("fault injection (reliability tests; never on a real contributor)")
    f.add_argument(
        "--fail-rate", type=float, default=0.0, help="probability of failing an assignment"
    )
    f.add_argument("--delay-ms", type=int, default=0, help="sleep before computing")
    f.add_argument("--die-after-start", action="store_true", help="vanish after /start")
    f.add_argument("--corrupt-result", action="store_true", help="return wrong statistics")
    f.add_argument("--late-result-ms", type=int, default=0, help="hold the result back")
    f.add_argument("--fault-seed", type=int, default=None)
    f.add_argument("--attack", default="none", choices=attacks.MODES, help="Part 2 attack mode")
    f.add_argument("--attack-after", type=int, default=0, help="honest results before cheating")
    f.add_argument("--attack-prob", type=float, default=1.0, help="probability of cheating")
    args = ap.parse_args()
    faults = Faults(
        fail_rate=args.fail_rate,
        delay_ms=args.delay_ms,
        die_after_start=args.die_after_start,
        corrupt_result=args.corrupt_result,
        late_result_ms=args.late_result_ms,
        seed=args.fault_seed,
        attack=args.attack,
        attack_after=args.attack_after,
        attack_prob=args.attack_prob,
    )
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
            faults=faults,
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
