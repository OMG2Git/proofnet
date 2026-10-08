"""Verification (ARCHITECTURE 17): acceptance states shared by intake, aggregation and rewards.

The decision logic lives in `verification.verifier` (audit by backend recomputation) and the math in
`trust.pwav`; this module only holds the vocabulary so that importing it stays dependency-free.
"""

VERIFIED = "verified"  # audited: the backend recomputed the chunk and it matched
ACCEPTED_UNVERIFIED = "accepted_unverified"  # not audited this time (adaptive sampling)
REJECTED_VERIFICATION = "rejected_verification"  # audited and wrong: never merged
ACCEPTED = (VERIFIED, ACCEPTED_UNVERIFIED)  # acceptance states that count towards the result

MODES = ("off", "adaptive", "full")


def normalize_mode(policy: dict[str, str] | None, default: str = "adaptive") -> str:
    """Task verification mode. Tasks created before Part 2 carry {"mode": "none"} = off."""
    mode = (policy or {}).get("mode", default)
    return "off" if mode == "none" else (mode if mode in MODES else default)
