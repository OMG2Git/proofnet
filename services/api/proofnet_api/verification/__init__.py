"""Verification hook (ARCHITECTURE 9.2, 17). MVP: accept everything as unverified.

Part 2 replaces the implementation (replica/audit/challenge handling, tolerance checks) without
changing callers: result intake only talks to `VerificationHook`.
"""

from typing import Any, Protocol

ACCEPTED_UNVERIFIED = "accepted_unverified"


class VerificationHook(Protocol):
    def on_partial(
        self, task: dict[str, Any], chunk: dict[str, Any], payload: dict[str, Any]
    ) -> str:
        """Return the acceptance state to store on the partial result."""
        ...


class AcceptUnverified:
    def on_partial(
        self, task: dict[str, Any], chunk: dict[str, Any], payload: dict[str, Any]
    ) -> str:
        return ACCEPTED_UNVERIFIED


DEFAULT_HOOK: VerificationHook = AcceptUnverified()
