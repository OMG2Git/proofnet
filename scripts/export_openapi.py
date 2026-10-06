"""Export the FastAPI OpenAPI schema to apps/web/lib/api/openapi.json (source for TS types)."""

import json
from pathlib import Path

from proofnet_api.main import app

OUT = Path(__file__).resolve().parents[1] / "apps" / "web" / "lib" / "api" / "openapi.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
