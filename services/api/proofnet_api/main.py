"""FastAPI app factory. One long-running instance; the reconciler runs as a background task."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import reconciler
from .auth.routes import router as auth_router
from .config import Settings, get_settings
from .contracts import (
    ApiError,
    ErrorBody,
    FailRequest,
    PartialResultEnvelope,
)
from .datasets.routes import router as datasets_router
from .db import Db, connect
from .devices.routes import router as devices_router
from .errors import install_error_handlers
from .network.routes import router as network_router
from .tasks.routes import router as tasks_router
from .worker_gateway.routes import router as worker_router
from .worker_gateway.runtime import build_bundle

_NOT_IMPLEMENTED: dict[int | str, dict[str, object]] = {501: {"model": ApiError}}


def _stub() -> JSONResponse:
    body = ApiError(error=ErrorBody(code="NOT_IMPLEMENTED", message="Implemented in P4"))
    return JSONResponse(status_code=501, content=body.model_dump())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db: Db = await connect(settings)
        app.state.db = db
        task = asyncio.create_task(reconciler.run_forever(db, settings))
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await db.close()

    app = FastAPI(
        title="ProofNet API",
        version="0.1.0",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.bundle = build_bundle()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix="/api/v1")

    @api.get("/health")
    async def health() -> dict[str, str]:
        db: Db = app.state.db
        await db.client.admin.command("ping")
        return {"status": "ok", "db": "ok"}

    for r in (
        auth_router,
        devices_router,
        worker_router,
        datasets_router,
        tasks_router,
        network_router,
    ):
        api.include_router(r)

    # Worker result intake arrives in P4; the stubs keep the contract in OpenAPI.
    @api.post("/worker/assignments/{assignment_id}/result", responses=_NOT_IMPLEMENTED)
    async def submit_result(assignment_id: str, body: PartialResultEnvelope) -> JSONResponse:
        return _stub()

    @api.post("/worker/assignments/{assignment_id}/fail", responses=_NOT_IMPLEMENTED)
    async def fail_assignment(assignment_id: str, body: FailRequest) -> JSONResponse:
        return _stub()

    app.include_router(api)
    return app


app = create_app()
