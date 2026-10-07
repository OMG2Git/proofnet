"""FastAPI app factory. One long-running instance; the reconciler runs as a background task."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import reconciler
from .auth.routes import router as auth_router
from .config import Settings, get_settings
from .contracts import ApiError
from .datasets.routes import router as datasets_router
from .db import Db, connect
from .devices.routes import router as devices_router
from .errors import install_error_handlers
from .network.routes import router as network_router
from .tasks.monitor import router as monitor_router
from .tasks.routes import router as tasks_router
from .worker_gateway.assignments import router as assignments_router
from .worker_gateway.routes import router as worker_router
from .worker_gateway.runtime import build_bundle


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
    app.state.background = set()
    app.state.bundle = build_bundle()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    error_models: dict[int | str, dict[str, Any]] = {
        code: {"model": ApiError} for code in (401, 403, 404, 409, 413, 422)
    }
    api = APIRouter(prefix="/api/v1", responses=error_models)

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
        assignments_router,
        monitor_router,
    ):
        api.include_router(r)

    app.include_router(api)
    return app


app = create_app()
