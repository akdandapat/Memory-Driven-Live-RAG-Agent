"""FastAPI entrypoint: API + static demo UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import PROJECT_ROOT, get_settings
from app.services import Services

logger = logging.getLogger(__name__)
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    services = Services(settings)
    await services.startup()
    app.state.services = services
    try:
        yield
    finally:
        await services.shutdown()


app = FastAPI(
    title="Agentic RAG over Live Data and Memory",
    version="1.0.0",
    description=(
        "Agentic retrieval over a live issue tracker: MCP tool layer, dynamic planning, "
        "multi-hop evidence collection, persistent memory, citations and full tracing."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
async def health_alias() -> JSONResponse:
    services = getattr(app.state, "services", None)
    if services is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return JSONResponse(await services.status())


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    run()
