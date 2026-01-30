from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import asyncio
from app.core.ws import manager

from app.api.health import router as health_router
from app.api.streams import router as streams_router
from app.api.metrics import router as metrics_router

from app.api.ws_metrics import router as ws_router

app = FastAPI(
    title="Smart Traffic & Crowd Analytics",
    version="0.1.0"
)

# -----------------------------
# API routes
# -----------------------------
app.include_router(health_router, prefix="/api")
app.include_router(streams_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(ws_router)

# -----------------------------
# Frontend
# -----------------------------
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)


@app.on_event("startup")
async def on_startup():
    manager.set_loop(asyncio.get_running_loop())