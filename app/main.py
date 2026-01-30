from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router

app = FastAPI(
    title="Smart Traffic & Crowd Analytics",
    version="0.1.0"
)

# Register API routes
app.include_router(health_router, prefix="/api")

# Serve frontend (HTML/CSS/JS)
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)