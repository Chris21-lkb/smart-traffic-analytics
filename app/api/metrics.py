from fastapi import APIRouter
from app.core.state import STATE

router = APIRouter()


@router.get("/metrics")
def get_metrics():
    return {
        "running": STATE["running"],
        "people": STATE["people"],
        "vehicles": STATE["vehicles"],
    }