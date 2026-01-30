from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.ws import manager

router = APIRouter()


@router.websocket("/ws/metrics")
async def metrics_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from the client (push-only)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
