from fastapi import APIRouter
from app.workers.video_worker import VideoWorker

router = APIRouter()
worker = VideoWorker()


@router.post("/start")
def start_stream(video_path: str = "data/videos/input_video.mp4"):
    worker.start(video_path)
    return {"status": "started", "video": video_path}


@router.post("/stop")
def stop_stream():
    worker.stop()
    return {"status": "stopped"}