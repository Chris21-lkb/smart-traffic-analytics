import threading

from app.cv.pipeline import run_video_pipeline
from app.core.state import STATE
from app.core.ws import manager


class VideoWorker:
    """
    Background worker that runs the CV pipeline in a separate thread
    and reports analytics to:
      - shared STATE (REST)
      - WebSocket clients (real-time)
    """

    def __init__(self):
        self.thread = None

    def start(self, video_path: str):
        if STATE["running"]:
            return

        STATE["running"] = True

        def on_update(people, vehicles):
            STATE["people"] = people
            STATE["vehicles"] = vehicles

            payload = {
                "people": people,
                "vehicles": vehicles,
            }

            # SAFE: schedules coroutine on FastAPI event loop
            manager.broadcast(payload)

        def target():
            try:
                run_video_pipeline(
                    video_path=video_path,
                    target_fps=5,
                    on_update=on_update,
                )
            finally:
                STATE["running"] = False

        self.thread = threading.Thread(
            target=target,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        STATE["running"] = False
