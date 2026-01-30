import cv2
import time
from pathlib import Path


def run_video_pipeline(video_path: str, target_fps: int = 10):
    """
    Basic offline video pipeline:
    - Load video
    - Read frames
    - Control FPS
    - Display frames
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Failed to open video file")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(original_fps // target_fps), 1)

    print(f"[INFO] Original FPS: {original_fps}")
    print(f"[INFO] Target FPS: {target_fps}")
    print(f"[INFO] Frame interval: {frame_interval}")

    frame_count = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            cv2.imshow("Offline Video Pipeline", frame)

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing finished.")