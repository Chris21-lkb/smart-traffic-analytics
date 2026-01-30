import cv2
from pathlib import Path

from app.cv.detector import ObjectDetector
from app.cv.tracker import IoUTracker


def run_video_pipeline(video_path: str, target_fps: int = 5):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(original_fps // target_fps), 1)

    print(f"[INFO] Original FPS: {original_fps}")
    print(f"[INFO] Target FPS: {target_fps}")
    print(f"[INFO] Frame interval: {frame_interval}")

    detector = ObjectDetector(score_threshold=0.6)
    tracker = IoUTracker(iou_threshold=0.3, max_age=20)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            # Draw tracks (ID + class)
            for tr in tracks:
                x1, y1, x2, y2 = tr["bbox"]
                label = tr["label"]
                tid = tr["track_id"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    frame,
                    f"{label} #{tid}",
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("Detection + Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking finished.")