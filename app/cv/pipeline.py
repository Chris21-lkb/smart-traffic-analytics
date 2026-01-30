import cv2
from pathlib import Path

from app.cv.detector import ObjectDetector
from app.cv.tracker import IoUTracker
from app.analytics.people import PeopleAnalytics
from app.analytics.traffic import VehicleAnalytics


def run_video_pipeline(
    video_path: str,
    target_fps: int = 5,
    on_update=None,
):
    """
    Offline video pipeline (callable from a background worker):

    - Read video from disk
    - Run object detection (CPU)
    - Track people & vehicles (IDs)
    - Compute people analytics (count, dwell time)
    - Compute vehicle analytics (counts, congestion proxy)
    - Optionally report analytics via callback (FastAPI integration)
    - Visualize results (OpenCV window)

    Parameters
    ----------
    video_path : str
        Path to the input video file
    target_fps : int
        Effective FPS for inference (frame skipping)
    on_update : callable or None
        Callback function receiving analytics dicts:
        on_update(people={...}, vehicles={...})
    """

    # ------------------------------------------------------------------
    # Video setup
    # ------------------------------------------------------------------
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(original_fps // target_fps), 1)

    print(f"[INFO] Original FPS: {original_fps:.2f}")
    print(f"[INFO] Target FPS: {target_fps}")
    print(f"[INFO] Frame interval: {frame_interval}")

    # ------------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------------
    detector = ObjectDetector(score_threshold=0.6)
    tracker = IoUTracker(iou_threshold=0.3, max_age=20)

    people_analytics = PeopleAnalytics()
    vehicle_analytics = VehicleAnalytics()

    frame_count = 0

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to control compute load
        if frame_count % frame_interval == 0:

            # ----------------------------------------------------------
            # Detection
            # ----------------------------------------------------------
            detections = detector.detect(frame)

            # ----------------------------------------------------------
            # Tracking
            # ----------------------------------------------------------
            tracks = tracker.update(detections)

            # ----------------------------------------------------------
            # Analytics update
            # ----------------------------------------------------------
            people_analytics.update(tracks)
            vehicle_analytics.update(tracks)

            # ----------------------------------------------------------
            # Report analytics to FastAPI (if callback provided)
            # ----------------------------------------------------------
            if on_update is not None:
                on_update(
                    people={
                        "current": people_analytics.current_count(),
                        "unique": people_analytics.unique_count(),
                        "avg_dwell": people_analytics.average_dwell_time(),
                    },
                    vehicles={
                        "current": vehicle_analytics.current_count(),
                        "per_class": vehicle_analytics.current_counts_per_class(),
                        "congestion": vehicle_analytics.congestion_level(),
                    },
                )

            # ----------------------------------------------------------
            # Draw tracked objects
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # People analytics overlay
            # ----------------------------------------------------------
            cv2.putText(
                frame,
                f"People now: {people_analytics.current_count()}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Unique people: {people_analytics.unique_count()}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Avg dwell: {people_analytics.average_dwell_time():.1f}s",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            # ----------------------------------------------------------
            # Vehicle analytics overlay
            # ----------------------------------------------------------
            vehicle_total = vehicle_analytics.current_count()
            vehicle_counts = vehicle_analytics.current_counts_per_class()
            congestion = vehicle_analytics.congestion_level()

            y = 130
            cv2.putText(
                frame,
                f"Vehicles now: {vehicle_total}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )

            y += 30
            for cls, cnt in vehicle_counts.items():
                cv2.putText(
                    frame,
                    f"{cls.capitalize()}: {cnt}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                y += 25

            cv2.putText(
                frame,
                f"Congestion: {congestion}",
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if congestion == "HIGH" else (0, 255, 255),
                2,
            )

            # ----------------------------------------------------------
            # Display
            # ----------------------------------------------------------
            cv2.imshow("Smart Traffic & Crowd Analytics", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline finished.")