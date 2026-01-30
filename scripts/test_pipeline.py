from app.cv.pipeline import run_video_pipeline

if __name__ == "__main__":
    run_video_pipeline(
        video_path="data/videos/input_video.mp4",
        target_fps=2
    )