from pathlib import Path

from reality_to_simulation.detector import VehicleDetector
from reality_to_simulation.video import open_video, read_frames


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")


def run_detection(video_path: str):
    detector = VehicleDetector()
    capture = open_video(video_path)

    frame_count = 0
    detection_count = 0

    for frame in read_frames(capture):
        detections = detector.detect(frame)

        frame_count += 1
        detection_count += len(detections)

    capture.release()

    return {
        "frames_processed": frame_count,
        "vehicle_detections": detection_count,
    }


if __name__ == "__main__":
    results = run_detection(str(VIDEO_PATH))

    print("DETECTION COMPLETE")
    print("FRAMES PROCESSED:", results["frames_processed"])
    print("TOTAL VEHICLE DETECTIONS:", results["vehicle_detections"])