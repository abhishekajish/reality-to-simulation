from pathlib import Path

import cv2
from ultralytics import YOLO

from reality_to_simulation.trajectory import TrajectoryStore


DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
}


class VehicleTracker:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        tracker_config: str = DEFAULT_TRACKER,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config

    def track_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        result = results[0]
        tracks = []

        if result.boxes.id is None:
            return tracks

        for box, track_id in zip(result.boxes, result.boxes.id):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[class_id]

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            tracks.append(
                {
                    "track_id": int(track_id),
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                }
            )

        return tracks


def track_video(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    tracker_config: str = DEFAULT_TRACKER,
):
    tracker = VehicleTracker(model_path, tracker_config)
    trajectory_store = TrajectoryStore()

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    total_tracks = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        tracks = tracker.track_frame(frame)

        frame_count += 1
        total_tracks += len(tracks)

        trajectory_store.add_tracks(
            frame_id=frame_count,
            tracks=tracks,
        )

    capture.release()

    return {
        "frames_processed": frame_count,
        "total_track_instances": total_tracks,
        "unique_track_ids": trajectory_store.get_track_count(),
        "trajectories": trajectory_store.get_all_trajectories(),
    }


if __name__ == "__main__":
    video_path = Path("data/raw/videos/traffic_demo.mp4")

    results = track_video(str(video_path))

    print("TRACKING COMPLETE")
    print("FRAMES PROCESSED:", results["frames_processed"])
    print("TOTAL TRACK INSTANCES:", results["total_track_instances"])
    print("UNIQUE TRACK IDs:", results["unique_track_ids"])

    trajectory_lengths = [
        len(trajectory)
        for trajectory in results["trajectories"].values()
    ]

    if trajectory_lengths:
        print("LONGEST TRAJECTORY:", max(trajectory_lengths))
        print("AVERAGE TRAJECTORY LENGTH:", sum(trajectory_lengths) / len(trajectory_lengths))