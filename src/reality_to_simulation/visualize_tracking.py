from collections import defaultdict
from pathlib import Path

import cv2

from reality_to_simulation.tracker import VehicleTracker
from reality_to_simulation.video import open_video


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")
OUTPUT_PATH = Path("data/interim/tracking_visualization.mp4")


def draw_tracking_visualization(
    video_path: str,
    output_path: str,
):
    tracker = VehicleTracker()

    capture = open_video(video_path)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError(
            f"Could not create output video: {output_path}"
        )

    trajectory_history = defaultdict(list)

    frame_id = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_id += 1

        tracks = tracker.track_frame(frame)

        for track in tracks:
            track_id = track["track_id"]
            x1, y1, x2, y2 = map(
                int,
                track["bbox"],
            )

            center_x, center_y = map(
                int,
                track["center"],
            )

            trajectory_history[track_id].append(
                (center_x, center_y)
            )

            points = trajectory_history[track_id]

            for index in range(1, len(points)):
                cv2.line(
                    frame,
                    points[index - 1],
                    points[index],
                    (0, 255, 255),
                    2,
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"ID {track_id} | "
                f"{track['class_name']}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    capture.release()
    writer.release()

    return {
        "frames_processed": frame_id,
        "output_path": str(output_file),
    }


if __name__ == "__main__":
    results = draw_tracking_visualization(
        str(VIDEO_PATH),
        str(OUTPUT_PATH),
    )

    print("TRACKING VISUALIZATION COMPLETE")
    print("FRAMES PROCESSED:", results["frames_processed"])
    print("OUTPUT:", results["output_path"])