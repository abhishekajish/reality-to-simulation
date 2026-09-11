import csv
from pathlib import Path

from reality_to_simulation.tracker import track_video


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")
OUTPUT_PATH = Path("data/processed/trajectories.csv")


def export_trajectories(video_path: str, output_path: str):
    results = track_video(video_path)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for trajectory in results["trajectories"].values():
        rows.extend(trajectory)

    rows.sort(
        key=lambda row: (
            row["track_id"],
            row["frame_id"],
        )
    )

    fieldnames = [
        "frame_id",
        "track_id",
        "class_name",
        "x",
        "y",
        "confidence",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return {
        "frames_processed": results["frames_processed"],
        "unique_track_ids": results["unique_track_ids"],
        "trajectory_rows": len(rows),
        "output_path": str(output_file),
    }


if __name__ == "__main__":
    results = export_trajectories(
        str(VIDEO_PATH),
        str(OUTPUT_PATH),
    )

    print("TRAJECTORY EXPORT COMPLETE")
    print("FRAMES PROCESSED:", results["frames_processed"])
    print("UNIQUE TRACK IDs:", results["unique_track_ids"])
    print("TRAJECTORY ROWS:", results["trajectory_rows"])
    print("OUTPUT:", results["output_path"])