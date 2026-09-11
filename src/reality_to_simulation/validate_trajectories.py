import csv
from collections import defaultdict
from pathlib import Path


TRAJECTORY_PATH = Path("data/processed/trajectories.csv")

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


def load_trajectories(csv_path: str):
    trajectories = defaultdict(list)

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            trajectories[int(row["track_id"])].append(
                {
                    "frame_id": int(row["frame_id"]),
                    "class_name": row["class_name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "confidence": float(row["confidence"]),
                }
            )

    return dict(trajectories)


def validate_trajectories(
    trajectories: dict,
    video_width: int,
    video_height: int,
):
    track_lengths = []
    confidence_values = []

    invalid_coordinates = 0
    frame_gaps = 0

    for trajectory in trajectories.values():
        track_lengths.append(len(trajectory))

        previous_frame = None

        for point in trajectory:
            confidence_values.append(point["confidence"])

            if not (
                0 <= point["x"] <= video_width
                and 0 <= point["y"] <= video_height
            ):
                invalid_coordinates += 1

            if previous_frame is not None:
                if point["frame_id"] != previous_frame + 1:
                    frame_gaps += 1

            previous_frame = point["frame_id"]

    total_points = sum(track_lengths)

    return {
        "unique_tracks": len(trajectories),
        "total_points": total_points,
        "short_tracks": sum(
            1 for length in track_lengths if length < 5
        ),
        "longest_track": max(track_lengths)
        if track_lengths
        else 0,
        "average_track_length": (
            total_points / len(track_lengths)
            if track_lengths
            else 0
        ),
        "average_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0
        ),
        "invalid_coordinates": invalid_coordinates,
        "frame_gaps": frame_gaps,
    }


if __name__ == "__main__":
    trajectories = load_trajectories(
        str(TRAJECTORY_PATH)
    )

    results = validate_trajectories(
        trajectories,
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
    )

    print("TRAJECTORY VALIDATION COMPLETE")
    print("UNIQUE TRACKS:", results["unique_tracks"])
    print("TOTAL TRAJECTORY POINTS:", results["total_points"])
    print("SHORT TRACKS (<5 FRAMES):", results["short_tracks"])
    print("LONGEST TRACK:", results["longest_track"])
    print(
        "AVERAGE TRACK LENGTH:",
        round(results["average_track_length"], 2),
    )
    print(
        "AVERAGE CONFIDENCE:",
        round(results["average_confidence"], 4),
    )
    print(
        "INVALID COORDINATES:",
        results["invalid_coordinates"],
    )
    print("FRAME GAPS:", results["frame_gaps"])