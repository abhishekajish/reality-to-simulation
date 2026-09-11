import csv
from pathlib import Path


INPUT_PATH = Path("data/processed/trajectories.csv")
OUTPUT_PATH = Path("data/processed/trajectory_features.csv")


def load_trajectories(csv_path: str):
    trajectories = {}

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            track_id = int(row["track_id"])

            trajectories.setdefault(track_id, []).append(
                {
                    "frame_id": int(row["frame_id"]),
                    "track_id": track_id,
                    "class_name": row["class_name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "confidence": float(row["confidence"]),
                }
            )

    return trajectories


def calculate_motion_features(trajectories: dict):
    rows = []

    for trajectory in trajectories.values():
        previous = None
        previous_velocity = None

        for point in trajectory:
            if previous is None:
                velocity_x = 0.0
                velocity_y = 0.0
                speed = 0.0
                acceleration = 0.0
            else:
                frame_delta = point["frame_id"] - previous["frame_id"]

                if frame_delta <= 0:
                    continue

                velocity_x = (
                    point["x"] - previous["x"]
                ) / frame_delta

                velocity_y = (
                    point["y"] - previous["y"]
                ) / frame_delta

                speed = (
                    velocity_x**2 + velocity_y**2
                ) ** 0.5

                if previous_velocity is None:
                    acceleration = 0.0
                else:
                    previous_speed = (
                        previous_velocity["speed"]
                    )

                    acceleration = (
                        speed - previous_speed
                    ) / frame_delta

            heading = 0.0

            if speed > 0:
                import math

                heading = math.degrees(
                    math.atan2(
                        velocity_y,
                        velocity_x,
                    )
                )

            rows.append(
                {
                    "frame_id": point["frame_id"],
                    "track_id": point["track_id"],
                    "class_name": point["class_name"],
                    "x": point["x"],
                    "y": point["y"],
                    "confidence": point["confidence"],
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                    "speed": speed,
                    "acceleration": acceleration,
                    "heading": heading,
                }
            )

            previous = point

            previous_velocity = {
                "speed": speed,
            }

    return rows


def save_features(rows: list[dict], output_path: str):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_id",
        "track_id",
        "class_name",
        "x",
        "y",
        "confidence",
        "velocity_x",
        "velocity_y",
        "speed",
        "acceleration",
        "heading",
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


if __name__ == "__main__":
    trajectories = load_trajectories(
        str(INPUT_PATH)
    )

    rows = calculate_motion_features(
        trajectories
    )

    save_features(
        rows,
        str(OUTPUT_PATH),
    )

    print("MOTION FEATURE EXTRACTION COMPLETE")
    print("INPUT TRAJECTORIES:", len(trajectories))
    print("FEATURE ROWS:", len(rows))
    print("OUTPUT:", OUTPUT_PATH)