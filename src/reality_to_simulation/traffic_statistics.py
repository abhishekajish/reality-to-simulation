import csv
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/trajectory_features.csv"
)


def calculate_statistics(csv_path: str):
    rows = []

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(
            "No trajectory data found."
        )

    track_ids = {
        int(row["track_id"])
        for row in rows
    }

    speeds = [
        float(row["speed"])
        for row in rows
        if float(row["speed"]) > 0
    ]

    accelerations = [
        float(row["acceleration"])
        for row in rows
    ]

    average_speed = (
        sum(speeds) / len(speeds)
        if speeds
        else 0.0
    )

    average_acceleration = (
        sum(accelerations)
        / len(accelerations)
        if accelerations
        else 0.0
    )

    return {
        "unique_vehicles": len(track_ids),
        "trajectory_points": len(rows),
        "average_speed_pixels_per_frame": average_speed,
        "average_acceleration_pixels_per_frame2":
            average_acceleration,
    }


if __name__ == "__main__":
    statistics = calculate_statistics(
        str(INPUT_PATH)
    )

    print(
        "REAL TRAFFIC STATISTICS"
    )

    print(
        "UNIQUE VEHICLES:",
        statistics["unique_vehicles"],
    )

    print(
        "TRAJECTORY POINTS:",
        statistics["trajectory_points"],
    )

    print(
        "AVERAGE SPEED:",
        round(
            statistics[
                "average_speed_pixels_per_frame"
            ],
            4,
        ),
        "pixels/frame",
    )

    print(
        "AVERAGE ACCELERATION:",
        round(
            statistics[
                "average_acceleration_pixels_per_frame2"
            ],
            4,
        ),
        "pixels/frame²",
    )