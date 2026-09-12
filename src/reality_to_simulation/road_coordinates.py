from pathlib import Path

import csv


INPUT_PATH = Path(
    "data/processed/trajectory_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/road_coordinates.csv"
)

ROAD_CENTER_X = 640.0
ROAD_START_Y = 720.0


def convert_coordinates(
    input_path: str,
    output_path: str,
):
    rows = []

    with Path(input_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            x = float(row["x"])
            y = float(row["y"])

            longitudinal = (
                ROAD_START_Y - y
            )

            lateral = (
                x - ROAD_CENTER_X
            )

            rows.append(
                {
                    "frame_id": int(
                        row["frame_id"]
                    ),
                    "track_id": int(
                        row["track_id"]
                    ),
                    "class_name": row[
                        "class_name"
                    ],
                    "longitudinal": longitudinal,
                    "lateral": lateral,
                    "confidence": float(
                        row["confidence"]
                    ),
                }
            )

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "frame_id",
        "track_id",
        "class_name",
        "longitudinal",
        "lateral",
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

    return len(rows)


if __name__ == "__main__":
    row_count = convert_coordinates(
        str(INPUT_PATH),
        str(OUTPUT_PATH),
    )

    print(
        "ROAD COORDINATE CONVERSION COMPLETE"
    )

    print(
        "ROWS:",
        row_count,
    )

    print(
        "COORDINATE SYSTEM:",
        "ROAD-RELATIVE",
    )

    print(
        "OUTPUT:",
        OUTPUT_PATH,
    )