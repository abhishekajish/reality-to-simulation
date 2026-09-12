import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/trajectory_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/prediction_dataset.csv"
)

INPUT_FRAMES = 10
PREDICTION_FRAMES = 5

FEATURE_COLUMNS = [
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "speed",
    "acceleration",
    "heading",
]


def load_trajectories(csv_path: str):
    trajectories = defaultdict(list)

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            track_id = int(row["track_id"])

            trajectories[track_id].append(
                {
                    "frame_id": int(row["frame_id"]),
                    "track_id": track_id,
                    "class_name": row["class_name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "velocity_x": float(
                        row["velocity_x"]
                    ),
                    "velocity_y": float(
                        row["velocity_y"]
                    ),
                    "speed": float(row["speed"]),
                    "acceleration": float(
                        row["acceleration"]
                    ),
                    "heading": float(row["heading"]),
                }
            )

    return dict(trajectories)


def create_prediction_samples(
    trajectories: dict,
):
    samples = []

    required_length = (
        INPUT_FRAMES + PREDICTION_FRAMES
    )

    for track_id, trajectory in trajectories.items():
        trajectory.sort(
            key=lambda point: point["frame_id"]
        )

        if len(trajectory) < required_length:
            continue

        for start in range(
            len(trajectory) - required_length + 1
        ):
            window = trajectory[
                start : start + required_length
            ]

            frame_ids = [
                point["frame_id"]
                for point in window
            ]

            expected_frames = list(
                range(
                    frame_ids[0],
                    frame_ids[0] + required_length,
                )
            )

            if frame_ids != expected_frames:
                continue

            input_window = window[
                :INPUT_FRAMES
            ]

            target_window = window[
                INPUT_FRAMES:
            ]

            last_observed_x = input_window[-1]["x"]
            last_observed_y = input_window[-1]["y"]

            row = {
                "track_id": track_id,
                "input_start_frame": input_window[0][
                    "frame_id"
                ],
                "input_end_frame": input_window[-1][
                    "frame_id"
                ],
                "target_start_frame": target_window[0][
                    "frame_id"
                ],
                "target_end_frame": target_window[-1][
                    "frame_id"
                ],
                "last_observed_x": last_observed_x,
                "last_observed_y": last_observed_y,
            }

            for frame_index, point in enumerate(
                input_window
            ):
                for feature in FEATURE_COLUMNS:
                    row[
                        f"input_{frame_index}_{feature}"
                    ] = point[feature]

            for frame_index, point in enumerate(
                target_window
            ):
                row[
                    f"target_{frame_index}_dx"
                ] = (
                    point["x"]
                    - last_observed_x
                )

                row[
                    f"target_{frame_index}_dy"
                ] = (
                    point["y"]
                    - last_observed_y
                )

            samples.append(row)

    return samples


def save_samples(
    samples: list[dict],
    output_path: str,
):
    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not samples:
        raise ValueError(
            "No valid prediction samples were created."
        )

    fieldnames = list(
        samples[0].keys()
    )

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
        writer.writerows(samples)


if __name__ == "__main__":
    trajectories = load_trajectories(
        str(INPUT_PATH)
    )

    samples = create_prediction_samples(
        trajectories
    )

    save_samples(
        samples,
        str(OUTPUT_PATH),
    )

    print(
        "PREDICTION DATASET CREATED"
    )
    print(
        "INPUT TRAJECTORIES:",
        len(trajectories),
    )
    print(
        "INPUT FRAMES:",
        INPUT_FRAMES,
    )
    print(
        "PREDICTION FRAMES:",
        PREDICTION_FRAMES,
    )
    print(
        "TARGET TYPE: FUTURE DISPLACEMENT"
    )
    print(
        "SAMPLES:",
        len(samples),
    )
    print(
        "OUTPUT:",
        OUTPUT_PATH,
    )