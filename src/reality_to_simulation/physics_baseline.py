import csv
import math
from pathlib import Path

from reality_to_simulation.data_split import (
    split_track_ids,
)


INPUT_PATH = Path(
    "data/processed/prediction_dataset.csv"
)

INPUT_FRAMES = 10
PREDICTION_FRAMES = 5


def load_samples(csv_path):
    samples = []

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            samples.append(row)

    return samples


def predict_constant_velocity(row):
    velocity_x = float(
        row[
            f"input_{INPUT_FRAMES - 1}_velocity_x"
        ]
    )

    velocity_y = float(
        row[
            f"input_{INPUT_FRAMES - 1}_velocity_y"
        ]
    )

    predictions = []

    for step in range(
        1,
        PREDICTION_FRAMES + 1,
    ):
        predictions.append(
            (
                velocity_x * step,
                velocity_y * step,
            )
        )

    return predictions


def calculate_metrics(samples):
    total_ade = 0.0
    total_fde = 0.0

    for row in samples:
        predictions = (
            predict_constant_velocity(row)
        )

        errors = []

        for step, (
            predicted_dx,
            predicted_dy,
        ) in enumerate(predictions):
            actual_dx = float(
                row[
                    f"target_{step}_dx"
                ]
            )

            actual_dy = float(
                row[
                    f"target_{step}_dy"
                ]
            )

            error = math.sqrt(
                (
                    predicted_dx
                    - actual_dx
                ) ** 2
                + (
                    predicted_dy
                    - actual_dy
                ) ** 2
            )

            errors.append(error)

        total_ade += (
            sum(errors) / len(errors)
        )

        total_fde += errors[-1]

    if not samples:
        return 0.0, 0.0

    return (
        total_ade / len(samples),
        total_fde / len(samples),
    )


def main():
    samples = load_samples(
        str(INPUT_PATH)
    )

    track_ids = sorted(
        {
            int(row["track_id"])
            for row in samples
        }
    )

    _, _, test_track_ids = (
        split_track_ids(track_ids)
    )

    test_samples = [
        row
        for row in samples
        if int(row["track_id"])
        in test_track_ids
    ]

    ade, fde = calculate_metrics(
        test_samples
    )

    print(
        "PHYSICS BASELINE COMPLETE"
    )

    print(
        "TEST TRACKS:",
        len(test_track_ids),
    )

    print(
        "TEST SAMPLES:",
        len(test_samples),
    )

    print(
        "PREDICTION HORIZON:",
        PREDICTION_FRAMES,
    )

    print(
        "TARGET TYPE:",
        "FUTURE DISPLACEMENT",
    )

    print(
        "ADE (PIXELS):",
        round(ade, 4),
    )

    print(
        "FDE (PIXELS):",
        round(fde, 4),
    )


if __name__ == "__main__":
    main()