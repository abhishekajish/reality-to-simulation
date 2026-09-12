import csv
from pathlib import Path

import torch

from reality_to_simulation.lstm_model import (
    TrajectoryLSTM,
)


INPUT_PATH = Path(
    "data/processed/prediction_dataset.csv"
)

MODEL_PATH = Path(
    "models/lstm_trajectory.pt"
)

INPUT_FRAMES = 10
PREDICTION_FRAMES = 5

INPUT_FEATURES = [
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "speed",
    "acceleration",
    "heading",
]


def load_dataset(csv_path):
    inputs = []
    targets = []
    track_ids = []

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            input_sequence = []

            for frame_index in range(
                INPUT_FRAMES
            ):
                input_sequence.append(
                    [
                        float(
                            row[
                                f"input_{frame_index}_{feature}"
                            ]
                        )
                        for feature in INPUT_FEATURES
                    ]
                )

            target_sequence = []

            for frame_index in range(
                PREDICTION_FRAMES
            ):
                target_sequence.append(
                    [
                        float(
                            row[
                                f"target_{frame_index}_dx"
                            ]
                        ),
                        float(
                            row[
                                f"target_{frame_index}_dy"
                            ]
                        ),
                    ]
                )

            inputs.append(input_sequence)
            targets.append(target_sequence)
            track_ids.append(
                int(row["track_id"])
            )

    return (
        torch.tensor(
            inputs,
            dtype=torch.float32,
        ),
        torch.tensor(
            targets,
            dtype=torch.float32,
        ),
        track_ids,
    )


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    test_track_ids = set(
        checkpoint["test_track_ids"]
    )

    (
        inputs,
        targets,
        track_ids,
    ) = load_dataset(
        str(INPUT_PATH)
    )

    test_indices = [
        index
        for index, track_id in enumerate(
            track_ids
        )
        if track_id in test_track_ids
    ]

    test_inputs = inputs[test_indices]
    test_targets = targets[test_indices]

    mean = checkpoint["input_mean"]
    std = checkpoint["input_std"]

    test_inputs = (
        test_inputs - mean
    ) / std

    model = TrajectoryLSTM()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        predictions = model(
            test_inputs
        )

    errors = torch.sqrt(
        torch.sum(
            (
                predictions
                - test_targets
            )
            ** 2,
            dim=2,
        )
    )

    ade = errors.mean().item()

    fde = (
        errors[:, -1]
        .mean()
        .item()
    )

    print(
        "LSTM EVALUATION COMPLETE"
    )

    print(
        "TEST TRACKS:",
        len(test_track_ids),
    )

    print(
        "TEST SAMPLES:",
        len(test_inputs),
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