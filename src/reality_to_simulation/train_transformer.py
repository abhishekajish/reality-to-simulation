import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reality_to_simulation.data_split import split_track_ids
from reality_to_simulation.transformer_model import (
    TrajectoryTransformer,
)


INPUT_PATH = Path(
    "data/processed/prediction_dataset.csv"
)

MODEL_PATH = Path(
    "models/trajectory_transformer.pt"
)

INPUT_FRAMES = 10
PREDICTION_FRAMES = 5

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

SEED = 42

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


def filter_by_tracks(
    inputs,
    targets,
    track_ids,
    allowed_tracks,
):
    indices = [
        index
        for index, track_id in enumerate(
            track_ids
        )
        if track_id in allowed_tracks
    ]

    return (
        inputs[indices],
        targets[indices],
    )


def normalize_inputs(
    train_inputs,
    validation_inputs,
    test_inputs,
):
    mean = train_inputs.mean(
        dim=(0, 1),
        keepdim=True,
    )

    std = train_inputs.std(
        dim=(0, 1),
        keepdim=True,
    )

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std,
    )

    return (
        (train_inputs - mean) / std,
        (validation_inputs - mean) / std,
        (test_inputs - mean) / std,
        mean,
        std,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        inputs,
        targets,
        track_ids,
    ) = load_dataset(
        str(INPUT_PATH)
    )

    unique_track_ids = sorted(
        set(track_ids)
    )

    (
        train_ids,
        validation_ids,
        test_ids,
    ) = split_track_ids(
        unique_track_ids
    )

    (
        train_inputs,
        train_targets,
    ) = filter_by_tracks(
        inputs,
        targets,
        track_ids,
        train_ids,
    )

    (
        validation_inputs,
        validation_targets,
    ) = filter_by_tracks(
        inputs,
        targets,
        track_ids,
        validation_ids,
    )

    (
        test_inputs,
        test_targets,
    ) = filter_by_tracks(
        inputs,
        targets,
        track_ids,
        test_ids,
    )

    (
        train_inputs,
        validation_inputs,
        test_inputs,
        mean,
        std,
    ) = normalize_inputs(
        train_inputs,
        validation_inputs,
        test_inputs,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_inputs,
            train_targets,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        TensorDataset(
            validation_inputs,
            validation_targets,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TrajectoryTransformer().to(
        device
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float(
        "inf"
    )

    print(
        "DEVICE:",
        device,
    )

    print(
        "TRAIN TRACKS:",
        len(train_ids),
    )

    print(
        "VALIDATION TRACKS:",
        len(validation_ids),
    )

    print(
        "TEST TRACKS:",
        len(test_ids),
    )

    print(
        "TRAIN SAMPLES:",
        len(train_inputs),
    )

    print(
        "VALIDATION SAMPLES:",
        len(validation_inputs),
    )

    print(
        "TEST SAMPLES:",
        len(test_inputs),
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        train_loss = 0.0
        train_count = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(
                device
            )

            batch_targets = batch_targets.to(
                device
            )

            optimizer.zero_grad()

            predictions = model(
                batch_inputs
            )

            loss = criterion(
                predictions,
                batch_targets,
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item()
                * batch_inputs.size(0)
            )

            train_count += (
                batch_inputs.size(0)
            )

        model.eval()

        validation_loss = 0.0
        validation_count = 0

        with torch.no_grad():
            for (
                batch_inputs,
                batch_targets,
            ) in validation_loader:
                batch_inputs = batch_inputs.to(
                    device
                )

                batch_targets = batch_targets.to(
                    device
                )

                predictions = model(
                    batch_inputs
                )

                loss = criterion(
                    predictions,
                    batch_targets,
                )

                validation_loss += (
                    loss.item()
                    * batch_inputs.size(0)
                )

                validation_count += (
                    batch_inputs.size(0)
                )

        train_loss /= train_count
        validation_loss /= validation_count

        if validation_loss < best_validation_loss:
            best_validation_loss = (
                validation_loss
            )

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "input_mean": mean,
                    "input_std": std,
                    "input_frames":
                        INPUT_FRAMES,
                    "prediction_frames":
                        PREDICTION_FRAMES,
                    "test_track_ids":
                        sorted(test_ids),
                    "train_track_ids":
                        sorted(train_ids),
                    "validation_track_ids":
                        sorted(validation_ids),
                },
                MODEL_PATH,
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Validation Loss: "
                f"{validation_loss:.6f}"
            )

    print()
    print(
        "TRANSFORMER TRAINING COMPLETE"
    )

    print(
        "BEST VALIDATION LOSS:",
        round(
            best_validation_loss,
            6,
        ),
    )

    print(
        "MODEL:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()