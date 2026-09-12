import csv
from pathlib import Path

import torch
from torch import nn


INPUT_PATH = Path(
    "data/processed/prediction_dataset.csv"
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


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = 7,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        output_size: int = 2,
        prediction_frames: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.prediction_frames = prediction_frames

        self.input_projection = nn.Linear(
            input_size,
            model_dim,
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(
                1,
                INPUT_FRAMES,
                model_dim,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            model_dim,
            prediction_frames * output_size,
        )

    def forward(self, x):
        x = self.input_projection(x)

        x = (
            x
            + self.positional_encoding
        )

        encoded = self.encoder(x)

        last_output = encoded[:, -1, :]

        prediction = self.output_layer(
            last_output
        )

        prediction = prediction.view(
            -1,
            self.prediction_frames,
            2,
        )

        return prediction


def load_dataset(csv_path: str):
    inputs = []
    targets = []

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

    return (
        torch.tensor(
            inputs,
            dtype=torch.float32,
        ),
        torch.tensor(
            targets,
            dtype=torch.float32,
        ),
    )


if __name__ == "__main__":
    inputs, targets = load_dataset(
        str(INPUT_PATH)
    )

    model = TrajectoryTransformer()

    predictions = model(inputs)

    print(
        "TRANSFORMER MODEL INITIALIZED"
    )
    print(
        "INPUT SHAPE:",
        tuple(inputs.shape),
    )
    print(
        "TARGET SHAPE:",
        tuple(targets.shape),
    )
    print(
        "PREDICTION SHAPE:",
        tuple(predictions.shape),
    )
    print(
        "TARGET TYPE: FUTURE DISPLACEMENT"
    )
    print(
        "TRAINABLE PARAMETERS:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )