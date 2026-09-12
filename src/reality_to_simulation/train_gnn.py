import random
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from reality_to_simulation.data_split import (
    split_track_ids,
)
from reality_to_simulation.gnn_dataset import (
    INPUT_FRAMES,
    PREDICTION_FRAMES,
    INPUT_PATH,
    create_dataset,
    load_track_data,
)
from reality_to_simulation.gnn_model import (
    TrajectoryGNN,
)


MODEL_PATH = Path(
    "models/trajectory_gnn.pt"
)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

SEED = 42


def filter_samples(
    samples,
    track_ids,
):
    return [
        sample
        for sample in samples
        if int(sample.target_track_id)
        in track_ids
    ]


def get_global_target_indices(batch):
    target_indices = []

    for graph_index in range(
        batch.num_graphs
    ):
        graph_start = int(
            batch.ptr[graph_index]
        )

        target_node = int(
            batch.target_node[
                graph_index
            ]
        )

        target_indices.append(
            graph_start + target_node
        )

    return torch.tensor(
        target_indices,
        dtype=torch.long,
        device=batch.x.device,
    )


def calculate_loss(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        if training:
            optimizer.zero_grad()

        predictions = model(
            batch.x,
            batch.edge_index,
        )

        target_indices = (
            get_global_target_indices(
                batch
            )
        )

        target_predictions = (
            predictions[target_indices]
        )

        targets = batch.y.view(
            batch.num_graphs,
            PREDICTION_FRAMES,
            2,
        )

        loss = criterion(
            target_predictions,
            targets,
        )

        if training:
            loss.backward()
            optimizer.step()

        batch_size = batch.num_graphs

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return (
        total_loss / total_samples
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tracks = load_track_data(
        str(INPUT_PATH)
    )

    # Create valid graph samples FIRST.
    # Only tracks represented by these samples
    # are eligible for the prediction experiment.
    all_samples = create_dataset(
        tracks
    )

    experiment_track_ids = sorted(
        {
            int(sample.target_track_id)
            for sample in all_samples
        }
    )

    (
        train_ids,
        validation_ids,
        test_ids,
    ) = split_track_ids(
        experiment_track_ids
    )

    train_samples = filter_samples(
        all_samples,
        train_ids,
    )

    validation_samples = filter_samples(
        all_samples,
        validation_ids,
    )

    test_samples = filter_samples(
        all_samples,
        test_ids,
    )

    print(
        "DEVICE:",
        device,
    )

    print(
        "RAW TRACKS:",
        len(tracks),
    )

    print(
        "EXPERIMENT TRACKS:",
        len(experiment_track_ids),
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
        "TOTAL GRAPH SAMPLES:",
        len(all_samples),
    )

    print(
        "TRAIN SAMPLES:",
        len(train_samples),
    )

    print(
        "VALIDATION SAMPLES:",
        len(validation_samples),
    )

    print(
        "TEST SAMPLES:",
        len(test_samples),
    )

    train_loader = DataLoader(
        train_samples,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_samples,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TrajectoryGNN(
        node_input_size=7,
        hidden_size=64,
        output_size=2,
        prediction_frames=PREDICTION_FRAMES,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        train_loss = calculate_loss(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        validation_loss = calculate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        if (
            validation_loss
            < best_validation_loss
        ):
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
                    "input_frames":
                        INPUT_FRAMES,
                    "prediction_frames":
                        PREDICTION_FRAMES,
                    "input_features": 7,
                    "hidden_size": 64,
                    "best_validation_loss":
                        best_validation_loss,
                    "train_track_ids":
                        sorted(train_ids),
                    "validation_track_ids":
                        sorted(validation_ids),
                    "test_track_ids":
                        sorted(test_ids),
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
        "GNN TRAINING COMPLETE"
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