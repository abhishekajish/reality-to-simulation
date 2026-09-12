from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from reality_to_simulation.gnn_dataset import (
    INPUT_PATH,
    PREDICTION_FRAMES,
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


def calculate_metrics(
    model,
    loader,
    device,
):
    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

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

            errors = torch.sqrt(
                torch.sum(
                    (
                        target_predictions
                        - targets
                    ) ** 2,
                    dim=2,
                )
            )

            ade = errors.mean(
                dim=1
            )

            fde = errors[:, -1]

            total_ade += (
                ade.sum().item()
            )

            total_fde += (
                fde.sum().item()
            )

            total_samples += (
                batch.num_graphs
            )

    if total_samples == 0:
        return 0.0, 0.0

    return (
        total_ade / total_samples,
        total_fde / total_samples,
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

    tracks = load_track_data(
        str(INPUT_PATH)
    )

    all_samples = create_dataset(
        tracks
    )

    test_samples = filter_samples(
        all_samples,
        test_track_ids,
    )

    if not test_samples:
        raise ValueError(
            "No GNN test samples found."
        )

    test_loader = DataLoader(
        test_samples,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TrajectoryGNN(
        node_input_size=7,
        hidden_size=checkpoint[
            "hidden_size"
        ],
        output_size=2,
        prediction_frames=PREDICTION_FRAMES,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    ade, fde = calculate_metrics(
        model,
        test_loader,
        device,
    )

    print(
        "GNN EVALUATION COMPLETE"
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
        "POSITION REPRESENTATION:",
        "TARGET-RELATIVE",
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