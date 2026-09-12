import torch
from torch import nn
from torch_geometric.nn import GCNConv


class TrajectoryGNN(nn.Module):
    def __init__(
        self,
        node_input_size: int = 7,
        hidden_size: int = 64,
        output_size: int = 2,
        prediction_frames: int = 5,
    ):
        super().__init__()

        self.prediction_frames = prediction_frames

        self.conv1 = GCNConv(
            node_input_size,
            hidden_size,
        )

        self.conv2 = GCNConv(
            hidden_size,
            hidden_size,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            prediction_frames * output_size,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x,
        edge_index,
    ):
        x = self.conv1(
            x,
            edge_index,
        )

        x = self.activation(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = self.activation(x)

        predictions = self.output_layer(x)

        predictions = predictions.view(
            -1,
            self.prediction_frames,
            2,
        )

        return predictions


if __name__ == "__main__":
    node_features = torch.randn(
        15,
        7,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 3, 2, 5, 4],
        ],
        dtype=torch.long,
    )

    model = TrajectoryGNN()

    predictions = model(
        node_features,
        edge_index,
    )

    print("GNN MODEL INITIALIZED")
    print(
        "NODE FEATURE SHAPE:",
        tuple(node_features.shape),
    )
    print(
        "EDGE INDEX SHAPE:",
        tuple(edge_index.shape),
    )
    print(
        "PREDICTION SHAPE:",
        tuple(predictions.shape),
    )
    print(
        "PREDICTION TYPE:",
        "5 FUTURE DISPLACEMENTS",
    )
    print(
        "TRAINABLE PARAMETERS:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )