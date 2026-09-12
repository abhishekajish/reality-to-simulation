import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data


INPUT_PATH = Path(
    "data/processed/trajectory_features.csv"
)

INPUT_FRAMES = 10
PREDICTION_FRAMES = 5

DISTANCE_THRESHOLD = 150.0

NODE_FEATURES = [
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "speed",
    "acceleration",
    "heading",
]


def load_track_data(csv_path: str):
    tracks = defaultdict(list)

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            tracks[int(row["track_id"])].append(
                {
                    "frame_id": int(row["frame_id"]),
                    "track_id": int(row["track_id"]),
                    "class_name": row["class_name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "velocity_x": float(row["velocity_x"]),
                    "velocity_y": float(row["velocity_y"]),
                    "speed": float(row["speed"]),
                    "acceleration": float(row["acceleration"]),
                    "heading": float(row["heading"]),
                }
            )

    for trajectory in tracks.values():
        trajectory.sort(
            key=lambda point: point["frame_id"]
        )

    return dict(tracks)


def create_graph_sample(
    target_track_id: int,
    target_start_index: int,
    tracks: dict,
):
    target_trajectory = tracks[target_track_id]

    required_length = (
        INPUT_FRAMES + PREDICTION_FRAMES
    )

    target_window = target_trajectory[
        target_start_index:
        target_start_index + required_length
    ]

    if len(target_window) != required_length:
        return None

    input_window = target_window[:INPUT_FRAMES]
    target_future = target_window[INPUT_FRAMES:]

    frame_ids = [
        point["frame_id"]
        for point in target_window
    ]

    expected_frames = list(
        range(
            frame_ids[0],
            frame_ids[0] + required_length,
        )
    )

    if frame_ids != expected_frames:
        return None

    last_frame = input_window[-1]["frame_id"]

    vehicles = []

    for trajectory in tracks.values():
        for point in trajectory:
            if point["frame_id"] == last_frame:
                vehicles.append(point)
                break

    target_vehicle = None

    for vehicle in vehicles:
        if vehicle["track_id"] == target_track_id:
            target_vehicle = vehicle
            break

    if target_vehicle is None:
        return None

    nearby_vehicles = []

    for vehicle in vehicles:
        dx = (
            vehicle["x"]
            - target_vehicle["x"]
        )

        dy = (
            vehicle["y"]
            - target_vehicle["y"]
        )

        distance = math.sqrt(
            dx**2 + dy**2
        )

        if distance <= DISTANCE_THRESHOLD:
            nearby_vehicles.append(vehicle)

    if not nearby_vehicles:
        return None

    nearby_vehicles.sort(
        key=lambda vehicle: (
            0
            if vehicle["track_id"] == target_track_id
            else 1,
            math.sqrt(
                (
                    vehicle["x"]
                    - target_vehicle["x"]
                ) ** 2
                + (
                    vehicle["y"]
                    - target_vehicle["y"]
                ) ** 2
            ),
        )
    )

    target_last_x = target_vehicle["x"]
    target_last_y = target_vehicle["y"]

    node_features = []

    for vehicle in nearby_vehicles:
        node_features.append(
            [
                vehicle["x"] - target_last_x,
                vehicle["y"] - target_last_y,
                vehicle["velocity_x"],
                vehicle["velocity_y"],
                vehicle["speed"],
                vehicle["acceleration"],
                vehicle["heading"],
            ]
        )

    node_features = torch.tensor(
        node_features,
        dtype=torch.float32,
    )

    edge_pairs = []

    for i, source in enumerate(
        nearby_vehicles
    ):
        for j, destination in enumerate(
            nearby_vehicles
        ):
            if i == j:
                continue

            dx = (
                destination["x"]
                - source["x"]
            )

            dy = (
                destination["y"]
                - source["y"]
            )

            distance = math.sqrt(
                dx**2 + dy**2
            )

            if distance <= DISTANCE_THRESHOLD:
                edge_pairs.append([i, j])

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    target = []

    for point in target_future:
        target.append(
            [
                point["x"] - target_last_x,
                point["y"] - target_last_y,
            ]
        )

    target = torch.tensor(
        target,
        dtype=torch.float32,
    )

    target_node_index = 0

    return Data(
        x=node_features,
        edge_index=edge_index,
        y=target,
        target_node=torch.tensor(
            target_node_index,
            dtype=torch.long,
        ),
        target_track_id=torch.tensor(
            target_track_id,
            dtype=torch.long,
        ),
        frame_id=torch.tensor(
            last_frame,
            dtype=torch.long,
        ),
    )


def create_dataset(tracks: dict):
    samples = []

    required_length = (
        INPUT_FRAMES + PREDICTION_FRAMES
    )

    for track_id, trajectory in tracks.items():
        if len(trajectory) < required_length:
            continue

        for start_index in range(
            len(trajectory)
            - required_length
            + 1
        ):
            sample = create_graph_sample(
                target_track_id=track_id,
                target_start_index=start_index,
                tracks=tracks,
            )

            if sample is not None:
                samples.append(sample)

    return samples


if __name__ == "__main__":
    tracks = load_track_data(
        str(INPUT_PATH)
    )

    samples = create_dataset(tracks)

    if not samples:
        raise ValueError(
            "No GNN samples were created."
        )

    sample = samples[0]

    print("GNN DATASET CREATED")
    print(
        "TRAJECTORIES:",
        len(tracks),
    )
    print(
        "GRAPH SAMPLES:",
        len(samples),
    )
    print(
        "SAMPLE NODES:",
        sample.x.shape[0],
    )
    print(
        "NODE FEATURE SIZE:",
        sample.x.shape[1],
    )
    print(
        "SAMPLE EDGES:",
        sample.edge_index.shape[1],
    )
    print(
        "TARGET SHAPE:",
        tuple(sample.y.shape),
    )
    print(
        "POSITION REPRESENTATION:",
        "TARGET-RELATIVE",
    )