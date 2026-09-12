import csv
import math
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/trajectory_features.csv"
)

DISTANCE_THRESHOLD = 150.0


def load_frame_data(csv_path: str):
    frames = defaultdict(list)

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            frames[
                int(row["frame_id"])
            ].append(
                {
                    "track_id": int(
                        row["track_id"]
                    ),
                    "class_name": row[
                        "class_name"
                    ],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "velocity_x": float(
                        row["velocity_x"]
                    ),
                    "velocity_y": float(
                        row["velocity_y"]
                    ),
                    "speed": float(
                        row["speed"]
                    ),
                    "acceleration": float(
                        row["acceleration"]
                    ),
                    "heading": float(
                        row["heading"]
                    ),
                }
            )

    return dict(frames)


def build_interaction_graph(
    vehicles: list[dict],
    distance_threshold: float = DISTANCE_THRESHOLD,
):
    nodes = []
    edges = []

    for vehicle in vehicles:
        nodes.append(
            {
                "track_id": vehicle[
                    "track_id"
                ],
                "features": [
                    vehicle["x"],
                    vehicle["y"],
                    vehicle["velocity_x"],
                    vehicle["velocity_y"],
                    vehicle["speed"],
                    vehicle["acceleration"],
                    vehicle["heading"],
                ],
            }
        )

    for i, source in enumerate(vehicles):
        for j, target in enumerate(vehicles):
            if i == j:
                continue

            dx = (
                target["x"]
                - source["x"]
            )

            dy = (
                target["y"]
                - source["y"]
            )

            distance = math.sqrt(
                dx**2 + dy**2
            )

            if distance <= distance_threshold:
                edges.append(
                    {
                        "source": source[
                            "track_id"
                        ],
                        "target": target[
                            "track_id"
                        ],
                        "features": [
                            dx,
                            dy,
                            distance,
                            (
                                target[
                                    "velocity_x"
                                ]
                                - source[
                                    "velocity_x"
                                ]
                            ),
                            (
                                target[
                                    "velocity_y"
                                ]
                                - source[
                                    "velocity_y"
                                ]
                            ),
                        ],
                    }
                )

    return {
        "nodes": nodes,
        "edges": edges,
    }


if __name__ == "__main__":
    frames = load_frame_data(
        str(INPUT_PATH)
    )

    frame_ids = sorted(frames)

    if not frame_ids:
        raise ValueError(
            "No frame data found."
        )

    sample_frame_id = frame_ids[
        len(frame_ids) // 2
    ]

    graph = build_interaction_graph(
        frames[sample_frame_id]
    )

    print(
        "INTERACTION GRAPH CREATED"
    )
    print(
        "SAMPLE FRAME:",
        sample_frame_id,
    )
    print(
        "NODES:",
        len(graph["nodes"]),
    )
    print(
        "EDGES:",
        len(graph["edges"]),
    )

    if graph["nodes"]:
        print(
            "NODE FEATURE SIZE:",
            len(
                graph["nodes"][0][
                    "features"
                ]
            ),
        )

    if graph["edges"]:
        print(
            "EDGE FEATURE SIZE:",
            len(
                graph["edges"][0][
                    "features"
                ]
            ),
        )