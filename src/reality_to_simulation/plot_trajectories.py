import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("data/processed/trajectory_features.csv")
OUTPUT_PATH = Path("data/interim/trajectory_plot.png")


def load_trajectories(csv_path: str):
    trajectories = defaultdict(list)

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            trajectories[int(row["track_id"])].append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )

    return dict(trajectories)


def plot_trajectories(
    trajectories: dict,
    output_path: str,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    for track_id, trajectory in trajectories.items():
        if len(trajectory) < 5:
            continue

        x_values = [point["x"] for point in trajectory]
        y_values = [point["y"] for point in trajectory]

        plt.plot(
            x_values,
            y_values,
            linewidth=1,
            alpha=0.7,
        )

        plt.scatter(
            x_values[0],
            y_values[0],
            s=12,
        )

    plt.gca().invert_yaxis()

    plt.xlabel("Image X (pixels)")
    plt.ylabel("Image Y (pixels)")
    plt.title("Tracked Vehicle Trajectories")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()


if __name__ == "__main__":
    trajectories = load_trajectories(
        str(INPUT_PATH)
    )

    plot_trajectories(
        trajectories,
        str(OUTPUT_PATH),
    )

    print("TRAJECTORY PLOT COMPLETE")
    print("TRACKS LOADED:", len(trajectories))
    print("OUTPUT:", OUTPUT_PATH)