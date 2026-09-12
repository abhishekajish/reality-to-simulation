from pathlib import Path

import csv
import json


RESULTS_PATH = Path(
    "experiments/results/trajectory_benchmark.json"
)


RESULTS = {
    "experiment": "Trajectory Prediction Benchmark",
    "input_frames": 10,
    "prediction_frames": 5,
    "target_type": "future displacement",
    "test_tracks": 7,
    "test_samples": 711,
    "models": [
        {
            "name": "Constant Velocity",
            "type": "physics baseline",
            "ade_pixels": 4.8001,
            "fde_pixels": 7.5732,
        },
        {
            "name": "LSTM",
            "type": "sequence model",
            "ade_pixels": 2.9472,
            "fde_pixels": 4.6309,
        },
        {
            "name": "Transformer",
            "type": "sequence model",
            "ade_pixels": 2.9004,
            "fde_pixels": 4.0899,
        },
        {
            "name": "GNN",
            "type": "interaction model",
            "ade_pixels": 6.2800,
            "fde_pixels": 9.8735,
        },
    ],
}


def calculate_improvement(
    baseline,
    model,
):
    return (
        (baseline - model)
        / baseline
    ) * 100


def save_results(results):
    output_file = RESULTS_PATH

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def main():
    models = RESULTS["models"]

    baseline = models[0]

    print(
        "TRAJECTORY PREDICTION BENCHMARK"
    )
    print("=" * 60)

    print(
        "INPUT FRAMES:",
        RESULTS["input_frames"],
    )

    print(
        "PREDICTION FRAMES:",
        RESULTS["prediction_frames"],
    )

    print(
        "TEST TRACKS:",
        RESULTS["test_tracks"],
    )

    print(
        "TEST SAMPLES:",
        RESULTS["test_samples"],
    )

    print()

    print(
        f"{'MODEL':<20}"
        f"{'ADE':>12}"
        f"{'FDE':>12}"
    )

    print("-" * 44)

    for model in models:
        print(
            f"{model['name']:<20}"
            f"{model['ade_pixels']:>12.4f}"
            f"{model['fde_pixels']:>12.4f}"
        )

    print()

    print(
        "IMPROVEMENT OVER CONSTANT VELOCITY"
    )
    print("-" * 44)

    for model in models[1:]:
        ade_improvement = (
            calculate_improvement(
                baseline["ade_pixels"],
                model["ade_pixels"],
            )
        )

        fde_improvement = (
            calculate_improvement(
                baseline["fde_pixels"],
                model["fde_pixels"],
            )
        )

        print(
            f"{model['name']:<20}"
            f"ADE: {ade_improvement:>7.2f}% | "
            f"FDE: {fde_improvement:>7.2f}%"
        )

    best_ade = min(
        models,
        key=lambda model:
        model["ade_pixels"],
    )

    best_fde = min(
        models,
        key=lambda model:
        model["fde_pixels"],
    )

    print()
    print(
        "BEST ADE MODEL:",
        best_ade["name"],
    )

    print(
        "BEST FDE MODEL:",
        best_fde["name"],
    )

    save_results(RESULTS)

    print()
    print(
        "RESULTS SAVED:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()