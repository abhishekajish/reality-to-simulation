import json
from pathlib import Path

from reality_to_simulation.signal_experiment import (
    run_signal_experiment,
)


GREEN_DURATIONS = [
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
]

RED_DURATION = 30

OUTPUT_PATH = Path(
    "experiments/results/signal_optimization.json"
)


def main():
    results = []

    print(
        "SIGNAL TIMING OPTIMIZATION"
    )
    print("=" * 60)

    for green_duration in GREEN_DURATIONS:
        print(
            f"\nTesting "
            f"{green_duration}s green / "
            f"{RED_DURATION}s red..."
        )

        result = run_signal_experiment(
            green_duration=green_duration,
            red_duration=RED_DURATION,
        )

        results.append(result)

    best = min(
        results,
        key=lambda result:
        result["average_waiting_time"],
    )

    optimization_result = {
        "experiment": "Traffic Signal Timing Optimization",
        "red_duration": RED_DURATION,
        "tested_green_durations": GREEN_DURATIONS,
        "objective": "minimize_average_waiting_time",
        "best_scenario": best,
        "all_results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            optimization_result,
            file,
            indent=4,
        )

    print()
    print(
        "SIGNAL OPTIMIZATION RESULTS"
    )
    print("=" * 72)

    print(
        f"{'GREEN':<12}"
        f"{'ARRIVED':<12}"
        f"{'AVG SPEED':<15}"
        f"{'AVG WAIT':<15}"
    )

    print("-" * 72)

    for result in results:
        print(
            f"{result['green_duration']:<12}"
            f"{result['vehicles_arrived']:<12}"
            f"{result['average_speed']:<15.2f}"
            f"{result['average_waiting_time']:<15.2f}"
        )

    print()
    print(
        "OPTIMAL GREEN DURATION:",
        best["green_duration"],
        "seconds",
    )

    print(
        "AVERAGE SPEED:",
        round(
            best["average_speed"],
            2,
        ),
        "m/s",
    )

    print(
        "AVERAGE WAITING TIME:",
        round(
            best["average_waiting_time"],
            2,
        ),
        "seconds/vehicle",
    )

    print()
    print(
        "RESULTS SAVED:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()