from reality_to_simulation.signal_experiment import (
    run_signal_experiment,
)


SCENARIOS = [
    ("Baseline", 30, 30),
    ("Long Green", 45, 30),
    ("Short Green", 20, 30),
]


def main():
    results = []

    for name, green, red in SCENARIOS:
        print()
        print(
            f"Running {name}: "
            f"{green}s green / {red}s red"
        )

        result = run_signal_experiment(
            green_duration=green,
            red_duration=red,
        )

        results.append(
            {
                "name": name,
                **result,
            }
        )

    print()
    print(
        "SIGNAL SCENARIO COMPARISON"
    )
    print("=" * 72)

    print(
        f"{'SCENARIO':<16}"
        f"{'GREEN':>8}"
        f"{'ARRIVED':>10}"
        f"{'AVG SPEED':>14}"
        f"{'AVG WAIT':>14}"
    )

    print("-" * 72)

    for result in results:
        print(
            f"{result['name']:<16}"
            f"{result['green_duration']:>8}s"
            f"{result['vehicles_arrived']:>10}"
            f"{result['average_speed']:>14.2f}"
            f"{result['average_waiting_time']:>14.2f}"
        )

    best_wait = min(
        results,
        key=lambda result:
        result["average_waiting_time"],
    )

    best_speed = max(
        results,
        key=lambda result:
        result["average_speed"],
    )

    print()
    print(
        "BEST FOR WAITING TIME:",
        best_wait["name"],
    )

    print(
        "BEST FOR AVERAGE SPEED:",
        best_speed["name"],
    )


if __name__ == "__main__":
    main()