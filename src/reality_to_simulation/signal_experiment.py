import os
import sys
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set."
    )

sys.path.append(
    os.path.join(
        os.environ["SUMO_HOME"],
        "tools",
    )
)

import traci


CONFIG_PATH = Path(
    "configs/sumo/traffic_twin.sumocfg"
)

TLS_ID = "junction"


def run_signal_experiment(
    green_duration=30,
    red_duration=30,
):
    traci.start(
        [
            "sumo",
            "-c",
            str(CONFIG_PATH),
        ]
    )

    total_steps = 300

    cycle_length = (
        green_duration + red_duration
    )

    total_waiting_time = 0.0
    total_speed = 0.0
    speed_samples = 0

    total_departed = 0
    total_arrived = 0

    for step in range(total_steps):

        cycle_position = (
            step % cycle_length
        )

        if cycle_position < green_duration:
            traci.trafficlight.setRedYellowGreenState(
                TLS_ID,
                "GG",
            )
        else:
            traci.trafficlight.setRedYellowGreenState(
                TLS_ID,
                "rr",
            )

        traci.simulationStep()

        total_departed += (
            traci.simulation.getDepartedNumber()
        )

        total_arrived += (
            traci.simulation.getArrivedNumber()
        )

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        for vehicle_id in vehicle_ids:
            speed = traci.vehicle.getSpeed(
                vehicle_id
            )

            total_speed += speed
            speed_samples += 1

            # Each simulation step represents
            # one second. Count a vehicle as
            # waiting when it is effectively stopped.
            if speed < 0.1:
                total_waiting_time += 1.0

    traci.close()

    average_speed = (
        total_speed / speed_samples
        if speed_samples
        else 0.0
    )

    average_waiting_time = (
        total_waiting_time
        / total_departed
        if total_departed
        else 0.0
    )

    return {
        "green_duration": green_duration,
        "red_duration": red_duration,
        "vehicles_departed": total_departed,
        "vehicles_arrived": total_arrived,
        "average_speed": average_speed,
        "total_waiting_time": total_waiting_time,
        "average_waiting_time": average_waiting_time,
    }


if __name__ == "__main__":
    results = run_signal_experiment(
        green_duration=30,
        red_duration=30,
    )

    print(
        "SIGNAL EXPERIMENT COMPLETE"
    )

    print(
        "GREEN:",
        results["green_duration"],
        "seconds",
    )

    print(
        "RED:",
        results["red_duration"],
        "seconds",
    )

    print(
        "VEHICLES DEPARTED:",
        results["vehicles_departed"],
    )

    print(
        "VEHICLES ARRIVED:",
        results["vehicles_arrived"],
    )

    print(
        "AVERAGE SPEED:",
        round(
            results["average_speed"],
            2,
        ),
        "m/s",
    )

    print(
        "TOTAL WAITING TIME:",
        round(
            results["total_waiting_time"],
            2,
        ),
        "vehicle-seconds",
    )

    print(
        "AVERAGE WAITING TIME:",
        round(
            results["average_waiting_time"],
            2,
        ),
        "seconds/vehicle",
    )