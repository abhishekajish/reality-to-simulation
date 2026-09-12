import os
import sys
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set. "
        "Please set the SUMO_HOME environment variable."
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


def run_simulation():
    traci.start(
        [
            "sumo",
            "-c",
            str(CONFIG_PATH),
            "--duration-log.statistics",
        ]
    )

    total_steps = 300

    total_departed = 0
    total_arrived = 0

    for _ in range(total_steps):
        traci.simulationStep()

        total_departed += (
            traci.simulation.getDepartedNumber()
        )

        total_arrived += (
            traci.simulation.getArrivedNumber()
        )

    running = traci.vehicle.getIDCount()

    traci.close()

    return {
        "simulation_steps": total_steps,
        "vehicles_departed": total_departed,
        "vehicles_arrived": total_arrived,
        "vehicles_running": running,
    }


if __name__ == "__main__":
    results = run_simulation()

    print("SUMO SIMULATION COMPLETE")

    print(
        "SIMULATION STEPS:",
        results["simulation_steps"],
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
        "VEHICLES STILL RUNNING:",
        results["vehicles_running"],
    )