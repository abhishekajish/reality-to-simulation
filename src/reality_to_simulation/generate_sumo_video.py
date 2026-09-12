import os
import sys
from pathlib import Path

import cv2


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

OUTPUT_PATH = Path(
    "data/interim/sumo_simulation.mp4"
)


def draw_simulation_frame(
    frame,
    vehicle_ids,
    vehicle_positions,
    traffic_light_state,
    step,
):
    height, width = frame.shape[:2]

    # Road
    cv2.rectangle(
        frame,
        (150, 250),
        (1130, 470),
        (60, 60, 60),
        -1,
    )

    # Lane markings
    cv2.line(
        frame,
        (150, 360),
        (1130, 360),
        (255, 255, 255),
        2,
    )

    # Intersection
    cv2.rectangle(
        frame,
        (560, 250),
        (720, 470),
        (80, 80, 80),
        -1,
    )

    # Traffic light
    light_color = (
        (0, 255, 0)
        if traffic_light_state == "green"
        else (0, 0, 255)
    )

    cv2.circle(
        frame,
        (640, 180),
        25,
        light_color,
        -1,
    )

    cv2.putText(
        frame,
        f"Traffic Light: {traffic_light_state.upper()}",
        (500, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Vehicles
    for vehicle_id in vehicle_ids:

        x, y = vehicle_positions[
            vehicle_id
        ]

        # Convert SUMO coordinates
        # into the visualization area.
        pixel_x = int(
            150
            + (
                x / 500
            ) * 980
        )

        pixel_y = 330

        cv2.rectangle(
            frame,
            (
                pixel_x - 15,
                pixel_y - 8,
            ),
            (
                pixel_x + 15,
                pixel_y + 8,
            ),
            (255, 200, 0),
            -1,
        )

        cv2.putText(
            frame,
            vehicle_id.split(".")[-1],
            (
                pixel_x - 12,
                pixel_y - 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

    # Simulation information
    cv2.putText(
        frame,
        f"Simulation time: {step}s",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Vehicles: {len(vehicle_ids)}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame


def generate_sumo_video():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    width = 1280
    height = 720
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(
            "Could not create SUMO video."
        )

    traci.start(
        [
            "sumo",
            "-c",
            str(CONFIG_PATH),
            "--no-step-log",
        ]
    )

    total_steps = 300

    for step in range(total_steps):

        if step % 60 < 30:

            traci.trafficlight.setRedYellowGreenState(
                "junction",
                "GG",
            )

            traffic_light_state = "green"

        else:

            traci.trafficlight.setRedYellowGreenState(
                "junction",
                "rr",
            )

            traffic_light_state = "red"

        traci.simulationStep()

        vehicle_ids = (
            traci.vehicle.getIDList()
        )

        vehicle_positions = {}

        for vehicle_id in vehicle_ids:

            position = (
                traci.vehicle.getPosition(
                    vehicle_id
                )
            )

            vehicle_positions[
                vehicle_id
            ] = position

        frame = (
            255
            * __import__("numpy").ones(
                (
                    height,
                    width,
                    3,
                ),
                dtype="uint8",
            )
        )

        frame = draw_simulation_frame(
            frame,
            vehicle_ids,
            vehicle_positions,
            traffic_light_state,
            step,
        )

        writer.write(frame)

    traci.close()
    writer.release()

    return OUTPUT_PATH


if __name__ == "__main__":

    output = generate_sumo_video()

    print(
        "SUMO VIDEO GENERATED"
    )

    print(
        "OUTPUT:",
        output,
    )