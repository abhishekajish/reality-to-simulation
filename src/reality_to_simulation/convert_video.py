import subprocess
from pathlib import Path

import imageio_ffmpeg


INPUT_PATH = Path(
    "data/interim/sumo_simulation.mp4"
)

OUTPUT_PATH = Path(
    "data/interim/sumo_simulation_browser.mp4"
)


def convert_video():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(INPUT_PATH),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT_PATH),
    ]

    subprocess.run(
        command,
        check=True,
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    output = convert_video()

    print(
        "VIDEO CONVERSION COMPLETE"
    )

    print(
        "OUTPUT:",
        output,
    )

    print(
        "CODEC:",
        "H.264 / yuv420p",
    )