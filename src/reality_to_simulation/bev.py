import cv2
import numpy as np
from pathlib import Path


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")
OUTPUT_PATH = Path("data/interim/bev_preview.jpg")


SOURCE_POINTS = np.float32(
    [
        [830, 278],
        [1098, 301],
        [1128, 348],
        [731, 314],
    ]
)


OUTPUT_WIDTH = 800
OUTPUT_HEIGHT = 600


def create_bev_preview(
    video_path: str,
    output_path: str,
):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise ValueError(
            "Could not read the first frame."
        )

    destination_points = np.float32(
        [
            [0, 0],
            [OUTPUT_WIDTH, 0],
            [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            [0, OUTPUT_HEIGHT],
        ]
    )

    homography_matrix = cv2.getPerspectiveTransform(
        SOURCE_POINTS,
        destination_points,
    )

    bev = cv2.warpPerspective(
        frame,
        homography_matrix,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_file),
        bev,
    )

    if not success:
        raise ValueError(
            f"Could not save BEV preview: {output_path}"
        )

    return homography_matrix


if __name__ == "__main__":
    matrix = create_bev_preview(
        str(VIDEO_PATH),
        str(OUTPUT_PATH),
    )

    print("BEV PREVIEW COMPLETE")
    print("OUTPUT:", OUTPUT_PATH)
    print()
    print("HOMOGRAPHY MATRIX:")
    print(matrix)