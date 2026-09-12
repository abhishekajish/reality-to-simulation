import cv2
from pathlib import Path


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")

selected_points = []


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(selected_points) >= 4:
        return

    selected_points.append((x, y))

    print(
        f"Point {len(selected_points)}: "
        f"({x}, {y})"
    )


def draw_overlay(frame):
    display = frame.copy()

    for index, point in enumerate(selected_points):
        cv2.circle(
            display,
            point,
            7,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            display,
            f"P{index + 1}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if len(selected_points) >= 2:
        for index in range(1, len(selected_points)):
            cv2.line(
                display,
                selected_points[index - 1],
                selected_points[index],
                (0, 255, 0),
                2,
            )

    if len(selected_points) == 4:
        cv2.line(
            display,
            selected_points[3],
            selected_points[0],
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "4 POINTS SELECTED - PRESS Q",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return display


def main():
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {VIDEO_PATH}"
        )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise ValueError(
            "Could not read the first frame."
        )

    window_name = "Homography Point Selection"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    print()
    print("HOMOGRAPHY POINT SELECTION")
    print("---------------------------")
    print()
    print("We want a SMALL, relatively straight")
    print("section of the ROAD SURFACE.")
    print()
    print("Choose four points around one road/lane area.")
    print("Do NOT use:")
    print("  - cars")
    print("  - sidewalks")
    print("  - traffic lights")
    print("  - median")
    print("  - grass")
    print()
    print("Press R to reset.")
    print("Press Q when finished.")
    print()

    while True:
        display = draw_overlay(frame)

        cv2.imshow(
            window_name,
            display,
        )

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            selected_points.clear()
            print("Points reset.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    print()
    print("SELECTED POINTS:")
    print(selected_points)


if __name__ == "__main__":
    main()