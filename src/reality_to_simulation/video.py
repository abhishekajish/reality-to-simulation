import cv2


def open_video(video_path: str):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    return capture


def read_first_frame(capture):
    success, frame = capture.read()

    if not success:
        raise ValueError("Could not read the first frame from the video.")

    return frame


def get_video_metadata(capture):
    return {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }