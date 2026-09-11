from pathlib import Path

from reality_to_simulation.video import (
    get_video_metadata,
    open_video,
    read_first_frame,
)


VIDEO_PATH = Path("data/raw/videos/traffic_demo.mp4")


def test_package_import():
    import reality_to_simulation

    assert reality_to_simulation is not None


def test_open_video():
    capture = open_video(str(VIDEO_PATH))

    assert capture.isOpened()

    capture.release()


def test_read_first_frame():
    capture = open_video(str(VIDEO_PATH))

    frame = read_first_frame(capture)

    assert frame is not None
    assert frame.shape == (720, 1280, 3)

    capture.release()


def test_video_metadata():
    capture = open_video(str(VIDEO_PATH))

    metadata = get_video_metadata(capture)

    assert metadata["width"] == 1280
    assert metadata["height"] == 720
    assert metadata["fps"] == 30.0
    assert metadata["frame_count"] == 251

    capture.release()