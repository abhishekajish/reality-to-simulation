from pathlib import Path

from reality_to_simulation.video import (
    get_video_metadata,
    open_video,
    read_first_frame,
    read_frames,
    save_frame,
)
from reality_to_simulation.detector import VehicleDetector

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

def test_read_frames():
    capture = open_video(str(VIDEO_PATH))

    frames = list(read_frames(capture))

    assert len(frames) == 251
    assert frames[0].shape == (720, 1280, 3)
    assert frames[-1].shape == (720, 1280, 3)

    capture.release()

def test_save_frame(tmp_path):
    capture = open_video(str(VIDEO_PATH))

    frame = read_first_frame(capture)

    output_path = tmp_path / "test_frame.jpg"

    save_frame(frame, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    capture.release()

def test_vehicle_detector():
    detector = VehicleDetector()

    capture = open_video(str(VIDEO_PATH))
    frame = read_first_frame(capture)

    detections = detector.detect(frame)

    assert len(detections) > 0

    for detection in detections:
        assert detection["class_name"] in {
            "car",
            "truck",
            "bus",
            "motorcycle",
        }

        assert 0.0 <= detection["confidence"] <= 1.0

        assert len(detection["bbox"]) == 4

    capture.release()