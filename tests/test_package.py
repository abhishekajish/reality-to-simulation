from pathlib import Path

from reality_to_simulation.detector import VehicleDetector
from reality_to_simulation.trajectory import TrajectoryStore
from reality_to_simulation.video import (
    get_video_metadata,
    open_video,
    read_first_frame,
    read_frames,
    save_frame,
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
        assert len(detection["center"]) == 2

    capture.release()


def test_trajectory_store():
    store = TrajectoryStore()

    tracks = [
        {
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": [100, 200, 140, 240],
            "center": [120, 220],
        }
    ]

    store.add_tracks(frame_id=1, tracks=tracks)
    store.add_tracks(
        frame_id=2,
        tracks=[
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.92,
                "bbox": [110, 205, 150, 245],
                "center": [130, 225],
            }
        ],
    )

    trajectory = store.get_trajectory(1)

    assert len(trajectory) == 2
    assert trajectory[0]["frame_id"] == 1
    assert trajectory[0]["x"] == 120
    assert trajectory[0]["y"] == 220
    assert trajectory[1]["frame_id"] == 2
    assert trajectory[1]["x"] == 130
    assert trajectory[1]["y"] == 225
    assert store.get_track_count() == 1