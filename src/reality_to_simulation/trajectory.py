from collections import defaultdict


class TrajectoryStore:
    def __init__(self):
        self.trajectories = defaultdict(list)

    def add_tracks(self, frame_id: int, tracks: list[dict]):
        for track in tracks:
            self.trajectories[track["track_id"]].append(
                {
                    "frame_id": frame_id,
                    "track_id": track["track_id"],
                    "class_name": track["class_name"],
                    "x": track["center"][0],
                    "y": track["center"][1],
                    "confidence": track["confidence"],
                }
            )

    def get_trajectory(self, track_id: int):
        return self.trajectories.get(track_id, [])

    def get_all_trajectories(self):
        return dict(self.trajectories)

    def get_track_count(self):
        return len(self.trajectories)