import random


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
SEED = 42


def split_track_ids(track_ids):
    track_ids = list(track_ids)

    random.seed(SEED)
    random.shuffle(track_ids)

    total_tracks = len(track_ids)

    train_end = int(
        total_tracks * TRAIN_RATIO
    )

    validation_end = int(
        total_tracks
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    return (
        set(track_ids[:train_end]),
        set(
            track_ids[
                train_end:validation_end
            ]
        ),
        set(
            track_ids[
                validation_end:
            ]
        ),
    )