"""Per-frame feature vector and sequence resampling.

This module is the single definition of the model's input. Training
(``scripts/preprocess.py``), offline inference (``scripts/infer_video.py``) and
the browser app (``web/js/features.js``, kept in sync by
``tests/test_js_parity.py``) all produce features exactly this way.

Feature layout per frame (108 values):
    [ pose 11..22 (12 x 2) | left hand (21 x 2) | right hand (21 x 2) ]
Each part is min-max normalised to [-1, 1] on x and y independently.
A part that was not detected stays all-zero.
"""

import numpy as np

POSE_INDICES = list(range(11, 23))  # shoulders, elbows, wrists, hand points
NUM_POSE = len(POSE_INDICES)
NUM_HAND = 21
FEATURE_DIM = (NUM_POSE + 2 * NUM_HAND) * 2  # 108
SEQ_LEN = 65


def normalize_part(points: np.ndarray) -> np.ndarray:
    """Scale an (N, 2) landmark array to [-1, 1] per axis; all-zero input is returned unchanged."""
    points = np.asarray(points, dtype=np.float32)
    if not points.any():
        return points.copy()
    out = points.copy()
    for axis in (0, 1):
        lo, hi = out[:, axis].min(), out[:, axis].max()
        if hi > lo:
            out[:, axis] = 2 * (out[:, axis] - lo) / (hi - lo) - 1
    return out


def frame_features(pose=None, left_hand=None, right_hand=None) -> np.ndarray:
    """Build the 108-d feature vector from raw (x, y) landmarks.

    pose: (12, 2) upper-body points (MediaPipe pose indices 11-22) or None.
    left_hand / right_hand: (21, 2) hand points or None.
    """
    pose = np.zeros((NUM_POSE, 2), np.float32) if pose is None else pose
    left_hand = np.zeros((NUM_HAND, 2), np.float32) if left_hand is None else left_hand
    right_hand = np.zeros((NUM_HAND, 2), np.float32) if right_hand is None else right_hand
    return np.concatenate([
        normalize_part(pose).ravel(),
        normalize_part(left_hand).ravel(),
        normalize_part(right_hand).ravel(),
    ]).astype(np.float32)


def resample_sequence(frames, length: int = SEQ_LEN) -> np.ndarray:
    """Resample a (T, D) sequence to (length, D).

    Longer sequences are sampled uniformly; shorter ones are linearly
    interpolated between neighbouring frames.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        raise ValueError("cannot resample an empty sequence")
    if len(frames) == length:
        return frames
    if len(frames) > length:
        return frames[np.linspace(0, len(frames) - 1, length).astype(int)]

    positions = np.linspace(0, len(frames) - 1, length)
    out = np.empty((length, frames.shape[1]), np.float32)
    for k, pos in enumerate(positions):
        i = int(pos)
        if i >= len(frames) - 1:
            out[k] = frames[-1]
            continue
        alpha = pos - i
        out[k] = frames[i] * (1 - alpha) + frames[i + 1] * alpha
    return out
