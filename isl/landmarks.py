"""MediaPipe Tasks landmark extraction (hands + pose) for video frames.

Uses the same ``.task`` model files as the browser app, so features computed
here match what the web UI feeds the classifier.
"""

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe's C++ info/warning logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from .features import NUM_HAND, POSE_INDICES, frame_features  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

# Detection thresholds used by the original app.
MIN_HAND_DETECTION = 0.7
MIN_POSE_DETECTION = 0.7


def ensure_models(model_dir: Path = MODEL_DIR) -> dict:
    """Download the MediaPipe model files if missing; return name -> path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = model_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        paths[name] = path
    return paths


def assign_hands(hand_landmarks, handedness):
    """Split detected hands into (left, right) slots using MediaPipe's label as-is.

    This matches the original training features (verified against the feature
    overlay in the 2024 sample videos). The 2024 Streamlit app swapped the
    labels, which fed the model mirrored hand slots and hurt live accuracy.
    If two hands get the same label, the later one wins.
    """
    left = right = None
    for landmarks, labels in zip(hand_landmarks, handedness):
        coords = np.array([[p.x, p.y] for p in landmarks], np.float32)
        if coords.shape[0] != NUM_HAND:
            continue
        if labels[0].category_name == "Left":
            left = coords
        else:
            right = coords
    return left, right


class LandmarkExtractor:
    """Streaming extractor: call :meth:`features` once per frame, in order."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        paths = ensure_models(model_dir)
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        mode = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(paths["hand_landmarker.task"])),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=MIN_HAND_DETECTION))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(paths["pose_landmarker_full.task"])),
            running_mode=mode, min_pose_detection_confidence=MIN_POSE_DETECTION))
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Return raw (pose, left_hand, right_hand) arrays (None when not detected)."""
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # must strictly increase
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        hand_result = self.hands.detect_for_video(image, timestamp_ms)
        pose_result = self.pose.detect_for_video(image, timestamp_ms)

        pose = None
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            pose = np.array([[lm[i].x, lm[i].y] for i in POSE_INDICES], np.float32)
        left, right = assign_hands(hand_result.hand_landmarks, hand_result.handedness)
        return pose, left, right

    def features(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        return frame_features(*self.detect(frame_bgr, timestamp_ms))

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_video_frames(path, fps: float | None = None, transform=None):
    """Yield (timestamp_ms, frame_bgr) from a video file.

    fps: if given, frames are subsampled to roughly this rate.
    transform: optional callable applied to each frame (used for augmentation).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1.0 if not fps or fps >= src_fps else src_fps / fps
    next_keep, idx = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= next_keep:
                next_keep += step
                if transform is not None:
                    frame = transform(frame)
                yield int(idx * 1000 / src_fps), frame
            idx += 1
    finally:
        cap.release()


def extract_video_features(path, extractor: LandmarkExtractor | None = None,
                           fps: float | None = None, transform=None) -> np.ndarray:
    """Return a (T, 108) feature array for every (kept) frame of a video."""
    own = extractor is None
    extractor = extractor or LandmarkExtractor()
    try:
        feats = [extractor.features(frame, ts) for ts, frame in iter_video_frames(path, fps, transform)]
    finally:
        if own:
            extractor.close()
    return np.stack(feats) if feats else np.zeros((0, 108), np.float32)
