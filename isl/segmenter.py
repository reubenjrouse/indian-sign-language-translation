"""Motion-based segmentation of a continuous feature stream into single signs.

Reference implementation of the paper's real-time algorithm. The browser app
uses a line-for-line port (``web/js/segmenter.js``); ``tests/test_js_parity.py``
checks that both produce the same segments.

A sign starts when the mean absolute change between consecutive feature
vectors exceeds ``start_threshold``. It ends after ``static_frames``
consecutive low-motion frames (or at ``max_frames``). The emitted segment
includes those trailing still frames, as in the original app, but is only
emitted if at least ``min_frames`` of its frames had motion. (The original
counted the still frames too, so any twitch became a "sign".) Thresholds are
per *processed* frame, so the processing rate matters (``fps``).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmenterConfig:
    fps: float = 15.0
    start_threshold: float = 0.05
    static_threshold: float = 0.05
    static_frames: int = 20
    min_frames: int = 8
    max_frames: int = 200


class MotionSegmenter:
    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()
        self.reset()

    def reset(self):
        self.prev = None
        self.frames = []
        self.static_count = 0
        self.active = False
        self.motion = 0.0

    def flush(self):
        """End of stream: return the sign in progress (if long enough), else None."""
        segment, moving = self.frames, len(self.frames) - self.static_count
        active = self.active
        self.reset()
        if active and moving >= self.config.min_frames:
            return np.stack(segment)
        return None

    def push(self, features: np.ndarray):
        """Feed one frame. Returns a (T, D) array when a sign completes, else None."""
        cfg = self.config
        features = np.asarray(features, np.float32)
        prev, self.prev = self.prev, features
        if prev is None:
            return None

        self.motion = float(np.mean(np.abs(features - prev)))
        if self.motion > cfg.start_threshold and not self.active:
            self.active = True
            self.frames = []
            self.static_count = 0
        if not self.active:
            return None

        self.frames.append(features)
        self.static_count = self.static_count + 1 if self.motion < cfg.static_threshold else 0
        if self.static_count >= cfg.static_frames or len(self.frames) >= cfg.max_frames:
            segment, moving = self.frames, len(self.frames) - self.static_count
            self.active = False
            self.frames = []
            self.static_count = 0
            if moving >= cfg.min_frames:
                return np.stack(segment)
        return None


# ---------------------------------------------------------------------------
# Hand-position segmenter (2026 rework)
# ---------------------------------------------------------------------------

@dataclass
class HandSegmenterConfig:
    """Times are in seconds and positions in shoulder widths, so nothing depends on fps.

    raise_on / raise_off: how far below the shoulders the higher wrist may be for a
    hand to count as raised (hysteresis: start below raise_on, end above raise_off).
    On INCLUDE's validation clips the resting wrist sits 2.6-3.2 shoulder widths below
    the shoulders and the signing wrist -1.1 to 0.6, so these have a wide margin.
    """
    fps: float = 15.0
    raise_on: float = 1.8
    raise_off: float = 2.2
    preroll: float = 0.4       # frames kept from before the hand rose (clips start at rest)
    postroll: float = 0.3      # frames kept after it dropped (clips end at rest)
    min_sign: float = 0.4      # shortest raised stretch that counts as a sign
    max_sign: float = 6.0
    still_speed: float = 0.35  # shoulder widths per second: below this the hands are holding still
    pause: float = 0.8         # a hold this long with the hands up ends a sign; long enough not to split
                               # compound signs like GOOD-MORNING (0.5-1.0 s score the same on val sentences)
    move_speed: float = 0.7    # after such a hold, moving faster than this starts the next sign


def wrist_signals(pose):
    """(height of the higher wrist below the shoulders, wrist positions, shoulder width) in shoulder widths.

    pose: (12, 2) raw landmarks for MediaPipe pose points 11-22, or None / NaN when not detected.
    """
    if pose is None or np.isnan(pose[:2]).any():
        return None, None, None
    shoulders = pose[0], pose[1]
    width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    if width < 1e-6:
        return None, None, None
    sy = (shoulders[0][1] + shoulders[1][1]) / 2
    wrists = pose[4:6]
    heights = (wrists[:, 1] - sy) / width
    return float(np.nanmin(heights)), wrists / width, width


class HandSegmenter:
    """Cut signs from a stream by where the hands are, not by how much the features change.

    A sign starts when a hand rises to the chest and ends when the hands drop back to
    rest, or when they hold still for `pause` seconds (so signs can follow each other
    without lowering the hands). Segments include a little rest before and after,
    like the isolated training clips. Same interface as MotionSegmenter, plus `pose`.
    """

    def __init__(self, config: HandSegmenterConfig | None = None):
        self.config = config or HandSegmenterConfig()
        c = self.config
        # round half up, like JavaScript's Math.round (Python's round() would give 4 for 4.5)
        self.n = {k: max(1, int(np.floor(getattr(c, k) * c.fps + 0.5)))
                  for k in ("preroll", "postroll", "min_sign", "max_sign", "pause")}
        self.reset()

    def reset(self):
        from collections import deque
        self.state = "idle"
        self.buffer = deque(maxlen=self.n["preroll"])
        self.frames, self.core, self.still, self.post_left = [], 0, 0, 0
        self.prev_wrists = None
        self.motion = 0.0   # wrist speed, shoulder widths per second (shown in the trace)
        self.height = None  # higher wrist below the shoulders, shoulder widths

    @property
    def active(self):
        return self.state in ("active", "post")

    def _start(self, features):
        self.frames = [*self.buffer, features]
        self.core, self.still = 1, 0
        self.state = "active"

    def _emit(self, next_state):
        segment, core = self.frames, self.core
        self.state = next_state
        self.frames, self.core, self.still = [], 0, 0
        return np.stack(segment) if core >= self.n["min_sign"] else None

    def push(self, features, pose=None):
        c = self.config
        features = np.asarray(features, np.float32)
        height, wrists, _ = wrist_signals(pose)
        if wrists is not None and self.prev_wrists is not None:
            step = np.nan_to_num(np.linalg.norm(wrists - self.prev_wrists, axis=1), nan=0.0)
            self.motion = float(step.max() * c.fps)
        if wrists is not None:
            self.prev_wrists = wrists
            self.height = height
        height = self.height  # keep the last known position through a missed frame
        raised_on = height is not None and height < c.raise_on
        lowered = height is not None and height > c.raise_off
        out = None

        if self.state == "idle":
            if raised_on:
                self._start(features)
            else:
                self.buffer.append(features)
        elif self.state == "held":
            self.buffer.append(features)
            if lowered:
                self.state = "idle"
            elif self.motion > c.move_speed:
                self._start(features)
        elif self.state == "active":
            self.frames.append(features)
            self.core += 1
            if lowered:
                self.state, self.post_left = "post", self.n["postroll"]
            elif self.motion < c.still_speed:
                self.still += 1
                if self.still >= self.n["pause"]:
                    self.frames = self.frames[: len(self.frames) - self.still + self.n["postroll"]]
                    self.core -= self.still
                    out = self._emit("held")
                    self.buffer.clear()
            else:
                self.still = 0
            if self.state == "active" and self.core >= self.n["max_sign"]:
                out = self._emit("held")
        elif self.state == "post":
            if raised_on:  # rose again before the post-roll finished: a new sign
                out = self._emit("idle")
                self._start(features)
            else:
                self.frames.append(features)
                self.post_left -= 1
                if self.post_left <= 0:
                    out = self._emit("idle")
                    self.buffer.clear()
        return out

    def flush(self):
        out = self._emit("idle") if self.active else None
        self.reset()
        return out
