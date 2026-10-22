"""Video-level augmentation (paper Sec. III-B).

Augmentations are applied to the RGB frames *before* landmark extraction,
so each augmented copy is a new MediaPipe pass over the video:

* Gaussian pixel noise, std 15 or 30
* rotation within +/-7 degrees
* cutout: one random occluding rectangle covering 10-20% of the frame
* combinations of the above

No horizontal flips: they swap handedness and can change a sign's meaning.
Rotation angle and cutout box are fixed per video; noise is redrawn per frame.
"""

import cv2
import numpy as np

# Cycled in order, so a video with k augmented copies gets the first k recipes.
RECIPES = [
    ("noise15",),
    ("rotate",),
    ("cutout",),
    ("noise30",),
    ("rotate", "noise15"),
    ("cutout", "noise15"),
    ("rotate", "cutout"),
    ("rotate", "noise30"),
    ("cutout", "noise30"),
    ("rotate", "cutout", "noise15"),
]


class VideoAugmentation:
    """Callable frame transform with per-video random parameters."""

    def __init__(self, recipe, seed: int, max_angle: float = 7.0, cutout_area=(0.10, 0.20)):
        self.recipe = tuple(recipe)
        self.rng = np.random.default_rng(seed)
        self.angle = float(self.rng.uniform(-max_angle, max_angle))
        self.cutout_frac = float(self.rng.uniform(*cutout_area))
        self.cutout_aspect = float(self.rng.uniform(0.5, 2.0))
        self.cutout_center = self.rng.uniform(0.2, 0.8, size=2)
        self._rotation = None

    @property
    def name(self) -> str:
        return "+".join(self.recipe)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        for op in self.recipe:
            if op == "rotate":
                frame = self._rotate(frame)
            elif op == "cutout":
                frame = self._cutout(frame)
            elif op.startswith("noise"):
                frame = self._noise(frame, float(op[len("noise"):]))
            else:
                raise ValueError(f"unknown augmentation: {op}")
        return frame

    def _rotate(self, frame):
        h, w = frame.shape[:2]
        if self._rotation is None or self._rotation[0] != (h, w):
            self._rotation = ((h, w), cv2.getRotationMatrix2D((w / 2, h / 2), self.angle, 1.0))
        return cv2.warpAffine(frame, self._rotation[1], (w, h), borderMode=cv2.BORDER_REFLECT)

    def _cutout(self, frame):
        h, w = frame.shape[:2]
        area = self.cutout_frac * h * w
        ch = int(min(h, np.sqrt(area / self.cutout_aspect)))
        cw = int(min(w, area / max(ch, 1)))
        cy, cx = int(self.cutout_center[0] * h), int(self.cutout_center[1] * w)
        y0, x0 = max(0, cy - ch // 2), max(0, cx - cw // 2)
        frame = frame.copy()
        frame[y0:y0 + ch, x0:x0 + cw] = 0
        return frame

    def _noise(self, frame, std):
        # cv2.randn is several times faster than numpy for 1080p frames. Fill a
        # single-channel view: on a 3-channel array the std would only apply to channel 0.
        noise = np.empty(frame.shape, np.int16)
        cv2.setRNGSeed(int(self.rng.integers(2**31)))
        cv2.randn(noise.reshape(frame.shape[0], -1), 0, std)
        return cv2.add(frame.astype(np.int16), noise, dtype=cv2.CV_16S).clip(0, 255).astype(np.uint8)


def copies_needed(num_originals: int, target_per_class: int) -> int:
    """Augmented copies per original video so a class reaches ~target_per_class samples."""
    if num_originals == 0:
        return 0
    return max(0, int(np.ceil(target_per_class / num_originals)) - 1)
