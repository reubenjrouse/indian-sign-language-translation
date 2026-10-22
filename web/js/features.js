// Per-frame feature vector and sequence resampling.
// Port of isl/features.py + isl/landmarks.assign_hands; tests/test_js_parity.py
// checks both produce identical output. No browser APIs, so Node can import it.

export const POSE_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
export const NUM_HAND = 21;
export const FEATURE_DIM = (POSE_INDICES.length + 2 * NUM_HAND) * 2; // 108
export const SEQ_LEN = 65;

/** Min-max scale flat [x0, y0, x1, y1, ...] to [-1, 1] per axis, writing into out at offset. */
function normalizePartInto(points, out, offset) {
  const n = points.length / 2;
  if (!points.some((v) => v !== 0)) return; // undetected part stays zero
  for (let axis = 0; axis < 2; axis++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = points[2 * i + axis];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    for (let i = 0; i < n; i++) {
      const v = points[2 * i + axis];
      out[offset + 2 * i + axis] = hi > lo ? (2 * (v - lo)) / (hi - lo) - 1 : v;
    }
  }
}

const flatten = (landmarks) => landmarks.flatMap((p) => [p.x, p.y]);

/**
 * Split hands into slots using MediaPipe's handedness label as-is
 * (matches the training data; see isl/landmarks.py).
 * hands: [{landmarks: [{x, y}, ...21], label: "Left" | "Right"}]
 */
export function assignHands(hands) {
  let left = null;
  let right = null;
  for (const { landmarks, label } of hands) {
    if (landmarks.length !== NUM_HAND) continue;
    if (label === "Left") left = landmarks;
    else right = landmarks;
  }
  return { left, right };
}

/**
 * poseLandmarks: full 33-point pose array or null.
 * hands: see assignHands.
 * Returns Float32Array(108): [pose 11-22 | left hand | right hand].
 */
export function frameFeatures(poseLandmarks, hands) {
  const out = new Float32Array(FEATURE_DIM);
  const { left, right } = assignHands(hands);
  if (poseLandmarks) {
    normalizePartInto(Float32Array.from(flatten(POSE_INDICES.map((i) => poseLandmarks[i]))), out, 0);
  }
  if (left) normalizePartInto(Float32Array.from(flatten(left)), out, POSE_INDICES.length * 2);
  if (right) normalizePartInto(Float32Array.from(flatten(right)), out, (POSE_INDICES.length + NUM_HAND) * 2);
  return out;
}

/** Resample an array of Float32Array frames to `length` frames; returns one flat Float32Array. */
export function resampleSequence(frames, length = SEQ_LEN) {
  if (frames.length === 0) throw new Error("cannot resample an empty sequence");
  const dim = frames[0].length;
  const out = new Float32Array(length * dim);
  const n = frames.length;
  // Same arithmetic as numpy.linspace(0, n - 1, length): k * step, endpoint exact.
  const step = length > 1 ? (n - 1) / (length - 1) : 0;
  for (let k = 0; k < length; k++) {
    const pos = k === length - 1 ? n - 1 : k * step;
    if (n >= length) {
      out.set(frames[Math.trunc(pos)], k * dim);
      continue;
    }
    const i = Math.trunc(pos);
    if (i >= n - 1) {
      out.set(frames[n - 1], k * dim);
      continue;
    }
    const alpha = pos - i;
    const a = frames[i];
    const b = frames[i + 1];
    for (let d = 0; d < dim; d++) out[k * dim + d] = a[d] * (1 - alpha) + b[d] * alpha;
  }
  return out;
}
