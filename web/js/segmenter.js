// Motion-based sign segmentation. Port of isl/segmenter.py (kept in sync by
// tests/test_js_parity.py). Thresholds are per processed frame, so frames must
// be pushed at a steady rate (config.fps).

export const DEFAULT_SEGMENTER = {
  fps: 15,
  startThreshold: 0.05,
  staticThreshold: 0.05,
  staticFrames: 20,
  minFrames: 8, // frames with motion; trailing still frames don't count
  maxFrames: 200,
};

export class MotionSegmenter {
  constructor(config = {}) {
    this.config = { ...DEFAULT_SEGMENTER, ...config };
    this.reset();
  }

  reset() {
    this.prev = null;
    this.frames = [];
    this.staticCount = 0;
    this.active = false;
    this.motion = 0;
  }

  /** End of stream: return the sign in progress (if long enough), else null. */
  flush() {
    const segment = this.frames;
    const moving = segment.length - this.staticCount;
    const active = this.active;
    this.reset();
    return active && moving >= this.config.minFrames ? segment : null;
  }

  /** Feed one frame (Float32Array). Returns the frames of a completed sign, or null. */
  push(features) {
    const cfg = this.config;
    const prev = this.prev;
    this.prev = features;
    if (prev === null) return null;

    let sum = 0;
    for (let i = 0; i < features.length; i++) sum += Math.abs(features[i] - prev[i]);
    this.motion = sum / features.length;

    if (this.motion > cfg.startThreshold && !this.active) {
      this.active = true;
      this.frames = [];
      this.staticCount = 0;
    }
    if (!this.active) return null;

    this.frames.push(features);
    this.staticCount = this.motion < cfg.staticThreshold ? this.staticCount + 1 : 0;
    if (this.staticCount >= cfg.staticFrames || this.frames.length >= cfg.maxFrames) {
      const segment = this.frames;
      const moving = segment.length - this.staticCount;
      this.active = false;
      this.frames = [];
      this.staticCount = 0;
      if (moving >= cfg.minFrames) return segment;
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Hand-position segmenter: port of isl/segmenter.py HandSegmenter (kept in sync by
// tests/test_js_parity.py). Times in seconds, positions in shoulder widths.
// ---------------------------------------------------------------------------

export const DEFAULT_HAND_SEGMENTER = {
  fps: 15,
  raiseOn: 1.8,
  raiseOff: 2.2,
  preroll: 0.4,
  postroll: 0.3,
  minSign: 0.4,
  maxSign: 6.0,
  stillSpeed: 0.35,
  pause: 0.8,
  moveSpeed: 0.7,
};

/** pose12: 12 [x, y] points (MediaPipe pose 11-22) or null. Returns {height, wrists} or null. */
export function wristSignals(pose12) {
  if (!pose12) return null;
  const [l, r] = pose12;
  const width = Math.hypot(l[0] - r[0], l[1] - r[1]);
  if (!(width > 1e-6)) return null;
  const sy = (l[1] + r[1]) / 2;
  const wrists = [pose12[4], pose12[5]].map(([x, y]) => [x / width, y / width]);
  const height = Math.min((pose12[4][1] - sy) / width, (pose12[5][1] - sy) / width);
  return { height, wrists };
}

export class HandSegmenter {
  constructor(config = {}) {
    this.config = { ...DEFAULT_HAND_SEGMENTER, ...config };
    const c = this.config;
    const n = (s) => Math.max(1, Math.round(s * c.fps));
    this.n = { preroll: n(c.preroll), postroll: n(c.postroll), minSign: n(c.minSign), maxSign: n(c.maxSign), pause: n(c.pause) };
    this.reset();
  }

  reset() {
    this.state = "idle";
    this.buffer = [];
    this.frames = [];
    this.core = 0;
    this.still = 0;
    this.postLeft = 0;
    this.prevWrists = null;
    this.motion = 0; // wrist speed, shoulder widths per second
    this.height = null; // higher wrist below the shoulders, shoulder widths
  }

  get active() {
    return this.state === "active" || this.state === "post";
  }

  _buffer(features) {
    this.buffer.push(features);
    if (this.buffer.length > this.n.preroll) this.buffer.shift();
  }

  _start(features) {
    this.frames = [...this.buffer, features];
    this.core = 1;
    this.still = 0;
    this.state = "active";
  }

  _emit(next) {
    const segment = this.frames;
    const core = this.core;
    this.state = next;
    this.frames = [];
    this.core = 0;
    this.still = 0;
    return core >= this.n.minSign ? segment : null;
  }

  /** Feed one frame: features (Float32Array) and pose12 (or null). Returns a finished sign's frames, or null. */
  push(features, pose12 = null) {
    const c = this.config;
    const sig = wristSignals(pose12);
    if (sig && this.prevWrists) {
      const step = sig.wrists.map((w, i) => Math.hypot(w[0] - this.prevWrists[i][0], w[1] - this.prevWrists[i][1]));
      this.motion = Math.max(...step.map((s) => (Number.isNaN(s) ? 0 : s))) * c.fps;
    }
    if (sig) {
      this.prevWrists = sig.wrists;
      this.height = sig.height;
    }
    const h = this.height;
    const raisedOn = h !== null && h < c.raiseOn;
    const lowered = h !== null && h > c.raiseOff;
    let out = null;

    if (this.state === "idle") {
      if (raisedOn) this._start(features);
      else this._buffer(features);
    } else if (this.state === "held") {
      this._buffer(features);
      if (lowered) this.state = "idle";
      else if (this.motion > c.moveSpeed) this._start(features);
    } else if (this.state === "active") {
      this.frames.push(features);
      this.core += 1;
      if (lowered) {
        this.state = "post";
        this.postLeft = this.n.postroll;
      } else if (this.motion < c.stillSpeed) {
        this.still += 1;
        if (this.still >= this.n.pause) {
          this.frames = this.frames.slice(0, this.frames.length - this.still + this.n.postroll);
          this.core -= this.still;
          out = this._emit("held");
          this.buffer = [];
        }
      } else {
        this.still = 0;
      }
      if (this.state === "active" && this.core >= this.n.maxSign) out = this._emit("held");
    } else if (this.state === "post") {
      if (raisedOn) {
        out = this._emit("idle");
        this._start(features);
      } else {
        this.frames.push(features);
        this.postLeft -= 1;
        if (this.postLeft <= 0) {
          out = this._emit("idle");
          this.buffer = [];
        }
      }
    }
    return out;
  }

  flush() {
    const out = this.active ? this._emit("idle") : null;
    this.reset();
    return out;
  }
}
