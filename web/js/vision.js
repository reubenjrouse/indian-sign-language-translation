// MediaPipe Tasks (hands + pose) in the browser. Uses the same .task models as
// isl/landmarks.py so features match the Python pipeline.
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

// Thresholds used by the original app.
const MIN_HAND_DETECTION = 0.7;
const MIN_POSE_DETECTION = 0.7;

async function create(delegate) {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const [hands, pose] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: MIN_HAND_DETECTION,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate },
      runningMode: "VIDEO",
      minPoseDetectionConfidence: MIN_POSE_DETECTION,
    }),
  ]);
  return { hands, pose };
}

export class Tracker {
  static async load() {
    let models;
    let delegate = "GPU";
    try {
      models = await create("GPU");
    } catch (err) {
      console.warn("GPU delegate unavailable, using CPU", err);
      delegate = "CPU";
      models = await create("CPU");
    }
    return new Tracker(models, delegate);
  }

  constructor({ hands, pose }, delegate) {
    this.hands = hands;
    this.pose = pose;
    this.delegate = delegate;
    this.lastTs = 0;
  }

  /** Returns {pose: landmarks|null, hands: [{landmarks, label}]} for one video frame. */
  detect(video) {
    const ts = Math.max(performance.now(), this.lastTs + 1); // must strictly increase
    this.lastTs = ts;
    const handResult = this.hands.detectForVideo(video, ts);
    const poseResult = this.pose.detectForVideo(video, ts);
    return {
      pose: poseResult.landmarks?.[0] ?? null,
      hands: handResult.landmarks.map((landmarks, i) => ({
        landmarks,
        label: handResult.handedness[i]?.[0]?.categoryName ?? "Right",
      })),
    };
  }
}

const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS.map(({ start, end }) => [start, end]);
const UPPER_BODY = PoseLandmarker.POSE_CONNECTIONS.map(({ start, end }) => [start, end]).filter(
  ([a, b]) => a >= 11 && a <= 22 && b >= 11 && b <= 22,
);

const UPPER_BODY_POINTS = Array.from({ length: 12 }, (_, i) => i + 11);
const HAND_POINTS = Array.from({ length: 21 }, (_, i) => i);

function strokeSkeleton(ctx, points, connections, dots, color, w, h, radius) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  for (const [a, b] of connections) {
    ctx.moveTo(points[a].x * w, points[a].y * h);
    ctx.lineTo(points[b].x * w, points[b].y * h);
  }
  ctx.stroke();
  for (const i of dots) {
    const p = points[i];
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Draw the tracked skeleton; colours come from CSS custom properties. */
export function drawLandmarks(canvas, detection, colors) {
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, w / 320);
  if (detection.pose) {
    strokeSkeleton(ctx, detection.pose, UPPER_BODY, UPPER_BODY_POINTS, colors.pose, w, h, Math.max(2, w / 300));
  }
  for (const hand of detection.hands) {
    const color = hand.label === "Left" ? colors.left : colors.right;
    strokeSkeleton(ctx, hand.landmarks, HAND_CONNECTIONS, HAND_POINTS, color, w, h, Math.max(2, w / 360));
  }
}
