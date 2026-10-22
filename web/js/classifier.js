// BiLSTM classifier exported to ONNX (scripts/export_onnx.py), run with onnxruntime-web.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import { FEATURE_DIM, resampleSequence } from "./features.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // the model is small; threads need cross-origin isolation

export class Classifier {
  static async load(base = "model/isl60") {
    const meta = await (await fetch(`${base}.json`)).json();
    const session = await ort.InferenceSession.create(`${base}.onnx`, { executionProviders: ["wasm"] });
    const classifier = new Classifier(session, meta);
    await classifier.predict([new Float32Array(FEATURE_DIM)]); // warm up so the first real sign isn't slow
    return classifier;
  }

  constructor(session, meta) {
    this.session = session;
    this.labels = meta.labels;
    this.categories = meta.categories ?? meta.labels.map(() => "Signs");
    this.test = meta.test ?? null; // held-out accuracy, when exported with --metrics
    this.seqLen = meta.seq_len;
  }

  /** frames: Float32Array[] of any length -> [{label, prob}] sorted, plus inference ms. */
  async predict(frames) {
    const input = new ort.Tensor("float32", resampleSequence(frames, this.seqLen), [1, this.seqLen, FEATURE_DIM]);
    const start = performance.now();
    const { probs } = await this.session.run({ frames: input });
    const ms = performance.now() - start;
    const ranked = Array.from(probs.data, (prob, i) => ({ label: this.labels[i], prob })).sort(
      (a, b) => b.prob - a.prob,
    );
    return { ranked, ms };
  }
}
