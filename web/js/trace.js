// The hand-height trace: how high the signer's higher hand is over the last few seconds,
// the "raised" line where a sign starts, and each detected sign as a shaded span named
// with what the model recognised. It shows how signs are cut out of continuous video.

const NS = "http://www.w3.org/2000/svg";
const PAD = { left: 70, right: 6, top: 20, bottom: 18 };
// Height above the shoulders in shoulder widths. Resting hands sit about 2.6-3.2 below.
const MIN = -3.4;
const MAX = 1.2;

function node(name, attrs, text) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}

export class HandTrace {
  constructor(svg, { seconds = 12, raisedAt = -1.8 } = {}) {
    this.svg = svg;
    this.seconds = seconds;
    this.raisedAt = raisedAt;
    this.reset();
    new ResizeObserver(() => this.render()).observe(svg);
  }

  reset() {
    this.points = []; // [time s, height or null when no body was found]
    this.spans = []; // {start, end, label, confident}
    this.now = 0;
    this.render();
  }

  /** One processed frame. t in seconds (wall clock for the camera, video time for a file). */
  push(t, height, active) {
    this.now = t;
    this.points.push([t, height]);
    const last = this.spans.at(-1);
    if (active && (!last || last.end !== null)) this.spans.push({ start: t, end: null, label: null });
    if (!active && last && last.end === null) last.end = t;
    const cutoff = t - this.seconds - 1;
    while (this.points.length && this.points[0][0] < cutoff) this.points.shift();
    while (this.spans.length && this.spans[0].end !== null && this.spans[0].end < cutoff) this.spans.shift();
    this.render();
  }

  /** Close the open span (end of a video file). */
  end() {
    const last = this.spans.at(-1);
    if (last && last.end === null) last.end = this.now;
    this.render();
  }

  /** Name the most recent finished span with the model's answer. */
  label(text, confident) {
    const span = [...this.spans].reverse().find((s) => s.end !== null && s.label === null) ?? this.spans.at(-1);
    if (!span) return;
    span.label = text;
    span.confident = confident;
    this.render();
  }

  render() {
    const svg = this.svg;
    const w = svg.clientWidth || 600;
    const h = svg.clientHeight || 120;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom;
    const t0 = this.now - this.seconds;
    const x = (t) => PAD.left + ((t - t0) / this.seconds) * plotW;
    const y = (v) => PAD.top + ((MAX - Math.max(MIN, Math.min(MAX, v))) / (MAX - MIN)) * plotH;
    const parts = [];

    for (const s of this.spans) {
      const x1 = Math.max(PAD.left, x(s.start));
      const x2 = x(s.end ?? this.now);
      if (x2 <= PAD.left) continue;
      parts.push(node("rect", { class: "span", x: x1, y: PAD.top, width: Math.max(1, x2 - x1), height: plotH }));
      parts.push(node("line", { class: "span-edge", x1, x2: x1, y1: PAD.top, y2: PAD.top + plotH }));
      if (s.label) {
        parts.push(node("text", { class: `span-label${s.confident ? "" : " unsure"}`, x: x1 + 2, y: PAD.top - 6 }, s.label));
      }
    }

    parts.push(node("line", { class: "grid-line", x1: PAD.left, x2: w - PAD.right, y1: y(0), y2: y(0) }));
    parts.push(node("line", { class: "baseline", x1: PAD.left, x2: w - PAD.right, y1: y(MIN), y2: y(MIN) }));
    parts.push(node("line", { class: "threshold", x1: PAD.left, x2: w - PAD.right, y1: y(this.raisedAt), y2: y(this.raisedAt) }));
    for (const [v, text] of [[0, "shoulders"], [this.raisedAt, "raised"], [-2.9, "at rest"]]) {
      parts.push(node("text", { x: PAD.left - 8, y: y(v) + 4, "text-anchor": "end" }, text));
    }

    // the line, broken where no body was found
    let run = [];
    const flush = () => {
      if (run.length > 1) parts.push(node("polyline", { class: "line", points: run.join(" ") }));
      run = [];
    };
    for (const [t, v] of this.points) {
      if (t < t0 - 0.5) continue;
      if (v === null) flush();
      else run.push(`${Math.max(PAD.left, x(t)).toFixed(1)},${y(v).toFixed(1)}`);
    }
    flush();

    for (let s = this.seconds; s >= 0; s -= 3) {
      const anchor = s === this.seconds ? "start" : s === 0 ? "end" : "middle";
      parts.push(node("text", { x: x(this.now - s), y: h - 3, "text-anchor": anchor }, s === 0 ? "now" : `-${s} s`));
    }
    svg.replaceChildren(...parts);
  }
}
