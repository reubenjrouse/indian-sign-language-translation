import { alignSentence } from "./align.js";
import { Classifier } from "./classifier.js";
import { POSE_INDICES, frameFeatures } from "./features.js";
import { DEFAULT_HAND_SEGMENTER, HandSegmenter } from "./segmenter.js";
import { HandTrace } from "./trace.js";
import { Tracker, drawLandmarks } from "./vision.js";

const SAMPLE_URL = "samples/good-morning-we-alright.webm";
const ALT_MIN = 0.05; // show a runner-up guess on a tile from this probability

const $ = (id) => document.getElementById(id);
const el = {
  stage: $("stage"),
  backdrop: $("backdrop"),
  viewport: $("viewport"),
  video: $("video"),
  overlay: $("overlay"),
  emptyTitle: $("empty-title"),
  emptyBody: $("empty-body"),
  status: $("status"),
  tiles: $("tiles"),
  links: $("links"),
  sentence: $("sentence"),
  llmLabel: $("llm-label"),
  countdown: $("countdown"),
  translate: $("btn-translate"),
  clear: $("btn-clear"),
  history: $("history"),
  signGroups: $("sign-groups"),
  signCount: $("sign-count"),
  factTracking: $("fact-tracking"),
  factModel: $("fact-model"),
  factTest: $("fact-test"),
  camera: $("btn-camera"),
  cameraSelect: $("camera-select"),
  sample: $("btn-sample"),
  fileLabel: $("btn-file-label"),
  fileInput: $("file-input"),
  settingsBtn: $("btn-settings"),
  settings: $("settings"),
};

// ---------- settings ----------
const DEFAULT_SETTINGS = {
  fps: DEFAULT_HAND_SEGMENTER.fps,
  holdSeconds: DEFAULT_HAND_SEGMENTER.pause,
  minConfidence: 0.5,
  pauseSeconds: 5,
};
const SETTINGS_KEY = "isl-settings-v2";

function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable: settings last for this visit only */
  }
}

let settings = loadSettings();

// ---------- state ----------
let tracker = null;
let classifier = null;
let segmenter = makeSegmenter();
const trace = new HandTrace($("trace"), { raisedAt: -DEFAULT_HAND_SEGMENTER.raiseOn });
let stream = null;
let mode = "sentence";
let lastProcess = 0;
let processedTimes = [];
let lastInferenceMs = null;
let lastFactUpdate = 0;
let llm = { ready: false, model: "" };
let fileRun = 0; // bumped to cancel an in-progress video file
let analysing = false;

/**
 * The band's contents. Sentence mode: the signs collected so far ({label, ranked}),
 * then the translation once it arrives. Word mode: the last sign's ranked guesses.
 */
let band = { signs: [], translation: null, translating: false, lastSignAt: 0, word: null };

function makeSegmenter() {
  return new HandSegmenter({ fps: settings.fps, pause: settings.holdSeconds });
}

const gloss = (label) => label.trim().toUpperCase().split(/\s+/).join("-");
const pct = (p) => `${Math.round(p * 100)}%`;
const clock = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ---------- startup ----------
async function init() {
  renderBand();
  try {
    [tracker, classifier] = await Promise.all([Tracker.load(), Classifier.load()]);
  } catch (err) {
    console.error(err);
    el.viewport.querySelector(".empty").classList.add("empty-error");
    showEmpty("The models didn't load.",
      "Hand tracking loads from a CDN, so check the internet connection, then reload. Details are in the browser console.");
    el.factModel.textContent = "Didn't load";
    return;
  }
  showEmpty("Turn on the camera, open a video, or try the sample.",
    "Sign one word, pause briefly, then sign the next. Everything runs in this tab and your video stays on this device.");
  for (const c of [el.camera, el.sample, el.fileInput]) c.disabled = false;
  el.fileLabel.removeAttribute("aria-disabled");
  el.factModel.textContent = `${classifier.labels.length} signs, BiLSTM with attention`;
  if (classifier.test) {
    el.factTest.textContent = `${(classifier.test.accuracy * 100).toFixed(1)}% on ${classifier.test.samples} videos`;
  }
  renderSignList();
  listCameras(); // shows names right away if camera permission was granted before
  checkLLM();
  requestAnimationFrame(tick);
}

async function checkLLM() {
  try {
    const health = await (await fetch("api/health")).json();
    llm = { ready: Boolean(health.llm), model: health.model ?? "" };
  } catch {
    llm = { ready: false, model: "" };
  }
  el.llmLabel.textContent = llm.ready ? `English by ${llm.model}` : "No LLM key set, so the signs are shown as written";
  el.llmLabel.dataset.tone = llm.ready ? "" : "off";
}

function renderSignList() {
  el.signCount.textContent = String(classifier.labels.length);
  const groups = new Map();
  classifier.labels.forEach((label, i) => {
    const cat = classifier.categories[i];
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat).push(label);
  });
  el.signGroups.replaceChildren(
    ...[...groups].map(([cat, labels]) => {
      const div = document.createElement("div");
      div.className = "sign-group";
      const h3 = document.createElement("h3");
      h3.textContent = cat;
      const ul = document.createElement("ul");
      ul.append(...labels.map((label) => {
        const li = document.createElement("li");
        li.textContent = label;
        li.dataset.label = label;
        return li;
      }));
      div.append(h3, ul);
      return div;
    }),
  );
}

// ---------- sources ----------
const CAMERA_KEY = "isl-camera";

function savedCamera() {
  try {
    return localStorage.getItem(CAMERA_KEY) || "";
  } catch {
    return "";
  }
}

/** Fill the camera picker. Device names are only available after camera permission is granted. */
async function listCameras() {
  let cameras = [];
  try {
    cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  } catch {
    /* enumerateDevices unavailable: keep the picker as it is */
  }
  const current = stream?.getVideoTracks()[0]?.getSettings().deviceId || savedCamera();
  if (!cameras.length || !cameras[0].deviceId) {
    // Some browsers briefly report no cameras while one is in use; keep the list we have.
    for (const option of el.cameraSelect.options) option.selected = option.value === current;
    return;
  }
  el.cameraSelect.replaceChildren(
    ...cameras.map((cam, i) => {
      const option = document.createElement("option");
      option.value = cam.deviceId;
      option.textContent = cam.label || `Camera ${i + 1}`;
      option.selected = cam.deviceId === current;
      return option;
    }),
  );
  el.cameraSelect.disabled = false;
}

const CAMERA_ERRORS = {
  NotAllowedError: ["Camera access was blocked.",
    "Allow camera access for this page in the browser's site settings, then press Turn on camera again."],
  NotReadableError: ["That camera is busy.",
    "Another app (Zoom, Teams, the Camera app) may be using it. Close that app, or choose another camera."],
  NotFoundError: ["No camera was found.", "Connect a camera, or try the sample instead."],
};
const CAMERA_UNAVAILABLE = ["That camera isn't available.",
  "It may have been unplugged or switched off. Choose another camera."];

/**
 * deviceId: a specific camera, or "" for the browser's default.
 * chosen: true when the person just picked this camera (show an error if it fails);
 * false for a remembered camera (quietly fall back to the default if it's gone).
 */
async function startCamera(deviceId = savedCamera(), chosen = false) {
  stopSource();
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = "user";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  } catch (err) {
    stream = null;
    const missing = err.name === "OverconstrainedError" || err.name === "NotFoundError";
    if (deviceId && missing && !chosen) {
      try {
        localStorage.removeItem(CAMERA_KEY);
      } catch {
        /* ignore */
      }
      return startCamera("");
    }
    const [title, body] = deviceId && missing
      ? CAMERA_UNAVAILABLE
      : CAMERA_ERRORS[err.name] ?? ["The camera couldn't start.", `${err.name}: ${err.message}`];
    showEmpty(title, body);
    listCameras();
    return;
  }
  el.video.srcObject = stream;
  el.video.loop = false;
  el.stage.dataset.mirror = "true"; // display only; features use the unmirrored frame
  await el.video.play();
  el.viewport.dataset.state = "live";
  el.stage.dataset.live = "true";
  el.camera.textContent = "Turn off camera";
  el.camera.setAttribute("aria-pressed", "true");
  listCameras();
}

el.cameraSelect.addEventListener("change", () => {
  const id = el.cameraSelect.value;
  try {
    localStorage.setItem(CAMERA_KEY, id);
  } catch {
    /* remembered for this visit only */
  }
  startCamera(id, true);
});
navigator.mediaDevices?.addEventListener?.("devicechange", () => listCameras());

const once = (target, event) =>
  new Promise((resolve, reject) => {
    target.addEventListener(event, resolve, { once: true });
    target.addEventListener("error", reject, { once: true });
  });

/**
 * Step through a video (a File or a URL) at a fixed rate (settings.fps) instead of
 * playing it, so results don't depend on device speed and match scripts/infer_video.py.
 */
async function openVideo(source) {
  stopSource();
  const run = fileRun;
  const v = el.video;
  v.src = typeof source === "string" ? source : URL.createObjectURL(source);
  el.stage.dataset.mirror = "false";
  try {
    await once(v, "loadeddata");
  } catch {
    return; // reported by the "error" listener
  }
  el.viewport.dataset.state = "live";
  el.stage.dataset.live = "true";
  analysing = true;
  const step = 1 / settings.fps;
  for (let t = 0; t <= v.duration; t += step) {
    v.currentTime = t;
    await once(v, "seeked");
    if (run !== fileRun) return; // another source was opened
    await processFrame(performance.now(), t);
    setStatus(`Reading the video, ${Math.min(100, Math.round((t / v.duration) * 100))}%`);
  }
  analysing = false;
  const segment = segmenter.flush(); // the clip may end mid-sign
  trace.end();
  if (segment) await classify(segment);
  setStatus("Finished. Try it with the camera next.");
  if (mode === "sentence" && band.signs.length && !band.translation) translateSentence();
}

el.video.addEventListener("error", () => {
  if (!el.video.getAttribute("src")) return; // cleared on purpose
  stopSource();
  showEmpty("This video can't be played in the browser.",
    "Its format isn't supported. Try an MP4 (H.264) or WebM file, for example one recorded on a phone.");
});

function stopSource() {
  fileRun++; // cancels a video being read
  analysing = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  if (el.video.src.startsWith("blob:")) URL.revokeObjectURL(el.video.src);
  el.video.removeAttribute("src");
  el.video.srcObject = null;
  el.camera.textContent = "Turn on camera";
  el.camera.setAttribute("aria-pressed", "false");
  segmenter.reset();
  trace.reset();
  setStatus("");
  el.stage.dataset.live = "false";
  el.factTracking.textContent = "Not running";
  el.overlay.getContext("2d").clearRect(0, 0, el.overlay.width, el.overlay.height);
}

function showEmpty(title, body) {
  el.viewport.dataset.state = "empty";
  el.emptyTitle.textContent = title;
  el.emptyBody.textContent = body;
}

// ---------- main loop ----------
function tick(now) {
  requestAnimationFrame(tick);
  const v = el.video;
  const playing = v.readyState >= 2 && !v.paused && !v.ended;
  if (playing && now - lastProcess >= 1000 / settings.fps - 2) {
    lastProcess = now;
    processFrame(now);
  }

  const waiting = mode === "sentence" && band.signs.length && !band.translation && !band.translating && !segmenter.active;
  const left = settings.pauseSeconds - (now - band.lastSignAt) / 1000;
  el.countdown.textContent = waiting && left > 0 ? `Translating in ${left.toFixed(1)} s, or keep signing` : "";
  if (waiting && left <= 0 && !analysing) translateSentence();
}

/** t: the frame's time in seconds for the trace (video time for a file). */
function processFrame(now, t = now / 1000) {
  const v = el.video;
  if (el.overlay.width !== v.videoWidth) {
    el.overlay.width = v.videoWidth;
    el.overlay.height = v.videoHeight;
  }
  const detection = tracker.detect(v);
  drawLandmarks(el.overlay, detection, cssColors());
  backdropCtx.drawImage(v, 0, 0, el.backdrop.width, el.backdrop.height); // tiny frame, blurred by CSS

  const pose12 = detection.pose ? POSE_INDICES.map((i) => [detection.pose[i].x, detection.pose[i].y]) : null;
  const segment = segmenter.push(frameFeatures(detection.pose, detection.hands), pose12);
  trace.push(t, pose12 && segmenter.height !== null ? -segmenter.height : null, segmenter.active);
  if (!analysing) {
    if (segmenter.active) setStatus("Signing", "signing");
    else if (el.status.dataset.tone !== "unsure") setStatus(pose12 ? "Raise a hand to sign" : "Step back so your shoulders are in view");
  }

  processedTimes.push(now);
  while (processedTimes.length && now - processedTimes[0] > 1000) processedTimes.shift();
  if (now - lastFactUpdate > 500) {
    lastFactUpdate = now;
    el.factTracking.textContent = analysing
      ? `${tracker.delegate}, reading the video at ${settings.fps} frames/s`
      : `${tracker.delegate}, ${processedTimes.length} frames/s`;
  }
  return segment ? classify(segment) : Promise.resolve();
}

const backdropCtx = el.backdrop.getContext("2d");

let colorCache = null;
function cssColors() {
  if (!colorCache) {
    const s = getComputedStyle(document.documentElement);
    colorCache = { left: s.getPropertyValue("--left"), right: s.getPropertyValue("--right"), pose: s.getPropertyValue("--pose") };
  }
  return colorCache;
}

function setStatus(text, tone = "") {
  el.status.textContent = text;
  el.status.dataset.tone = tone;
}

// ---------- recognition ----------
async function classify(frames) {
  const { ranked, ms } = await classifier.predict(frames);
  lastInferenceMs = ms;
  el.factModel.textContent = `${classifier.labels.length} signs, ${ms.toFixed(0)} ms per sign`;
  const top = ranked[0];
  const confident = top.prob >= settings.minConfidence;
  trace.label(confident ? gloss(top.label) : `unsure (${pct(top.prob)})`, confident);
  if (!confident) {
    setStatus(`Not sure (${pct(top.prob)} ${gloss(top.label)}), so it was skipped. Sign it again.`, "unsure");
    setTimeout(() => el.status.dataset.tone === "unsure" && setStatus(""), 3000);
    return;
  }
  flashSign(top.label);

  if (mode === "word") {
    band.word = ranked;
    renderBand();
    addHistory(gloss(top.label), `${pct(top.prob)} sure`);
    return;
  }
  if (band.translation) band = { ...band, signs: [], translation: null }; // a new sentence starts
  if (band.signs.at(-1)?.label !== top.label) band.signs.push({ label: top.label, ranked: ranked.slice(0, 3) });
  band.lastSignAt = performance.now();
  renderBand();
}

function flashSign(label) {
  const li = el.signGroups.querySelector(`[data-label="${CSS.escape(label)}"]`);
  if (!li) return;
  li.dataset.hit = "true";
  setTimeout(() => (li.dataset.hit = "false"), 2500);
}

// ---------- the band ----------
function tile({ label, prob, alt, removable, onRemove, used, cls }) {
  const li = document.createElement("li");
  li.className = `tile ${cls ?? ""}`;
  const name = document.createElement("span");
  name.className = "tile-name";
  name.textContent = gloss(label);
  li.append(name);
  if (removable) {
    const x = document.createElement("button");
    x.type = "button";
    x.className = "tile-x";
    x.textContent = "×";
    x.setAttribute("aria-label", `Remove ${label}`);
    x.addEventListener("click", onRemove);
    li.append(x);
  } else {
    li.append(document.createElement("span"));
  }
  const conf = document.createElement("span");
  conf.className = "tile-conf";
  const bar = document.createElement("i");
  bar.style.width = `${(prob * 100).toFixed(1)}%`;
  conf.append(bar);
  const sub = document.createElement("span");
  sub.className = "tile-alt num";
  sub.textContent = alt;
  if (used) li.dataset.used = used;
  li.append(conf, sub);
  return li;
}

function renderBand() {
  if (mode === "word") return renderWord();
  const t = band.translation;
  const alignment = t ? alignSentence(t.sentence, band.signs.map((s) => ({
    label: s.label,
    alternatives: s.ranked.slice(1).filter((r) => r.prob >= ALT_MIN).map((r) => r.label),
  }))) : null;

  el.tiles.replaceChildren(...band.signs.map((s, i) => {
    const [first, second] = s.ranked;
    const used = alignment?.signs[i].used;
    const readAs = used && used !== s.label ? used : null;
    return tile({
      label: s.label,
      prob: first.prob,
      alt: readAs ? `read as ${gloss(readAs)}` :
        second && second.prob >= ALT_MIN ? `${pct(first.prob)}, then ${gloss(second.label)} ${pct(second.prob)}` : pct(first.prob),
      used: readAs ?? undefined,
      removable: !t && !band.translating,
      onRemove: () => {
        band.signs.splice(i, 1);
        renderBand();
      },
    });
  }));

  el.sentence.replaceChildren();
  el.sentence.dataset.state = "";
  if (t) {
    alignment.words.forEach((w, i) => {
      const span = document.createElement("span");
      span.className = `w${w.sign === null ? " added" : ""}`;
      if (w.sign !== null) span.dataset.sign = w.sign;
      span.style.animationDelay = `${i * 60}ms`;
      span.textContent = w.text;
      el.sentence.append(span, " ");
    });
  } else if (band.translating) {
    el.sentence.dataset.state = "pending";
    el.sentence.textContent = "Translating…";
  } else {
    el.sentence.dataset.state = "placeholder";
    el.sentence.textContent = band.signs.length
      ? "Keep signing, or pause to translate."
      : "Sign a word, pause, then sign the next. The English appears here.";
  }
  el.translate.disabled = !band.signs.length || Boolean(t) || band.translating;
  el.clear.disabled = !band.signs.length || band.translating;
  requestAnimationFrame(() => drawLinks(Boolean(t)));
}

function renderWord() {
  const ranked = band.word;
  el.links.replaceChildren();
  el.sentence.replaceChildren();
  if (!ranked) {
    el.tiles.replaceChildren();
    el.sentence.dataset.state = "placeholder";
    el.sentence.textContent = "Sign one word and pause. Its name and the model's top guesses appear here.";
    return;
  }
  el.tiles.replaceChildren(...ranked.slice(0, 3).map((r, i) =>
    tile({ label: r.label, prob: r.prob, alt: pct(r.prob), cls: i === 0 ? "first" : "dim" })));
  el.sentence.dataset.state = "";
  const span = document.createElement("span");
  span.className = "w";
  span.textContent = ranked[0].label;
  el.sentence.append(span);
}

/** A line from each sign tile to the first word it became. */
function drawLinks(animate) {
  const svg = el.links;
  svg.replaceChildren();
  if (mode !== "sentence" || !band.translation) return;
  // When the tiles wrap onto several rows (phones, long sentences) lines would cross
  // tiles and words; the order and the lighter added words still read without them.
  if (new Set([...el.tiles.children].map((t) => t.offsetTop)).size > 1) return;
  const box = svg.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  [...el.tiles.children].forEach((tileEl, i) => {
    const word = el.sentence.querySelector(`.w[data-sign="${i}"]`);
    if (!word) return;
    const a = tileEl.getBoundingClientRect();
    const b = word.getBoundingClientRect();
    const x1 = a.left + a.width / 2 - box.left;
    const y1 = a.bottom - box.top;
    const x2 = b.left + Math.min(b.width / 2, 40) - box.left;
    const y2 = b.top - box.top + b.height * 0.18;
    const mid = (y1 + y2) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`);
    svg.append(path);
    if (animate) {
      path.style.setProperty("--len", path.getTotalLength());
      path.style.animationDelay = `${150 + i * 90}ms`;
      path.classList.add("draw");
    }
  });
}
addEventListener("resize", () => drawLinks(false));
document.fonts?.ready.then(() => drawLinks(false));

// ---------- translation ----------
async function translateSentence() {
  if (!band.signs.length || band.translating || band.translation) return;
  band.translating = true;
  renderBand();
  const labels = band.signs.map((s) => s.label);
  const alternatives = band.signs.map((s) => s.ranked.map((r) => [r.label, Number(r.prob.toFixed(4))]));
  let result;
  try {
    const res = await fetch("api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels, alternatives }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    result = await res.json();
  } catch (err) {
    result = { sentence: labels.join(" "), glosses: labels.map(gloss).join(" "), source: "fallback", error: String(err) };
  }
  band.translating = false;
  band.translation = result;
  renderBand();
  addHistory(result.sentence, result.glosses);
  if (result.source !== "llm" && llm.ready) {
    el.llmLabel.textContent = `Translation failed, so the signs are shown as written (${result.error ?? "unknown error"})`;
    el.llmLabel.dataset.tone = "off";
  }
}

function addHistory(text, sub) {
  const li = document.createElement("li");
  li.innerHTML = '<span class="h-text"></span><span class="h-time num"></span><span class="h-sub"></span>';
  li.children[0].textContent = text;
  li.children[1].textContent = clock();
  li.children[2].textContent = sub;
  el.history.prepend(li);
  while (el.history.children.length > 12) el.history.lastChild.remove();
}

// ---------- controls ----------
el.camera.addEventListener("click", () => {
  if (stream) {
    stopSource();
    showEmpty("Camera is off.", "Turn it back on, open a video, or try the sample.");
  } else {
    startCamera();
  }
});

el.sample.addEventListener("click", () => openVideo(SAMPLE_URL));

el.fileInput.addEventListener("change", () => {
  const file = el.fileInput.files?.[0];
  if (file) openVideo(file);
  el.fileInput.value = "";
});

el.translate.addEventListener("click", translateSentence);
el.clear.addEventListener("click", () => {
  band = { ...band, signs: [], translation: null };
  renderBand();
});

document.querySelectorAll('input[name="mode"]').forEach((input) =>
  input.addEventListener("change", () => {
    mode = input.value;
    document.body.dataset.mode = mode;
    band = { signs: [], translation: null, translating: false, lastSignAt: 0, word: null };
    renderBand();
  }),
);

// Settings dialog
const settingsForm = el.settings.querySelector("form");

function syncSettingsForm() {
  for (const [key, value] of Object.entries(settings)) {
    const input = settingsForm.elements[key];
    if (input) input.value = value;
    const out = settingsForm.querySelector(`output[data-for="${key}"]`);
    if (out) out.textContent = key === "minConfidence" ? pct(value) : key.endsWith("Seconds") ? `${value} s` : String(value);
  }
}

settingsForm.addEventListener("input", (e) => {
  if (!e.target.name) return;
  settings[e.target.name] = Number(e.target.value);
  saveSettings();
  segmenter = makeSegmenter();
  syncSettingsForm();
});

$("btn-reset-settings").addEventListener("click", () => {
  settings = { ...DEFAULT_SETTINGS };
  saveSettings();
  segmenter = makeSegmenter();
  syncSettingsForm();
});

el.settingsBtn.addEventListener("click", () => {
  syncSettingsForm();
  el.settings.showModal();
});

init();
