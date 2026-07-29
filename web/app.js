// StoryPal front-end: MediaRecorder -> POST /api/turn -> playback + panels.
// No build step, no dependencies.

let target = "";
let recorder = null;
let chunks = [];
let mode = "reading"; // "warmup" until the mic check passes

const $ = (id) => document.getElementById(id);

const ICONS = {
  tool: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
  ok: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
  warn: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
};

// --- StoryPal's face: one mood per state of the turn -------------------
// A 7-year-old reads the face long before they read the words.

const FACE = (eyes, mouth) =>
  '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#fff" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + eyes + mouth + "</svg>";

const EYES = {
  open: '<circle cx="9" cy="10" r="1.4" fill="#fff"/><circle cx="15" cy="10" r="1.4" fill="#fff"/>',
  happy: '<path d="M7.4 10.6c.5-1 1.6-1 2.1 0"/><path d="M14.5 10.6c.5-1 1.6-1 2.1 0"/>',
  up: '<circle cx="9" cy="9.2" r="1.4" fill="#fff"/><circle cx="15" cy="9.2" r="1.4" fill="#fff"/>',
  squint: '<path d="M7.5 10h2.6"/><circle cx="15" cy="10" r="1.4" fill="#fff"/>',
};
const MOUTH = {
  smile: '<path d="M8.5 14.4c1 1.2 2.2 1.8 3.5 1.8s2.5-.6 3.5-1.8"/>',
  grin: '<path d="M7.8 13.8c1.2 2 2.6 3 4.2 3s3-1 4.2-3z" fill="#fff" stroke="none"/>',
  o: '<circle cx="12" cy="15" r="2.1" fill="#fff"/>',
  flat: '<path d="M9.5 15.2h5"/>',
  wavy: '<path d="M9 15.4c.8-.9 1.6.9 2.4 0s1.6-.9 2.4 0"/>',
};

const MOODS = {
  idle: { face: FACE(EYES.open, MOUTH.smile), label: "ready!", cls: "" },
  listening: { face: FACE(EYES.open, MOUTH.o), label: "listening 👂", cls: "listening" },
  thinking: { face: FACE(EYES.up, MOUTH.flat), label: "thinking…", cls: "thinking" },
  happy: { face: FACE(EYES.happy, MOUTH.grin), label: "perfect! ⭐", cls: "happy" },
  practice: { face: FACE(EYES.open, MOUTH.smile), label: "let's practice 💪", cls: "practice" },
  confused: { face: FACE(EYES.squint, MOUTH.wavy), label: "didn't catch that 🤔", cls: "confused" },
};

// --- Pipeline bar: the turn's real stages and their real durations ----

function pipelineRunning(on) {
  $("pipeline").classList.toggle("running", on);
  if (on) {
    document.querySelectorAll(".pipeline li").forEach((li) => {
      li.className = "";
      li.querySelector("i").textContent = "";
    });
  }
}

function pipelineDone(turn) {
  pipelineRunning(false);
  const ms = turn.timings_ms || {};
  const trusted = turn.signals.S2.reliable;
  const shown = {
    asr: ms.asr_ms != null ? ms.asr_ms + "ms" : "",
    grade: "0ms",
    trust: trusted ? "ok" : "failed",
    agent: ms.agent_ms != null ? ms.agent_ms + "ms" : "",
    tts: ms.tts_ms != null ? ms.tts_ms + "ms" : "",
  };
  document.querySelectorAll(".pipeline li").forEach((li) => {
    const stage = li.dataset.stage;
    li.className = stage === "trust" && !trusted ? "flagged" : "done";
    li.querySelector("i").textContent = shown[stage] || "";
  });
}

function setMood(name) {
  const mood = MOODS[name] || MOODS.idle;
  const face = $("palFace");
  face.innerHTML = mood.face;
  face.className = "pal-face " + mood.cls;
  $("palMood").textContent = mood.label;
}

function moodForTurn(turn) {
  if (!turn.signals.S2.reliable) return "confused";
  if (turn.drill_words) return turn.signals.S1.score >= 0.75 ? "happy" : "practice";
  if (turn.signals.S1.score >= 0.99) return "happy";
  return "practice";
}

async function init() {
  setMood("idle");
  target = (await getJSON("/api/story")).target;
  showSentence(target, []);
  refreshPanels();

  // Tap to start, tap again to finish (click also fires for Space/Enter).
  $("recBtn").addEventListener("click", () => {
    if (recorder && recorder.state === "recording") stopRecording();
    else startRecording();
  });

  $("startBtn").addEventListener("click", async () => {
    $("startBtn").disabled = true;
    try {
      const greeting = await getJSON2("/api/greet");
      target = greeting.target;
      mode = "warmup";
      showWarmupPrompt();
      $("reply").textContent = greeting.text;
      new Audio(greeting.audio_url).play().catch(() => {});
    } catch (e) {
      showSentence(target, []); // greeting is a nicety; never block the session
    }
    $("welcome").classList.add("hidden");
  });

  $("nextBtn").addEventListener("click", async () => {
    const res = await getJSON2("/api/next");
    target = res.target;
    showSentence(target, []);
    $("transcript").textContent = "";
    $("asrFlag").innerHTML = "";
    $("reply").textContent = "Here's a new one! Tap the button and read it out loud.";
  });
}

const getJSON2 = async (url) => (await fetch(url, { method: "POST" })).json();

let audioCtx = null;
let meterRaf = null;

async function startRecording() {
  if (recorder && recorder.state === "recording") return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => submitTurn(new Blob(chunks, { type: recorder.mimeType }));
  recorder.start();
  startMeter(stream);
  setMood("listening");
  $("recBtn").classList.add("recording");
  $("recLabel").textContent = "Tap when you're done";
}

function stopRecording() {
  if (!recorder || recorder.state !== "recording") return;
  recorder.stop();
  recorder.stream.getTracks().forEach((t) => t.stop());
  stopMeter();
  $("recBtn").classList.remove("recording");
  $("recBtn").disabled = true;
  $("recLabel").textContent = "Thinking…";
}

// --- Live level meter: proof the mic is hearing something -------------

function startMeter(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  const canvas = $("meter");
  const ctx = canvas.getContext("2d");
  $("meterWrap").hidden = false;

  const BARS = 24;
  const draw = () => {
    analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const step = Math.floor(data.length / BARS);
    const barWidth = canvas.width / BARS;
    for (let i = 0; i < BARS; i++) {
      const level = data[i * step] / 255; // 0..1
      const barHeight = Math.max(4, level * (canvas.height - 8));
      const x = i * barWidth + barWidth * 0.2;
      const y = (canvas.height - barHeight) / 2;
      ctx.fillStyle = level > 0.55 ? "#DC2626" : "#F97316";
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth * 0.6, barHeight, 4);
      ctx.fill();
    }
    meterRaf = requestAnimationFrame(draw);
  };
  draw();
}

function stopMeter() {
  if (meterRaf) cancelAnimationFrame(meterRaf);
  meterRaf = null;
  if (audioCtx) audioCtx.close().catch(() => {});
  audioCtx = null;
  $("meterWrap").hidden = true;
}

async function submitTurn(blob) {
  setMood("thinking");
  pipelineRunning(true);
  $("reply").textContent = "Hmm, let me listen…";
  try {
    const form = new FormData();
    form.append("audio", blob, "read.webm");
    if (mode === "warmup") {
      const res = await fetch("/api/warmup", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      renderWarmup(await res.json());
    } else {
      form.append("target", target);
      const res = await fetch("/api/turn", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      render(await res.json());
    }
  } catch (err) {
    setMood("confused");
    pipelineRunning(false);
    $("reply").textContent = "Oops, something went wrong: " + err.message;
  } finally {
    $("recBtn").disabled = false;
    $("recLabel").textContent = mode === "warmup" ? "Tap and say hello" : "Tap to read";
  }
}

function renderWarmup(result) {
  setMood(result.heard ? "happy" : "confused");
  // Hide the inline TTS control tags from the visible text.
  $("reply").textContent = result.text.replace(/<\|[^|]*\|>/g, " ").replace(/\s+/g, " ");
  if (result.transcript) $("transcript").textContent = 'I heard: "' + result.transcript + '"';
  new Audio(result.audio_url).play().catch(() => {});
  if (result.heard) {
    mode = "reading";
    target = result.target;
    showSentence(target, []); // words appear while StoryPal reads them aloud
    $("transcript").textContent = "";
  }
}

function showWarmupPrompt() {
  $("sentence").innerHTML =
    '<span class="word-chip">Say</span><span class="word-chip">hello</span>' +
    '<span class="word-chip">to</span><span class="word-chip">StoryPal!</span>';
  $("legend").hidden = true;
  $("recLabel").textContent = "Tap and say hello";
}

function render(turn) {
  // In drill mode the grading covered only the practiced words — show
  // those chips, then bring the full sentence back a moment later.
  const graded = turn.graded_target || target;
  showSentence(graded, turn.assessment);
  if (graded !== target) {
    setTimeout(() => showSentence(target, []), 4000);
  }
  $("transcript").textContent = 'I heard: "' + turn.transcript + '"';

  setMood(moodForTurn(turn));
  pipelineDone(turn);
  lastSignals = turn.signals;
  renderSignals(null); // S1/S2 now; S3/S4 fill in when the judges finish

  const s2 = turn.signals.S2;
  $("asrFlag").innerHTML = s2.reliable
    ? '<div class="flag ok">' + ICONS.ok + " ears trusted — " + esc(s2.reasons[0]) + "</div>"
    : '<div class="flag bad">' + ICONS.warn + " ears NOT trusted — " + esc(s2.reasons.join("; ")) + "</div>";

  $("reply").textContent = turn.reply;
  $("tools").innerHTML = (turn.tool_calls || [])
    .map((t) => '<span class="pill">' + ICONS.tool + esc(t.name) + "</span>").join("");
  $("prompt").textContent = turn.prompt;
  // Per-stage timings live in the pipeline bar now; the footer keeps
  // only what the bar cannot show.
  const total = Object.values(turn.timings_ms || {}).reduce((a, b) => a + b, 0);
  $("latency").innerHTML =
    '<span class="lat">voice style <b>' + esc(turn.style) + "</b></span>" +
    '<span class="lat">turn total <b>' + total + "ms</b></span>";

  new Audio(turn.audio_url).play().catch(() => {});
  if (turn.next_target && turn.next_target !== target) {
    target = turn.next_target;
    setTimeout(() => showSentence(target, []), 4000);
  }
  // The judges are two more model calls running after this reply was
  // sent, so their verdict lands seconds later. Poll until the verdict
  // carries this turn's number rather than checking once and giving up.
  awaitJudgment(turn.turn);
}

async function awaitJudgment(turnNumber, attempt = 0) {
  await refreshPanels();
  const judged = lastJudgment && lastJudgment.turn === turnNumber;
  if (!judged && attempt < 12) {
    setTimeout(() => awaitJudgment(turnNumber, attempt + 1), 1200);
  }
}

function showSentence(text, assessment) {
  // Verdicts arrive in target-word order, so zip them positionally with
  // the display words — repeated words each get their own verdict.
  const verdicts = (assessment || []).filter((v) => v.target_word);
  const words = text.split(" ");
  const graded = verdicts.length > 0;

  $("sentence").innerHTML = words.map((word, i) => {
    const v = verdicts[i];
    const status = v ? "w-" + v.status : "";
    let said = "";
    if (v && (v.status === "near_miss" || v.status === "substituted")) {
      said = '<span class="said">you said &#8220;' + esc(v.heard_word) + '&#8221;</span>';
    } else if (v && v.status === "missed") {
      said = '<span class="said">skipped</span>';
    }
    return '<span class="word-chip ' + status + '" style="animation-delay:' + (i * 45) + 'ms">' +
           esc(word) + said + "</span>";
  }).join("");
  $("legend").hidden = !graded;
}

// --- Instrument panels -------------------------------------------------

let lastSignals = null; // S1/S2 from the turn; S3/S4 arrive with the judges
let lastJudgment = null;

function metric(value, label) {
  return '<div class="metric"><b>' + value + "</b><span>" + label + "</span></div>";
}

function signalRow(name, valueText, state, reason) {
  return (
    '<div class="row"><span class="dot ' + state + '"></span>' +
    '<span class="k">' + name + "</span>" +
    '<span class="v">' + valueText + "</span>" +
    (reason ? '<div class="sig-reason">' + esc(reason) + "</div>" : "") +
    "</div>"
  );
}

function chip(label, value, state) {
  return '<div class="chip-metric"><span>' + label + "</span>" +
         '<b class="' + state + '">' + value + "</b></div>";
}

function renderSignals(judgment) {
  if (!lastSignals) return;
  const s1 = lastSignals.S1, s2 = lastSignals.S2;
  let html =
    '<div class="chip-row">' +
    chip("Reading", Math.round(s1.score * 100) + "%",
         s1.score >= 0.99 ? "good" : s1.score > 0 ? "info" : "bad") +
    chip("Ears", s2.reliable ? "Trusted" : "Doubted", s2.reliable ? "good" : "bad") +
    "</div>";
  html += '<div class="rows">';
  html += signalRow("S1 · reading", Math.round(s1.score * 100) + "%",
                    s1.score >= 0.99 ? "good" : s1.score > 0 ? "info" : "bad", s1.reasons[0]);
  html += signalRow("S2 · ears", s2.reliable ? "trusted" : "not trusted",
                    s2.reliable ? "good" : "bad", s2.reasons.join("; "));
  if (judgment && judgment.S3) {
    html += signalRow("S3 · grounded", judgment.S3.score.toFixed(1),
                      judgment.S3.score >= 0.5 ? "good" : "bad", judgment.S3.reason);
    html += signalRow("S4 · teaching", judgment.S4.score.toFixed(1),
                      judgment.S4.score >= 0.5 ? "good" : "bad", judgment.S4.reason);
  } else {
    html += signalRow("S3 · grounded", "judging…", "");
    html += signalRow("S4 · teaching", "judging…", "");
  }
  html += "</div>";
  if (judgment && judgment.route) {
    const flagged = judgment.route !== "archive";
    html += '<div class="route-line ' + (flagged ? "route-flag" : "route-archive") + '">' +
            (flagged ? ICONS.warn : ICONS.ok) + " routed → " + esc(judgment.route) + "</div>";
  }
  $("signals").innerHTML = html;
}

// Misses per opportunity, not raw misses: "the" appears in nearly every
// sentence, so by volume it would always look like the hardest word.
const SMOOTHING = 2;
const rank = (counts, attempts) =>
  Object.entries(counts || {})
    .map(([key, misses]) => {
      const tries = (attempts || {})[key] || misses;
      return { key, misses, tries, rate: misses / (tries + SMOOTHING) };
    })
    .sort((a, b) => b.rate - a.rate);

async function refreshPanels() {
  const profile = await getJSON("/api/profile");
  const sounds = rank(profile.weak_phonemes, profile.phoneme_attempts);
  let html = '<div class="metrics">' + metric(profile.level, "level") +
             metric(profile.total_turns, "graded turns") + "</div>";
  if (sounds.length) {
    html += '<div class="rows">' + sounds.slice(0, 5).map((s) =>
      '<div class="row"><span class="k">/' + esc(s.key) + '/</span>' +
      '<span class="track"><i class="warnfill" style="width:' +
      Math.round(s.rate * 100) + '%"></i></span>' +
      '<span class="v">' + s.misses + "/" + s.tries + "</span></div>").join("") + "</div>";
  } else {
    html += '<p class="muted">No weak sounds recorded yet.</p>';
  }
  const words = rank(profile.missed_words, profile.word_attempts).slice(0, 6);
  if (words.length) {
    html += '<h2 style="margin:16px 0 6px">Hardest words</h2><div class="wordchips">' +
            words.map((w) => "<span>" + esc(w.key) + " · " + w.misses + "/" + w.tries +
                      "</span>").join("") + "</div>";
  }
  $("profile").innerHTML = html;

  const curated = await getJSON("/api/curated");
  const piles = curated.piles;
  $("curated").innerHTML =
    '<div class="metrics">' + metric(piles.review_queue.count, "review queue") +
    metric(piles.finetune_set.count, "finetune set") + "</div>" +
    '<p class="muted">Review holds turns we could not trust. Finetune holds replies the judges faulted — each with context and a slot for a corrected reply.</p>';

  lastJudgment = curated.last_judgment;
  renderSignals(lastJudgment);
}

async function resetAll() {
  await fetch("/api/reset", { method: "POST" });
  setMood("idle");
  target = (await getJSON("/api/story")).target;
  showSentence(target, []);
  refreshPanels();
  $("reply").textContent = "Fresh start! Tap the button and read the sentence.";
  $("transcript").textContent = "";
  $("asrFlag").innerHTML = "";
}

const getJSON = async (url) => (await fetch(url)).json();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => "&#" + c.charCodeAt(0) + ";");

init();
