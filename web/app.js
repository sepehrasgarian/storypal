// StoryPal front-end: MediaRecorder -> POST /api/turn -> playback + panels.
// No build step, no dependencies.

let target = "";
let recorder = null;
let chunks = [];

const $ = (id) => document.getElementById(id);

const ICONS = {
  tool: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
  ok: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
  warn: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
};

async function init() {
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
      showSentence(target, []);
      $("reply").textContent = greeting.text;
      new Audio(greeting.audio_url).play().catch(() => {});
    } catch (e) { /* greeting is a nicety; never block the session on it */ }
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
  $("reply").textContent = "Hmm, let me listen…";
  try {
    const form = new FormData();
    form.append("audio", blob, "read.webm");
    form.append("target", target);
    const res = await fetch("/api/turn", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    render(await res.json());
  } catch (err) {
    $("reply").textContent = "Oops, something went wrong: " + err.message;
  } finally {
    $("recBtn").disabled = false;
    $("recLabel").textContent = "Tap to read";
  }
}

function render(turn) {
  showSentence(target, turn.assessment);
  $("transcript").textContent = 'I heard: "' + turn.transcript + '"';

  const s2 = turn.signals.S2;
  $("asrFlag").innerHTML = s2.reliable
    ? '<div class="flag ok">' + ICONS.ok + " ears trusted — " + esc(s2.reasons[0]) + "</div>"
    : '<div class="flag bad">' + ICONS.warn + " ears NOT trusted — " + esc(s2.reasons.join("; ")) + "</div>";

  $("reply").textContent = turn.reply;
  $("tools").innerHTML = (turn.tool_calls || [])
    .map((t) => '<span class="pill">' + ICONS.tool + esc(t.name) + "</span>").join("");
  $("prompt").textContent = turn.prompt;
  $("latency").innerHTML =
    '<span class="lat">style <b>' + esc(turn.style) + "</b></span>" +
    '<span class="lat">ears <b>' + turn.timings_ms.asr_ms + "ms</b></span>" +
    '<span class="lat">brain <b>' + turn.timings_ms.agent_ms + "ms</b></span>" +
    '<span class="lat">voice <b>' + turn.timings_ms.tts_ms + "ms</b></span>";

  new Audio(turn.audio_url).play().catch(() => {});
  if (turn.next_target && turn.next_target !== target) {
    target = turn.next_target;
    setTimeout(() => showSentence(target, []), 4000);
  }
  setTimeout(refreshPanels, 1500); // judges run async; give them a moment
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

async function refreshPanels() {
  const profile = await getJSON("/api/profile");
  const sounds = Object.entries(profile.weak_phonemes || {}).sort((a, b) => b[1] - a[1]);
  const maxMisses = sounds.length ? sounds[0][1] : 1;
  let html =
    '<div class="stat-row">' +
    '<div class="stat"><b>' + profile.level + "</b><span>level</span></div>" +
    '<div class="stat"><b>' + profile.total_turns + "</b><span>turns</span></div>" +
    "</div>";
  if (sounds.length) {
    html += '<ul class="sound-list">' + sounds.map(([p, n]) =>
      '<li><span class="sound-chip">' + esc(p) + '</span><span class="bar"><i style="width:' +
      Math.round((n / maxMisses) * 100) + '%"></i></span>' + n + "</li>").join("") + "</ul>";
  } else {
    html += '<p class="muted">No tricky sounds recorded yet.</p>';
  }
  $("profile").innerHTML = html;

  const curated = await getJSON("/api/curated");
  const piles = curated.piles;
  let cur =
    '<div class="stat-row">' +
    '<div class="stat"><b>' + piles.review_queue.count + "</b><span>review</span></div>" +
    '<div class="stat"><b>' + piles.finetune_set.count + "</b><span>finetune</span></div>" +
    "</div>";
  if (curated.last_judgment && curated.last_judgment.route) {
    const j = curated.last_judgment;
    cur += '<div class="verdict"><div class="flag ' + (j.route === "archive" ? "ok" : "bad") + '">' +
           (j.route === "archive" ? ICONS.ok : ICONS.warn) + " last turn → " + esc(j.route) + "</div>" +
           judgeRow("S3 grounded", j.S3) + judgeRow("S4 kind & on-target", j.S4) + "</div>";
  }
  $("curated").innerHTML = cur;
}

function judgeRow(label, judge) {
  const cls = judge.score >= 0.5 ? "score-good" : "score-bad";
  return '<div class="judge-row"><span class="judge-score ' + cls + '">' + judge.score.toFixed(1) +
         "</span>" + esc(label) + '<span class="muted" style="font-size:11.5px"> — ' + esc(judge.reason) + "</span></div>";
}

async function resetAll() {
  await fetch("/api/reset", { method: "POST" });
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
