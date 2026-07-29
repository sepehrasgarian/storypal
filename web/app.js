// StoryPal front-end: MediaRecorder -> POST /api/turn -> playback + panels.
// No build step, no dependencies.

let target = "";
let recorder = null;
let chunks = [];

const $ = (id) => document.getElementById(id);

async function init() {
  target = (await getJSON("/api/story")).target;
  showSentence(target, []);
  refreshPanels();

  const btn = $("recBtn");
  btn.addEventListener("mousedown", startRecording);
  btn.addEventListener("mouseup", stopRecording);
  btn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
  btn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => submitTurn(new Blob(chunks, { type: recorder.mimeType }));
  recorder.start();
  $("recBtn").classList.add("recording");
  $("recBtn").textContent = "🔴 Reading… release when done";
}

function stopRecording() {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    recorder.stream.getTracks().forEach((t) => t.stop());
  }
  const btn = $("recBtn");
  btn.classList.remove("recording");
  btn.textContent = "🎤 Hold to read";
  btn.disabled = true;
}

async function submitTurn(blob) {
  $("reply").textContent = "Listening…";
  try {
    const form = new FormData();
    form.append("audio", blob, "read.webm");
    form.append("target", target);
    const res = await fetch("/api/turn", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    render(await res.json());
  } catch (err) {
    $("reply").textContent = "Something went wrong: " + err.message;
  } finally {
    $("recBtn").disabled = false;
  }
}

function render(turn) {
  showSentence(target, turn.assessment);
  $("transcript").textContent = 'heard: "' + turn.transcript + '"';

  const s2 = turn.signals.S2;
  $("asrFlag").innerHTML = s2.reliable
    ? '<div class="flag ok">ears trusted — ' + esc(s2.reasons[0]) + "</div>"
    : '<div class="flag bad">⚠ ears NOT trusted — ' + esc(s2.reasons.join("; ")) + "</div>";

  $("reply").textContent = turn.reply;
  $("tools").innerHTML = (turn.tool_calls || [])
    .map((t) => '<span class="pill">🔧 ' + esc(t.name) + "</span>").join("");
  $("prompt").textContent = turn.prompt;
  $("latency").textContent =
    "style: " + turn.style +
    "  ·  asr " + turn.timings_ms.asr_ms + "ms" +
    "  ·  agent " + turn.timings_ms.agent_ms + "ms" +
    "  ·  tts " + turn.timings_ms.tts_ms + "ms";

  new Audio(turn.audio_url).play().catch(() => {});
  if (turn.next_target && turn.next_target !== target) {
    target = turn.next_target;
    setTimeout(() => showSentence(target, []), 4000);
  }
  setTimeout(refreshPanels, 1500); // judges run async; give them a moment
}

function showSentence(text, assessment) {
  const verdictByWord = {};
  (assessment || []).forEach((v) => {
    if (v.target_word) verdictByWord[v.target_word.toLowerCase()] = v.status;
  });
  $("sentence").innerHTML = text.split(" ").map((word) => {
    const key = word.toLowerCase().replace(/[^a-z0-9']/g, "");
    const status = verdictByWord[key] || "";
    return '<span class="w-' + status + '">' + esc(word) + "</span>";
  }).join(" ");
}

async function refreshPanels() {
  const profile = await getJSON("/api/profile");
  const sounds = Object.entries(profile.weak_phonemes || {})
    .sort((a, b) => b[1] - a[1])
    .map(([p, n]) => "<li>'" + esc(p) + "' — " + n + " misses</li>").join("");
  $("profile").innerHTML =
    "level " + profile.level + " · " + profile.total_turns + " turns" +
    (sounds ? "<ul>" + sounds + "</ul>" : "<br>no weak sounds recorded");

  const curated = await getJSON("/api/curated");
  const piles = curated.piles;
  let html = "review queue: <b>" + piles.review_queue.count + "</b><br>" +
             "finetune set: <b>" + piles.finetune_set.count + "</b>";
  if (curated.last_judgment && curated.last_judgment.route) {
    const j = curated.last_judgment;
    html += '<div class="flag ' + (j.route === "archive" ? "ok" : "bad") + '">last turn → ' +
            esc(j.route) + "<br>S3 " + j.S3.score + " · S4 " + j.S4.score + "</div>";
  }
  $("curated").innerHTML = html;
}

async function resetAll() {
  await fetch("/api/reset", { method: "POST" });
  target = (await getJSON("/api/story")).target;
  showSentence(target, []);
  refreshPanels();
  $("reply").textContent = "Fresh start! Hold the button and read the sentence.";
}

const getJSON = async (url) => (await fetch(url)).json();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => "&#" + c.charCodeAt(0) + ";");

init();
