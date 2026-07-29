"""FastAPI wiring: one turn of the pipeline per POST /api/turn.

Sync pre-loop (deterministic): ASR -> assessment -> S1/S2 -> prompt ->
agent -> TTS -> respond. Async post-loop (background): S3/S4 judges ->
triage -> curated piles. Per-stage timings are measured and returned.
"""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from storypal import observability
from storypal.agent.judges import s3_grounding, s4_pedagogy
from storypal.agent.loop import run_turn
from storypal.agent.tools import ToolContext
from storypal.config import DATA_DIR, PROFILE_PATH, STORIES, TRAJECTORY_PATH, phonemes_in_word
from storypal.learning.kb import best_tactic
from storypal.session import problem_words
from storypal.core.trajectory import TrajectoryLog, TurnRecord, reward_from_signals
from storypal.core.triage import route_turn
from storypal.learning import profile as profile_mod
from storypal.learning.curated import CuratedStore
from storypal.learning.kb import TacticStats
from storypal.learning.prompt import build_prompt
from storypal.session import grade_turn, update_expectations
from storypal.speech.tts import choose_style

load_dotenv()


@dataclass
class Services:
    """The three external dependencies, injectable so tests fake them."""

    asr: object
    tts: object
    llm: object
    judge_llm: object


def default_services() -> Services:
    from storypal.speech.asr import WhisperASR
    from storypal.agent.llm import GeminiLLM
    from storypal.speech.tts import HiggsTTS

    llm = GeminiLLM()
    return Services(
        asr=WhisperASR(),
        tts=HiggsTTS(api_key=os.environ["BOSON_API_KEY"]),
        llm=llm,
        judge_llm=llm,
    )


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="StoryPal")
    state = app.state
    state.services = services  # resolved lazily so tests never build real ones
    state.profile = profile_mod.load(PROFILE_PATH)
    state.tactic_stats = TacticStats(Path(DATA_DIR) / "tactics.json")
    state.trajectory = TrajectoryLog(TRAJECTORY_PATH)
    state.curated = CuratedStore()
    state.session_id = uuid.uuid4().hex[:8]
    state.turn_count = 0
    state.target = STORIES[0].text
    state.seen = set()
    state.pending_drill = None  # words the child is expected to repeat next
    state.last_tactic = None  # tactic awaiting its outcome signal
    state.attempts = 0  # tries on the current sentence
    state.streak = 0  # consecutive accepted reads
    state.last_target = None
    state.last_prompt = ""
    state.last_judgment = {}
    state.exporter = observability.from_env()  # None unless Langfuse keys set

    def services_now() -> Services:
        if state.services is None:
            state.services = default_services()
        return state.services

    # src/storypal/api/main.py -> repo root is three levels up from storypal/
    web_dir = Path(__file__).parents[3] / "web"
    app.mount("/web", StaticFiles(directory=web_dir), name="web")

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        """Force the browser to revalidate the page assets.

        Starlette serves an ETag but no Cache-Control, so browsers fall
        back to heuristic caching and can run a stale script for a long
        time. That cost real debugging once: a fixed front end sat on
        disk and in the response while the browser kept the old one.
        """
        response = await call_next(request)
        if request.url.path.startswith("/web") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(web_dir / "index.html")

    @app.get("/api/story")
    def story():
        return {"target": state.target}

    @app.post("/api/greet")
    def greet():
        """Session-opening welcome, spoken by the tutor. Fixed scripts
        (new vs returning learner) so the audio caches after one call."""
        svc = services_now()
        if state.profile.total_turns == 0:
            text = ("Hi there, I'm StoryPal! I'm so happy to read with you today. "
                    "First, let's check I can hear you: tap the big orange "
                    "button, say hello to me, and tap again when you're done!")
        else:
            text = ("Welcome back, my friend! I missed reading with you. "
                    "First, tap the big orange button and say hello, so I "
                    "know I can hear you!")
        audio_file = svc.tts.synthesize(text, style="celebrate")
        return {
            "text": text,
            "audio_url": f"/api/audio/{Path(audio_file).name}",
            "target": state.target,
        }

    @app.post("/api/warmup")
    async def warmup(audio: UploadFile = File(...)):
        """Mic-check warm-up: the child says hello, we confirm we heard
        them. Nothing is graded and nothing touches memory — this exists
        to verify the audio pipeline (and build confidence) before the
        first real turn. Fixed replies so the audio caches."""
        svc = services_now()
        with NamedTemporaryFile(suffix=Path(audio.filename or "a.webm").suffix, delete=False) as f:
            f.write(await audio.read())
            audio_path = f.name
        try:
            result = svc.asr.transcribe(audio_path)
        except Exception:
            from storypal.core.signals import AsrTelemetry
            from storypal.speech.asr import TranscriptionResult

            result = TranscriptionResult("", AsrTelemetry(no_speech_prob=1.0))
        finally:
            os.unlink(audio_path)

        heard = bool(result.transcript.strip()) and result.telemetry.no_speech_prob <= 0.5
        if heard:
            # Model the first sentence out loud (slowed) so the child
            # knows their job is to repeat it.
            text = ("I can hear you loud and clear! Hello to you too! "
                    "Here is our first sentence. Listen to me first: "
                    f"<|prosody:speed_slow|> {state.target} "
                    "Now tap the button and read it back to me!")
        else:
            text = ("Hmm, I couldn't quite hear you. Make sure your microphone "
                    "is on, and try saying hello one more time!")
        audio_file = svc.tts.synthesize(text, style="celebrate" if heard else "encourage")
        return {
            "heard": heard,
            "transcript": result.transcript,
            "text": text,
            "audio_url": f"/api/audio/{Path(audio_file).name}",
            "target": state.target,
        }

    def _advance_target() -> str:
        """Pick a new sentence for the child's level, biased toward
        their weakest sound, never repeating until the pool runs dry."""
        from storypal.learning import kb
        from storypal.learning.profile import weakest_phoneme

        state.seen.add(state.target)
        story = kb.next_sentence(
            state.profile.level, weakest_phoneme(state.profile), exclude=state.seen
        )
        if story is None:  # everything seen: start the rotation over
            state.seen = {state.target}
            story = kb.next_sentence(state.profile.level, weakest_phoneme(state.profile))
        state.target = story.text
        return state.target

    @app.post("/api/next")
    def next_story():
        """Manual skip. The agent's next_sentence tool and the
        auto-advance on an accepted read use the same picker."""
        return {"target": _advance_target()}

    @app.post("/api/turn")
    async def turn(background: BackgroundTasks, audio: UploadFile = File(...), target: str = Form(...)):
        svc = services_now()
        timings: dict[str, int] = {}

        def timed(name, fn):
            start = time.monotonic()
            result = fn()
            timings[name] = int((time.monotonic() - start) * 1000)
            return result

        with NamedTemporaryFile(suffix=Path(audio.filename or "a.webm").suffix, delete=False) as f:
            f.write(await audio.read())
            audio_path = f.name
        try:
            asr_result = timed("asr_ms", lambda: svc.asr.transcribe(audio_path))
        except Exception:
            # Empty or undecodable audio is just another form of unreliable
            # perception: treat it as certain silence and let the normal
            # S2 path produce the gentle "read it once more" behaviour.
            from storypal.speech.asr import TranscriptionResult
            from storypal.core.signals import AsrTelemetry

            timings.setdefault("asr_ms", 0)
            asr_result = TranscriptionResult("", AsrTelemetry(no_speech_prob=1.0))
        finally:
            os.unlink(audio_path)

        graded = grade_turn(
            target, asr_result.transcript, asr_result.telemetry, state.pending_drill
        )
        s1, s2, assessment = graded.s1, graded.s2, graded.assessment

        # Tier 2: unreliable turns never touch memory, and conversation
        # is not evidence of reading ability.
        if not graded.chat_turn:
            profile_mod.update_from_turn(state.profile, assessment, s2)
            profile_mod.save(state.profile, PROFILE_PATH)

        # Strategy KB: a drill response is the outcome signal for the
        # tactic that was used to teach it.
        if graded.drill_words is not None and s2.reliable and state.last_tactic is not None:
            state.tactic_stats.record_outcome(state.last_tactic, worked=graded.drill_worked)
            state.last_tactic = None

        state.attempts = state.attempts + 1 if target == state.last_target else 1
        state.last_target = target

        # The strategy KB is consulted for us, not only when the model
        # remembers to call drill_sound: every corrective turn now gets
        # the tactic with the best track record for this child.
        tactic = _tactic_for(assessment, graded, state)
        if tactic is not None:
            state.tactic_stats.record_usage(tactic)

        # Tier 1: instructions rebuilt from this turn's state.
        system_prompt = build_prompt(
            target, assessment, s1, s2, state.profile, tactic=tactic,
            drill_words=graded.drill_words, conversational=graded.chat_turn,
            attempts=state.attempts, streak=state.streak,
        )
        state.last_prompt = system_prompt

        ctx = ToolContext(
            profile=state.profile,
            tactic_stats=state.tactic_stats,
            asr_reliable=s2.reliable,
            sentences_seen={target},
        )
        result = timed("agent_ms", lambda: run_turn(
            system_prompt, f'The child just read aloud: "{asr_result.transcript}"', svc.llm, ctx
        ))

        style = choose_style(s1.score, s2.reliable)
        audio_file = timed("tts_ms", lambda: svc.tts.synthesize(result.reply, style))

        state.streak = state.streak + 1 if graded.accepted else 0
        if ctx.next_story:
            state.target = ctx.next_story  # the agent chose the next sentence
        elif graded.accepted:
            _advance_target()  # accepted full read: move on automatically

        state.pending_drill, state.last_tactic = update_expectations(
            graded, ctx.tactic_used or tactic, state.pending_drill, state.last_tactic
        )

        state.turn_count += 1
        signals = {"S1": s1, "S2": s2}
        record = TurnRecord(
            session_id=state.session_id,
            turn=state.turn_count,
            timestamp=time.time(),
            state={"target": target, "transcript": asr_result.transcript,
                   "profile": asdict(state.profile)},
            action={"reply": result.reply, "tool_calls": [(n, a) for n, a, _ in result.tool_calls],
                    "style": style},
            reward=reward_from_signals(signals),
            route=route_turn(signals, chat_turn=graded.chat_turn).route.value,
        )
        state.trajectory.append(record)
        background.add_task(_post_loop, state, svc, s1, s2, result.reply,
                            record, timings, graded.chat_turn)

        return {
            "turn": state.turn_count,
            "transcript": asr_result.transcript,
            "graded_target": graded.graded_target,  # differs from target in drill mode
            "drill_words": graded.drill_words,
            "assessment": [asdict(v) | {"status": v.status.value} for v in assessment.verdicts],
            "signals": {k: asdict(v) for k, v in signals.items()},
            "reply": result.reply,
            "audio_url": f"/api/audio/{Path(audio_file).name}",
            "tool_calls": [{"name": n, "args": a, "result": r} for n, a, r in result.tool_calls],
            "next_target": state.target,
            "prompt": system_prompt,
            "style": style,
            "timings_ms": timings,
        }

    @app.get("/api/audio/{name}")
    def audio_file(name: str):
        return FileResponse(Path(DATA_DIR) / "tts_cache" / Path(name).name, media_type="audio/mpeg")

    @app.get("/api/profile")
    def get_profile():
        return asdict(state.profile)

    @app.get("/api/curated")
    def get_curated():
        return {"piles": state.curated.summary(), "last_judgment": state.last_judgment}

    @app.post("/api/reset")
    def reset():
        state.profile = profile_mod.Profile()
        profile_mod.save(state.profile, PROFILE_PATH)
        state.session_id = uuid.uuid4().hex[:8]
        state.turn_count = 0
        state.target = STORIES[0].text
        state.seen = set()
        state.pending_drill = None
        state.last_tactic = None
        state.attempts = 0
        state.streak = 0
        state.last_target = None
        return {"ok": True}

    return app


def _tactic_for(assessment, graded, state):
    """The best-performing tactic for the first troublesome sound this
    turn, or None when there is nothing to teach."""
    if graded.chat_turn or not graded.s2.reliable or graded.drill_words is not None:
        return None
    for word in problem_words(assessment):
        for phoneme in phonemes_in_word(word):
            tactic = best_tactic(phoneme, state.tactic_stats)
            if tactic is not None:
                return tactic
    return None


def _post_loop(state, svc, s1, s2, reply, record: TurnRecord, timings: dict,
               chat_turn: bool = False) -> None:
    """S3/S4 judges + triage + observability. Runs after the response is sent."""
    summary = "; ".join(s1.reasons) + f" (accuracy {s1.score:.0%}; ASR {'ok' if s2.reliable else 'UNRELIABLE'})"
    # The judges are independent, so run them together rather than one
    # after the other: the verdict is what the panel waits on.
    with ThreadPoolExecutor(max_workers=2) as pool:
        grounding = pool.submit(s3_grounding, summary, reply, svc.judge_llm)
        pedagogy = pool.submit(s4_pedagogy, summary, reply, svc.judge_llm)
        s3, s4 = grounding.result(), pedagogy.result()
    signals = {"S1": s1, "S2": s2, "S3": s3, "S4": s4}
    decision = route_turn(signals, chat_turn=chat_turn)
    if state.exporter is not None:
        state.exporter.export_turn(
            session_id=record.session_id, turn=record.turn, timestamp=record.timestamp,
            target=record.state["target"], transcript=record.state["transcript"],
            reply=reply, prompt=state.last_prompt, signals=signals,
            timings_ms=timings, route=decision.route.value,
        )
    state.last_judgment = {
        "turn": record.turn,  # so the UI knows which turn this verdict is for
        "S3": {"score": s3.score, "reason": s3.reasons[0]},
        "S4": {"score": s4.score, "reason": s4.reasons[0]},
        "route": decision.route.value,
        "why": decision.reason,
    }
    state.curated.add(decision.route, {
        "context": record.state,
        "reply": reply,
        "verdicts": {k: {"score": v.score, "reasons": list(v.reasons)} for k, v in signals.items()},
        "route_reason": decision.reason,
    })


app = create_app()
