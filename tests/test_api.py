"""End-to-end integration test of POST /api/turn with all services faked.

The full pipeline runs — ASR, assessment, signals, profile gating,
prompt, agent, TTS, trajectory, judges, triage — with zero network.
"""

import pytest
from fastapi.testclient import TestClient

from storypal.speech.asr import TranscriptionResult
from storypal.agent.llm import FakeLLM, LLMReply
from storypal.api.main import Services, create_app
from storypal.core.signals import AsrTelemetry

TARGET = "The cat sat on the mat."


class FakeASR:
    """Returns whatever transcript+telemetry the test scripts."""

    def __init__(self):
        self.next = TranscriptionResult("the cat sat on the mat", AsrTelemetry(-0.2, 0.05, 1.2))

    def transcribe(self, path):
        return self.next


class FakeTTS:
    def __init__(self, tmp_path):
        self.dir = tmp_path
        self.styles = []

    def synthesize(self, text, style="neutral"):
        self.styles.append(style)
        path = self.dir / "fake.mp3"
        path.write_bytes(b"MP3")
        return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("storypal.api.main.PROFILE_PATH", str(tmp_path / "profile.json"))
    monkeypatch.setattr("storypal.api.main.TRAJECTORY_PATH", str(tmp_path / "traj.jsonl"))
    monkeypatch.setattr("storypal.api.main.DATA_DIR", str(tmp_path))
    asr = FakeASR()
    tts = FakeTTS(tmp_path)
    tutor_llm = FakeLLM([LLMReply(text="Wonderful reading! You got every word.")] * 20)
    judge_llm = FakeLLM([LLMReply(text='{"score": 1.0, "reason": "fine"}')] * 40)
    app = create_app(Services(asr=asr, tts=tts, llm=tutor_llm, judge_llm=judge_llm))
    app.state.curated.directory = tmp_path / "curated"
    return {"client": TestClient(app), "asr": asr, "tts": tts, "app": app}


def post_turn(env):
    return env["client"].post(
        "/api/turn",
        data={"target": TARGET},
        files={"audio": ("read.webm", b"fake-audio-bytes", "audio/webm")},
    )


class TestGoodTurn:
    def test_full_pipeline_responds(self, env):
        body = post_turn(env).json()
        assert body["transcript"] == "the cat sat on the mat"
        assert body["signals"]["S1"]["score"] == 1.0
        assert body["signals"]["S2"]["reliable"] is True
        assert body["reply"].startswith("Wonderful")
        assert set(body["timings_ms"]) == {"asr_ms", "agent_ms", "tts_ms"}
        assert env["tts"].styles == ["celebrate"]

    def test_profile_and_trajectory_are_updated(self, env):
        post_turn(env)
        profile = env["client"].get("/api/profile").json()
        assert profile["total_turns"] == 1
        assert len(env["app"].state.trajectory.read_all()) == 1

    def test_prompt_endpoint_shows_last_prompt(self, env):
        post_turn(env)
        prompt = env["client"].get("/api/prompt").json()["prompt"]
        assert TARGET in prompt


class TestHallucinatedTurn:
    def test_unreliable_turn_flips_behaviour(self, env):
        # The demo moment: silence in, "thanks for watching" out.
        env["asr"].next = TranscriptionResult(
            "thanks for watching", AsrTelemetry(avg_logprob=-1.6, no_speech_prob=0.92, compression_ratio=1.1)
        )
        body = post_turn(env).json()
        assert body["signals"]["S2"]["reliable"] is False
        assert "do NOT correct" in body["prompt"]
        assert body["style"] == "encourage"  # never celebrate an unverified read
        # Tier 2 gate: memory untouched.
        assert env["client"].get("/api/profile").json()["total_turns"] == 0
        # Tier 3 gate: routed to the review queue by the background loop.
        curated = env["client"].get("/api/curated").json()
        assert curated["piles"]["review_queue"]["count"] == 1
        assert curated["piles"]["finetune_set"]["count"] == 0


class TestUndecodableAudio:
    def test_broken_audio_becomes_gentle_reask_not_500(self, env):
        # A too-short press sends an empty webm; decoding fails. That is
        # unreliable perception, not a server error.
        class ExplodingASR:
            def transcribe(self, path):
                raise RuntimeError("EOF: empty container")

        env["app"].state.services.asr = ExplodingASR()
        response = post_turn(env)
        assert response.status_code == 200
        body = response.json()
        assert body["transcript"] == ""
        assert body["signals"]["S2"]["reliable"] is False
        assert "do NOT correct" in body["prompt"]


class TestGreeting:
    def test_new_learner_gets_intro_with_audio(self, env):
        body = env["client"].post("/api/greet").json()
        assert "I'm StoryPal" in body["text"]
        assert body["audio_url"].startswith("/api/audio/")
        assert body["target"] == "The cat sat on the mat."

    def test_returning_learner_is_welcomed_back(self, env):
        post_turn(env)  # one real turn makes the learner "returning"
        body = env["client"].post("/api/greet").json()
        assert "Welcome back" in body["text"]


class TestAutoAdvance:
    def test_perfect_trusted_read_advances_to_next_sentence(self, env):
        body = post_turn(env).json()  # FakeASR reads the target perfectly
        assert body["signals"]["S1"]["score"] == 1.0
        assert body["next_target"] != TARGET

    def test_flawed_read_stays_on_the_same_sentence(self, env):
        env["asr"].next = TranscriptionResult("the cat sat on the", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert body["next_target"] == TARGET

    def test_unreliable_perfect_looking_read_does_not_advance(self, env):
        # Even a transcript that matches the target must not advance if
        # the recognizer itself is untrustworthy.
        env["asr"].next = TranscriptionResult(
            "the cat sat on the mat", AsrTelemetry(avg_logprob=-1.8, no_speech_prob=0.1, compression_ratio=1.2)
        )
        body = post_turn(env).json()
        assert body["signals"]["S2"]["reliable"] is False
        assert body["next_target"] == TARGET


class TestConversationalTurns:
    """'Yes, I do!' is the child answering the tutor - it must be
    answered, never graded as a failed reading. (Seen in real logs:
    these turns were polluting the profile with fake misses.)"""

    def test_yes_i_do_is_not_graded(self, env):
        env["asr"].next = TranscriptionResult("Yes, I do.", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert "not a reading attempt" in body["signals"]["S1"]["reasons"][0]
        assert "talking TO you" in body["prompt"]
        # No fake misses recorded, no advance earned.
        assert env["client"].get("/api/profile").json()["missed_words"] == {}
        assert body["next_target"] == TARGET

    def test_frustration_is_answered_not_graded(self, env):
        env["asr"].next = TranscriptionResult("Damn it!", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert "talking TO you" in body["prompt"]
        assert env["client"].get("/api/profile").json()["missed_words"] == {}

    def test_real_partial_read_is_still_graded(self, env):
        # "the cat sat" contains story words - reading, not chat.
        env["asr"].next = TranscriptionResult("the cat sat", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert "talking TO you" not in body["prompt"]
        assert env["client"].get("/api/profile").json()["missed_words"] != {}


class TestDrillFollowup:
    """After a flawed read, repeating just the practiced word must be
    graded as a drill - not as skipping the rest of the sentence."""

    def flawed_read(self, env):
        # Target: "The cat sat on the mat." - child misses "mat".
        env["asr"].next = TranscriptionResult("the cat sat on the", AsrTelemetry(-0.3, 0.1, 1.2))
        return post_turn(env).json()

    def test_single_word_answer_is_graded_as_drill(self, env):
        self.flawed_read(env)
        env["asr"].next = TranscriptionResult("mat", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert body["drill_words"] == ["mat"]
        assert body["graded_target"] == "mat"
        assert body["signals"]["S1"]["score"] == 1.0  # they got the word!
        assert "practicing just the word" in body["prompt"]

    def test_drill_does_not_pollute_the_profile(self, env):
        self.flawed_read(env)
        profile_after_miss = env["client"].get("/api/profile").json()
        env["asr"].next = TranscriptionResult("mat", AsrTelemetry(-0.3, 0.1, 1.2))
        post_turn(env)
        profile = env["client"].get("/api/profile").json()
        # Saying just "mat" must not record cat/sat/on/the as missed.
        assert profile["missed_words"] == profile_after_miss["missed_words"]

    def test_drill_success_does_not_auto_advance(self, env):
        self.flawed_read(env)
        env["asr"].next = TranscriptionResult("mat", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        # The child still owes a full read of the sentence.
        assert body["next_target"] == TARGET

    def test_full_reread_is_still_graded_as_full_sentence(self, env):
        self.flawed_read(env)
        env["asr"].next = TranscriptionResult("the cat sat on the mat", AsrTelemetry(-0.3, 0.1, 1.2))
        body = post_turn(env).json()
        assert body["drill_words"] is None
        assert body["graded_target"] == TARGET
        assert body["next_target"] != TARGET  # accepted: advances

    def test_drill_outcome_updates_tactic_scoreboard(self, env):
        from storypal.learning.kb import TACTICS
        tactic = next(t for t in TACTICS if t.phoneme == "th")
        self.flawed_read(env)
        env["app"].state.last_tactic = tactic  # as if the agent had drilled
        env["asr"].next = TranscriptionResult("mat", AsrTelemetry(-0.3, 0.1, 1.2))
        post_turn(env)
        stats = env["app"].state.tactic_stats
        assert stats.success_rate(tactic) > 0.5  # one success recorded


class TestWarmup:
    def post_warmup(self, env):
        return env["client"].post(
            "/api/warmup", files={"audio": ("hi.webm", b"fake", "audio/webm")}
        ).json()

    def test_hearing_the_child_confirms_and_models_the_sentence(self, env):
        env["asr"].next = TranscriptionResult("hello story pal", AsrTelemetry(-0.3, 0.1, 1.2))
        body = self.post_warmup(env)
        assert body["heard"] is True
        assert "loud and clear" in body["text"]
        # StoryPal must speak the first sentence so the child can repeat it.
        assert body["target"] in body["text"]
        assert "read it back" in body["text"]

    def test_silence_asks_to_try_again(self, env):
        env["asr"].next = TranscriptionResult("", AsrTelemetry(0.0, 1.0, 1.0))
        body = self.post_warmup(env)
        assert body["heard"] is False
        assert "try saying hello" in body["text"]

    def test_warmup_never_touches_memory(self, env):
        env["asr"].next = TranscriptionResult("hello", AsrTelemetry(-0.3, 0.1, 1.2))
        self.post_warmup(env)
        assert env["client"].get("/api/profile").json()["total_turns"] == 0


class TestNextSentence:
    def test_manual_skip_changes_the_target(self, env):
        first = env["client"].get("/api/story").json()["target"]
        second = env["client"].post("/api/next").json()["target"]
        assert second != first

    def test_rotation_never_runs_dry(self, env):
        # Skipping more times than there are sentences must keep working.
        from storypal.config import STORIES
        targets = {env["client"].post("/api/next").json()["target"] for _ in range(len(STORIES) + 3)}
        assert all(targets)


class TestReset:
    def test_reset_clears_session(self, env):
        post_turn(env)
        env["client"].post("/api/reset")
        assert env["client"].get("/api/profile").json()["total_turns"] == 0
