"""End-to-end integration test of POST /api/turn with all services faked.

The full pipeline runs — ASR, assessment, signals, profile gating,
prompt, agent, TTS, trajectory, judges, triage — with zero network.
"""

import pytest
from fastapi.testclient import TestClient

from api.asr import TranscriptionResult
from api.llm import FakeLLM, LLMReply
from api.main import Services, create_app
from api.signals import AsrTelemetry

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
    monkeypatch.setattr("api.main.PROFILE_PATH", str(tmp_path / "profile.json"))
    monkeypatch.setattr("api.main.TRAJECTORY_PATH", str(tmp_path / "traj.jsonl"))
    monkeypatch.setattr("api.main.DATA_DIR", str(tmp_path))
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


class TestReset:
    def test_reset_clears_session(self, env):
        post_turn(env)
        env["client"].post("/api/reset")
        assert env["client"].get("/api/profile").json()["total_turns"] == 0
