"""Tests for the Tier 1 prompt builder.

The key property: the same inputs always produce the same prompt, and
the reliable/unreliable versions differ exactly as designed.
"""

from api.assessment import assess
from api.kb import TacticStats, best_tactic
from api.profile import Profile
from api.prompt import build_prompt
from api.signals import AsrTelemetry, s1_reading_accuracy, s2_asr_reliability

TARGET = "The bird flew through the trees."
CONFIDENT = AsrTelemetry(avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.3)
HALLUCINATING = AsrTelemetry(avg_logprob=-1.5, no_speech_prob=0.9, compression_ratio=1.1)


def prompt_for(transcript, telemetry=CONFIDENT, tactic=None):
    assessment = assess(TARGET, transcript)
    return build_prompt(
        target=TARGET,
        assessment=assessment,
        s1=s1_reading_accuracy(assessment),
        s2=s2_asr_reliability(telemetry, assessment),
        profile=Profile(),
        tactic=tactic,
    )


class TestReliableTurns:
    def test_perfect_read_celebrates(self):
        prompt = prompt_for("the bird flew through the trees")
        assert "celebrate" in prompt
        assert "ASR: reliable" in prompt

    def test_missed_word_is_drilled(self):
        prompt = prompt_for("the bird flew the trees")
        assert "'through'" in prompt
        assert "praise what went right" in prompt

    def test_tactic_is_included_when_given(self, tmp_path):
        tactic = best_tactic("th", TacticStats(tmp_path / "t.json"))
        prompt = prompt_for("the bird flew the trees", tactic=tactic)
        assert tactic.instructions in prompt


class TestUnreliableTurns:
    def test_hallucination_flips_the_instructions(self):
        prompt = prompt_for("thanks for watching", telemetry=HALLUCINATING)
        assert "do NOT correct" in prompt
        assert "read the sentence one more time" in prompt
        # It must not leak the hallucinated words as things to drill.
        assert "praise what went right" not in prompt

    def test_reliable_and_unreliable_prompts_differ_only_in_the_live_section(self):
        reliable = prompt_for("the bird flew through the trees")
        unreliable = prompt_for("thanks for watching", telemetry=HALLUCINATING)
        # Persona and profile sections are shared.
        assert reliable.split("-- this turn")[0] == unreliable.split("-- this turn")[0]


class TestReproducibility:
    def test_same_inputs_same_prompt(self):
        assert prompt_for("the bird flew the trees") == prompt_for("the bird flew the trees")
