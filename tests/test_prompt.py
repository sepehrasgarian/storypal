"""Tests for the Tier 1 prompt builder.

The key property: the same inputs always produce the same prompt, and
the reliable/unreliable versions differ exactly as designed.
"""

from storypal.core.assessment import assess
from storypal.learning.kb import TacticStats, best_tactic
from storypal.learning.profile import Profile
from storypal.learning.prompt import build_prompt
from storypal.core.signals import AsrTelemetry, s1_reading_accuracy, s2_asr_reliability

TARGET = "The bird flew through the trees."
CONFIDENT = AsrTelemetry(avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.3)
HALLUCINATING = AsrTelemetry(avg_logprob=-1.5, no_speech_prob=0.9, compression_ratio=1.1)


def prompt_for(transcript, telemetry=CONFIDENT, tactic=None, attempts=1, streak=0, level=1):
    assessment = assess(TARGET, transcript)
    return build_prompt(
        target=TARGET,
        assessment=assessment,
        s1=s1_reading_accuracy(assessment),
        s2=s2_asr_reliability(telemetry, assessment),
        profile=Profile(level=level),
        tactic=tactic,
        attempts=attempts,
        streak=streak,
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


class TestToolGuidance:
    """Tools only get called when the prompt shows concrete grounds.
    Real logs had 36 turns with drill_sound never once invoked."""

    def test_drill_tool_is_always_offered_on_a_teaching_turn(self):
        assert "drill_sound" in prompt_for("the bird flew the trees")

    def test_streak_prompts_a_level_up(self):
        prompt = prompt_for("the bird flew the trees", streak=3, level=1)
        assert "set_difficulty(level=2" in prompt

    def test_repeated_attempts_prompt_easing_off_and_a_human_flag(self):
        prompt = prompt_for("the bird flew the trees", attempts=3, level=2)
        assert "set_difficulty(level=1" in prompt
        assert "flag_for_review" in prompt

    def test_untrusted_turn_offers_no_tools_at_all(self):
        # Perception failed: nothing to teach, nothing to decide.
        prompt = prompt_for("thanks for watching", telemetry=HALLUCINATING)
        assert "drill_sound" not in prompt
        assert "set_difficulty" not in prompt


class TestReproducibility:
    def test_same_inputs_same_prompt(self):
        assert prompt_for("the bird flew the trees") == prompt_for("the bird flew the trees")
