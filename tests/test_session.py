"""Tests for the turn-level decision logic, now directly testable
without HTTP."""

from storypal.core.signals import AsrTelemetry
from storypal.session import grade_turn, problem_words, update_expectations

TARGET = "The sun is hot."
CONFIDENT = AsrTelemetry(avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2)


def graded(transcript, pending=None, telemetry=CONFIDENT):
    return grade_turn(TARGET, transcript, telemetry, pending)


class TestGradeTurn:
    def test_full_read_frames(self):
        g = graded("the sun is hot")
        assert g.graded_target == TARGET
        assert g.accepted
        assert not g.chat_turn and g.drill_words is None

    def test_chat_frame(self):
        g = graded("Yes, I do.")
        assert g.chat_turn
        assert not g.accepted
        assert "not graded" in g.s1.reasons[0]

    def test_drill_frame(self):
        g = graded("hot", pending=["hot"])
        assert g.drill_words == ["hot"]
        assert g.graded_target == "hot"
        assert g.drill_worked
        assert not g.accepted  # a drill never advances the sentence

    def test_unreliable_never_accepts(self):
        silence = AsrTelemetry(avg_logprob=-1.8, no_speech_prob=0.9, compression_ratio=1.1)
        g = graded("the sun is hot", telemetry=silence)
        assert not g.accepted


class TestUpdateExpectations:
    def test_flawed_read_sets_up_drill(self):
        g = graded("the sun is")
        pending, tactic = update_expectations(g, "TACTIC", None, None)
        assert pending == ["hot"]
        assert tactic == "TACTIC"

    def test_accepted_read_clears_everything(self):
        g = graded("the sun is hot")
        assert update_expectations(g, None, ["hot"], "TACTIC") == (None, None)

    def test_successful_drill_clears_everything(self):
        g = graded("hot", pending=["hot"])
        assert update_expectations(g, None, ["hot"], "TACTIC") == (None, None)

    def test_failed_drill_keeps_expectations(self):
        g = graded("zzz", pending=["hot"])  # matches neither the word nor the sentence
        pending, tactic = update_expectations(g, None, ["hot"], "TACTIC")
        # Whether graded as failed drill or failed full read, the child
        # still owes the same word.
        assert pending == ["hot"]

    def test_chat_changes_nothing(self):
        g = graded("Yes, I do.")
        assert update_expectations(g, None, ["hot"], "TACTIC") == (["hot"], "TACTIC")


class TestProblemWords:
    def test_orders_by_sentence_position(self):
        g = graded("sun is")
        assert problem_words(g.assessment) == ["the", "hot"]


class TestAdversarialRegressions:
    """Both of these shipped broken and were caught by writing cases
    designed to break the system rather than to pass."""

    def test_invented_tail_is_not_a_perfect_read(self):
        # Was: S1 1.00, trusted, and the child advanced on a hallucination.
        g = graded("the sun is hot thanks for watching")
        assert not g.s2.reliable
        assert not g.accepted

    def test_scattered_self_talk_is_still_trusted(self):
        # The precision side of the same rule: children narrate themselves.
        g = graded("um the sun is hot i did it")
        assert g.s2.reliable

    def test_failed_drill_is_graded_as_a_drill_not_the_sentence(self):
        # Was: saying "fun" while drilling "sun" scored 0.12 against the
        # whole sentence and blamed words nobody asked the child to say.
        g = graded("fun", pending=["sun"])
        assert g.drill_words == ["sun"]
        assert g.graded_target == "sun"
        assert not g.drill_worked
