"""Tests for the turn-level decision logic, now directly testable
without HTTP."""

from storypal.core.signals import AsrTelemetry
from storypal.session import grade_turn, problem_words, update_expectations

TARGET = "The cat sat on the mat."
CONFIDENT = AsrTelemetry(avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2)


def graded(transcript, pending=None, telemetry=CONFIDENT):
    return grade_turn(TARGET, transcript, telemetry, pending)


class TestGradeTurn:
    def test_full_read_frames(self):
        g = graded("the cat sat on the mat")
        assert g.graded_target == TARGET
        assert g.accepted
        assert not g.chat_turn and g.drill_words is None

    def test_chat_frame(self):
        g = graded("Yes, I do.")
        assert g.chat_turn
        assert not g.accepted
        assert "not graded" in g.s1.reasons[0]

    def test_drill_frame(self):
        g = graded("mat", pending=["mat"])
        assert g.drill_words == ["mat"]
        assert g.graded_target == "mat"
        assert g.drill_worked
        assert not g.accepted  # a drill never advances the sentence

    def test_unreliable_never_accepts(self):
        silence = AsrTelemetry(avg_logprob=-1.8, no_speech_prob=0.9, compression_ratio=1.1)
        g = graded("the cat sat on the mat", telemetry=silence)
        assert not g.accepted


class TestUpdateExpectations:
    def test_flawed_read_sets_up_drill(self):
        g = graded("the cat sat on the")
        pending, tactic = update_expectations(g, "TACTIC", None, None)
        assert pending == ["mat"]
        assert tactic == "TACTIC"

    def test_accepted_read_clears_everything(self):
        g = graded("the cat sat on the mat")
        assert update_expectations(g, None, ["mat"], "TACTIC") == (None, None)

    def test_successful_drill_clears_everything(self):
        g = graded("mat", pending=["mat"])
        assert update_expectations(g, None, ["mat"], "TACTIC") == (None, None)

    def test_failed_drill_keeps_expectations(self):
        g = graded("moo", pending=["mat"])  # near-miss counts as drill, low score
        pending, tactic = update_expectations(g, None, ["mat"], "TACTIC")
        # Whether graded as failed drill or failed full read, the child
        # still owes the same word.
        assert pending == ["mat"]

    def test_chat_changes_nothing(self):
        g = graded("Yes, I do.")
        assert update_expectations(g, None, ["mat"], "TACTIC") == (["mat"], "TACTIC")


class TestProblemWords:
    def test_orders_by_sentence_position(self):
        g = graded("cat on the mat")
        assert problem_words(g.assessment) == ["the", "sat"]
