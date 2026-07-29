"""Session-level guarantees, checked by simulating whole sessions with
different children. These are the promises that only break over many
turns, so single-turn tests cannot catch them."""

import pytest

from eval.personas import PERSONAS
from eval.simulate import run_session

SESSIONS = {p.name: run_session(p) for p in PERSONAS}


@pytest.mark.parametrize("name", list(SESSIONS))
def test_no_child_is_ever_falsely_corrected(name):
    """The harm the architecture exists to prevent: telling a child who
    read the sentence perfectly that they got it wrong."""
    assert SESSIONS[name].false_corrections == 0


@pytest.mark.parametrize("name", list(SESSIONS))
def test_no_child_is_falsely_praised(name):
    assert SESSIONS[name].false_praise == 0


class TestMisheardChild:
    """Reads correctly, is mangled by the recogniser every time."""

    def test_their_turns_are_discarded_not_graded(self):
        session = SESSIONS["tired_mumbler"]
        assert session.discarded > session.graded

    def test_their_progress_stalls_which_is_the_honest_cost(self):
        # Documented, not accidental: perfect safety buys zero progress.
        # The fix is a better acoustic model, not a looser threshold.
        assert SESSIONS["tired_mumbler"].advances == 0


class TestStrugglingChild:
    def test_a_real_weakness_is_identified_quickly(self):
        session = SESSIONS["th_struggler"]
        assert session.detected_at is not None
        assert session.detected_at <= 3

    def test_their_misses_are_graded_not_discarded(self):
        # Real mistakes must not look like recogniser failures, or the
        # tutor would never teach anything.
        assert SESSIONS["th_struggler"].graded > 0


class TestChattyAndFrustratedChildren:
    def test_talking_back_is_answered_not_scored(self):
        assert SESSIONS["chatterbox"].chat > 0

    def test_frustration_and_silence_never_become_reading_scores(self):
        session = SESSIONS["frustrated_child"]
        assert session.chat + session.discarded > session.graded


class TestConfidentChild:
    def test_a_good_reader_makes_progress(self):
        assert SESSIONS["confident_reader"].advances > 0
