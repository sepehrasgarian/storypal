"""Turn-level decision logic, independent of HTTP.

Given what was heard, decide how to grade it (full read, drill
follow-up, or conversation), whether the session advances, and what to
expect next turn. Pure functions over small dataclasses so every
branch is directly testable — main.py only wires them to routes.
"""

from dataclasses import dataclass

from storypal.config import (
    AUTO_ADVANCE_ACCURACY, DRILL_FULL_MISMATCH, DRILL_MATCH_ACCURACY,
)
from storypal.core.assessment import Assessment, WordStatus, assess, is_conversational
from storypal.core.signals import AsrTelemetry, Signal, s1_reading_accuracy, s2_asr_reliability

PROBLEM_STATUSES = (WordStatus.MISSED, WordStatus.NEAR_MISS, WordStatus.SUBSTITUTED)


@dataclass(frozen=True)
class GradedTurn:
    """Everything the grading pre-loop decided about one recording."""

    graded_target: str  # what the child was actually scored against
    assessment: Assessment
    s1: Signal
    s2: Signal
    chat_turn: bool  # the child was talking TO the tutor
    drill_words: list[str] | None  # set when a drill response was graded

    @property
    def drill_worked(self) -> bool:
        return self.drill_words is not None and self.s1.score >= DRILL_MATCH_ACCURACY

    @property
    def accepted(self) -> bool:
        """A trusted, accurate full read: the session may move on."""
        return (
            not self.chat_turn
            and self.drill_words is None
            and self.s2.reliable
            and self.s1.score >= AUTO_ADVANCE_ACCURACY
        )


def grade_turn(
    target: str,
    transcript: str,
    telemetry: AsrTelemetry,
    pending_drill: list[str] | None,
) -> GradedTurn:
    """Grade one recording in the right frame of reference.

    Priority: conversation is answered, not graded; a reply matching
    the pending drill words (but not the sentence) is graded as a
    drill; everything else is graded against the full sentence.
    """
    assessment = assess(target, transcript)
    graded_target = target
    chat_turn = is_conversational(transcript, assessment)

    drill_words = None
    if not chat_turn and pending_drill and assessment.accuracy < DRILL_FULL_MISMATCH:
        mini_target = " ".join(pending_drill)
        mini = assess(mini_target, transcript)
        if mini.accuracy >= DRILL_MATCH_ACCURACY:
            drill_words = list(pending_drill)
            assessment = mini
            graded_target = mini_target

    if chat_turn:
        s1 = Signal(
            id="S1", score=assessment.accuracy, reliable=True,
            reasons=("conversational reply, not a reading attempt - not graded",),
        )
    else:
        s1 = s1_reading_accuracy(assessment)
    s2 = s2_asr_reliability(telemetry, assessment)

    return GradedTurn(graded_target, assessment, s1, s2, chat_turn, drill_words)


def problem_words(assessment: Assessment) -> list[str]:
    """Target words the child struggled with, in sentence order."""
    return [v.target_word for v in assessment.verdicts if v.status in PROBLEM_STATUSES]


def update_expectations(
    graded: GradedTurn,
    tactic_used,
    current_drill: list[str] | None,
    current_tactic,
) -> tuple[list[str] | None, object]:
    """What should the next turn expect? Returns (pending_drill, tactic).

    A flawed but trusted full read sets up a drill (and remembers the
    tactic used, so its outcome can be scored). A successful drill or
    an accepted read clears both. Chat and unreliable turns change
    nothing - the child still owes the same reading.
    """
    if not graded.s2.reliable or graded.chat_turn:
        return current_drill, current_tactic

    if graded.drill_words is None:
        if 0 < graded.s1.score < AUTO_ADVANCE_ACCURACY:
            return problem_words(graded.assessment) or None, tactic_used
        if graded.s1.score >= AUTO_ADVANCE_ACCURACY:
            return None, None
        return current_drill, current_tactic

    if graded.drill_worked:
        return None, None
    return current_drill, current_tactic
