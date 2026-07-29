"""Routing turns into the curated data piles.

Policy lives in one declarative table, separate from measurement, so
thresholds and destinations can change without touching any scorer.
First matching rule wins.

Routes:
- review_queue: a human should look (the assessment itself is suspect)
- finetune_set: the tutor replied badly; worth training on
- archive:      normal turn, kept for completeness
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from storypal.config import JUDGE_FAIL_THRESHOLD
from storypal.core.signals import Signal


class Route(Enum):
    REVIEW_QUEUE = "review_queue"
    FINETUNE_SET = "finetune_set"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    reason: str


Signals = Mapping[str, Signal]  # e.g. {"S1": ..., "S2": ...}; S3/S4 arrive later


@dataclass(frozen=True)
class TurnContext:
    """What routing decides from: the signals, plus what kind of turn
    this was. Conversation and reading fail in different ways."""

    signals: Signals
    chat_turn: bool = False  # the child answered the tutor instead of reading


def _asr_unreliable(ctx: TurnContext) -> bool:
    # Conversation trips the novelty check by design — chat words match
    # nothing in the target sentence. That is not a recognition failure,
    # and there is no reading assessment to doubt, so no human review.
    if ctx.chat_turn:
        return False
    s2 = ctx.signals.get("S2")
    return s2 is not None and not s2.reliable


def _judge_failed(signal_id: str) -> Callable[[TurnContext], bool]:
    def check(ctx: TurnContext) -> bool:
        judge = ctx.signals.get(signal_id)
        return judge is not None and judge.score < JUDGE_FAIL_THRESHOLD
    return check


# The policy. Order matters: first match wins. Judge rules apply to
# every turn — replying badly to a child's "hello" is still worth
# learning from.
RULES: list[tuple[str, Callable[[TurnContext], bool], Route]] = [
    ("ASR unreliable: assessment cannot be trusted", _asr_unreliable, Route.REVIEW_QUEUE),
    ("harmful or off-target correction (S4)", _judge_failed("S4"), Route.FINETUNE_SET),
    ("tutor claim not grounded in the assessment (S3)", _judge_failed("S3"), Route.FINETUNE_SET),
]

DEFAULT = RoutingDecision(Route.ARCHIVE, "normal turn")


def route_turn(signals: Signals, chat_turn: bool = False) -> RoutingDecision:
    """Decide which pile this turn's record belongs in."""
    ctx = TurnContext(signals=signals, chat_turn=chat_turn)
    for description, applies, route in RULES:
        if applies(ctx):
            return RoutingDecision(route, description)
    return DEFAULT
