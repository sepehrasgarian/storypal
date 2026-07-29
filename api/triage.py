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

from api.config import JUDGE_FAIL_THRESHOLD
from api.signals import Signal


class Route(Enum):
    REVIEW_QUEUE = "review_queue"
    FINETUNE_SET = "finetune_set"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    reason: str


Signals = Mapping[str, Signal]  # e.g. {"S1": ..., "S2": ...}; S3/S4 arrive later


def _asr_unreliable(signals: Signals) -> bool:
    s2 = signals.get("S2")
    return s2 is not None and not s2.reliable


def _judge_failed(signal_id: str) -> Callable[[Signals], bool]:
    def check(signals: Signals) -> bool:
        judge = signals.get(signal_id)
        return judge is not None and judge.score < JUDGE_FAIL_THRESHOLD
    return check


# The policy. Order matters: first match wins.
RULES: list[tuple[str, Callable[[Signals], bool], Route]] = [
    ("ASR unreliable: assessment cannot be trusted", _asr_unreliable, Route.REVIEW_QUEUE),
    ("harmful or off-target correction (S4)", _judge_failed("S4"), Route.FINETUNE_SET),
    ("tutor claim not grounded in the assessment (S3)", _judge_failed("S3"), Route.FINETUNE_SET),
]

DEFAULT = RoutingDecision(Route.ARCHIVE, "normal turn")


def route_turn(signals: Signals) -> RoutingDecision:
    """Decide which pile this turn's record belongs in."""
    for description, applies, route in RULES:
        if applies(signals):
            return RoutingDecision(route, description)
    return DEFAULT
