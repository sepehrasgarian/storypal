"""The flight recorder: every turn saved as one (state, action, reward) step.

Records are shaped the way an RL post-training framework consumes
rollouts: the signals compose the reward. A session is an episode; the
file is JSONL, one step per line.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from storypal.core.signals import Signal


@dataclass(frozen=True)
class TurnRecord:
    """One step of a tutoring episode."""

    session_id: str
    turn: int
    timestamp: float  # supplied by the caller so this module stays pure
    state: dict  # target sentence, profile snapshot, incoming signals
    action: dict  # tutor reply, tool calls, tts tags
    reward: dict = field(default_factory=dict)  # per-signal scores + composite
    route: str = "archive"  # triage decision for this turn


def reward_from_signals(signals: dict[str, Signal]) -> dict:
    """Flatten signal scores into a reward dict with a simple mean composite."""
    scores = {signal_id: s.score for signal_id, s in signals.items()}
    composite = sum(scores.values()) / len(scores) if scores else 0.0
    return {**scores, "composite": round(composite, 4)}


class TrajectoryLog:
    """Append-only JSONL log of turn records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: TurnRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def read_all(self) -> list[TurnRecord]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [TurnRecord(**json.loads(line)) for line in f if line.strip()]

    def episode(self, session_id: str) -> list[TurnRecord]:
        """All steps of one session, in turn order."""
        steps = [r for r in self.read_all() if r.session_id == session_id]
        return sorted(steps, key=lambda r: r.turn)
