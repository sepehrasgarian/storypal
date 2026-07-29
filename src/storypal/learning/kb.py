"""The two knowledge bases.

Content KB: target sentences tagged by level and the phonemes they
exercise, retrieved by the child's current needs. Keyword/tag lookup —
a vector store is unjustified at this corpus size.

Strategy KB: teaching tactics per phoneme, each with a scoreboard of
how often it actually worked for THIS child. Retrieval prefers tactics
with the best track record, so the KB itself learns from outcomes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from storypal.config import STORIES, Story


@dataclass(frozen=True)
class Tactic:
    name: str
    phoneme: str
    instructions: str  # what the tutor should do
    example_words: tuple[str, ...]


TACTICS: list[Tactic] = [
    Tactic(
        "slow_demonstration", "th",
        "Say the word very slowly, stretching the 'th': tongue peeks between the teeth.",
        ("three", "throw", "through"),
    ),
    Tactic(
        "minimal_pairs", "th",
        "Contrast the sound with a similar one the child already knows: tree/three, tank/thank.",
        ("tree", "three", "tank", "thank"),
    ),
    Tactic(
        "slow_demonstration", "r",
        "Model the 'r' slowly with a small growl sound, then blend it into the word.",
        ("red", "run", "roar"),
    ),
    Tactic(
        "word_families", "ch",
        "Practice a family of words sharing the sound: chip, chat, chin.",
        ("chip", "chat", "chin"),
    ),
    Tactic(
        "word_families", "sh",
        "Practice a family of words sharing the sound: ship, shop, shine.",
        ("ship", "shop", "shine"),
    ),
]


# --- Content KB ---------------------------------------------------------

def next_sentence(
    level: int,
    focus_phoneme: str | None = None,
    exclude: set[str] | None = None,
) -> Story | None:
    """Pick the next target sentence.

    Prefers sentences at the requested level exercising the focus
    phoneme; falls back to any sentence at the level, then any at all.
    """
    exclude = exclude or set()
    candidates = [s for s in STORIES if s.text not in exclude]
    if not candidates:
        return None

    def rank(story: Story) -> tuple:
        return (
            story.level == level,  # right difficulty first
            focus_phoneme in story.phonemes if focus_phoneme else False,
        )

    return max(candidates, key=rank)


# --- Strategy KB --------------------------------------------------------

class TacticStats:
    """Persistent scoreboard: how often each tactic was used and worked."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stats: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _entry(self, tactic: Tactic) -> dict:
        key = f"{tactic.phoneme}:{tactic.name}"
        return self._stats.setdefault(key, {"used": 0, "worked": 0})

    def record_usage(self, tactic: Tactic) -> None:
        self._entry(tactic)["used"] += 1
        self._save()

    def record_outcome(self, tactic: Tactic, worked: bool) -> None:
        if worked:
            self._entry(tactic)["worked"] += 1
        self._save()

    def success_rate(self, tactic: Tactic) -> float:
        """Laplace-smoothed so untried tactics start at 0.5, not 0."""
        entry = self._entry(tactic)
        return (entry["worked"] + 1) / (entry["used"] + 2)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._stats, indent=2))


def best_tactic(phoneme: str, stats: TacticStats) -> Tactic | None:
    """The tactic with the best track record for this child and sound."""
    candidates = [t for t in TACTICS if t.phoneme == phoneme]
    if not candidates:
        return None
    return max(candidates, key=stats.success_rate)
