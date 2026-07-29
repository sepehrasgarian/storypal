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
    """One way to teach one sound.

    Tactics are drawn from established reading instruction rather than
    invented: Orton-Gillingham articulatory cues (feel the sound in the
    mouth and throat), Elkonin sound boxes (segment, then blend),
    minimal pairs (contrast with the sound the child substitutes), word
    families (onset-rime), and gradual release ("I do, we do, you do").
    Each is adapted to an audio-only channel - nothing here needs paper,
    counters or a mirror, only the child's own voice and hands.
    """

    name: str
    phoneme: str
    instructions: str  # scripted guidance handed to the tutor
    example_words: tuple[str, ...]
    source: str = ""  # the method this comes from, for provenance


TACTICS: list[Tactic] = [
    # --- th ------------------------------------------------------------
    Tactic(
        "articulatory_cue", "th",
        "Tell them to put a hand on their throat and poke their tongue "
        "just between their teeth. 'thumb' is quiet, 'mother' buzzes - "
        "same tongue, different throat.",
        ("thumb", "this", "three"),
        source="Orton-Gillingham articulatory cue (voiced/unvoiced contrast)",
    ),
    Tactic(
        "minimal_pairs", "th",
        "Contrast it with the sound they swapped in: tree/three, "
        "tank/thank. Say both, ask which one has the tongue peeking out.",
        ("tree", "three", "tank", "thank"),
        source="minimal pairs (phonological contrast)",
    ),
    Tactic(
        "sound_boxes", "th",
        "Segment then blend: have them tap a finger for each sound - "
        "/th/ /i/ /s/ - then sweep the taps together into 'this'.",
        ("this", "that", "with"),
        source="Elkonin sound boxes, adapted to taps for an audio-only channel",
    ),
    # --- sh ------------------------------------------------------------
    Tactic(
        "articulatory_cue", "sh",
        "Lips pushed forward like blowing a kiss, teeth nearly shut, "
        "long quiet air: 'shhh' - the sound for asking a room to be quiet.",
        ("ship", "shop", "shed"),
        source="Orton-Gillingham articulatory cue",
    ),
    Tactic(
        "minimal_pairs", "sh",
        "Contrast with plain /s/: sip/ship, sell/shell. One hisses "
        "sharp, one is soft and wide.",
        ("sip", "ship", "sell", "shell"),
        source="minimal pairs (phonological contrast)",
    ),
    # --- ch ------------------------------------------------------------
    Tactic(
        "articulatory_cue", "ch",
        "A tiny sneeze, or a train pulling away: ch-ch-ch. Stop the air "
        "first, then let it burst.",
        ("chip", "chat", "chin"),
        source="Orton-Gillingham articulatory cue",
    ),
    Tactic(
        "word_families", "ch",
        "Practise a family that shares the sound so only the ending "
        "changes: chip, chat, chin.",
        ("chip", "chat", "chin"),
        source="onset-rime word families",
    ),
    # --- r -------------------------------------------------------------
    Tactic(
        "articulatory_cue", "r",
        "Growl like a small tiger: tongue pulled back, touching nothing, "
        "throat humming - rrr-ed.",
        ("red", "run", "rug"),
        source="Orton-Gillingham articulatory cue",
    ),
    Tactic(
        "minimal_pairs", "r",
        "Children often swap /w/ for /r/. Contrast them: run/won, "
        "red/wed - lips round for /w/, tongue back for /r/.",
        ("run", "won", "red", "wed"),
        source="minimal pairs (targets the common r/w substitution)",
    ),
    # --- s -------------------------------------------------------------
    Tactic(
        "articulatory_cue", "s",
        "A snake sound: teeth almost closed, tongue behind them, thin "
        "air hissing out - sss-un. No buzzing in the throat.",
        ("sun", "sit", "six"),
        source="Orton-Gillingham articulatory cue",
    ),
    Tactic(
        "word_families", "s",
        "Practise a family sharing the sound: sun, sit, sad, six.",
        ("sun", "sit", "sad", "six"),
        source="onset-rime word families",
    ),
    # --- t -------------------------------------------------------------
    Tactic(
        "articulatory_cue", "t",
        "Tongue taps the ridge behind the top teeth and springs off - "
        "quick, crisp, no voice: t-t-top.",
        ("top", "ten", "tap"),
        source="Orton-Gillingham articulatory cue",
    ),
    Tactic(
        "gradual_release", "t",
        "I do, we do, you do: say the word yourself first, then invite "
        "them to say it with you, then let them say it alone.",
        ("top", "ten", "tap"),
        source="gradual release of responsibility (explicit instruction)",
    ),
    # --- d -------------------------------------------------------------
    Tactic(
        "articulatory_cue", "d",
        "The same tongue tap as /t/, but with a hand on the throat to "
        "feel it hum: d-d-dog. /t/ is silent there, /d/ buzzes.",
        ("dog", "dad", "dig"),
        source="Orton-Gillingham articulatory cue (voiced/unvoiced contrast)",
    ),
    Tactic(
        "minimal_pairs", "d",
        "Contrast with /t/: tap/dab, ten/den. Same mouth, one hums.",
        ("tap", "dab", "ten", "den"),
        source="minimal pairs (phonological contrast)",
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
