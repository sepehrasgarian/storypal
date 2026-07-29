"""Tier 2 memory: the learner profile.

A small JSON document accumulated across turns and sessions — weak
phonemes, missed words, level. It is the only thing that survives a
restart. Turns flagged unreliable by S2 never update it, so
recognition artifacts cannot poison memory.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from storypal.config import TRACKED_PHONEMES, phonemes_in_word  # noqa: F401 (re-exported)
from storypal.core.assessment import Assessment, WordStatus
from storypal.core.signals import Signal


# Misses are counted against opportunities, and a raw count is not the
# same thing as a difficulty. "the" appears in nearly every sentence, so
# by volume it will always look like the hardest word in English. The
# smoothing constant keeps a single miss on a rarely seen word from
# jumping to the top of the list.
SMOOTHING = 2


@dataclass
class Profile:
    level: int = 1
    total_turns: int = 0
    weak_phonemes: dict = field(default_factory=dict)  # phoneme -> miss count
    missed_words: dict = field(default_factory=dict)  # word -> miss count
    phoneme_attempts: dict = field(default_factory=dict)  # phoneme -> times presented
    word_attempts: dict = field(default_factory=dict)  # word -> times presented


def miss_rate(misses: int, attempts: int) -> float:
    """Share of opportunities missed, smoothed so thin evidence ranks low."""
    return misses / (attempts + SMOOTHING)


def _opportunities(misses: int, attempts: dict, key: str) -> int:
    """Attempts can never be fewer than misses. A profile written before
    attempts were tracked has the misses but not the denominator, which
    would otherwise render as "missed 12 of 4"."""
    return max(attempts.get(key, 0), misses)


def _rank(counts: dict, attempts: dict) -> list[tuple[str, float, int, int]]:
    rows = [
        (key, miss_rate(n, _opportunities(n, attempts, key)), n,
         _opportunities(n, attempts, key))
        for key, n in counts.items()
    ]
    return sorted(rows, key=lambda r: -r[1])


def ranked_phonemes(profile: Profile) -> list[tuple[str, float, int, int]]:
    """(phoneme, rate, misses, attempts), hardest first."""
    return _rank(profile.weak_phonemes, profile.phoneme_attempts)


def ranked_words(profile: Profile) -> list[tuple[str, float, int, int]]:
    """(word, rate, misses, attempts), hardest first."""
    return _rank(profile.missed_words, profile.word_attempts)


def update_from_turn(profile: Profile, assessment: Assessment, s2: Signal) -> Profile:
    """Fold one turn into the profile. Unreliable turns are skipped entirely:
    we refuse to remember things we are not sure we heard."""
    if not s2.reliable:
        return profile

    profile.total_turns += 1

    # Every target word presented this turn is an opportunity, whether or
    # not the child missed it. Without this denominator the profile
    # measures how often a word appears, not how hard it is.
    for verdict in assessment.verdicts:
        word = verdict.target_word
        if word is None:
            continue
        profile.word_attempts[word] = profile.word_attempts.get(word, 0) + 1
        for phoneme in phonemes_in_word(word):
            profile.phoneme_attempts[phoneme] = profile.phoneme_attempts.get(phoneme, 0) + 1

    troubled = (
        assessment.words_with_status(WordStatus.MISSED)
        + assessment.words_with_status(WordStatus.NEAR_MISS)
        + assessment.words_with_status(WordStatus.SUBSTITUTED)
    )
    for verdict in troubled:
        word = verdict.target_word
        profile.missed_words[word] = profile.missed_words.get(word, 0) + 1
        for phoneme in phonemes_in_word(word):
            profile.weak_phonemes[phoneme] = profile.weak_phonemes.get(phoneme, 0) + 1
    return profile


def weakest_phoneme(profile: Profile) -> str | None:
    """The sound missed most often relative to how often it came up."""
    ranked = ranked_phonemes(profile)
    return ranked[0][0] if ranked else None


def render(profile: Profile) -> str:
    """The profile as prompt text for the start of a session."""
    if profile.total_turns == 0:
        return "This is a new learner: no history yet. Start gently at level 1."
    lines = [f"Learner history ({profile.total_turns} turns, level {profile.level}):"]
    sounds = ranked_phonemes(profile)[:4]
    if sounds:
        rendered = ", ".join(f"'{p}' (missed {n} of {a})" for p, _, n, a in sounds)
        lines.append(f"- struggles with sounds: {rendered}")
    words = ranked_words(profile)[:5]
    if words:
        rendered = ", ".join(f"{w} ({n}/{a})" for w, _, n, a in words)
        lines.append(f"- hardest words so far: {rendered}")
    return "\n".join(lines)


def load(path: str | Path) -> Profile:
    path = Path(path)
    if not path.exists():
        return Profile()
    return Profile(**json.loads(path.read_text()))


def save(profile: Profile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2))
