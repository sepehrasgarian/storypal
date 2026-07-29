"""Tier 2 memory: the learner profile.

A small JSON document accumulated across turns and sessions — weak
phonemes, missed words, level. It is the only thing that survives a
restart. Turns flagged unreliable by S2 never update it, so
recognition artifacts cannot poison memory.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from api.assessment import Assessment, WordStatus
from api.signals import Signal

# Sounds we track, longest first so 'th' wins over 't' when scanning a word.
TRACKED_PHONEMES = ("th", "ch", "sh", "r", "s", "t", "d")


@dataclass
class Profile:
    level: int = 1
    total_turns: int = 0
    weak_phonemes: dict = field(default_factory=dict)  # phoneme -> miss count
    missed_words: dict = field(default_factory=dict)  # word -> miss count


def phonemes_in_word(word: str) -> list[str]:
    """The tracked sounds a word exercises: 'through' -> ['th', 'r']."""
    found = []
    remaining = word
    for phoneme in TRACKED_PHONEMES:
        if phoneme in remaining:
            found.append(phoneme)
            remaining = remaining.replace(phoneme, "")
    return found


def update_from_turn(profile: Profile, assessment: Assessment, s2: Signal) -> Profile:
    """Fold one turn into the profile. Unreliable turns are skipped entirely:
    we refuse to remember things we are not sure we heard."""
    if not s2.reliable:
        return profile

    profile.total_turns += 1
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
    """The sound with the most recorded misses, if any."""
    if not profile.weak_phonemes:
        return None
    return max(profile.weak_phonemes, key=profile.weak_phonemes.get)


def render(profile: Profile) -> str:
    """The profile as prompt text for the start of a session."""
    if profile.total_turns == 0:
        return "This is a new learner: no history yet. Start gently at level 1."
    lines = [f"Learner history ({profile.total_turns} turns, level {profile.level}):"]
    if profile.weak_phonemes:
        sounds = ", ".join(
            f"'{p}' ({n} misses)"
            for p, n in sorted(profile.weak_phonemes.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- struggles with sounds: {sounds}")
    if profile.missed_words:
        words = ", ".join(sorted(profile.missed_words, key=profile.missed_words.get, reverse=True)[:5])
        lines.append(f"- hardest words so far: {words}")
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
