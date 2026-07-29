"""Word-level alignment of an ASR transcript against the target sentence.

The target sentence is known exactly, so grading is deterministic:
normalize both sides, align them word by word, and classify each
difference. No AI involved.
"""

import re
from dataclasses import dataclass
from enum import Enum

from storypal.config import (
    NEAR_MISS_MAX_EDIT_DISTANCE,
    SHORT_WORD_LEN,
    SHORT_WORD_MAX_EDIT_DISTANCE,
)

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


class WordStatus(Enum):
    CORRECT = "correct"
    NEAR_MISS = "near_miss"  # heard word is close to the target: likely mispronounced
    SUBSTITUTED = "substituted"  # heard a clearly different word
    MISSED = "missed"  # target word not read
    ADDED = "added"  # heard word with no counterpart in the target


@dataclass(frozen=True)
class WordVerdict:
    """The outcome for one position in the alignment."""

    status: WordStatus
    target_word: str | None  # None for ADDED
    heard_word: str | None  # None for MISSED


@dataclass(frozen=True)
class Assessment:
    target: str
    transcript: str
    verdicts: tuple[WordVerdict, ...]

    @property
    def accuracy(self) -> float:
        """Fraction of target words read correctly (near-misses count half)."""
        target_positions = [v for v in self.verdicts if v.target_word is not None]
        if not target_positions:
            return 0.0
        score = sum(
            1.0 if v.status is WordStatus.CORRECT
            else 0.5 if v.status is WordStatus.NEAR_MISS
            else 0.0
            for v in target_positions
        )
        return score / len(target_positions)

    def words_with_status(self, status: WordStatus) -> list[WordVerdict]:
        return [v for v in self.verdicts if v.status is status]


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, spell out digits: '2 Birds!' -> ['two', 'birds']."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [_DIGIT_WORDS.get(w, w) for w in words]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two words."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,  # delete
                current[j - 1] + 1,  # insert
                previous[j - 1] + (ca != cb),  # substitute
            ))
        previous = current
    return previous[-1]


def is_near_miss(target_word: str, heard_word: str) -> bool:
    """Close enough in spelling to be a mispronunciation of the target.

    Short words use a tighter threshold: at distance 2 almost every
    3-letter word resembles some other word.
    """
    limit = (
        SHORT_WORD_MAX_EDIT_DISTANCE
        if min(len(target_word), len(heard_word)) <= SHORT_WORD_LEN
        else NEAR_MISS_MAX_EDIT_DISTANCE
    )
    return edit_distance(target_word, heard_word) <= limit


def assess(target: str, transcript: str) -> Assessment:
    """Grade a transcript against the target sentence."""
    target_words = normalize(target)
    heard_words = normalize(transcript)
    ops = _align(target_words, heard_words)
    verdicts = tuple(_classify(op, t, h) for op, t, h in ops)
    return Assessment(target=target, transcript=transcript, verdicts=verdicts)


def _classify(op: str, target_word: str | None, heard_word: str | None) -> WordVerdict:
    if op == "match":
        return WordVerdict(WordStatus.CORRECT, target_word, heard_word)
    if op == "delete":
        return WordVerdict(WordStatus.MISSED, target_word, None)
    if op == "insert":
        return WordVerdict(WordStatus.ADDED, None, heard_word)
    # Substitution: a close spelling suggests a mispronunciation of the
    # right word rather than a different word entirely.
    status = WordStatus.NEAR_MISS if is_near_miss(target_word, heard_word) else WordStatus.SUBSTITUTED
    return WordVerdict(status, target_word, heard_word)


def _substitution_cost(target_word: str, heard_word: str) -> float:
    """Cheaper to pair similar words, so 'floo' aligns with 'flew'
    (a near-miss) rather than with an unrelated target word."""
    if target_word == heard_word:
        return 0.0
    if is_near_miss(target_word, heard_word):
        return 0.5
    return 1.0


def _align(target: list[str], heard: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Minimal-edit alignment of the two word lists.

    Returns (op, target_word, heard_word) triples where op is one of
    match / substitute / delete (missed) / insert (added).
    """
    rows, cols = len(target) + 1, len(heard) + 1
    cost = [[0.0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = float(i)
    for j in range(1, cols):
        cost[0][j] = float(j)
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(
                cost[i - 1][j] + 1,  # delete target word (missed)
                cost[i][j - 1] + 1,  # insert heard word (added)
                cost[i - 1][j - 1] + _substitution_cost(target[i - 1], heard[j - 1]),
            )

    # Walk back from the corner to recover the operations.
    # Costs are multiples of 0.5, which floats represent exactly, so
    # equality comparisons here are safe.
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = len(target), len(heard)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = _substitution_cost(target[i - 1], heard[j - 1])
            if cost[i][j] == cost[i - 1][j - 1] + sub:
                op = "match" if sub == 0.0 else "substitute"
                ops.append((op, target[i - 1], heard[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("delete", target[i - 1], None))
            i -= 1
        else:
            ops.append(("insert", None, heard[j - 1]))
            j -= 1
    ops.reverse()
    return ops
