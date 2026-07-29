"""The deterministic signals computed on every turn.

S1: how well did the child read? (from the assessment)
S2: can we trust our own ears?   (from ASR telemetry + the assessment)

Each signal is a pure function returning (score, reason) so it can be
tested alone and swapped without touching anything else. S2 gates S1:
when the recognizer is unreliable, the assessment must not be acted on.
"""

from dataclasses import dataclass, field

from storypal.core.assessment import Assessment, WordStatus, is_near_miss, normalize
from storypal.config import (
    ASR_AVG_LOGPROB_UNRELIABLE,
    ASR_COMPRESSION_RATIO_UNRELIABLE,
    ASR_NO_SPEECH_UNRELIABLE,
    ASR_NOVEL_RUN_UNRELIABLE,
    ASR_NOVEL_WORD_RATIO_UNRELIABLE,
    FUNCTION_WORDS,
)


@dataclass(frozen=True)
class AsrTelemetry:
    """Confidence numbers the recognizer reports alongside the transcript."""

    avg_logprob: float = 0.0  # closer to 0 = more confident
    no_speech_prob: float = 0.0  # probability the audio was silence
    compression_ratio: float = 1.0  # high = repetitive, a hallucination signature


@dataclass(frozen=True)
class Signal:
    id: str
    score: float  # 0..1; for S2, 1.0 = fully trustworthy
    reliable: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def s1_reading_accuracy(assessment: Assessment) -> Signal:
    """How accurately the target sentence was read."""
    reasons = []
    for verdict in assessment.verdicts:
        if verdict.status is WordStatus.MISSED:
            reasons.append(f"missed '{verdict.target_word}'")
        elif verdict.status is WordStatus.NEAR_MISS:
            reasons.append(f"mispronounced '{verdict.target_word}' as '{verdict.heard_word}'")
        elif verdict.status is WordStatus.SUBSTITUTED:
            reasons.append(f"read '{verdict.heard_word}' instead of '{verdict.target_word}'")
    accuracy = assessment.accuracy
    return Signal(
        id="S1",
        score=accuracy,
        reliable=True,  # trustworthiness is S2's call, not S1's
        reasons=tuple(reasons) if reasons else ("perfect read",),
    )


def s2_asr_reliability(telemetry: AsrTelemetry, assessment: Assessment) -> Signal:
    """Whether the recognizer's output can be trusted at all.

    Two independent checks:
    1. The recognizer's own confidence numbers (hallucination signatures).
    2. Target-anchored novelty: heard content that matches nothing in the
       target sentence. A real misreading is a near-miss of an expected
       word; fabrication is coherent text from nowhere.
    """
    reasons = []

    if telemetry.no_speech_prob > ASR_NO_SPEECH_UNRELIABLE:
        reasons.append(f"audio was probably silence (no_speech_prob={telemetry.no_speech_prob:.2f})")
    if telemetry.avg_logprob < ASR_AVG_LOGPROB_UNRELIABLE:
        reasons.append(f"low decode confidence (avg_logprob={telemetry.avg_logprob:.2f})")
    if telemetry.compression_ratio > ASR_COMPRESSION_RATIO_UNRELIABLE:
        reasons.append(f"repetitive output (compression_ratio={telemetry.compression_ratio:.2f})")

    novel_count, heard_count, longest_run = _novel_words(assessment)
    if heard_count and novel_count / heard_count > ASR_NOVEL_WORD_RATIO_UNRELIABLE:
        reasons.append(
            f"{novel_count / heard_count:.0%} of heard words match nothing in the target"
        )
    elif longest_run >= ASR_NOVEL_RUN_UNRELIABLE:
        # An invented phrase, as opposed to scattered self-talk.
        reasons.append(
            f"{longest_run} words in a row match nothing in the target"
        )

    reliable = not reasons
    return Signal(
        id="S2",
        score=1.0 if reliable else 0.0,
        reliable=reliable,
        reasons=tuple(reasons) if reasons else ("telemetry and content look trustworthy",),
    )


def _novel_words(assessment: Assessment) -> tuple[int, int, int]:
    """(novel, heard, longest consecutive run of novel words).

    A heard word is 'explained' if it matched a target word, is a
    near-miss of one, or is a common function word the recognizer may
    insert harmlessly."""
    target_words = normalize(assessment.target)
    heard = [v for v in assessment.verdicts if v.heard_word is not None]
    if not heard:
        return 0, 0, 0

    def explained(verdict) -> bool:
        if verdict.status in (WordStatus.CORRECT, WordStatus.NEAR_MISS):
            return True
        if verdict.heard_word in FUNCTION_WORDS:
            return True
        # An added/substituted word still close to *some* target word is a
        # misplacement, not fabrication.
        return any(is_near_miss(t, verdict.heard_word) for t in target_words)

    flags = [not explained(v) for v in heard]
    longest = run = 0
    for is_novel in flags:
        run = run + 1 if is_novel else 0
        longest = max(longest, run)
    return sum(flags), len(heard), longest
