"""Tier 1: the tutor's instructions, rebuilt from scratch every turn.

No accumulated chat history — the prompt is a pure function of
(target, assessment, signals, profile, tactic), which makes behaviour
reproducible from state and lets the UI show exactly why the tutor
acted as it did.

The architectural point of the project lives here: when S2 says the
recognizer is unreliable, the instructions flip from "correct the
child" to "do not correct the child".
"""

from api.assessment import Assessment, WordStatus
from api.kb import Tactic
from api.profile import Profile, render as render_profile
from api.signals import Signal

PERSONA = (
    "You are StoryPal, a warm, patient reading tutor for a young child. "
    "Keep replies to 2-3 short sentences a 7-year-old understands. "
    "Always start with something positive."
)


def build_prompt(
    target: str,
    assessment: Assessment,
    s1: Signal,
    s2: Signal,
    profile: Profile,
    tactic: Tactic | None = None,
) -> str:
    """Assemble the full system prompt for this turn."""
    sections = [
        PERSONA,
        "",
        render_profile(profile),
        "",
        "-- this turn (generated live) --",
        f'Target sentence: "{target}"',
        f'Heard: "{assessment.transcript}"',
    ]
    if s2.reliable:
        sections += _reliable_instructions(assessment, s1, tactic)
    else:
        sections += _unreliable_instructions(s2)
    return "\n".join(sections)


def _reliable_instructions(assessment: Assessment, s1: Signal, tactic: Tactic | None) -> list[str]:
    lines = [f"Assessment: {'; '.join(s1.reasons)} (accuracy {s1.score:.0%})", "ASR: reliable", ""]
    if s1.score == 1.0:
        lines.append("Instructions: celebrate the perfect read, then invite the next sentence.")
        return lines

    problem_words = [
        v.target_word
        for v in assessment.verdicts
        if v.status in (WordStatus.MISSED, WordStatus.NEAR_MISS, WordStatus.SUBSTITUTED)
    ]
    lines.append(
        "Instructions: praise what went right first, then gently work on: "
        + ", ".join(f"'{w}'" for w in problem_words)
        + "."
    )
    if tactic is not None:
        lines.append(
            f"Teaching tactic ({tactic.name}): {tactic.instructions} "
            f"Example words: {', '.join(tactic.example_words)}."
        )
    return lines


def _unreliable_instructions(s2: Signal) -> list[str]:
    return [
        f"ASR: UNRELIABLE — {'; '.join(s2.reasons)}",
        "",
        "Instructions: we cannot trust what was heard, so do NOT correct "
        "the child — you might correct a child who read perfectly. Do not "
        "mention any specific words. Warmly ask them to read the sentence "
        "one more time.",
    ]
