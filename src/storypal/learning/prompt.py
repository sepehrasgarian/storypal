"""Tier 1: the tutor's instructions, rebuilt from scratch every turn.

No accumulated chat history — the prompt is a pure function of
(target, assessment, signals, profile, tactic), which makes behaviour
reproducible from state and lets the UI show exactly why the tutor
acted as it did.

The architectural point of the project lives here: when S2 says the
recognizer is unreliable, the instructions flip from "correct the
child" to "do not correct the child".
"""

from storypal.core.assessment import Assessment, WordStatus
from storypal.learning.kb import Tactic
from storypal.learning.profile import Profile, render as render_profile
from storypal.core.signals import Signal

PERSONA = (
    "You are StoryPal, a warm, patient reading tutor for a young child. "
    "Keep replies to 2-3 short sentences a 7-year-old understands. "
    "Always start with something positive. Never ask yes-or-no questions "
    "- always end by telling the child exactly what to read next, because "
    "anything they say back will be recorded as a reading attempt."
)


def _tool_guidance(attempts: int, streak: int, level: int) -> list[str]:
    """Tell the model when its judgement tools apply, and show the
    numbers that justify using them. Without concrete grounds in the
    prompt, a model simply answers in prose and never calls a tool."""
    from storypal.config import LEVEL_UP_STREAK, MAX_LEVEL, STUCK_ATTEMPTS

    lines = [
        "",
        f"Session state: attempt {attempts} at this sentence; "
        f"{streak} perfect read(s) in a row; level {level}.",
        "Tools available to you:",
        "- drill_sound(phoneme): fetch the practice trick that has worked "
        "best for THIS child. Call it whenever you are about to teach a "
        "sound - your own wording is not personalised, the tool's is.",
    ]
    if streak >= LEVEL_UP_STREAK and level < MAX_LEVEL:
        lines.append(
            f"- set_difficulty(level={level + 1}, reason=...): {streak} perfect "
            "reads in a row - this level is too easy. Call it now."
        )
    elif attempts >= STUCK_ATTEMPTS and level > 1:
        lines.append(
            f"- set_difficulty(level={level - 1}, reason=...): attempt {attempts} "
            "on one sentence - this level is too hard. Call it now."
        )
    if attempts >= STUCK_ATTEMPTS:
        lines.append(
            f"- flag_for_review(reason=...): {attempts} attempts on the same "
            "sentence. If the child sounds upset or stuck, call it so a "
            "grown-up can look."
        )
    lines.append(
        "- next_sentence(focus_phoneme=...): move on once this sentence is read well."
    )
    return lines


def build_prompt(
    target: str,
    assessment: Assessment,
    s1: Signal,
    s2: Signal,
    profile: Profile,
    tactic: Tactic | None = None,
    drill_words: list[str] | None = None,
    conversational: bool = False,
    attempts: int = 1,
    streak: int = 0,
) -> str:
    """Assemble the full system prompt for this turn.

    When drill_words is set, the child answered a drill by repeating
    just those words (not the whole sentence) — the instructions must
    react to the drill, never scold about unread sentence words.
    """
    sections = [
        PERSONA,
        "",
        render_profile(profile),
        "",
        "-- this turn (generated live) --",
        f'Target sentence: "{target}"',
        f'Heard: "{assessment.transcript}"',
    ]
    # Conversation outranks the unreliable-ASR branch: chat words that
    # match nothing in the target are exactly what trips the novelty
    # check, but "the child answered you" is the better explanation.
    if conversational:
        sections += _conversational_instructions(assessment, target)
    elif not s2.reliable:
        # Perception failed: no tools, no teaching, just ask again.
        sections += _unreliable_instructions(s2)
        return "\n".join(sections)
    elif drill_words is not None:
        sections += _drill_followup_instructions(drill_words, s1, target)
    else:
        sections += _reliable_instructions(assessment, s1, tactic)
    sections += _tool_guidance(attempts, streak, profile.level)
    return "\n".join(sections)


def _conversational_instructions(assessment: Assessment, target: str) -> list[str]:
    return [
        "Context: the child is talking TO you, not reading - this was "
        "conversation, so there is nothing to grade.",
        "",
        "Instructions: answer them naturally and warmly in one short "
        f'sentence, then ask them to read the sentence: "{target}"',
    ]


def _drill_followup_instructions(drill_words: list[str], s1: Signal, target: str) -> list[str]:
    words = ", ".join(f"'{w}'" for w in drill_words)
    lines = [
        f"Context: the child was practicing just the word(s) {words} and "
        "repeated only those, not the whole sentence. That is exactly what "
        "was asked - do NOT treat the other sentence words as skipped.",
        f"Drill result: {'; '.join(s1.reasons)} (accuracy {s1.score:.0%})",
        "ASR: reliable",
        "",
    ]
    if s1.score >= 0.75:
        lines.append(
            "Instructions: celebrate that they got the practiced word! Then "
            f'ask them to read the whole sentence again: "{target}"'
        )
    else:
        lines.append(
            "Instructions: encourage warmly, model the practiced word slowly "
            "one more time, and ask them to try just that word again."
        )
    return lines


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
