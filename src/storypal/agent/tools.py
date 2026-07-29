"""The four tools the tutor agent may call, with their guard rails.

Every tool validates its arguments and returns a plain dict — a
structured error dict on bad input (so the model can retry), never an
exception. drill_sound is additionally gated by S2: we refuse to drill
based on a transcript we do not trust.
"""

from dataclasses import dataclass, field

from storypal.learning import kb
from storypal.config import STORIES
from storypal.learning.kb import TacticStats
from storypal.learning.profile import Profile

VALID_PHONEMES = sorted({t.phoneme for t in kb.TACTICS})
MIN_LEVEL, MAX_LEVEL = 1, max(s.level for s in STORIES)


@dataclass
class ToolContext:
    """Everything a tool handler may need to read or change."""

    profile: Profile
    tactic_stats: TacticStats
    asr_reliable: bool = True
    sentences_seen: set = field(default_factory=set)
    flagged: list = field(default_factory=list)  # reasons raised for human review
    tactic_used: kb.Tactic | None = None  # recorded so outcomes can be scored later
    next_story: str | None = None  # sentence chosen for the next turn


# --- Handlers -----------------------------------------------------------

def next_sentence(ctx: ToolContext, focus_phoneme: str | None = None, difficulty: int | None = None) -> dict:
    if focus_phoneme is not None and focus_phoneme not in VALID_PHONEMES:
        return _error(f"unknown phoneme '{focus_phoneme}'; known: {', '.join(VALID_PHONEMES)}")
    level = difficulty if difficulty is not None else ctx.profile.level
    if not MIN_LEVEL <= level <= MAX_LEVEL:
        return _error(f"difficulty must be between {MIN_LEVEL} and {MAX_LEVEL}")
    story = kb.next_sentence(level, focus_phoneme, exclude=ctx.sentences_seen)
    if story is None:
        return _error("no sentences left; call with different filters")
    ctx.sentences_seen.add(story.text)
    ctx.next_story = story.text
    return {"sentence": story.text, "level": story.level, "phonemes": list(story.phonemes)}


def drill_sound(ctx: ToolContext, phoneme: str) -> dict:
    if phoneme not in VALID_PHONEMES:
        return _error(f"unknown phoneme '{phoneme}'; known: {', '.join(VALID_PHONEMES)}")
    if not ctx.asr_reliable:
        # The S2 gate, enforced at the tool layer as well as in the prompt.
        return _error("recognition was unreliable this turn; do not drill — ask the child to read again")
    tactic = kb.best_tactic(phoneme, ctx.tactic_stats)
    if tactic is None:
        return _error(f"no tactic available for '{phoneme}'")
    ctx.tactic_stats.record_usage(tactic)
    ctx.tactic_used = tactic
    return {
        "tactic": tactic.name,
        "instructions": tactic.instructions,
        "example_words": list(tactic.example_words),
        "success_rate": round(ctx.tactic_stats.success_rate(tactic), 2),
    }


def set_difficulty(ctx: ToolContext, level: int, reason: str) -> dict:
    if not isinstance(level, int) or not MIN_LEVEL <= level <= MAX_LEVEL:
        return _error(f"level must be an integer between {MIN_LEVEL} and {MAX_LEVEL}")
    if not reason or not str(reason).strip():
        return _error("a reason is required so the change is auditable")
    ctx.profile.level = level
    return {"level": level, "reason": reason}


def flag_for_review(ctx: ToolContext, reason: str) -> dict:
    if not reason or not str(reason).strip():
        return _error("a reason is required")
    ctx.flagged.append(reason)
    return {"flagged": True, "reason": reason}


def _error(message: str) -> dict:
    return {"error": message}


HANDLERS = {
    "next_sentence": next_sentence,
    "drill_sound": drill_sound,
    "set_difficulty": set_difficulty,
    "flag_for_review": flag_for_review,
}

# Schemas in the JSON-schema shape LLM function-calling APIs expect.
SCHEMAS = [
    {
        "name": "next_sentence",
        "description": "Pick the next sentence for the child to read, optionally targeting a sound or difficulty.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus_phoneme": {"type": "string", "description": f"Sound to practice, one of: {', '.join(VALID_PHONEMES)}"},
                "difficulty": {"type": "integer", "description": f"Level {MIN_LEVEL}-{MAX_LEVEL}"},
            },
        },
    },
    {
        "name": "drill_sound",
        "description": "Get the best teaching tactic for a sound the child is struggling with.",
        "parameters": {
            "type": "object",
            "properties": {"phoneme": {"type": "string", "description": f"One of: {', '.join(VALID_PHONEMES)}"}},
            "required": ["phoneme"],
        },
    },
    {
        "name": "set_difficulty",
        "description": "Move the child to an easier or harder level. Requires a reason.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["level", "reason"],
        },
    },
    {
        "name": "flag_for_review",
        "description": "Something seems wrong (distress, repeated failures, off-topic audio); ask a human to look.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]
