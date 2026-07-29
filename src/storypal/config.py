"""Central configuration: stories, thresholds, and model names.

Every tunable number in the system lives here so behaviour can be
changed without touching logic code.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Story:
    """One target sentence a child can be asked to read."""

    text: str
    level: int  # 1 = easiest
    phonemes: tuple[str, ...] = field(default_factory=tuple)  # sounds it exercises


STORIES: list[Story] = [
    Story("The cat sat on the mat.", level=1, phonemes=("s", "t")),
    Story("A big red dog ran fast.", level=1, phonemes=("r", "d")),
    Story("The sun is hot today.", level=1, phonemes=("s", "t")),
    Story("The bird flew through the trees.", level=2, phonemes=("th", "r")),
    Story("Three small ships sailed north.", level=2, phonemes=("th", "s")),
    Story("She threw the ball over there.", level=2, phonemes=("th", "r")),
    Story("The children thought about their birthday.", level=3, phonemes=("th", "ch")),
    Story("Thunder rumbled through the thick clouds.", level=3, phonemes=("th", "r")),
]

# --- Assessment ---------------------------------------------------------
# A substituted word within this edit distance of the target word counts
# as a near-miss (likely mispronunciation) rather than a different word.
# Short words get a tighter threshold: at distance 2 almost every
# 3-letter word resembles another ("now"/"hot"), which would let
# fabricated chatter pass as near-misses.
NEAR_MISS_MAX_EDIT_DISTANCE = 2
SHORT_WORD_LEN = 3
SHORT_WORD_MAX_EDIT_DISTANCE = 1

# --- S2: ASR reliability thresholds (Whisper telemetry) -----------------
ASR_NO_SPEECH_UNRELIABLE = 0.5  # above this, probably silence
ASR_AVG_LOGPROB_UNRELIABLE = -1.0  # below this, low-confidence decode
ASR_COMPRESSION_RATIO_UNRELIABLE = 2.4  # above this, repetitive hallucination
# Fraction of heard words that match nothing in the target before the
# transcript is treated as fabricated content.
ASR_NOVEL_WORD_RATIO_UNRELIABLE = 0.5

# Common words the recognizer may add that carry no evidence of
# fabrication ("the", "a", ...). Excluded from the novelty check.
FUNCTION_WORDS = frozenset(
    "a an the and or but is are was were to of in on at it i".split()
)

# --- Session flow -------------------------------------------------------
# A trusted turn at or above this accuracy is "accepted": the session
# automatically advances to the next sentence.
AUTO_ADVANCE_ACCURACY = 0.99

# After a flawed read the child often repeats just the practiced word,
# not the whole sentence. If the recording matches the drilled words at
# or above DRILL_MATCH_ACCURACY while matching the full sentence below
# DRILL_FULL_MISMATCH, grade the drill instead of the sentence.
DRILL_MATCH_ACCURACY = 0.75
DRILL_FULL_MISMATCH = 0.5

# Children answer the tutor back ("Yes, I do!", "okay"). A short reply
# made only of these words, that also matches the target poorly, is
# conversation - it must never be graded as a failed reading.
# Deliberately conservative: no word here appears in any story sentence.
CHAT_WORDS = frozenset(
    "yes no okay ok yeah yep nope sure please thanks thank you i do dont "
    "what huh hmm um uh hello hi hey bye damn duh it one more again wait "
    "stop now me lets try another im ready".split()
)
CHAT_MAX_WORDS = 6

# --- Judges (S3 grounding, S4 pedagogy) ---------------------------------
# A judge score below this means the tutor's reply failed that check.
JUDGE_FAIL_THRESHOLD = 0.5

# --- Models -------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
BOSON_TTS_URL = os.getenv("BOSON_TTS_URL", "https://api.boson.ai/v1")
HIGGS_TTS_MODEL = os.getenv("HIGGS_TTS_MODEL", "higgs-tts-v3")

# --- Data locations -----------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")
CURATED_DIR = os.path.join(DATA_DIR, "curated")
TRAJECTORY_PATH = os.path.join(DATA_DIR, "trajectories.jsonl")
