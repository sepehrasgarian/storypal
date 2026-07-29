"""Central configuration: stories, thresholds, and model names.

Every tunable number in the system lives here so behaviour can be
changed without touching logic code.
"""

import os
import re
from dataclasses import dataclass

# Sounds we track. Lives here because both the learner profile and the
# story catalogue must agree on what a word exercises.
#
# Digraphs map to one sound reliably at either edge of a word (this,
# fish, much). Single consonants are only trusted word-initially:
# English mangles them elsewhere - the 'r' in "bird" is an r-coloured
# vowel, the 's' in "is" says /z/, the 't' in "listen" is silent. We
# would rather count nothing than count a sound the child never made.
TRACKED_DIGRAPHS = ("th", "ch", "sh")
TRACKED_INITIALS = ("r", "s", "t", "d")
TRACKED_PHONEMES = TRACKED_DIGRAPHS + TRACKED_INITIALS


def phonemes_in_word(word: str) -> list[str]:
    """The tracked sounds a word exercises, counted only where spelling
    predicts sound: 'this' -> ['th'], 'fish' -> ['sh'], 'red' -> ['r'],
    'through' -> ['th'] (not 'r' - it is buried in a 'thr' blend),
    'bird' -> [] (that 'r' is not a consonant sound)."""
    word = word.lower()
    found = [d for d in TRACKED_DIGRAPHS if word.startswith(d) or word.endswith(d)]
    if word[:1] in TRACKED_INITIALS and not found:
        found.append(word[0])
    return found


def phonemes_in_sentence(text: str) -> tuple[str, ...]:
    sounds: set[str] = set()
    for word in re.findall(r"[a-z']+", text.lower()):
        sounds.update(phonemes_in_word(word))
    return tuple(sorted(sounds))


@dataclass(frozen=True)
class Story:
    """One target sentence a child can be asked to read."""

    text: str
    level: int  # 1 = easiest

    @property
    def phonemes(self) -> tuple[str, ...]:
        """Derived from the words, never hand-tagged: hand-written tags
        drifted from reality (they missed the 'th' in "The"), so a child
        weak at a sound could be served sentences that never practise it."""
        return phonemes_in_sentence(self.text)


# Sentences are built from decodable words: the tracked sound sits at
# the start or end of the word, so a miss really does implicate that
# sound. This is why the catalogue avoids words like "through" or
# "bird", where the spelling would lie about what the child said.
STORIES: list[Story] = [
    Story("The sun is hot.", level=1),
    Story("A red dog ran fast.", level=1),
    Story("The cat sat on a mat.", level=1),
    Story("This fish is big.", level=1),
    Story("She has ten chips.", level=1),
    Story("That thick rug is soft.", level=2),
    Story("Three ships sailed to the dock.", level=2),
    Story("She chose the red socks.", level=2),
    Story("This chick shall rest with them.", level=2),
    Story("Thirty thin threads shook in the shed.", level=3),
    Story("The chef chopped such thick sticks.", level=3),
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

# Consecutive unexplained words before the transcript is treated as
# fabricated. Invented content arrives as a phrase ("thanks for
# watching", "please subscribe"); a child's self-talk arrives as
# scattered single words ("um ... i did it"). Both can total the same
# number of novel words, so contiguity - not the count - separates them.
ASR_NOVEL_RUN_UNRELIABLE = 2

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

# Grounds for the agent's judgement tools. Tools only get called when
# the prompt shows the model a concrete reason to call them.
LEVEL_UP_STREAK = 3  # perfect reads in a row -> offer a harder level
STUCK_ATTEMPTS = 3  # tries on one sentence -> ease off or flag a human
MAX_LEVEL = 3

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
