"""A second agent that plays the child, speaking through real audio.

Every other eval in this project fabricates the transcript, so the real
recogniser is never exercised. Here a child agent SPEAKS - Higgs TTS
renders an utterance to audio - and StoryPal's real Whisper has to
listen to it. Ground truth is what the child agent intended to say, so
the loop can measure what no text-level test can: whether the system
mistreats a child because the recogniser, not the child, made the error.

Honest limitation: Higgs preset voices are adults. Real children's
speech is markedly harder to recognise, so these numbers are an
optimistic bound - a floor on the error rate, not an estimate of it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from storypal.core.assessment import normalize


@dataclass(frozen=True)
class SpokenTurn:
    """What the child agent decided to do, before anyone listened."""

    behaviour: str
    said: str  # ground truth: the words actually voiced
    text_to_speak: str  # what is handed to TTS, tags included
    expect_trusted: bool  # should StoryPal believe the recogniser?
    expect_perfect: bool  # is `said` a correct reading of the target?


def _words(text: str) -> list[str]:
    return normalize(text)


def read_perfectly(target: str) -> SpokenTurn:
    said = " ".join(_words(target))
    return SpokenTurn("perfect", said, said, True, True)


def skip_a_word(target: str) -> SpokenTurn:
    words = _words(target)
    said = " ".join(words[:-1])  # trails off before the last word
    return SpokenTurn("skipped the last word", said, said, True, False)


def read_only_the_start(target: str) -> SpokenTurn:
    words = _words(target)
    said = " ".join(words[: max(1, len(words) // 2)])
    return SpokenTurn("gave up halfway", said, said, True, False)


def mispronounce(target: str) -> SpokenTurn:
    """Swap the hardest word for a near neighbour, the way a child
    substitutes a sound they cannot make yet."""
    swaps = {"thick": "tick", "three": "tree", "this": "dis", "shed": "sed",
             "chips": "ships", "sun": "son", "soft": "sof", "shall": "sal"}
    words = _words(target)
    said_words = [swaps.get(w, w) for w in words]
    said = " ".join(said_words)
    changed = said != " ".join(words)
    return SpokenTurn("mispronounced a word", said, said, True, not changed)


def speak_quietly(target: str) -> SpokenTurn:
    """Reads it correctly, but tired and indistinct. Any 'mistake' found
    here is the recogniser's fault, never the child's."""
    said = " ".join(_words(target))
    return SpokenTurn("read correctly but quietly", said,
                      "<|prosody:speed_slow|>" + said, True, True)


def go_off_script(_target: str) -> SpokenTurn:
    said = "can we play a game now"
    return SpokenTurn("talked about something else", said, said, False, False)


def answer_the_tutor(_target: str) -> SpokenTurn:
    said = "yes i do"
    return SpokenTurn("answered the tutor", said, said, False, False)


BEHAVIOURS: list[Callable[[str], SpokenTurn]] = [
    read_perfectly,
    skip_a_word,
    read_only_the_start,
    mispronounce,
    speak_quietly,
    go_off_script,
    answer_the_tutor,
]


class ChildVoice:
    """Speaks a turn out loud, caching each clip so reruns are free."""

    def __init__(self, tts, cache_dir: str | Path = "data/eval_audio"):
        self._tts = tts
        self._cache = Path(cache_dir)

    def speak(self, turn: SpokenTurn) -> Path:
        self._cache.mkdir(parents=True, exist_ok=True)
        return self._tts.synthesize(turn.text_to_speak, style="neutral")
