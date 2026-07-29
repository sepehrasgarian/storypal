"""Simulated children, for testing whole sessions rather than single turns.

Each persona produces an Utterance carrying BOTH what the child really
said and what the recogniser reported. Those two differing is the whole
point: it lets a simulation measure the failure that matters most -
correcting a child who actually read the sentence perfectly.
"""

import random
from dataclasses import dataclass
from typing import Callable

from storypal.core.assessment import normalize
from storypal.core.signals import AsrTelemetry

CLEAN = AsrTelemetry(avg_logprob=-0.3, no_speech_prob=0.08, compression_ratio=1.25)
SHAKY = AsrTelemetry(avg_logprob=-1.3, no_speech_prob=0.2, compression_ratio=1.4)
SILENT = AsrTelemetry(avg_logprob=-1.5, no_speech_prob=0.94, compression_ratio=1.1)


@dataclass(frozen=True)
class Utterance:
    truth: str  # what the child actually said (ground truth)
    transcript: str  # what the recogniser reported
    telemetry: AsrTelemetry

    @property
    def asr_was_wrong(self) -> bool:
        return normalize(self.truth) != normalize(self.transcript)


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    weakness: str | None  # the sound they genuinely struggle with, if any
    speak: Callable[[str, random.Random], Utterance]


def _drop_words_with(target: str, phoneme: str) -> str:
    """The child skips every word containing their weak sound."""
    from storypal.config import phonemes_in_word

    kept = [w for w in normalize(target) if phoneme not in phonemes_in_word(w)]
    return " ".join(kept)


def _confident(target: str, rng: random.Random) -> Utterance:
    """Reads well; the odd slip. Recognition is clean."""
    if rng.random() < 0.15:
        words = normalize(target)
        said = " ".join(words[:-1])  # trails off at the end
    else:
        said = " ".join(normalize(target))
    return Utterance(said, said, CLEAN)


def _struggler(target: str, rng: random.Random) -> Utterance:
    """Genuinely cannot say /th/ words yet. Recognition is clean, so
    every miss is real and the system SHOULD learn from it."""
    said = _drop_words_with(target, "th")
    if said == " ".join(normalize(target)) and rng.random() < 0.3:
        said = " ".join(normalize(target)[:-1])
    return Utterance(said, said, CLEAN)


def _mumbler(target: str, rng: random.Random) -> Utterance:
    """Tired or unwell: reads the sentence CORRECTLY but quietly, and
    the recogniser mangles it. The trap - every apparent mistake here is
    the system's fault, not the child's."""
    truth = " ".join(normalize(target))
    words = normalize(target)
    if rng.random() < 0.75:
        garbled = [w if rng.random() > 0.4 else w[:-1] + "uh" for w in words]
        return Utterance(truth, " ".join(garbled), SHAKY)
    return Utterance(truth, truth, SHAKY)


def _chatterbox(target: str, rng: random.Random) -> Utterance:
    """Talks back at least as often as they read."""
    if rng.random() < 0.5:
        said = rng.choice(["yes I do", "okay", "no more", "what now"])
        return Utterance(said, said, CLEAN)
    said = " ".join(normalize(target))
    return Utterance(said, said, CLEAN)


def _frustrated(target: str, rng: random.Random) -> Utterance:
    """Upset and disengaging: refusals, outbursts, and silences the
    recogniser fills with invented words."""
    roll = rng.random()
    if roll < 0.35:
        return Utterance("", rng.choice(["thanks for watching", "please subscribe"]), SILENT)
    if roll < 0.7:
        said = rng.choice(["no", "stop", "damn it", "i dont want to"])
        return Utterance(said, said, CLEAN)
    said = " ".join(normalize(target)[:2])
    return Utterance(said, said, CLEAN)


PERSONAS = [
    Persona("confident_reader", "reads well, occasional trail-off", None, _confident),
    Persona("th_struggler", "cannot manage /th/ words yet", "th", _struggler),
    Persona("tired_mumbler", "reads correctly but is misheard", None, _mumbler),
    Persona("chatterbox", "answers the tutor as often as reading", None, _chatterbox),
    Persona("frustrated_child", "refuses, goes quiet, gets misheard", None, _frustrated),
]
