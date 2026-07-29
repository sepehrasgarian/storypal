"""Voice output via Higgs TTS, with signal-driven delivery and a cache.

Delivery style is chosen by the turn's signals rather than fixed:
celebrate a strong read, encourage during corrections, slow down when
the recognizer failed and we ask for a re-read. Audio is cached by
content hash so repeated phrases are synthesized once.
"""

import hashlib
import os
from pathlib import Path

import httpx

from api.config import BOSON_TTS_URL, DATA_DIR, HIGGS_TTS_MODEL

# Style -> expressive tag prepended to the text. Empty string = plain.
# Kept in one place so tags can be tuned (or disabled) without code changes.
STYLE_TAGS = {
    "celebrate": "<|emotion:elation|>",
    "encourage": "<|emotion:enthusiasm|>",
    "model_word": "<|prosody:speed_slow|>",
    "neutral": "",
}

DEFAULT_VOICE = os.getenv("TTS_VOICE", "alloy")


def choose_style(s1_score: float, s2_reliable: bool) -> str:
    """Pick the delivery style from this turn's signals."""
    if not s2_reliable:
        return "encourage"  # gentle re-ask, never triumphant
    if s1_score == 1.0:
        return "celebrate"
    if s1_score >= 0.5:
        return "encourage"
    return "model_word"  # lots of trouble: slow down and demonstrate


class HiggsTTS:
    """Client for Boson's OpenAI-compatible /audio/speech endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = HIGGS_TTS_MODEL,
        voice: str = DEFAULT_VOICE,
        cache_dir: str | Path = Path(DATA_DIR) / "tts_cache",
        client: httpx.Client | None = None,  # injectable for tests
    ):
        self._model = model
        self._voice = voice
        self._cache_dir = Path(cache_dir)
        self._client = client or httpx.Client(
            base_url=BOSON_TTS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    def synthesize(self, text: str, style: str = "neutral") -> Path:
        """Return a path to MP3 audio for the text, from cache if possible."""
        tagged = STYLE_TAGS.get(style, "") + text
        path = self._cache_path(tagged)
        if path.exists():
            return path

        response = self._client.post(
            "/audio/speech",
            json={"model": self._model, "input": tagged, "voice": self._voice},
        )
        response.raise_for_status()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def _cache_path(self, tagged_text: str) -> Path:
        key = hashlib.sha256(f"{self._model}|{self._voice}|{tagged_text}".encode()).hexdigest()[:24]
        return self._cache_dir / f"{key}.mp3"
