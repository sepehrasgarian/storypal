"""Speech-to-text behind a swappable interface.

WhisperASR (faster-whisper, local) is the default: free, and it
reports the confidence telemetry S2 feeds on. HiggsSTT is a ready
adapter for Boson's hosted STT, inactive until the API key has access.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

import httpx

from api.config import BOSON_TTS_URL, WHISPER_MODEL
from api.signals import AsrTelemetry


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    telemetry: AsrTelemetry


def telemetry_from_segments(segments: list) -> AsrTelemetry:
    """Aggregate Whisper's per-segment confidence into one telemetry record.

    Pessimistic aggregation: the worst segment decides, because one
    hallucinated stretch poisons the whole transcript.
    """
    if not segments:
        return AsrTelemetry(avg_logprob=0.0, no_speech_prob=1.0, compression_ratio=1.0)
    return AsrTelemetry(
        avg_logprob=statistics.mean(s.avg_logprob for s in segments),
        no_speech_prob=max(s.no_speech_prob for s in segments),
        compression_ratio=max(s.compression_ratio for s in segments),
    )


class WhisperASR:
    """Local transcription via faster-whisper. The model loads once,
    lazily, so importing this module stays cheap."""

    def __init__(self, model_size: str = WHISPER_MODEL):
        self._model_size = model_size
        self._model = None

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: heavy import

            self._model = WhisperModel(self._model_size, compute_type="int8")
        segments, _info = self._model.transcribe(str(audio_path), language="en")
        segments = list(segments)  # the generator must be consumed once
        text = " ".join(s.text.strip() for s in segments).strip()
        return TranscriptionResult(transcript=text, telemetry=telemetry_from_segments(segments))


class HiggsSTT:
    """Boson-hosted STT via the OpenAI-compatible transcription endpoint.

    Written and tested against the API shape, but inactive by default:
    the current key exposes only TTS models. Note the hosted API does
    not return per-segment confidence, so telemetry is neutral — S2
    then relies on target-anchored novelty alone.
    """

    def __init__(self, api_key: str, model: str = "higgs-stt-3", base_url: str = BOSON_TTS_URL):
        self._model = model
        self._client = httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        with open(audio_path, "rb") as f:
            response = self._client.post(
                "/audio/transcriptions",
                data={"model": self._model},
                files={"file": (Path(audio_path).name, f)},
            )
        response.raise_for_status()
        return TranscriptionResult(
            transcript=response.json().get("text", "").strip(),
            telemetry=AsrTelemetry(),  # hosted API exposes no confidence numbers
        )
