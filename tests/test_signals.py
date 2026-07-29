"""Tests for S1 (reading accuracy) and S2 (ASR reliability).

The two headline scenarios:
- a clean read must come out reliable, and
- Whisper hallucinating on silence ("thanks for watching") must be caught.
"""

import pytest

from storypal.core.assessment import assess
from storypal.core.signals import AsrTelemetry, s1_reading_accuracy, s2_asr_reliability

TARGET = "The bird flew through the trees."

CONFIDENT = AsrTelemetry(avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=1.3)


class TestS1ReadingAccuracy:
    def test_perfect_read(self):
        signal = s1_reading_accuracy(assess(TARGET, "the bird flew through the trees"))
        assert signal.score == 1.0
        assert signal.reasons == ("perfect read",)

    def test_missed_word_is_named(self):
        signal = s1_reading_accuracy(assess(TARGET, "the bird flew the trees"))
        assert signal.score == pytest.approx(5 / 6)
        assert "missed 'through'" in signal.reasons

    def test_mispronunciation_is_named(self):
        signal = s1_reading_accuracy(assess(TARGET, "the bird floo through the trees"))
        assert "mispronounced 'flew' as 'floo'" in signal.reasons

    def test_substitution_is_named(self):
        signal = s1_reading_accuracy(assess(TARGET, "the bird jumped through the trees"))
        assert "read 'jumped' instead of 'flew'" in signal.reasons


class TestS2Telemetry:
    """The recognizer's own confidence numbers."""

    def test_confident_clean_read_is_reliable(self):
        signal = s2_asr_reliability(CONFIDENT, assess(TARGET, "the bird flew through the trees"))
        assert signal.reliable

    def test_silence_hallucination_is_caught(self):
        # The classic Whisper failure: silence in, "thanks for watching" out.
        telemetry = AsrTelemetry(avg_logprob=-1.4, no_speech_prob=0.9, compression_ratio=1.1)
        signal = s2_asr_reliability(telemetry, assess(TARGET, "thanks for watching"))
        assert not signal.reliable
        assert any("silence" in r for r in signal.reasons)

    def test_low_confidence_decode_is_flagged(self):
        telemetry = AsrTelemetry(avg_logprob=-1.8, no_speech_prob=0.1, compression_ratio=1.2)
        signal = s2_asr_reliability(telemetry, assess(TARGET, "the bird flew through the trees"))
        assert not signal.reliable

    def test_repetitive_hallucination_is_flagged(self):
        telemetry = AsrTelemetry(avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=3.5)
        signal = s2_asr_reliability(telemetry, assess(TARGET, "the the the the the the the the"))
        assert not signal.reliable


class TestS2TargetAnchoredNovelty:
    """Content checks: heard words that match nothing in the target."""

    def test_fabricated_content_with_good_telemetry_is_still_caught(self):
        # Telemetry alone can miss confident hallucinations; content saves us.
        signal = s2_asr_reliability(CONFIDENT, assess(TARGET, "please subscribe to my channel"))
        assert not signal.reliable
        assert any("match nothing" in r for r in signal.reasons)

    def test_a_real_mistake_is_not_fabrication(self):
        # Missing a word and mispronouncing another is a child reading,
        # not the recognizer inventing content. Must stay reliable.
        signal = s2_asr_reliability(CONFIDENT, assess(TARGET, "the bird floo the trees"))
        assert signal.reliable

    def test_added_function_words_are_harmless(self):
        signal = s2_asr_reliability(CONFIDENT, assess(TARGET, "the bird it flew through the trees"))
        assert signal.reliable

    def test_empty_transcript_has_no_novelty(self):
        # Nothing heard = nothing fabricated; silence is telemetry's job.
        signal = s2_asr_reliability(CONFIDENT, assess(TARGET, ""))
        assert signal.reliable
