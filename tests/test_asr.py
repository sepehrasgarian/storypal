"""Tests for ASR telemetry aggregation (pure logic; no model download)."""

from collections import namedtuple

from api.asr import telemetry_from_segments

Segment = namedtuple("Segment", "avg_logprob no_speech_prob compression_ratio text")


def seg(avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2):
    return Segment(avg_logprob, no_speech_prob, compression_ratio, "words")


class TestTelemetryAggregation:
    def test_no_segments_means_silence(self):
        telemetry = telemetry_from_segments([])
        assert telemetry.no_speech_prob == 1.0

    def test_single_segment_passes_through(self):
        telemetry = telemetry_from_segments([seg(avg_logprob=-0.5)])
        assert telemetry.avg_logprob == -0.5

    def test_worst_segment_decides_hallucination_signals(self):
        # One bad stretch poisons the transcript: max, not mean.
        segments = [seg(no_speech_prob=0.05, compression_ratio=1.1),
                    seg(no_speech_prob=0.95, compression_ratio=3.0)]
        telemetry = telemetry_from_segments(segments)
        assert telemetry.no_speech_prob == 0.95
        assert telemetry.compression_ratio == 3.0

    def test_logprob_is_averaged_across_segments(self):
        telemetry = telemetry_from_segments([seg(avg_logprob=-0.2), seg(avg_logprob=-0.8)])
        assert telemetry.avg_logprob == -0.5
