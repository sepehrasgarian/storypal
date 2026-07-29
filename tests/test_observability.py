"""Tests for the optional Langfuse exporter (network faked)."""

import httpx

from storypal.core.signals import Signal
from storypal.observability import LangfuseExporter, from_env

SIGNALS = {
    "S1": Signal("S1", score=0.83, reliable=True, reasons=("missed 'through'",)),
    "S2": Signal("S2", score=1.0, reliable=True, reasons=("looks trustworthy",)),
}


def make_exporter(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(207, json={"successes": [], "errors": []})

    client = httpx.Client(base_url="https://fake", transport=httpx.MockTransport(handler))
    return LangfuseExporter("pk", "sk", client=client)


def export(exporter):
    exporter.export_turn(
        session_id="abc", turn=1, timestamp=1700000000.0,
        target="The bird flew through the trees.", transcript="the bird flew the trees",
        reply="Great try!", prompt="You are StoryPal...", signals=SIGNALS,
        timings_ms={"asr_ms": 480, "agent_ms": 620, "tts_ms": 700}, route="archive",
    )


class TestConfiguration:
    def test_disabled_without_keys(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert from_env() is None

    def test_enabled_with_keys(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        assert from_env() is not None


class TestExport:
    def test_one_trace_three_spans_and_all_scores(self):
        captured = []
        export(make_exporter(captured))
        batch = captured[0].read().decode()
        import json
        events = json.loads(batch)["batch"]
        kinds = [e["type"] for e in events]
        assert kinds.count("trace-create") == 1
        assert kinds.count("span-create") == 3
        assert kinds.count("score-create") == len(SIGNALS)

    def test_scores_carry_values_and_reasons(self):
        captured = []
        export(make_exporter(captured))
        import json
        events = json.loads(captured[0].read().decode())["batch"]
        s1 = next(e["body"] for e in events if e["type"] == "score-create" and e["body"]["name"] == "S1")
        assert s1["value"] == 0.83
        assert "missed 'through'" in s1["comment"]

    def test_network_failure_is_swallowed(self):
        def exploding(request):
            raise httpx.ConnectError("no network")

        client = httpx.Client(base_url="https://fake", transport=httpx.MockTransport(exploding))
        export(LangfuseExporter("pk", "sk", client=client))  # must not raise
