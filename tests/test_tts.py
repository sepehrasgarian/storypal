"""Tests for signal-driven delivery style and the synthesis cache.

Network is faked with httpx.MockTransport: zero API cost.
"""

import httpx
import pytest

from api.tts import STYLE_TAGS, HiggsTTS, choose_style


class TestChooseStyle:
    def test_perfect_read_celebrates(self):
        assert choose_style(1.0, s2_reliable=True) == "celebrate"

    def test_small_trouble_encourages(self):
        assert choose_style(0.8, s2_reliable=True) == "encourage"

    def test_big_trouble_slows_down(self):
        assert choose_style(0.2, s2_reliable=True) == "model_word"

    def test_unreliable_turn_is_always_gentle(self):
        # Never celebrate a read we could not verify.
        assert choose_style(1.0, s2_reliable=False) == "encourage"


def make_tts(tmp_path, counter):
    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        counter["last_body"] = request.read().decode()
        return httpx.Response(200, content=b"FAKE_MP3_BYTES")

    client = httpx.Client(base_url="https://fake", transport=httpx.MockTransport(handler))
    return HiggsTTS(api_key="test", cache_dir=tmp_path / "cache", client=client)


class TestSynthesis:
    def test_audio_is_written_and_returned(self, tmp_path):
        counter = {"calls": 0}
        path = make_tts(tmp_path, counter).synthesize("Great job!", style="celebrate")
        assert path.read_bytes() == b"FAKE_MP3_BYTES"

    def test_style_tag_is_prepended(self, tmp_path):
        counter = {"calls": 0}
        make_tts(tmp_path, counter).synthesize("Great job!", style="celebrate")
        assert STYLE_TAGS["celebrate"] in counter["last_body"]

    def test_cache_prevents_repeat_synthesis(self, tmp_path):
        counter = {"calls": 0}
        tts = make_tts(tmp_path, counter)
        first = tts.synthesize("Great job!", style="celebrate")
        second = tts.synthesize("Great job!", style="celebrate")
        assert first == second
        assert counter["calls"] == 1

    def test_different_style_is_a_different_cache_entry(self, tmp_path):
        counter = {"calls": 0}
        tts = make_tts(tmp_path, counter)
        tts.synthesize("Great job!", style="celebrate")
        tts.synthesize("Great job!", style="neutral")
        assert counter["calls"] == 2

    def test_api_error_raises_not_caches(self, tmp_path):
        def failing(request):
            return httpx.Response(500, text="server error")

        client = httpx.Client(base_url="https://fake", transport=httpx.MockTransport(failing))
        tts = HiggsTTS(api_key="test", cache_dir=tmp_path / "c", client=client)
        with pytest.raises(httpx.HTTPStatusError):
            tts.synthesize("hello")
        assert not list((tmp_path / "c").glob("*.mp3")) if (tmp_path / "c").exists() else True
