"""Tests for the four tools and their guard rails."""

from api.kb import TacticStats
from api.profile import Profile
from api.tools import ToolContext, drill_sound, flag_for_review, next_sentence, set_difficulty


def make_ctx(tmp_path, reliable=True, level=2):
    return ToolContext(
        profile=Profile(level=level),
        tactic_stats=TacticStats(tmp_path / "tactics.json"),
        asr_reliable=reliable,
    )


class TestNextSentence:
    def test_returns_a_sentence_and_remembers_it(self, tmp_path):
        ctx = make_ctx(tmp_path)
        result = next_sentence(ctx, focus_phoneme="th")
        assert "th" in result["phonemes"]
        assert result["sentence"] in ctx.sentences_seen

    def test_does_not_repeat_sentences(self, tmp_path):
        ctx = make_ctx(tmp_path)
        first = next_sentence(ctx, focus_phoneme="th")
        second = next_sentence(ctx, focus_phoneme="th")
        assert first["sentence"] != second["sentence"]

    def test_unknown_phoneme_is_a_structured_error(self, tmp_path):
        result = next_sentence(make_ctx(tmp_path), focus_phoneme="x7")
        assert "unknown phoneme" in result["error"]

    def test_out_of_range_difficulty_is_rejected(self, tmp_path):
        assert "error" in next_sentence(make_ctx(tmp_path), difficulty=12)


class TestDrillSound:
    def test_returns_tactic_and_records_usage(self, tmp_path):
        ctx = make_ctx(tmp_path)
        result = drill_sound(ctx, phoneme="th")
        assert result["tactic"]
        assert ctx.tactic_used is not None

    def test_refused_when_asr_unreliable(self, tmp_path):
        # The S2 gate at the tool layer: never drill on an untrusted transcript.
        ctx = make_ctx(tmp_path, reliable=False)
        result = drill_sound(ctx, phoneme="th")
        assert "unreliable" in result["error"]
        assert ctx.tactic_used is None

    def test_unknown_phoneme_is_a_structured_error(self, tmp_path):
        assert "unknown phoneme" in drill_sound(make_ctx(tmp_path), phoneme="zz")["error"]


class TestSetDifficulty:
    def test_changes_the_profile_with_a_reason(self, tmp_path):
        ctx = make_ctx(tmp_path, level=2)
        result = set_difficulty(ctx, level=1, reason="three failed sentences in a row")
        assert ctx.profile.level == 1
        assert result["reason"]

    def test_reason_is_required(self, tmp_path):
        ctx = make_ctx(tmp_path, level=2)
        assert "error" in set_difficulty(ctx, level=1, reason="  ")
        assert ctx.profile.level == 2  # unchanged

    def test_level_must_be_in_range(self, tmp_path):
        assert "error" in set_difficulty(make_ctx(tmp_path), level=99, reason="x")


class TestFlagForReview:
    def test_flag_is_recorded(self, tmp_path):
        ctx = make_ctx(tmp_path)
        flag_for_review(ctx, reason="child sounded upset")
        assert ctx.flagged == ["child sounded upset"]

    def test_reason_is_required(self, tmp_path):
        assert "error" in flag_for_review(make_ctx(tmp_path), reason="")
