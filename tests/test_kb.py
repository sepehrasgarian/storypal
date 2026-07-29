"""Tests for the content and strategy knowledge bases."""

from storypal.learning.kb import TacticStats, best_tactic, next_sentence


class TestContentKB:
    def test_prefers_level_and_focus_phoneme(self):
        story = next_sentence(level=2, focus_phoneme="th")
        assert story.level == 2
        assert "th" in story.phonemes

    def test_excluded_sentences_are_skipped(self):
        first = next_sentence(level=2, focus_phoneme="th")
        second = next_sentence(level=2, focus_phoneme="th", exclude={first.text})
        assert second.text != first.text

    def test_falls_back_when_no_perfect_match(self):
        # No level-99 story exists; must still return something.
        assert next_sentence(level=99) is not None

    def test_everything_excluded_returns_none(self):
        from storypal.config import STORIES
        assert next_sentence(level=1, exclude={s.text for s in STORIES}) is None


class TestStrategyKB:
    def test_untried_tactics_start_at_half(self, tmp_path):
        stats = TacticStats(tmp_path / "tactics.json")
        tactic = best_tactic("th", stats)
        assert tactic is not None
        assert stats.success_rate(tactic) == 0.5

    def test_outcomes_change_which_tactic_wins(self, tmp_path):
        # slow_demonstration keeps failing; minimal_pairs keeps working.
        stats = TacticStats(tmp_path / "tactics.json")
        slow = next(t for t in _th_tactics() if t.name == "slow_demonstration")
        pairs = next(t for t in _th_tactics() if t.name == "minimal_pairs")
        for _ in range(3):
            stats.record_usage(slow)
            stats.record_outcome(slow, worked=False)
            stats.record_usage(pairs)
            stats.record_outcome(pairs, worked=True)
        assert best_tactic("th", stats).name == "minimal_pairs"

    def test_stats_persist_across_reloads(self, tmp_path):
        path = tmp_path / "tactics.json"
        slow = next(t for t in _th_tactics() if t.name == "slow_demonstration")
        stats = TacticStats(path)
        stats.record_usage(slow)
        stats.record_outcome(slow, worked=True)
        reloaded = TacticStats(path)
        assert reloaded.success_rate(slow) == (1 + 1) / (1 + 2)

    def test_unknown_phoneme_has_no_tactic(self, tmp_path):
        assert best_tactic("zz", TacticStats(tmp_path / "t.json")) is None


def _th_tactics():
    from storypal.learning.kb import TACTICS
    return [t for t in TACTICS if t.phoneme == "th"]
