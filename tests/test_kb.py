"""Tests for the content and strategy knowledge bases."""

from storypal.learning.kb import TacticStats, best_tactic, next_sentence


class TestDerivedPhonemeTags:
    """Tags come from the words themselves - hand-written ones drifted
    (they missed the 'th' in "The"), silently starving a th-weak child
    of th practice."""

    def test_tags_match_the_actual_words(self):
        from storypal.config import STORIES
        story = next(s for s in STORIES if s.text == "The sun is hot.")
        assert "th" in story.phonemes  # from "The"

    def test_weak_sound_is_practised_at_the_childs_own_level(self):
        story = next_sentence(level=1, focus_phoneme="th")
        assert story.level == 1
        assert "th" in story.phonemes


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


class TestTacticProvenance:
    """Tactics come from established reading instruction, not invention,
    and must work over an audio-only channel."""

    def test_every_tracked_sound_has_a_tactic(self):
        from storypal.config import TRACKED_PHONEMES
        from storypal.learning.kb import TACTICS
        covered = {t.phoneme for t in TACTICS}
        assert set(TRACKED_PHONEMES) <= covered

    def test_every_tactic_cites_its_method(self):
        from storypal.learning.kb import TACTICS
        assert all(t.source for t in TACTICS)

    def test_sounds_have_alternatives_to_choose_between(self):
        # A scoreboard is pointless if there is only ever one option.
        from storypal.learning.kb import TACTICS
        counts = {}
        for t in TACTICS:
            counts[t.phoneme] = counts.get(t.phoneme, 0) + 1
        assert all(n >= 2 for n in counts.values())


class TestStrategyKB:
    def test_untried_tactics_start_at_half(self, tmp_path):
        stats = TacticStats(tmp_path / "tactics.json")
        tactic = best_tactic("th", stats)
        assert tactic is not None
        assert stats.success_rate(tactic) == 0.5

    def test_outcomes_change_which_tactic_wins(self, tmp_path):
        # articulatory_cue keeps failing; minimal_pairs keeps working.
        stats = TacticStats(tmp_path / "tactics.json")
        cue = next(t for t in _th_tactics() if t.name == "articulatory_cue")
        pairs = next(t for t in _th_tactics() if t.name == "minimal_pairs")
        for _ in range(3):
            stats.record_usage(cue)
            stats.record_outcome(cue, worked=False)
            stats.record_usage(pairs)
            stats.record_outcome(pairs, worked=True)
        assert best_tactic("th", stats).name == "minimal_pairs"

    def test_stats_persist_across_reloads(self, tmp_path):
        path = tmp_path / "tactics.json"
        cue = next(t for t in _th_tactics() if t.name == "articulatory_cue")
        stats = TacticStats(path)
        stats.record_usage(cue)
        stats.record_outcome(cue, worked=True)
        reloaded = TacticStats(path)
        assert reloaded.success_rate(cue) == (1 + 1) / (1 + 2)

    def test_unknown_phoneme_has_no_tactic(self, tmp_path):
        assert best_tactic("zz", TacticStats(tmp_path / "t.json")) is None


def _th_tactics():
    from storypal.learning.kb import TACTICS
    return [t for t in TACTICS if t.phoneme == "th"]
