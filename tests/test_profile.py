"""Tests for the Tier 2 learner profile."""

from storypal.core.assessment import assess
from storypal.learning.profile import (
    Profile, load, phonemes_in_word, render, save, update_from_turn, weakest_phoneme,
)
from storypal.core.signals import Signal

TARGET = "That thick rug is soft."
RELIABLE = Signal("S2", score=1.0, reliable=True)
UNRELIABLE = Signal("S2", score=0.0, reliable=False)


class TestPhonemesInWord:
    """We only count a sound where spelling reliably predicts it:
    digraphs at either edge, single consonants word-initially. Counting
    nothing beats blaming a sound the child never made."""

    def test_digraph_at_the_start(self):
        assert phonemes_in_word("this") == ["th"]

    def test_digraph_at_the_end(self):
        assert phonemes_in_word("fish") == ["sh"]

    def test_initial_consonant(self):
        assert phonemes_in_word("red") == ["r"]

    def test_buried_consonant_is_not_counted(self):
        # The 'r' in "through" is inside a 'thr' blend, not a clean /r/.
        assert phonemes_in_word("through") == ["th"]

    def test_r_controlled_vowel_is_not_an_r_sound(self):
        # "bird" is not evidence about the /r/ consonant.
        assert phonemes_in_word("bird") == []

    def test_word_with_no_tracked_sounds(self):
        assert phonemes_in_word("bee") == []


class TestUpdateFromTurn:
    def test_missed_word_is_recorded_with_its_phonemes(self):
        profile = update_from_turn(Profile(), assess(TARGET, "that thick rug is"), RELIABLE)
        assert profile.missed_words == {"soft": 1}
        assert profile.weak_phonemes == {"s": 1}
        assert profile.total_turns == 1

    def test_unreliable_turn_changes_nothing(self):
        # The core guarantee: hallucinated transcripts never poison memory.
        profile = update_from_turn(Profile(), assess(TARGET, "thanks for watching"), UNRELIABLE)
        assert profile == Profile()

    def test_repeated_misses_accumulate(self):
        profile = Profile()
        for _ in range(3):
            update_from_turn(profile, assess(TARGET, "that thick rug is"), RELIABLE)
        assert profile.missed_words["soft"] == 3
        assert weakest_phoneme(profile) == "s"

    def test_perfect_read_only_counts_the_turn(self):
        profile = update_from_turn(Profile(), assess(TARGET, "that thick rug is soft"), RELIABLE)
        assert profile.total_turns == 1
        assert profile.missed_words == {}


class TestRenderAndPersistence:
    def test_new_learner_renders_gently(self):
        assert "new learner" in render(Profile())

    def test_history_renders_weak_sounds(self):
        profile = update_from_turn(Profile(), assess(TARGET, "that thick rug is"), RELIABLE)
        text = render(profile)
        assert "'s'" in text and "soft" in text

    def test_save_load_round_trip(self, tmp_path):
        profile = update_from_turn(Profile(), assess(TARGET, "that thick rug is"), RELIABLE)
        path = tmp_path / "profile.json"
        save(profile, path)
        assert load(path) == profile

    def test_loading_missing_file_gives_fresh_profile(self, tmp_path):
        assert load(tmp_path / "nope.json") == Profile()
