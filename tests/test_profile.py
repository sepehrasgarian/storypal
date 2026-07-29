"""Tests for the Tier 2 learner profile."""

from api.assessment import assess
from api.profile import (
    Profile, load, phonemes_in_word, render, save, update_from_turn, weakest_phoneme,
)
from api.signals import Signal

TARGET = "The bird flew through the trees."
RELIABLE = Signal("S2", score=1.0, reliable=True)
UNRELIABLE = Signal("S2", score=0.0, reliable=False)


class TestPhonemesInWord:
    def test_th_beats_t(self):
        assert phonemes_in_word("through") == ["th", "r"]

    def test_plain_word(self):
        assert phonemes_in_word("bird") == ["r", "d"]

    def test_word_with_no_tracked_sounds(self):
        assert phonemes_in_word("bee") == []


class TestUpdateFromTurn:
    def test_missed_word_is_recorded_with_its_phonemes(self):
        profile = update_from_turn(Profile(), assess(TARGET, "the bird flew the trees"), RELIABLE)
        assert profile.missed_words == {"through": 1}
        assert profile.weak_phonemes == {"th": 1, "r": 1}
        assert profile.total_turns == 1

    def test_unreliable_turn_changes_nothing(self):
        # The core guarantee: hallucinated transcripts never poison memory.
        profile = update_from_turn(Profile(), assess(TARGET, "thanks for watching"), UNRELIABLE)
        assert profile == Profile()

    def test_repeated_misses_accumulate(self):
        profile = Profile()
        for _ in range(3):
            update_from_turn(profile, assess(TARGET, "the bird flew the trees"), RELIABLE)
        assert profile.missed_words["through"] == 3
        assert weakest_phoneme(profile) in ("th", "r")

    def test_perfect_read_only_counts_the_turn(self):
        profile = update_from_turn(Profile(), assess(TARGET, "the bird flew through the trees"), RELIABLE)
        assert profile.total_turns == 1
        assert profile.missed_words == {}


class TestRenderAndPersistence:
    def test_new_learner_renders_gently(self):
        assert "new learner" in render(Profile())

    def test_history_renders_weak_sounds(self):
        profile = update_from_turn(Profile(), assess(TARGET, "the bird flew the trees"), RELIABLE)
        text = render(profile)
        assert "'th'" in text and "through" in text

    def test_save_load_round_trip(self, tmp_path):
        profile = update_from_turn(Profile(), assess(TARGET, "the bird flew the trees"), RELIABLE)
        path = tmp_path / "profile.json"
        save(profile, path)
        assert load(path) == profile

    def test_loading_missing_file_gives_fresh_profile(self, tmp_path):
        assert load(tmp_path / "nope.json") == Profile()
