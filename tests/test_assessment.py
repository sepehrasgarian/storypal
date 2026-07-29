"""Tests for word-level assessment: the deterministic heart of StoryPal."""

import pytest

from storypal.core.assessment import WordStatus, assess, edit_distance, normalize

TARGET = "The bird flew through the trees."


def statuses(assessment):
    return [v.status for v in assessment.verdicts]


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize("The bird, flew!") == ["the", "bird", "flew"]

    def test_spells_out_digits(self):
        assert normalize("2 birds saw 3 cats") == ["two", "birds", "saw", "three", "cats"]

    def test_keeps_apostrophes_inside_words(self):
        assert normalize("don't stop") == ["don't", "stop"]

    def test_empty_text(self):
        assert normalize("") == []


class TestEditDistance:
    @pytest.mark.parametrize(
        "a, b, expected",
        [("through", "through", 0), ("through", "though", 1), ("flew", "floo", 2), ("cat", "dog", 3)],
    )
    def test_known_distances(self, a, b, expected):
        assert edit_distance(a, b) == expected


class TestPerfectRead:
    def test_all_words_correct(self):
        result = assess(TARGET, "the bird flew through the trees")
        assert all(s is WordStatus.CORRECT for s in statuses(result))
        assert result.accuracy == 1.0

    def test_punctuation_and_case_do_not_matter(self):
        result = assess(TARGET, "THE BIRD FLEW THROUGH THE TREES!!!")
        assert result.accuracy == 1.0


class TestMistakes:
    def test_missed_word(self):
        result = assess(TARGET, "the bird flew the trees")
        missed = result.words_with_status(WordStatus.MISSED)
        assert [v.target_word for v in missed] == ["through"]
        assert result.accuracy == pytest.approx(5 / 6)

    def test_near_miss_is_probably_mispronunciation(self):
        result = assess(TARGET, "the bird floo through the trees")
        near = result.words_with_status(WordStatus.NEAR_MISS)
        assert [(v.target_word, v.heard_word) for v in near] == [("flew", "floo")]
        assert result.accuracy == pytest.approx((5 + 0.5) / 6)

    def test_clearly_different_word_is_substitution(self):
        result = assess(TARGET, "the bird jumped through the trees")
        subs = result.words_with_status(WordStatus.SUBSTITUTED)
        assert [(v.target_word, v.heard_word) for v in subs] == [("flew", "jumped")]

    def test_added_words(self):
        result = assess(TARGET, "the big bird flew through the trees")
        added = result.words_with_status(WordStatus.ADDED)
        assert [v.heard_word for v in added] == ["big"]
        assert result.accuracy == 1.0  # every target word was still read

    def test_multiple_mistakes_at_once(self):
        result = assess(TARGET, "bird floo the trees")
        assert [v.target_word for v in result.words_with_status(WordStatus.MISSED)] == ["the", "through"]
        assert len(result.words_with_status(WordStatus.NEAR_MISS)) == 1


class TestEdgeCases:
    def test_silence_means_everything_missed(self):
        result = assess(TARGET, "")
        assert all(s is WordStatus.MISSED for s in statuses(result))
        assert result.accuracy == 0.0

    def test_completely_unrelated_transcript(self):
        result = assess(TARGET, "thanks for watching")
        assert result.accuracy < 0.5

    def test_repeated_target_word_handled_positionally(self):
        # "the" appears twice in the target; reading it once should
        # count one correct and one missed, never two corrects.
        result = assess(TARGET, "bird flew through trees")
        correct_the = [
            v for v in result.verdicts
            if v.target_word == "the" and v.status is WordStatus.CORRECT
        ]
        assert len(correct_the) == 0
        assert len(result.words_with_status(WordStatus.MISSED)) == 2

    def test_digits_in_transcript_match_spelled_target(self):
        result = assess("Three small ships sailed north.", "3 small ships sailed north")
        assert result.accuracy == 1.0


class TestHomophones:
    """A child who reads 'son' for 'sun' produced the correct sound. A
    tutor that listens has no grounds to mark that wrong, and Whisper
    may write either spelling."""

    def test_homophone_counts_as_correct(self):
        result = assess("The sun is hot.", "the son is hot")
        assert result.accuracy == 1.0
        assert not result.words_with_status(WordStatus.NEAR_MISS)

    def test_genuinely_different_word_is_still_wrong(self):
        result = assess("The sun is hot.", "the fun is hot")
        assert result.accuracy < 1.0

    def test_homophone_does_not_leak_into_unrelated_words(self):
        result = assess("Three ships sailed to the dock.", "three ships sailed two the dock")
        assert result.accuracy == 1.0
