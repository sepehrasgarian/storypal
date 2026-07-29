"""Tests for the Tier 3 curated data store."""

from storypal.learning.curated import CuratedStore
from storypal.core.triage import Route


class TestCuratedStore:
    def test_archive_turns_are_not_stored(self, tmp_path):
        store = CuratedStore(tmp_path)
        store.add(Route.ARCHIVE, {"reply": "fine"})
        assert store.summary()["review_queue"]["count"] == 0
        assert store.summary()["finetune_set"]["count"] == 0

    def test_records_get_a_corrected_reply_slot(self, tmp_path):
        # The slot an offline curation step fills to form SFT/DPO pairs.
        store = CuratedStore(tmp_path)
        store.add(Route.FINETUNE_SET, {"reply": "bad reply"})
        sample = store.summary()["finetune_set"]["samples"][0]
        assert sample["corrected_reply"] is None

    def test_summary_counts_and_latest_samples(self, tmp_path):
        store = CuratedStore(tmp_path)
        for i in range(5):
            store.add(Route.REVIEW_QUEUE, {"reply": f"r{i}"})
        summary = store.summary(samples_per_pile=2)
        assert summary["review_queue"]["count"] == 5
        assert [s["reply"] for s in summary["review_queue"]["samples"]] == ["r3", "r4"]

    def test_empty_store_summarizes_cleanly(self, tmp_path):
        assert CuratedStore(tmp_path / "none").summary() == {
            "review_queue": {"count": 0, "samples": []},
            "finetune_set": {"count": 0, "samples": []},
        }
