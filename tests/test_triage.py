"""Tests for the triage routing table: every rule, plus precedence."""

from storypal.core.signals import Signal
from storypal.core.triage import Route, route_turn


def reliable(signal_id, score=1.0):
    return Signal(id=signal_id, score=score, reliable=True)


def unreliable(signal_id):
    return Signal(id=signal_id, score=0.0, reliable=False, reasons=("test",))


class TestEachRule:
    def test_unreliable_asr_goes_to_review_queue(self):
        decision = route_turn({"S2": unreliable("S2")})
        assert decision.route is Route.REVIEW_QUEUE
        assert "cannot be trusted" in decision.reason

    def test_harmful_correction_goes_to_finetune_set(self):
        decision = route_turn({"S2": reliable("S2"), "S4": reliable("S4", score=0.2)})
        assert decision.route is Route.FINETUNE_SET
        assert "S4" in decision.reason

    def test_ungrounded_claim_goes_to_finetune_set(self):
        decision = route_turn({"S2": reliable("S2"), "S3": reliable("S3", score=0.0)})
        assert decision.route is Route.FINETUNE_SET
        assert "S3" in decision.reason

    def test_normal_turn_is_archived(self):
        decision = route_turn({
            "S1": reliable("S1", score=0.8),
            "S2": reliable("S2"),
            "S3": reliable("S3", score=0.9),
            "S4": reliable("S4", score=0.9),
        })
        assert decision.route is Route.ARCHIVE


class TestPrecedenceAndEdges:
    def test_unreliable_asr_wins_over_judge_failures(self):
        # If the ears failed, the judges were judging garbage: review first.
        decision = route_turn({"S2": unreliable("S2"), "S4": reliable("S4", score=0.0)})
        assert decision.route is Route.REVIEW_QUEUE

    def test_missing_judges_is_normal(self):
        # S3/S4 run async and may not have arrived yet.
        decision = route_turn({"S1": reliable("S1"), "S2": reliable("S2")})
        assert decision.route is Route.ARCHIVE

    def test_no_signals_at_all_is_archived(self):
        assert route_turn({}).route is Route.ARCHIVE

    def test_judge_exactly_at_threshold_passes(self):
        decision = route_turn({"S2": reliable("S2"), "S4": reliable("S4", score=0.5)})
        assert decision.route is Route.ARCHIVE
