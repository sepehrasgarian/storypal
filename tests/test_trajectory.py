"""Tests for the trajectory flight recorder."""

from storypal.core.signals import Signal
from storypal.core.trajectory import TrajectoryLog, TurnRecord, reward_from_signals


def make_record(session="s1", turn=0, composite=1.0):
    return TurnRecord(
        session_id=session,
        turn=turn,
        timestamp=1000.0 + turn,
        state={"target": "The bird flew through the trees."},
        action={"reply": "Great job!", "tool_calls": []},
        reward={"S1": 1.0, "composite": composite},
        route="archive",
    )


class TestRewardFromSignals:
    def test_composite_is_mean_of_scores(self):
        signals = {
            "S1": Signal("S1", score=1.0, reliable=True),
            "S2": Signal("S2", score=0.0, reliable=False),
        }
        reward = reward_from_signals(signals)
        assert reward == {"S1": 1.0, "S2": 0.0, "composite": 0.5}

    def test_no_signals_gives_zero_composite(self):
        assert reward_from_signals({}) == {"composite": 0.0}


class TestTrajectoryLog:
    def test_append_then_read_round_trip(self, tmp_path):
        log = TrajectoryLog(tmp_path / "traj.jsonl")
        record = make_record()
        log.append(record)
        assert log.read_all() == [record]

    def test_creates_parent_directories(self, tmp_path):
        log = TrajectoryLog(tmp_path / "deep" / "nested" / "traj.jsonl")
        log.append(make_record())
        assert len(log.read_all()) == 1

    def test_reading_missing_file_is_empty(self, tmp_path):
        assert TrajectoryLog(tmp_path / "nope.jsonl").read_all() == []

    def test_episode_filters_and_orders_by_turn(self, tmp_path):
        log = TrajectoryLog(tmp_path / "traj.jsonl")
        log.append(make_record(session="a", turn=1))
        log.append(make_record(session="b", turn=0))
        log.append(make_record(session="a", turn=0))
        episode = log.episode("a")
        assert [r.turn for r in episode] == [0, 1]
        assert all(r.session_id == "a" for r in episode)
