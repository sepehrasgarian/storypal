"""Tests for the S3/S4 judges and their defensive parsing."""

from api.judges import s3_grounding, s4_pedagogy
from api.llm import FakeLLM, LLMReply


def judge_reply(text):
    return FakeLLM([LLMReply(text=text)])


class TestJudgeParsing:
    def test_clean_json_verdict(self):
        llm = judge_reply('{"score": 0.2, "reason": "praised a word the child missed"}')
        signal = s3_grounding("missed 'through'", "Perfect! You read every word!", llm)
        assert signal.id == "S3"
        assert signal.score == 0.2
        assert "praised" in signal.reasons[0]

    def test_json_wrapped_in_prose_is_still_found(self):
        llm = judge_reply('Sure! Here is my verdict: {"score": 0.9, "reason": "well grounded"} Hope that helps.')
        assert s4_pedagogy("perfect read", "Great job!", llm).score == 0.9

    def test_unparseable_output_abstains_with_pass(self):
        # A broken judge must not condemn a good reply.
        signal = s3_grounding("perfect read", "Great job!", judge_reply("I think it was fine?"))
        assert signal.score == 1.0
        assert "abstained" in signal.reasons[0]

    def test_out_of_range_score_is_clamped(self):
        assert s4_pedagogy("x", "y", judge_reply('{"score": 7, "reason": "r"}')).score == 1.0
        assert s4_pedagogy("x", "y", judge_reply('{"score": -3, "reason": "r"}')).score == 0.0

    def test_judge_sees_assessment_and_reply(self):
        llm = judge_reply('{"score": 1.0, "reason": "ok"}')
        s3_grounding("missed 'through'", "Let us practice through", llm)
        sent = llm.calls[0]["messages"][0].content
        assert "missed 'through'" in sent and "Let us practice through" in sent
