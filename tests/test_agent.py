"""Tests for the agent loop guard rails, using the scripted FakeLLM.

Zero network, zero cost: the fake model replays queued replies and the
loop's behaviour around them is what we assert.
"""

from storypal.agent.loop import MAX_TOOL_CALLS, run_turn
from storypal.learning.kb import TacticStats
from storypal.agent.llm import FakeLLM, LLMReply, ToolCall
from storypal.learning.profile import Profile
from storypal.agent.tools import ToolContext

SYSTEM = "You are StoryPal."
CHILD_READ = 'The child read: "the bird flew the trees"'


def make_ctx(tmp_path, reliable=True):
    return ToolContext(
        profile=Profile(level=2),
        tactic_stats=TacticStats(tmp_path / "tactics.json"),
        asr_reliable=reliable,
    )


class TestPlainTextTurns:
    def test_text_reply_needs_no_tools(self, tmp_path):
        llm = FakeLLM([LLMReply(text="Wonderful reading!")])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path))
        assert result.reply == "Wonderful reading!"
        assert result.tool_calls == []


class TestToolFlow:
    def test_tool_call_then_reply(self, tmp_path):
        llm = FakeLLM([
            LLMReply(tool_call=ToolCall("drill_sound", {"phoneme": "th"})),
            LLMReply(text="Let's practice thhh together!"),
        ])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path))
        assert result.reply == "Let's practice thhh together!"
        name, args, tool_result = result.tool_calls[0]
        assert name == "drill_sound" and tool_result["tactic"]
        # The model must have been shown the tool result before replying.
        assert "Result of drill_sound" in llm.calls[1]["messages"][-1].content

    def test_unknown_tool_gets_structured_error_and_model_recovers(self, tmp_path):
        llm = FakeLLM([
            LLMReply(tool_call=ToolCall("play_game", {})),
            LLMReply(text="Let's read instead!"),
        ])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path))
        assert result.reply == "Let's read instead!"
        assert "unknown tool" in result.tool_calls[0][2]["error"]

    def test_bad_arguments_get_structured_error(self, tmp_path):
        llm = FakeLLM([
            LLMReply(tool_call=ToolCall("drill_sound", {"wrong_arg": True})),
            LLMReply(text="Okay!"),
        ])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path))
        assert "bad arguments" in result.tool_calls[0][2]["error"]


class TestGuardRails:
    def test_tool_call_loop_is_capped_and_forced_to_text(self, tmp_path):
        # A model that only ever wants tools must still end in a text reply.
        endless = [LLMReply(tool_call=ToolCall("next_sentence", {}))] * 10
        llm = FakeLLM(endless + [LLMReply(text="fallback")])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path))
        assert len(result.tool_calls) == MAX_TOOL_CALLS
        assert result.reply  # never empty
        # The forcing call must offer no tools at all.
        assert llm.calls[-1]["tools"] == []

    def test_drill_refused_on_unreliable_turn_but_conversation_continues(self, tmp_path):
        llm = FakeLLM([
            LLMReply(tool_call=ToolCall("drill_sound", {"phoneme": "th"})),
            LLMReply(text="Could you read it once more for me?"),
        ])
        result = run_turn(SYSTEM, CHILD_READ, llm, make_ctx(tmp_path, reliable=False))
        assert "unreliable" in result.tool_calls[0][2]["error"]
        assert "once more" in result.reply
