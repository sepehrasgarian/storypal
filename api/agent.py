"""The agent loop: model -> (maybe tool) -> model -> reply.

Guard rails, all tested:
- plain text is a complete answer (most turns need no tool)
- a bad tool name or bad arguments produce a structured error the
  model sees, with limited retries — never a crashed turn
- hard cap on tool calls per turn, then a forced text reply
"""

import json
from dataclasses import dataclass, field

from api.llm import LLM, LLMReply, Message
from api.tools import HANDLERS, SCHEMAS, ToolContext

MAX_TOOL_CALLS = 2

# A child must never hear silence because a model misbehaved.
FALLBACK_REPLY = "Let's keep reading together! Can you read the sentence one more time?"


@dataclass
class AgentResult:
    reply: str
    tool_calls: list = field(default_factory=list)  # (name, args, result) triples


def run_turn(system_prompt: str, first_message: str, llm: LLM, ctx: ToolContext) -> AgentResult:
    """Drive one tutoring turn to a final text reply."""
    messages = [Message(role="user", content=first_message)]
    tool_calls = []

    for _ in range(MAX_TOOL_CALLS + 1):
        reply = llm.chat(system=system_prompt, messages=messages, tools=SCHEMAS)

        if reply.tool_call is None:
            return AgentResult(reply=reply.text or "", tool_calls=tool_calls)

        if len(tool_calls) >= MAX_TOOL_CALLS:
            break  # the model keeps asking for tools; force a text answer

        call = reply.tool_call
        result = _dispatch(call.name, call.args, ctx)
        tool_calls.append((call.name, call.args, result))
        messages.append(Message(
            role="tool",
            content=f"Result of {call.name}({json.dumps(call.args)}): {json.dumps(result)}. "
                    "Now answer the child in 2-3 warm sentences.",
        ))

    # Cap reached: one last call with tools disabled forces plain text.
    final = llm.chat(system=system_prompt, messages=messages, tools=[])
    return AgentResult(reply=final.text or FALLBACK_REPLY, tool_calls=tool_calls)


def _dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool '{name}'; available: {', '.join(HANDLERS)}"}
    try:
        return handler(ctx, **args)
    except TypeError:
        return {"error": f"bad arguments for {name}: {args}"}
