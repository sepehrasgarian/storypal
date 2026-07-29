"""Swappable LLM provider interface.

The rest of the system only knows this small surface: send a system
prompt plus messages (and tool schemas), get back either text or a
tool call. Tests use FakeLLM; production uses Gemini.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from storypal.config import LLM_MODEL


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


@dataclass(frozen=True)
class LLMReply:
    text: str | None = None
    tool_call: ToolCall | None = None


@dataclass
class Message:
    """One conversation item: a user/tool message the model should see."""

    role: str  # "user" | "tool"
    content: str


class LLM(Protocol):
    def chat(self, system: str, messages: list[Message], tools: list[dict]) -> LLMReply: ...


class FakeLLM:
    """Deterministic scripted model for tests: replays queued replies."""

    def __init__(self, replies: list[LLMReply]):
        self._replies = list(replies)
        self.calls: list[dict] = []  # every request, for assertions

    def chat(self, system: str, messages: list[Message], tools: list[dict]) -> LLMReply:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self._replies:
            return LLMReply(text="Okay!")
        return self._replies.pop(0)


class GeminiLLM:
    """Google Gemini via the google-genai SDK. Imported lazily so the
    package is only required when this provider is actually used."""

    def __init__(self, model: str = LLM_MODEL, api_key: str | None = None):
        from google import genai  # lazy: keeps tests dependency-free

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._model = model

    def chat(self, system: str, messages: list[Message], tools: list[dict]) -> LLMReply:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
        )
        contents = [m.content for m in messages]
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        call = response.function_calls[0] if response.function_calls else None
        if call is not None:
            return LLMReply(tool_call=ToolCall(name=call.name, args=dict(call.args or {})))
        return LLMReply(text=response.text or "")
