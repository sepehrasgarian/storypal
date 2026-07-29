"""Optional Langfuse export: one trace per turn, spans per pipeline
stage, and the four signals as scores.

Uses the public ingestion REST API directly (no SDK dependency).
Disabled unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set;
export failures are swallowed — observability must never break a turn.
"""

import os
import uuid
from datetime import datetime, timezone

import httpx

DEFAULT_HOST = "https://cloud.langfuse.com"


def from_env(client: httpx.Client | None = None) -> "LangfuseExporter | None":
    """Build an exporter from the environment, or None when unconfigured."""
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return None
    # Langfuse's own setup snippet emits LANGFUSE_BASE_URL; their SDKs
    # read LANGFUSE_HOST. Accept either so a pasted snippet just works.
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or DEFAULT_HOST
    return LangfuseExporter(public, secret, host, client=client)


class LangfuseExporter:
    def __init__(self, public_key: str, secret_key: str, host: str = DEFAULT_HOST,
                 client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            base_url=host, auth=(public_key, secret_key), timeout=10
        )

    def export_turn(self, *, session_id: str, turn: int, timestamp: float,
                    target: str, transcript: str, reply: str, prompt: str,
                    signals: dict, timings_ms: dict, route: str) -> None:
        """Ship one turn as trace + stage spans + signal scores."""
        try:
            self._client.post("/api/public/ingestion",
                              json={"batch": self._events(
                                  session_id, turn, timestamp, target, transcript,
                                  reply, prompt, signals, timings_ms, route)})
        except Exception:
            pass  # never let observability break the tutor

    def _events(self, session_id, turn, timestamp, target, transcript,
                reply, prompt, signals, timings_ms, route) -> list[dict]:
        trace_id = uuid.uuid4().hex
        now = _iso(timestamp)
        events = [_event("trace-create", {
            "id": trace_id,
            "name": "reading-turn",
            "sessionId": session_id,
            "timestamp": now,
            "input": {"target": target, "transcript": transcript},
            "output": {"reply": reply},
            "metadata": {"turn": turn, "route": route, "prompt": prompt},
        }, now)]

        start_ms = timestamp * 1000
        for stage in ("asr_ms", "agent_ms", "tts_ms"):
            duration = timings_ms.get(stage, 0)
            events.append(_event("span-create", {
                "id": uuid.uuid4().hex,
                "traceId": trace_id,
                "name": stage.removesuffix("_ms"),
                "startTime": _iso(start_ms / 1000),
                "endTime": _iso((start_ms + duration) / 1000),
            }, now))
            start_ms += duration

        for signal_id, signal in signals.items():
            events.append(_event("score-create", {
                "id": uuid.uuid4().hex,
                "traceId": trace_id,
                "name": signal_id,
                "value": signal.score,
                "comment": "; ".join(signal.reasons),
            }, now))
        return events


def _event(kind: str, body: dict, timestamp: str) -> dict:
    return {"id": uuid.uuid4().hex, "type": kind, "timestamp": timestamp, "body": body}


def _iso(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat()
