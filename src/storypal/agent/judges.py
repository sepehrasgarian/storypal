"""The LLM judges: S3 (grounding) and S4 (pedagogical fit).

They evaluate the tutor's REPLY, so they run after it is sent — never
on the latency path. Each returns a Signal like the deterministic
scorers, so triage treats all four uniformly.
"""

import json
import re

from storypal.agent.llm import LLM, Message
from storypal.core.signals import Signal

_S3_RUBRIC = """You are auditing a children's reading tutor.
Assessment of what the child actually read:
{assessment}

The tutor replied:
"{reply}"

Question: does the reply claim anything the assessment does not support
(praising words not read correctly, correcting words that were right,
inventing details)? Score 1.0 = fully grounded, 0.0 = clearly ungrounded.
Answer with only JSON: {{"score": <0..1>, "reason": "<one sentence>"}}"""

_S4_RUBRIC = """You are auditing a children's reading tutor for a 7-year-old.
Assessment of the child's reading:
{assessment}

The tutor replied:
"{reply}"

Question: is this reply pedagogically good — kind, starts positive,
age-appropriate, and drills the right thing? Score 1.0 = excellent,
0.0 = harmful (harsh, wrong target, or discouraging).
Answer with only JSON: {{"score": <0..1>, "reason": "<one sentence>"}}"""


def s3_grounding(assessment_summary: str, reply: str, llm: LLM) -> Signal:
    return _run_judge("S3", _S3_RUBRIC, assessment_summary, reply, llm)


def s4_pedagogy(assessment_summary: str, reply: str, llm: LLM) -> Signal:
    return _run_judge("S4", _S4_RUBRIC, assessment_summary, reply, llm)


def _run_judge(signal_id: str, rubric: str, assessment_summary: str, reply: str, llm: LLM) -> Signal:
    prompt = rubric.format(assessment=assessment_summary, reply=reply)
    response = llm.chat(system="You are a strict, fair auditor.",
                        messages=[Message(role="user", content=prompt)], tools=[])
    score, reason = _parse(response.text or "")
    return Signal(id=signal_id, score=score, reliable=True, reasons=(reason,))


def _parse(text: str) -> tuple[float, str]:
    """Extract {"score", "reason"} from the judge's reply; a judge that
    cannot be parsed abstains with a passing score rather than
    condemning a reply on garbage."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            score = max(0.0, min(1.0, float(data.get("score", 1.0))))
            return score, str(data.get("reason", "no reason given"))
        except (ValueError, TypeError):
            pass
    return 1.0, "judge output unparseable; abstained"
