"""Audio-to-audio adversarial evaluation: two agents, one real pipeline.

A child agent speaks (Higgs TTS), StoryPal's real Whisper listens, and
the real grading runs. Because the child agent knows what it voiced, the
loop can separate the child's mistakes from the recogniser's - which is
the distinction the whole architecture rests on.

Costs a few cents of TTS on the first run and nothing afterwards; clips
are cached by content. Traces export to Langfuse when configured.

Usage:  PYTHONPATH=src python -m eval.audio_loop
"""

import os
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from eval.audio_child import BEHAVIOURS, ChildVoice
from storypal import observability
from storypal.config import STORIES
from storypal.core.assessment import normalize
from storypal.core.triage import route_turn
from storypal.session import grade_turn

load_dotenv()

TARGETS = [STORIES[0].text, STORIES[3].text, STORIES[5].text]


@dataclass
class Row:
    target: str
    behaviour: str
    said: str
    heard: str
    asr_faithful: bool
    trusted: bool
    expect_trusted: bool
    s1: float
    expect_perfect: bool
    false_correction: bool
    false_praise: bool
    ms: int


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        """Only false corrections count as failures. A trust decision that
        differs from the guess in the script is information, not a bug:
        the recogniser genuinely varies, which is the point of testing on
        real audio."""
        return [
            f"FALSE CORRECTION on '{r.behaviour}': voiced {r.said!r}, "
            f"heard {r.heard!r}, scored {r.s1:.0%} and acted on it"
            for r in self.rows if r.false_correction
        ]


def run(exporter=None) -> Report:
    from storypal.speech.asr import WhisperASR
    from storypal.speech.tts import HiggsTTS

    voice = ChildVoice(HiggsTTS(api_key=os.environ["BOSON_API_KEY"]))
    ears = WhisperASR()
    report = Report()

    for target in TARGETS:
        for behaviour in BEHAVIOURS:
            turn = behaviour(target)
            clip = voice.speak(turn)

            start = time.monotonic()
            heard = ears.transcribe(clip)
            ms = int((time.monotonic() - start) * 1000)

            graded = grade_turn(target, heard.transcript, heard.telemetry, None)
            faithful = normalize(turn.said) == normalize(heard.transcript)
            # The harm case: the child voiced a correct reading, the system
            # believed the recogniser anyway, and marked them wrong.
            false_correction = (
                turn.expect_perfect and graded.s2.reliable and graded.s1.score < 1.0
            )
            # The mirror image, and the one we cannot defend against: the
            # child voiced a flawed reading, the recogniser tidied it up,
            # and the system praised a mistake.
            false_praise = (
                not turn.expect_perfect and graded.s2.reliable
                and graded.s1.score == 1.0 and not faithful
            )
            report.rows.append(Row(
                target=target, behaviour=turn.behaviour, said=turn.said,
                heard=heard.transcript, asr_faithful=faithful,
                trusted=graded.s2.reliable, expect_trusted=turn.expect_trusted,
                s1=graded.s1.score, expect_perfect=turn.expect_perfect,
                false_correction=false_correction, false_praise=false_praise, ms=ms,
            ))

            if exporter is not None:
                exporter.export_turn(
                    session_id="audio-eval", turn=len(report.rows),
                    timestamp=time.time(), target=target,
                    transcript=heard.transcript, reply=f"[{turn.behaviour}]",
                    prompt=f"child agent voiced: {turn.said!r}",
                    signals={"S1": graded.s1, "S2": graded.s2},
                    timings_ms={"asr_ms": ms},
                    route=route_turn({"S1": graded.s1, "S2": graded.s2},
                                     chat_turn=graded.chat_turn).route.value,
                )
    return report


def main() -> int:
    exporter = observability.from_env()
    print("# StoryPal audio-to-audio evaluation\n")
    print("A child agent speaks; the real Whisper listens.")
    print(f"Langfuse export: {'on' if exporter else 'off (no keys configured)'}\n")

    report = run(exporter)
    faithful = sum(r.asr_faithful for r in report.rows)
    total = len(report.rows)
    print(f"{'behaviour':28} {'trusted':8} {'S1':>5}  heard")
    print("-" * 92)
    for r in report.rows:
        mark = "" if r.asr_faithful else "  <- recogniser differs from what was voiced"
        print(f"{r.behaviour:28} {str(r.trusted):8} {r.s1:5.0%}  {r.heard[:34]!r}{mark}")

    praised = [r for r in report.rows if r.false_praise]
    print(f"\nrecogniser heard exactly what was voiced: {faithful}/{total}")
    print(f"false corrections (child was right, we said wrong): "
          f"{sum(r.false_correction for r in report.rows)}")
    print(f"false praise (child was wrong, recogniser tidied it up): {len(praised)}")
    for r in praised:
        print(f"    voiced {r.said!r}\n    heard  {r.heard!r}  scored 100%")
    print(f"median transcribe time: "
          f"{sorted(r.ms for r in report.rows)[len(report.rows) // 2]}ms")

    if report.failures:
        print("\nFAILURES:")
        for f in report.failures:
            print("  -", f)
        return 1
    print("\nNo child was mistreated for the recogniser's mistakes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
