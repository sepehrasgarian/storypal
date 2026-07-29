"""Replay the labeled cases through the real pipeline and report metrics.

Usage:  python -m eval.run_eval

For S2 the report is a proper confusion matrix over hallucination
detection (positive class = unreliable). Exit code 1 on any failure,
so this doubles as a regression gate.
"""

import json
import sys
from pathlib import Path

from storypal.core.assessment import WordStatus, assess
from storypal.core.signals import AsrTelemetry, s1_reading_accuracy, s2_asr_reliability

PROBLEM_STATUSES = (WordStatus.MISSED, WordStatus.NEAR_MISS, WordStatus.SUBSTITUTED)


def run_case(case: dict) -> list[str]:
    """Return a list of failure descriptions (empty = case passed)."""
    assessment = assess(case["target"], case["transcript"])
    s1 = s1_reading_accuracy(assessment)
    s2 = s2_asr_reliability(AsrTelemetry(**case["telemetry"]), assessment)
    expect = case["expect"]
    failures = []

    if s2.reliable != expect["s2_reliable"]:
        failures.append(f"S2: expected reliable={expect['s2_reliable']}, got {s2.reliable}")

    if expect["s2_reliable"]:  # S1 expectations only make sense on trusted turns
        if "accuracy_min" in expect and s1.score < expect["accuracy_min"]:
            failures.append(f"S1: accuracy {s1.score:.2f} < {expect['accuracy_min']}")
        if "accuracy_max" in expect and s1.score > expect["accuracy_max"]:
            failures.append(f"S1: accuracy {s1.score:.2f} > {expect['accuracy_max']}")
        if "problem_words" in expect:
            found = sorted(
                v.target_word for v in assessment.verdicts if v.status in PROBLEM_STATUSES
            )
            if found != sorted(expect["problem_words"]):
                failures.append(f"S1: problem words {found} != expected {sorted(expect['problem_words'])}")
    return failures


def main() -> int:
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    results = {c["name"]: run_case(c) for c in cases}

    # S2 confusion matrix: positive class = "unreliable" (hallucination).
    tp = fp = fn = tn = 0
    for case in cases:
        expected_bad = not case["expect"]["s2_reliable"]
        s2 = s2_asr_reliability(
            AsrTelemetry(**case["telemetry"]),
            assess(case["target"], case["transcript"]),
        )
        detected_bad = not s2.reliable
        tp += expected_bad and detected_bad
        fp += (not expected_bad) and detected_bad
        fn += expected_bad and (not detected_bad)
        tn += (not expected_bad) and (not detected_bad)

    passed = sum(1 for f in results.values() if not f)
    print("# StoryPal eval report\n")
    print(f"cases passed: {passed}/{len(cases)}\n")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    print("S2 hallucination detection:")
    print(f"  precision {precision:.2f}   recall {recall:.2f}   (tp={tp} fp={fp} fn={fn} tn={tn})\n")
    for name, failures in results.items():
        mark = "PASS" if not failures else "FAIL"
        print(f"[{mark}] {name}")
        for failure in failures:
            print(f"       - {failure}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
