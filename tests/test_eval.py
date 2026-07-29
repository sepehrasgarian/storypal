"""The eval suite doubles as a regression gate: every labeled case
must pass through the real pipeline."""

import pytest

from eval.run_eval import load_cases, run_case

CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_eval_case(case):
    failures = run_case(case)
    assert not failures, "; ".join(failures)
