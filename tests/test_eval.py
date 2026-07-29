"""The eval suite doubles as a regression gate: every labeled case
must pass through the real pipeline."""

import json
from pathlib import Path

import pytest

from eval.run_eval import run_case

CASES = json.loads((Path(__file__).parent.parent / "eval" / "cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_eval_case(case):
    failures = run_case(case)
    assert not failures, "; ".join(failures)
