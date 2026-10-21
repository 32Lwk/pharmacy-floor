"""Slow: loads Decider 2B (~4 GB, CPU). Run with RUN_DECIDER=1 python -m pytest tests/test_regression_decider.py"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DECIDER") != "1", reason="needs the local Decider model (RUN_DECIDER=1)")


def test_regression_cases_pass():
    from floor.regress import run
    bad = [r for r in run() if not r["ok"]]
    assert not bad, bad
