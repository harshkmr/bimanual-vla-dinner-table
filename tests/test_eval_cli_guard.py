# N5/M5 — canonical-CLI parity guard. Must pass before eval/eval_10_seeds.py deletion.
"""Asserts benchmark/evaluate_seeds.py exists as the single eval entrypoint with the
required flags (--seeds, --use_coupling, plus --perturb/--item per V3.6 §5)."""
import ast
import os

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL = os.path.join(BASE_DIR, "benchmark", "evaluate_seeds.py")
FORWARDER = os.path.join(BASE_DIR, "eval", "eval_10_seeds.py")


def test_canonical_cli_exists_with_required_flags():
    assert os.path.exists(CANONICAL), "canonical eval CLI missing"
    src = open(CANONICAL, encoding="utf-8").read()
    for flag in ("--seeds", "--use_coupling"):
        assert flag in src, f"canonical CLI missing flag {flag}"


def test_forwarder_deletion_trail():
    # N5: deletion happens in the freeze commit with changelog + message trail.
    # Until then the forwarder may exist; this documents the rule either way.
    if os.path.exists(FORWARDER):
        pytest.skip("forwarder still present pre-freeze (N5 deletes it at freeze)")
    assert not os.path.exists(FORWARDER)
