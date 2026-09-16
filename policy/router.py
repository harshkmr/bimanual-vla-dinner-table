# N1 — deterministic confidence router (pure function, config-driven, no heuristics).
"""Routes each action chunk to SmolVLA (primary) or ACT (fallback).

Rule: route to ACT only when mean(last-N chunk confidences) < tau, with tau/N read
from config/router_config.json. policy_selection flips the default when the N2
kill-switch fires (flag flip, never a code edit).
"""
import json

SMOLVLA = "smolvla"
ACT = "act"


def load_config(path="config/router_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decide_route(confidences, tau, window_n, policy_selection="smolvla-primary"):
    """Pure function: (sequence of chunk confidences, tau, window) -> route string."""
    if policy_selection == "act-primary":
        return ACT
    window = list(confidences)[-window_n:] if window_n > 0 else []
    if not window:
        return SMOLVLA
    prob = sum(window) / len(window)
    return ACT if prob < tau else SMOLVLA


def route_mix(decisions):
    """Count per-seed routing outcomes for the eval JSON (smolvla|act|mixed + counts)."""
    counts = {SMOLVLA: 0, ACT: 0}
    for d in decisions:
        counts[d] += 1
    kind = "mixed" if counts[SMOLVLA] and counts[ACT] else (SMOLVLA if counts[SMOLVLA] else ACT)
    return {"kind": kind, "counts": counts}
