# N1 — router determinism: same seeds + same config -> identical decisions 10/10.
import json
import os

from policy.router import decide_route, load_config, route_mix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE_DIR, "config", "router_config.json")


def test_router_determinism_10_of_10():
    cfg = load_config(CONFIG)
    tau, n = cfg["tau"], cfg["window_n"]
    confidences = [0.9, 0.4, 0.7, 0.3, 0.8, 0.2, 0.95, 0.55, 0.6, 0.1]
    first = [decide_route(confidences[:i + 1], tau, n) for i in range(10)]
    for _ in range(9):
        again = [decide_route(confidences[:i + 1], tau, n) for i in range(10)]
        assert again == first


def test_router_boundary_and_kill_switch():
    assert decide_route([0.9, 0.9], 0.65, 5) == "smolvla"
    assert decide_route([0.1, 0.1], 0.65, 5) == "act"
    assert decide_route([], 0.65, 5) == "smolvla"
    assert decide_route([0.99] * 5, 0.65, 5, policy_selection="act-primary") == "act"


def test_route_mix_shape():
    mix = route_mix(["smolvla", "smolvla", "act"])
    assert mix["kind"] == "mixed" and mix["counts"] == {"smolvla": 2, "act": 1}
    assert json.dumps(mix)  # JSON-serializable for per-seed eval output
