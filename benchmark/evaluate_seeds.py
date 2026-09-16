# Sprint: 10-seed dual scorecard -> output/10_seed_evaluation_report*.json
"""Oracle (coupling=1, full 6-phase flags) + autonomous BC-MLP (coupling=0).
Usage: python benchmark/evaluate_seeds.py --seeds 0-9"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.bimanual_so101_env import BimanualSO101Env  # noqa: E402
from data.scripted_expert import Oracle  # noqa: E402
from policy.train_mlp import BCMLP  # noqa: E402
from data.generate_demonstrations import obs_vec  # noqa: E402
from env import eval_scoring as S  # noqa: E402


def run_oracle(seeds, render=False):
    import numpy as np

    out = []
    for s in seeds:
        env = BimanualSO101Env(use_coupling=True, max_steps=1400)
        oc = Oracle(env)
        rep = oc.run(seed=s, item="fork", render_frames=render)
        if render:
            os.makedirs("output", exist_ok=True)
            np.savez_compressed(f"output/oracle_frames_{s}.npz", frames=np.array(oc.frames))
        out.append({"seed": s, "full": rep["full"], "steps": rep["steps"],
                    "mug_fill": round(rep["mug_fill"], 3), "flags": rep["flags"]})
        print("oracle", s, rep["full"], rep["flags"], flush=True)
    return out


def run_auto(seeds, weights):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    net = BCMLP(in_dim=int(ck["in_dim"])).eval()
    net.load_state_dict(ck["state"])
    mu, sd = ck["mu"], ck["sd"]
    out = []
    for s in seeds:
        env = BimanualSO101Env(use_coupling=False, max_steps=1200)
        obs = env.reset(seed=s)
        for _ in range(1200):
            x = torch.from_numpy(obs_vec(obs)).float()
            with torch.no_grad():
                a = net((x - mu) / sd).numpy()
            obs, done, _ = env.step(a[:12], float(a[12]))
            if done:
                break
        flags = {"drawer": S.check_drawer_open(obs),
                 "retrieved": bool(np.linalg.norm(obs["fork"] - obs["ee_a"]) < 0.10),
                 "handoff": False, "placed": S.check_item_placed(obs, "fork"),
                 "pour": bool(obs["mug_fill"] > 0.05), "mug": False}
        out.append({"seed": s, "full": S.full_task(flags), "steps": obs["t"],
                    "mug_fill": round(float(obs["mug_fill"]), 3), "flags": flags})
        print("auto", s, S.full_task(flags), flags, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--render-frames", action="store_true")
    ap.add_argument("--weights", default=os.path.join(REPO_ROOT, "weights", "mlp_bc10.pt"))
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    oracle = run_oracle(seeds, render=args.render_frames)
    auto = run_auto(seeds, args.weights)
    json.dump(oracle, open("output/10_seed_evaluation_report.json", "w"), indent=2)
    json.dump(auto, open("output/10_seed_evaluation_report_autonomous.json", "w"), indent=2)
    of = sum(r["full"] for r in oracle)
    af = sum(r["full"] for r in auto)
    print(f"oracle {of}/{len(oracle)} autonomous {af}/{len(auto)}")


if __name__ == "__main__":
    main()
