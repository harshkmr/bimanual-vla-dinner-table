# Sprint: evaluate the BC-MLP with coupling=0 (pure contact; no kinematic help).
"""Usage: python policy/eval_mlp.py --weights weights/mlp_bc.pt --seed 0 [--render-frames]
Reports per-phase flags via eval_scoring (drawer/retrieve/handoff/place/pour/mug)."""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.bimanual_so101_env import BimanualSO101Env  # noqa: E402
from env import eval_scoring as S  # noqa: E402
from policy.train_mlp import BCMLP  # noqa: E402
from data.generate_demonstrations import obs_vec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(REPO_ROOT, "weights", "mlp_bc.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--coupling", type=int, default=0)
    args = ap.parse_args()
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    net = BCMLP()
    net.load_state_dict(ck["state"])
    net.eval()
    mu, sd = ck["mu"], ck["sd"]
    env = BimanualSO101Env(use_coupling=bool(args.coupling), max_steps=args.steps)
    obs = env.reset(seed=args.seed)
    held_fork = False
    for _ in range(args.steps):
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
    print(f"seed={args.seed} steps={obs['t']} fill={obs['mug_fill']:.2f}")
    print("flags:", flags, "full:", S.full_task(flags))


if __name__ == "__main__":
    main()
