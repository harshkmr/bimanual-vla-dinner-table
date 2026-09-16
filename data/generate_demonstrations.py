# Sprint: oracle rollouts -> behavior-cloning dataset (obs vector, action) per step.
"""Usage: python data/generate_demonstrations.py --seeds 0-9 --out data/demos.npz
Obs (33D): qpos12 + ee6 + fork/plate/mug/bottle xyz (12) + drawer + mug_fill + bottle_liquid (3).
Action (13D): ctrl12 + drawer_cmd. Recorded from oracle WITH coupling (disclosed)."""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.bimanual_so101_env import BimanualSO101Env  # noqa: E402
from data.scripted_expert import Oracle, GRIP_OPEN, GRIP_CLOSED  # noqa: E402


def obs_vec(o, phase=0):
    base = np.concatenate([o["qpos_a"], o["qpos_b"], o["ee_a"], o["ee_b"], o["fork"],
                           o["plate"], o["mug"], o["bottle"],
                           [o["drawer_qpos"], o["mug_fill"], o["bottle_liquid"]]]).astype(np.float32)
    onehot = np.zeros(6, dtype=np.float32)
    onehot[int(phase)] = 1.0
    return np.concatenate([base, onehot])


def rollout(seed, item="fork"):
    env = BimanualSO101Env(use_coupling=True, max_steps=1400)
    oc = Oracle(env)
    obs = env.reset(seed=seed)
    X, Y = [], []
    orig_step = env.step

    def rec_step(ctrl12, drawer_cmd=0.0):
        X.append(obs_vec(env._obs(), oc.phase))
        c = np.concatenate([np.asarray(ctrl12).ravel(), [drawer_cmd]]).astype(np.float32)
        Y.append(c)
        return orig_step(ctrl12, drawer_cmd)

    env.step = rec_step
    rep = oc.run(seed=seed, item=item)
    return np.array(X), np.array(Y), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "demos.npz"))
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    Xs, Ys = [], []
    for s in range(lo, hi + 1):
        X, Y, rep = rollout(s)
        print(f"seed {s}: {len(X)} steps full={rep['full']}", flush=True)
        if rep["full"]:
            Xs.append(X)
            Ys.append(Y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    np.savez_compressed(args.out, X=X, Y=Y)
    print(f"saved {args.out}: X{X.shape} Y{Y.shape}")


if __name__ == "__main__":
    main()
