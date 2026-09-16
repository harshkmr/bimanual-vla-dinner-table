# Sprint oracle: 6-phase scripted expert (oracle baseline; IK used for DATA ONLY).
"""Phases: OPEN drawer -> RETRIEVE fork -> HANDOFF A->B -> PLACE by plate -> HOLD mug -> POUR.
Usage: python data/scripted_expert.py --seed 0 --item fork [--render-frames]
Writes output/oracle_frames_<seed>.npz when --render-frames (overhead 320x240 every 3rd step).
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.bimanual_so101_env import BimanualSO101Env
from env import eval_scoring as S
from env.kinematics import ik_dls

GRIP_OPEN, GRIP_CLOSED = 1.2, -0.1


class Oracle:
    def __init__(self, env):
        self.env = env
        self.m = env.model
        self.d = env.data
        self.ctrl = np.zeros(12)
        self.drawer = 0.0
        self.flags = {"drawer": False, "retrieved": False, "handoff": False,
                      "placed": False, "pour": False, "mug": False}
        self.frames = []

    def arm_state(self, prefix):
        return self.env.qpos_of(prefix)

    def solve(self, prefix, target):
        m, d = self.m, self.d
        saved = d.qpos.copy()
        jids = self.env.a_jids if prefix == "a" else self.env.b_jids
        err = ik_dls(m, d, f"{prefix}_gripperframe", target, jids)
        sol = self.arm_state(prefix).copy()
        d.qpos[:] = saved
        m and mujoco_forward(m, d)
        return sol, err

    def goto(self, qa=None, qb=None, ga=None, gb=None, drawer=None, steps=50, snap=False):
        cur = self.ctrl.copy()
        tgt = cur.copy()
        if qa is not None:
            tgt[0:6] = qa
        if qb is not None:
            tgt[6:12] = qb
        if ga is not None:
            tgt[5] = ga
        if gb is not None:
            tgt[11] = gb
        dc = self.drawer if drawer is None else drawer
        for i in range(steps):
            a = (i + 1) / steps
            self.ctrl = cur + (tgt - cur) * a
            self.drawer = self.drawer + (dc - self.drawer) * a if drawer is not None else self.drawer
            drawer_cmd = dc if drawer is not None else self.drawer
            obs, done, _ = self.env.step(self.ctrl, drawer_cmd)
            if snap and i % 3 == 0:
                self.frames.append(self.env.render("overhead", 320, 240))
            if done:
                break
        return obs

    def grasp_approach(self, prefix, target_fn, steps_per_round=6, max_rounds=30,
                       close_dist=0.03, snap=False):
        """Closed-loop grasp: servo toward live target, close the moment truly close."""
        idx = 5 if prefix == "a" else 11
        for _ in range(max_rounds):
            tgt = np.asarray(target_fn(), dtype=float)
            sol, _ = self.solve(prefix, tgt)
            self.goto(sol if prefix == "a" else None,
                      sol if prefix == "b" else None,
                      GRIP_OPEN if prefix == "a" else None,
                      GRIP_OPEN if prefix == "b" else None,
                      0.12, steps_per_round, snap)
            ee = self.env.ee(prefix)
            if float(np.linalg.norm(ee - tgt)) < close_dist:
                self.goto(None, None,
                          GRIP_CLOSED if prefix == "a" else None,
                          GRIP_CLOSED if prefix == "b" else None,
                          0.12, 12, snap)
                return True
        return False

    def run(self, seed=0, item="fork", render_frames=False):
        obs = self.env.reset(seed=seed)
        snap = render_frames
        self.phase = 0
        fork = obs["fork"]
        qa_home, _ = self.solve("a", [-0.20, 0.12, 0.62])
        qb_home, _ = self.solve("b", [0.20, 0.12, 0.62])
        # Lift clear of props first (jaw meshes are live; avoid transit clips).
        qa_high, _ = self.solve("a", [-0.20, 0.12, 0.74])
        qb_high, _ = self.solve("b", [0.20, 0.12, 0.74])
        self.goto(qa_high, qb_high, GRIP_OPEN, GRIP_OPEN, 0.0, 60, snap)
        self.goto(qa_home, qb_home, GRIP_OPEN, GRIP_OPEN, 0.0, 40, snap)
        # OPEN: grasp handle, pull +x while opening drawer.
        handle = np.array(self.d.site("drawer_handle").xpos)
        above, _ = self.solve("a", handle + [0, 0, 0.06])
        self.goto(above, None, GRIP_OPEN, None, 0.0, 40, snap)
        grasp, _ = self.solve("a", handle)
        self.goto(grasp, None, GRIP_CLOSED, None, 0.0, 40, snap)
        for k in range(1, 5):
            dx = 0.03 * k
            tgt, _ = self.solve("a", handle + [dx, 0, 0])
            self.goto(tgt, None, GRIP_CLOSED, None, dx, 25, snap)
        self.flags["drawer"] = S.check_drawer_open(self.env._obs())
        self.phase = 1
        self.goto(qa_home, None, GRIP_OPEN, None, 0.12, 30, snap)
        # RETRIEVE fork.
        fpos = self.env.obj_pos(item)
        pre, _ = self.solve("a", fpos + [0, 0, 0.09])
        self.goto(pre, None, GRIP_OPEN, None, 0.12, 45, snap)
        on, _ = self.solve("a", fpos + [0, 0, 0.015])
        self.goto(on, None, GRIP_CLOSED, None, 0.12, 40, snap)
        self.flags["retrieved"] = self.env.held["a"] == item
        lift, _ = self.solve("a", fpos + [0, 0, 0.16])
        self.goto(lift, None, GRIP_CLOSED, None, 0.12, 40, snap)
        self.phase = 2
        # HANDOFF at zone.
        zone = np.array([0, 0, 0.55])
        ha, _ = self.solve("a", zone + [-0.02, 0, 0])
        hb, _ = self.solve("b", zone + [0.02, 0, 0])
        self.goto(ha, hb, GRIP_CLOSED, GRIP_OPEN, 0.12, 55, snap)
        self.goto(None, None, GRIP_CLOSED, GRIP_CLOSED, 0.12, 25, snap)  # B closes -> transfer
        self.flags["handoff"] = self.env.held["b"] == item
        self.goto(None, None, GRIP_OPEN, GRIP_CLOSED, 0.12, 25, snap)  # A releases
        hold_b, _ = self.solve("b", zone + [0.10, 0.10, 0.05])
        self.goto(qa_home, hold_b, GRIP_OPEN, GRIP_CLOSED, 0.12, 45, snap)
        # PLACE by plate.
        place = np.array([0.06, -0.06, 0.50])
        pb, _ = self.solve("b", place)
        self.goto(None, pb, None, GRIP_CLOSED, 0.12, 50, snap)
        dn, _ = self.solve("b", np.array([0.06, -0.06, 0.455]))
        self.goto(None, dn, None, GRIP_CLOSED, 0.12, 35, snap)
        self.goto(None, None, None, GRIP_OPEN, 0.12, 30, snap)
        # settle
        self.goto(None, None, None, GRIP_OPEN, 0.12, 25, snap)
        self.flags["placed"] = S.check_item_placed(self.env._obs(), item)
        self.phase = 4
        up_b, _ = self.solve("b", [0.20, 0.12, 0.62])
        self.goto(None, up_b, None, GRIP_OPEN, 0.12, 40, snap)
        # HOLD mug with B.
        mpos = self.env.obj_pos("mug")
        mb_pre, _ = self.solve("b", mpos + [0, 0, 0.10])
        self.goto(None, mb_pre, None, GRIP_OPEN, 0.12, 45, snap)
        mb_on, _ = self.solve("b", mpos + [0, 0, 0.02])
        self.goto(None, mb_on, None, GRIP_CLOSED, 0.12, 40, snap)
        self.flags["mug"] = self.env.held["b"] == "mug"
        station, _ = self.solve("b", [-0.02, 0.02, 0.52])
        self.goto(None, station, None, GRIP_CLOSED, 0.12, 50, snap)
        self.phase = 5
        # POUR: A takes bottle by the neck (closed-loop: close only when truly on it).
        for _g in range(2):
            bpos = self.env.obj_pos("bottle")
            ba_pre, _ = self.solve("a", bpos + [0, 0, 0.12])
            self.goto(ba_pre, None, GRIP_OPEN, None, 0.12, 45, snap)
            ok = self.grasp_approach("a", lambda: self.env.obj_pos("bottle") + [0, 0, 0.015],
                                     snap=snap)
            if ok and self.env.held["a"] == "bottle":
                break
            bpos = self.env.obj_pos("bottle")
        lift_b, _ = self.solve("a", bpos + [0, 0, 0.16])
        self.goto(lift_b, None, GRIP_CLOSED, None, 0.12, 55, snap)
        rim = np.array(self.d.site("mug_rim_site").xpos)
        for _attempt in range(2):  # single honest re-approach on miss
            above_rim, _ = self.solve("a", rim + [0, 0, 0.06])
            self.goto(above_rim, None, GRIP_CLOSED, None, 0.12, 40, snap)
            self._tilt_bottle(65)
            for _ in range(25):
                self.env.pour_tick(True, 65)
                self.goto(None, None, None, None, 0.12, 2, snap)
            if self.env.mug_fill > 0.05:
                break
            rim = np.array(self.d.site("mug_rim_site").xpos)
        obs = self.env._obs()
        spout = self.env.obj_pos("bottle") + np.array([0, 0, 0.10])
        rim_now = np.array(self.d.site("mug_rim_site").xpos)
        self.flags["pour"] = S.check_pour_success(obs, 65, float(np.linalg.norm(spout[:2] - rim_now[:2])) < 0.09,
                                                  obs["mug_fill"] > 0.05)
        self._tilt_bottle(0)
        self.goto(qa_home, None, GRIP_OPEN, None, 0.12, 40, snap)
        obs = self.env._obs()
        return {"flags": dict(self.flags), "mug_fill": obs["mug_fill"],
                "full": S.full_task({k: v for k, v in self.flags.items()}),
                "steps": obs["t"]}

    def _tilt_bottle(self, deg):
        bid = self.m.body("bottle").id
        jid = self.m.body_jntadr[bid]
        adr = self.m.jnt_qposadr[jid]
        ang = np.deg2rad(deg)
        # tilt about Y in world: quaternion (cos(a/2), 0, sin(a/2), 0)
        self.d.qpos[adr + 3:adr + 7] = [np.cos(ang / 2), 0, np.sin(ang / 2), 0]


def mujoco_forward(m, d):
    import mujoco

    mujoco.mj_forward(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--item", default="fork", choices=["fork", "spoon", "both"])
    ap.add_argument("--render-frames", action="store_true")
    args = ap.parse_args()
    item = "fork" if args.item == "both" else args.item
    if item == "spoon":
        raise SystemExit("sprint scope: fork only (spoon deferred to full build)")
    env = BimanualSO101Env(use_coupling=True, max_steps=1200)
    oc = Oracle(env)
    rep = oc.run(seed=args.seed, item=item, render_frames=args.render_frames)
    print(f"seed={args.seed} item={item} full={rep['full']} steps={rep['steps']} fill={rep['mug_fill']:.2f}")
    print("flags:", rep["flags"])
    if args.render_frames:
        import numpy as np

        os.makedirs("output", exist_ok=True)
        np.savez_compressed(f"output/oracle_frames_{args.seed}.npz", frames=np.array(oc.frames))
    return rep


if __name__ == "__main__":
    main()
