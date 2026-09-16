# Sprint env: dual SO-101 dinner-table Gymnasium-style wrapper (MuJoCo-native, no gym dep).
"""13-D action: 12 arm position ctrls (a_* then b_*) + drawer_slide. 20 Hz (25x2ms)."""
import os

import numpy as np
import mujoco

from env.kinematics import check_bimanual_collision_risk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.join(REPO_ROOT, "assets", "scene", "dinner_table_scene.xml")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIP_CLOSED = 0.25  # normalized gripper ctrl below this = closed
GRASP_DIST = 0.06  # general attach window (fork/mug/plate + handoff transfer)
GRASP_DIST_BOTTLE = 0.035  # tall bottle: tight window avoids penetration pops
RELEASE_DIST = 0.12


class BimanualSO101Env:
    def __init__(self, xml_path=None, use_coupling=True, max_steps=300):
        self.model = mujoco.MjModel.from_xml_path(xml_path or DEFAULT_XML)
        self.data = mujoco.MjData(self.model)
        self.use_coupling = use_coupling
        self.max_steps = max_steps
        self.a_jids = [self.model.joint(f"a_{j}").id for j in ARM_JOINTS]
        self.b_jids = [self.model.joint(f"b_{j}").id for j in ARM_JOINTS]
        self.a_qadr = [self.model.jnt_qposadr[j] for j in self.a_jids]
        self.b_qadr = [self.model.jnt_qposadr[j] for j in self.b_jids]
        self.drawer_aid = self.model.actuator("drawer").id
        self._aids = [self.model.actuator(f"a_{j}").id for j in ARM_JOINTS]
        self._aids += [self.model.actuator(f"b_{j}").id for j in ARM_JOINTS]
        self.held = {ARM: None for ARM in ("a", "b")}
        self.bottle_liquid = 1.0
        self.mug_fill = 0.0
        self.t = 0
        self._still = 0
        self._last_ee = None

    # ---- helpers ----
    def qpos_of(self, prefix):
        adr = self.a_qadr if prefix == "a" else self.b_qadr
        return np.array([self.data.qpos[i] for i in adr])

    def ee(self, prefix):
        return np.array(self.data.site(f"{prefix}_gripperframe").xpos)

    def obj_pos(self, name):
        return np.array(self.data.body(name).xpos)

    def _grasp_zone(self, prefix, obj):
        return float(np.linalg.norm(self.ee(prefix) - self.obj_pos(obj)))

    # ---- core ----
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Sprint domain randomization (honest seed variation; full 6-axis lands in TICKET-07):
        # planar pose jitter on all props, recorded per seed for the report.
        self._seed_jitter = {}
        for obj, j in (("fork", 0.020), ("plate", 0.015), ("mug", 0.020), ("bottle", 0.015)):
            dx, dy = float(rng.uniform(-j, j)), float(rng.uniform(-j, j))
            self._seed_jitter[obj] = (dx, dy)
            bid = self.model.body(obj).id
            jid = self.model.body_jntadr[bid]
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr] += dx
            self.data.qpos[adr + 1] += dy
        self.held = {"a": None, "b": None}
        self.bottle_liquid, self.mug_fill = 1.0, 0.0
        self.t, self._still = 0, 0
        self._last_ee = None
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self):
        return {"qpos_a": self.qpos_of("a"), "qpos_b": self.qpos_of("b"),
                "ee_a": self.ee("a"), "ee_b": self.ee("b"),
                "fork": self.obj_pos("fork"), "plate": self.obj_pos("plate"),
                "mug": self.obj_pos("mug"), "bottle": self.obj_pos("bottle"),
                "drawer_qpos": float(self.data.qpos[self.model.jnt_qposadr[self.model.joint("drawer_slide").id]]),
                "mug_fill": self.mug_fill, "bottle_liquid": self.bottle_liquid, "t": self.t}

    def step(self, ctrl12, drawer_cmd=0.0):
        ctrl = np.asarray(ctrl12, dtype=float).ravel()
        assert ctrl.size == 12, f"expected 12 arm ctrls, got {ctrl.size}"
        aids = self._aids
        for i, c in enumerate(ctrl):
            lo, hi = self.model.actuator_ctrlrange[aids[i]]
            self.data.ctrl[aids[i]] = float(np.clip(c, lo, hi))
        lo, hi = self.model.actuator_ctrlrange[self.drawer_aid]
        self.data.ctrl[self.drawer_aid] = float(np.clip(drawer_cmd, lo, hi))
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        self._handle_grasps(ctrl)
        # Watchdog: end-effectors essentially frozen.
        ee = np.concatenate([self.ee("a"), self.ee("b")])
        if self._last_ee is not None and float(np.linalg.norm(ee - self._last_ee)) < 1e-5:
            self._still += 1
        else:
            self._still = 0
        self._last_ee = ee
        stuck = self._still > 60
        done = self.t >= self.max_steps or stuck
        return self._obs(), done, {"stuck": stuck}

    # ---- grasp coupling (oracle/data only when use_coupling=True) ----
    def _gripper_closed(self, ctrl12, prefix):
        idx = 5 if prefix == "a" else 11
        lo, hi = self.model.actuator_ctrlrange[self._aids[idx]]
        return (float(ctrl12[idx]) - lo) / (hi - lo) < GRIP_CLOSED

    def _handle_grasps(self, ctrl12):
        if not self.use_coupling:
            return
        graspables = ("fork", "mug", "bottle", "plate")
        for prefix in ("a", "b"):
            other = "b" if prefix == "a" else "a"
            closed = self._gripper_closed(ctrl12, prefix)
            held = self.held[prefix]
            if held is None and closed:
                near = []
                for o in graspables:
                    d = self._grasp_zone(prefix, o)
                    if o == "bottle":
                        # Tall object: allow a top-down capture funnel (jaw closes
                        # above the neck) in addition to the tight contact window,
                        # so the approach never has to shove through the body.
                        ee = self.ee(prefix)
                        bp = self.obj_pos(o)
                        dxy = float(np.linalg.norm(ee[:2] - bp[:2]))
                        dz = float(ee[2] - bp[2])
                        if d < GRASP_DIST_BOTTLE or (dxy < 0.045 and 0.0 < dz < 0.12):
                            near.append((o, d))
                    elif d < GRASP_DIST:
                        near.append((o, d))
                if near:
                    obj = min(near, key=lambda t: t[1])[0]
                    if self.held[other] == obj:
                        self.held[other] = None  # atomic handoff transfer
                    self.held[prefix] = obj
            elif held is not None and not closed:
                self.held[prefix] = None
        for prefix, obj in self.held.items():
            if obj is None:
                continue
            bid = self.model.body(obj).id
            jid = self.model.body_jntadr[bid]
            adr = self.model.jnt_qposadr[jid]
            target = self.ee(prefix) + np.array([0, 0, -0.02])
            self.data.qpos[adr:adr + 3] = target
            self.data.qvel[adr:adr + 6] = 0

    def pour_tick(self, active, tilt_deg=0.0):
        """Kinematic liquid proxy (labeled proxy in docs/HUD, not fluid sim).

        Pours when the bottle mouth is just above the rim in x/y and height:
        dxy < 6cm and rim slightly below the tilted mouth.
        """
        import math

        if not active or self.bottle_liquid <= 0:
            return
        pos = self.obj_pos("bottle")
        rim = np.array(self.data.site("mug_rim_site").xpos)
        mouth_z = pos[2] - 0.08 * math.cos(math.radians(tilt_deg))
        dxy = float(np.linalg.norm(pos[:2] - rim[:2]))
        if dxy < 0.08 and -0.02 < (rim[2] - mouth_z) < 0.15:
            flow = min(0.04, self.bottle_liquid)
            self.bottle_liquid -= flow
            self.mug_fill += flow

    def render(self, camera="overhead", width=640, height=480):
        r = mujoco.Renderer(self.model, height, width)
        r.update_scene(self.data, camera=camera)
        pix = r.render()
        r.close()
        return pix
