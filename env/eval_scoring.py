# F3/H5 — centralized scoring predicates. Pure functions over obs dicts; single source
# consumed by eval script, video HUD, and freeze gate. Pour uses the kinematic proxy.
"""All predicates return bools; full_task = AND of phases."""

DRAWER_OPEN_Q = 0.08
PLACE_DIST = 0.05
POUR_TILT_NOTE = "tilt alone is not success: requires spout-over-rim + fill rise"


def check_drawer_open(obs):
    return bool(obs["drawer_qpos"] > DRAWER_OPEN_Q)


def check_utensil_retrieved(obs, held_flag):
    return bool(held_flag)


def check_handoff(handoff_event):
    return bool(handoff_event)


def check_item_placed(obs, item="fork"):
    import numpy as np

    target = np.array([0.06, -0.06, 0.425])  # plate_target_site
    return bool(float(np.linalg.norm(np.array(obs[item]) - target)) < PLACE_DIST)


def check_pour_success(obs, tilt_deg, spout_over_rim, fill_risen):
    return bool(tilt_deg >= 45 and spout_over_rim and fill_risen)


def check_mug_stabilized(obs):
    import numpy as np

    home = np.array([-0.02, 0.02])  # pour station where B holds the mug
    return bool(float(np.linalg.norm(np.array(obs["mug"])[:2] - home)) < 0.06)


def full_task(flags):
    return bool(all(flags.values()))
