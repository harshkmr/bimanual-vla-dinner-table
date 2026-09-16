# Sprint: damped-least-squares IK on gripperframe sites + separation guard.
"""No scipy needed: iterative Jacobian IK, adequate for oracle waypoints."""
import numpy as np
import mujoco

MIN_SEPARATION_M = 0.05


def ee_pos(model, data, site_name):
    return np.array(data.site(site_name).xpos)


def ik_dls(model, data, site_name, target, joint_ids, iters=120, damping=0.08, tol=0.004):
    """Move data.qpos toward an IK solution in place; returns final error (m)."""
    target = np.array(target, dtype=float)
    dofs = [int(model.jnt_dofadr[j]) for j in joint_ids]  # dof addresses for Jacobian columns
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err = target - np.array(data.site(site_name).xpos)
        if float(np.linalg.norm(err)) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, data.site(site_name).id)
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), err)
        dq = np.clip(dq, -0.05, 0.05)
        for k, jid in enumerate(joint_ids):
            adr = int(model.jnt_qposadr[jid])
            lo, hi = model.jnt_range[jid]
            data.qpos[adr] = float(np.clip(data.qpos[adr] + dq[k], lo, hi))
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - np.array(data.site(site_name).xpos)))


def check_bimanual_collision_risk(data, min_dist=MIN_SEPARATION_M, in_handoff_zone=False):
    """True if end-effectors are too close outside the handoff zone."""
    if in_handoff_zone:
        return False
    a = np.array(data.site("a_gripperframe").xpos)
    b = np.array(data.site("b_gripperframe").xpos)
    return bool(float(np.linalg.norm(a - b)) < min_dist)
