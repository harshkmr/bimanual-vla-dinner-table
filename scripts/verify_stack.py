# TICKET-01 — 4 isolated stack checks with actionable errors.
"""1) python/torch 2) headless MuJoCo step 3) OpenVINO CPU 4) policy tensor shapes.

Exit 0 only if every check passes; otherwise exit 1 with per-check guidance.
Heavy deps are required here (this is the stack gate, not the unit suite).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def check_python_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        report("python/torch", False, "torch not importable — install the locked env (see requirements-cpu.lock / TICKET-01)")
        return
    import torch

    ok = sys.version_info >= (3, 10)
    report("python/torch", ok, f"python={sys.version.split()[0]} torch={torch.__version__}")


def check_mujoco_headless():
    try:
        import mujoco
    except ImportError:
        report("mujoco-headless", False, "mujoco not importable — pip install mujoco; headless needs MUJOCO_GL=egl|osmesa + libosmesa (see install_linux.sh)")
        return
    xml = """<mujoco><worldbody><light pos="0 0 3"/><geom name="floor" type="plane" size="1 1 0.1"/>"""
    xml += """<body pos="0 0 0.5"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>"""
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(10):
            mujoco.mj_step(model, data)
        report("mujoco-headless", True, f"mujoco={mujoco.__version__} stepped 10x, qpos[2]={data.qpos[2]:.3f}")
    except Exception as e:  # noqa: BLE001
        report("mujoco-headless", False, f"step failed ({e}) — set MUJOCO_GL=osmesa and install libosmesa6-dev")


def check_openvino_cpu():
    try:
        import openvino as ov
    except ImportError:
        report("openvino-cpu", False, "openvino not importable — install the snapshot-pinned build (F1: version must match convert env)")
        return
    try:
        from openvino import Core

        core = Core()
        devices = core.available_devices
        import numpy as np
        from openvino.runtime import opset8

        param = opset8.parameter([1, 4], dtype=np.float32, name="x")
        const = opset8.constant(np.ones((4, 2), dtype=np.float32))
        matmul = opset8.matmul(param, const, False, False)
        # Build via Function to avoid version-specific helpers:
        from openvino.runtime import Model

        model = Model([opset8.result(matmul)], [param], "tiny")
        compiled = core.compile_model(model, "CPU")
        out = compiled(np.ones((1, 4), dtype=np.float32))[0]
        ok = out.shape == (1, 2)
        report("openvino-cpu", ok, f"openvino={ov.__version__} devices={devices} tiny-matmul={'ok' if ok else 'shape-mismatch'}")
    except Exception as e:  # noqa: BLE001
        report("openvino-cpu", False, f"compile/infer failed ({e}) — check OpenVINO install matches the constraints literal")


def check_policy_shapes():
    try:
        sys.path.insert(0, REPO_ROOT)
        from policy.router import decide_route, route_mix  # noqa: E402
        import json  # noqa: E402

        cfg = json.load(open(os.path.join(REPO_ROOT, "config", "router_config.json"), encoding="utf-8"))
        seq = [decide_route([0.9] * (i + 1), cfg["tau"], cfg["window_n"]) for i in range(5)]
        mix = route_mix(seq)
        src = open(os.path.join(REPO_ROOT, "benchmark", "evaluate_seeds.py"), encoding="utf-8").read()
        flags_ok = all(f in src for f in ("--seeds", "--use_coupling", "--perturb", "--item"))
        ok = all(s == "smolvla" for s in seq) and mix["kind"] == "smolvla" and flags_ok
        report("policy-shapes", ok, f"router 5/5 smolvla, mix={mix['kind']}, eval-flags={'ok' if flags_ok else 'MISSING'}")
    except Exception as e:  # noqa: BLE001
        report("policy-shapes", False, f"router/config/eval-CLI check failed ({e})")


def main():
    check_python_torch()
    check_mujoco_headless()
    check_openvino_cpu()
    check_policy_shapes()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"--- {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed ---")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
