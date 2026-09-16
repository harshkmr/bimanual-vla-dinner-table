# Implementation Plan — Bimanual VLA Dinner-Table (Dual SO-101 / MuJoCo / OpenVINO)

> Intel Physical AI Online Challenge — Challenge Option: Setting Up a Dinner Table
> Robot: Simulated Dual SO-101 Arms | Sim: MuJoCo | Inference: OpenVINO | Deploy: Intel Core Ultra Series 2/3
> Policy: **SmolVLA primary + ACT grasp fallback** (Pi0.5 deprioritized) | Dev: **WSL2 Ubuntu 24.04 + cloud GPU** | Status: **No Core Ultra yet — CPU-preliminary, NPU/iGPU TBD**
> Video ref: https://youtu.be/HRG5qJPH8EQ (Build & Deploy Your Physical AI Project on Intel — Workshop; collect → train → export → deploy pattern)
> Docs: [Hack-a-thon Resources](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html) · [Pi0.5 OpenVINO Optimization](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/pi05-optimization.html) · [ACT OpenVINO](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_act.html) · [Heterogeneous Computing](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/heterogeneous_computing.html) · [Troubleshooting](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/troubleshooting.html) · [physicalai runtime](https://github.com/openvinotoolkit/physicalai)

## 0. Constraints & Ground Truth

- Current dev box (verified): `Intel i7-9750H / Win10 64-bit / Python 3.12`, repo empty (`.git/` only). **Cannot be the final demo box.**
- Final demo box per spec: `Ubuntu 24.04 LTS HWE kernel >= 6.8` on `Intel Core Ultra Series 2/3`, with `/dev/accel/accel0` (NPU), `clinfo` (iGPU), `vainfo` (media), Miniforge3 `intel_dev_env` (Python 3.11), Physical AI Studio, OpenVINO 2026.3, Anomalib 2.6, LeRobot with PyTorch-XPU swap, `verify_stack.py → ALL REQUIRED CHECKS PASSED`.
- Strategy: everything portable now (WSL2 + cloud train), all Intel calls config-driven (`AUTO` with CPU fallback), benchmark/video scripts hardware-agnostic. No NPU numbers claimed until measured on Core Ultra hardware (borrowed/hackathon lab).
- OpenVINO version pinned to `2026.3` for conversion == runtime (mismatch causes silent IR failures; check `<Runtime_version>` in `*.xml`).

## 1. Architecture (Observe → Understand → Plan → Act → Optimize)

```text
[MuJoCo: overhead + wrist_A + wrist_B + sim state]
        │ Observe (headless mujoco.Renderer → NumPy RGB, 640×480@30)
        ▼
[Task Planner (NL → subtasks + state tracker + re-plan)]
        │ Understand/Plan (hierarchical, keeps multi-step context)
        ▼
[SmolVLA finetuned (lerobot/smolvla_base) + ACT grasp fallback]
        │ Policy (LeRobot-native VLA, language-grounded)
        ▼
[Bimanual Coordinator (roles, workspace mutex, clamps, e-stop)]
        │ Act @30 Hz (SyncExecution → AsyncExecution if infer > 33 ms)
        ▼
[Dual SO-101 in MuJoCo dinner-table scene]
[OpenVINO IR FP16/INT8 on CPU/iGPU/NPU via physicalai.InferenceModel]
        Optimize (LATENCY for control, THROUGHPUT for bench)
```

- **Policy choice:** `SmolVLA` primary — lightweight, LeRobot-native, best NL grounding, 50-episode starting point, easiest OpenVINO path. Freeze vision encoder, train action expert. **ACT** small checkpoint for grasp subtask fallback only (`ckpt → torch.jit.trace(qpos 1,14 + images) → ov.convert_model`). **Pi0.5** stretch only (needs `sample_noise → zeros` + `sample_time → Beta-mean` ONNX hacks, heavy).
- **Deploy:** `physicalai.InferenceModel(backend="openvino", device="AUTO")` + `RobotRuntime(fps=30, PolicySource, cameras={wrist, overhead})` + `InferenceLatencyBenchmark` + `benchmark_app -hint latency`.
- **Workload map (Intel heterogeneous doc):** motion control/coordinator → CPU (deterministic); SmolVLA vision → GPU/NPU; LLM/VLM planner → iGPU; end-to-end VA large → dGPU/edge-server if present. Documented in README for Optimization points.

## 2. Target Repo Structure

```text
bimanual-dinner-table/  (repo root = C:\Nightscrawler hkton)
  IMPLEMENTATION_PLAN.md (this file)
  env/
    bimanual_so101_env.py      # headless Renderer, _handle_grasps (flag-gated), stepping
    kinematics.py              # SLSQP IK, check_bimanual_collision_risk(min_dist=0.05)
    domain_randomizer.py       # nominal-cache fix + full spec axes
  assets/scene/dinner_table_scene.xml  # implicitfast, 0.002, frame_skip 25, collision masking
  data/scripted_expert.py      # oracle with phase mutex + atomic handoff transfer
  policies/                    # smolvla configs, planner.py, coordinator.py, act_fallback/
  training/                    # record script, train_smolvla.sh, cloud RunPod/A100 recipe
  openvino_optimization/ov_bimanual_infer.py  # convert → FP16 → NNCF INT8, AUTO/CPU fallback
  benchmark/benchmark_intel.py # → output/benchmark_report.json (measured only)
  scripts/
    install_linux.sh           # OSMesa/EGL, MUJOCO_GL, OCL_ICD_VENDORS
    verify_stack.py            # 5 isolated checks with actionable errors
    record_demo_video.py       # VideoWriter mp4v, no imshow → output/demo_10_seeds.mp4
  eval/eval_10_seeds.py        # → output/10_seed_evaluation_report.json (oracle vs policy split)
  output/                      # benchmark_report.json, 10_seed_evaluation_report.json, demo_10_seeds.mp4
  docs/ARCHITECTURE.md, docs/SETUP_WSL2.md, docs/INTEL_RUNBOOK.md
```

Cameras: `overhead + wrist_A + wrist_B` via LeRobot `rename_map` to SmolVLA-expected names.

## 3. Phased Build (gates must pass in order)

### Phase 0 — Env + scaffold [Judging: Reproducibility 10pts]

- WSL2 Ubuntu 24.04 HWE + Miniforge `intel_dev_env` (Py3.11) + MuJoCo + LeRobot `[smolvla]` + OpenVINO (CPU on WSL). Cloud GPU (A40/A100, ffmpeg, HF token) for training only.
- `install_linux.sh`: `libosmesa6-dev libgl1-mesa-glx libglfw3`, `MUJOCO_GL=egl|osmesa`, `OCL_ICD_VENDORS=/etc/OpenCL/vendors`.
- `verify_stack.py`: python → torch → headless MuJoCo step → OV compile CPU → policy tensor shapes.
- **Gate:** `mujoco demo + ov CPU infer + lerobot-train --help` green; `available_devices=['CPU']` logged as preliminary.

### Phase 1 — Scene + stability [Judging: Bimanual 30pts]

- Base MJCF from `TheRobotStudio/SO-ARM100 Simulation/SO101`. Scene: 2× SO-101 facing table, drawer (slide joint), plate/mug/bottle/fork/spoon, `plate_target_site, mug_rim_site, handoff_zone`.
- `dinner_table_scene.xml`: `integrator="implicitfast" timestep="0.002" frame_skip=25` (20 Hz control / 500 Hz physics); actuator `Kp 15–55, Kv 1.5–5.5`.
- Self-collision mask: links 1–5 `contype=0 conaffinity=0`, gripper pads `1/1` (kills ~66 internal contacts; tracking error `0.44 → ~0.002 rad`).
- `kinematics.py`: bounded SLSQP IK, all targets `0.16–0.42 m` from base (max reach 0.53 m); `check_bimanual_collision_risk(0.05)` + negative reward outside handoff zone.
- Headless `mujoco.Renderer(h,w) → NumPy`, never GLFW/GLX; video via `VideoWriter`.
- **Gate:** scripted waypoints run 10/10 fixed seed, no QACC explosion.

### Phase 2 — Oracle → LeRobot dataset [Prereq for Objectives 1,4]

- `scripted_expert.py` phase mutex: B home while A opens drawer/retrieves; `COORDINATED_HANDOFF` atomic transfer (`attached_a=None, attached_b=fork` iff `grip>0.3 and dist<0.15 m`).
- `_handle_grasps` coupling (attach on close, detach at `plate_target_site` 12 cm / `z=0.425 m`, pour tilt 65° above `mug_rim_site` with B stabilizing) behind `--use_coupling=1` for data collection only; default `0` for eval.
- Collect 50–100 LeRobot episodes, consistent keys/layout, paraphrased NL templates, ≥1 handoff + hold+pour.
- **Gate:** dataset replays cleanly in LeRobot viewer.

### Phase 3 — Train SmolVLA + ACT fallback [Objectives 2,4; VLA 20pts]

- Finetune `lerobot/smolvla_base` 20–50k steps (batch 16 on 4090 ~5 h / 32–64 on A40/A100 ~2–3.5 h). Push `HF_USER/so101-bimanual-table-smolvla`.
- ACT grasp ckpt fallback; router: `grasp_conf < thr → ACT else SmolVLA chunk`.
- **Gate:** sim rollout `>70%` single-seed before randomization.

### Phase 4 — Planner + coordinator [VLA reasoning 20pts]

- `planner.py`: `open_drawer → retrieve → plate_place_A → mug_hold_B → pour_A → cutlery` + state tracker + re-plan.
- `coordinator.py`: static roles (A=plate/drawer/bottle, B=mug/cutlery), shared-zone mutex, joint/vel clamp, workspace box, e-stop. `SyncExecution@30 Hz` → `AsyncExecution` if slow.
- **Gate:** novel phrasing executes without code change.

### Phase 5 — OpenVINO optimize [Optimization 20pts]

- `ov_bimanual_infer.py`: `ov.convert_model → save FP16 (compress_to_fp16=True) → NNCF compress_weights INT8_ASYM all_layers`. `device=AUTO` default, `--devices CPU GPU NPU AUTO`, try/except fallback to CPU, hints `LATENCY`/`THROUGHPUT`.
- Validate `MSE < 1e-3` torch vs OV (`lerobot_ov_inferencing.py / is_same_tensor.py` pattern).
- `benchmark_intel.py → output/benchmark_report.json`: measured only (Torch CPU / OV FP32/FP16/INT8 ms + speedup). NPU/iGPU = `TBD_CoreUltra`. No projected FPS in README.
- Bench also via `benchmark_app -m model.xml -hint latency -d {CPU,GPU,AUTO} -shape "images[1,1,3,224,224],..."`.
- **Gate:** latency drop, no task-success regression on fixed seed.

### Phase 6 — Randomization [Robustness 15pts]

- Keep nominal-cache fix: cache `nominal_frictions` at init; `geom_friction = nom * rng(0.80,1.20)` (never `*=` compounding).
- Full spec axes: poses `±5 cm+`, mass `×0.5–2.0`, friction `±20%+`, shape variants, light angle/intensity `±25%+`, background textures, placement shuffles.
- **Gate:** oracle 10/10; policy shows variance (flat 100% = overfit signal).

### Phase 7 — Eval + deliverables [All 5 required deliverables]

1. Reproducible GitHub repo (this root).
2. Reproducible MuJoCo sim (`env/ + assets/scene/` + randomizer + eval config).
3. `benchmark/benchmark_intel.py` (latency/throughput/device/precision JSON).
4. `output/demo_10_seeds.mp4` — command overlay + seed variation + inference HUD + handoff close-up, per Recommended Sequence; `eval/eval_10_seeds.py → output/10_seed_evaluation_report.json` split `oracle_coupling=ON` vs `policy_coupling=OFF` per phase (drawer, retrieve, handoff, place, pour) + coordination score; freeze only if policy `≥7/10` full-task.
5. `README + docs/ARCHITECTURE.md` (arch, VLA choice, coord, training, robustness, OV mapping).
- `docs/INTEL_RUNBOOK.md`: `1_install_drivers.sh → reboot → /dev/accel/accel0, dmesg intel_vpu, clinfo, vainfo → 2_install_software.sh → verify_stack.py → bench + video` unmodified on Core Ultra.
- **Gate:** clean-clone reproduction of video + bench.

## 4. Risks → Mitigations (spec-tied)

| # | Risk | Spec impact | Mitigation | Verify |
|---|------|-------------|------------|--------|
| 1 | No Core Ultra | Deploy req + 20pts Optimization | `AUTO` + CPU fallback try/except; `available_devices` logged; bench JSON measured-only; NPU/iGPU `TBD`; convert==runtime 2026.3 | `verify_stack.py` CPU now, full `CPU,GPU,NPU` later, same code |
| 2 | WSL/display passthrough | Reproducibility 10pts | Headless `Renderer`, `MUJOCO_GL`, OSMesa/EGL, `VideoWriter` no `imshow`, `OCL_ICD_VENDORS`, mesa fixes | Headless step + OV compile green |
| 3 | Bimanual instability | Bimanual 30pts, hand-off demo req | `implicitfast@0.002×25`, Kp/Kv tune, link mask, IK in reach, 0.05 m guard + mutex + atomic transfer + clamps/e-stop; oracle before learned | Oracle 10/10 fixed, err ~0.002 rad |
| 4 | Grasp fail / overfit | VLA 20pts + Robustness 15pts + Training obj | Coupling oracle-only flag; SmolVLA finetune + ACT fallback + paraphrased NL; broad 6-axis randomizer with nominal-cache fix; MSE gate; 10-seed `≥7/10` gate | Split report oracle vs policy; no flat-100% claim |

## 5. Judging Map (100pts)

- End-to-End + Bimanual 30 → Phases 1,2,4,7 (sequencing, hand-off, accuracy, full task).
- VLA Reasoning 20 → Phases 3,4 (NL+vision, context, adapt/re-plan).
- Robustness 15 → Phase 6 + 10-seed report.
- OpenVINO + Core Ultra 20 → Phase 5 + runbook (latency/throughput/precision/device, quality preserved).
- Reproducibility 10 → Phase 0 + repo layout + setup docs.
- Innovation 5 → coordinator (mutex/ownership), fallback router, headless sim harness, clear demo.

## 6. Next Actions (after approval)

- [ ] Scaffold dirs/files per §2 (stubs, no logic yet).
- [ ] Write `scripts/install_linux.sh` + `scripts/verify_stack.py` skeleton.
- [ ] Stub `dinner_table_scene.xml` + `bimanual_so101_env.py` headless renderer.
- [ ] Stub `train_smolvla.sh` (cloud) + `benchmark_intel.py` (CPU table).
- Then start Phase 0 execution.
