# IMPLEMENTATION PLAN V3 — Bimanual VLA Dinner-Table (Dual SO-101 / MuJoCo / OpenVINO / Core Ultra)

> Intel Physical AI Online Challenge — Challenge Option: Setting Up a Dinner Table
> Robot: Simulated Dual SO-101 Arms | Sim: MuJoCo | Inference: OpenVINO 2026.3 | Deploy: Intel Core Ultra Series 2/3
> Policy: **SmolVLA finetuned (primary) + ACT grasp fallback** | Dev: **WSL2 Ubuntu 24.04 + cloud GPU** | Status: **No Core Ultra yet — CPU-preliminary, NPU/iGPU TBD**
> Sources: v1 `IMPLEMENTATION_PLAN.md` (spec safety) + v2 `implementation_planv2.md` (execution detail) + challenge doc + workshop video pattern (collect → train → export → deploy)
> Spec: `spec/SPEC-bimanual-vla-dinner-table.md` | Tickets: `tickets/` (one file per ticket, blocking edges as text, `ready-for-agent`)

## Why V3 (best of both)

- vs v1: keeps v1's spec safety (true language-conditioned VLA, LeRobot training + Hub push, measured-only benchmarks, full six-axis randomization, portable relative paths) but replaces stubs with v2's execution detail (exact physics values, file-by-file modify list, pytest suite, CLIs, NNCF calibration count, judge runbook steps). V1 alone builds slowly and under-specifies physics.
- vs v2: keeps v2's speed and stability (implicitfast solver, collision masking, phase mutex + atomic handoff, nominal-cache randomizer fix, zero-ONNX conversion, headless video, dual-scorecard eval) but fixes 5 spec disqualifiers:
  1. Language conditioning via SmolVLA primary (v2 Tier-2 had vision+proprio only; Tier-1 parser is not a VLM — renamed to planner).
  2. Training via `lerobot-train` + Hub push (v2 custom from-scratch `train.py` needs far more than 50 episodes; kept as fallback-only).
  3. Randomization broadened to shape + background with wider ranges (v2 had 4 axes, mass ±15% only).
  4. 8.51 ms labeled CPU-INT8 preliminary, NPU/iGPU TBD, zero projections in user docs (v2 mapped it to NPU + projected FPS).
  5. Relative paths + clean-clone gate (v2 used `c:/infra summit` absolutes and unverified 100%/14.18 MB claims).
- Net: v3 defends all 100pts (v2 risked ~35: VLA 20 + robustness gap + optimization honesty) while building as fast as v2.

## 1. Architecture (Observe → Understand → Plan → Act → Optimize)

```text
NL instruction + 3xRGB (overhead, wrist A/B) + 14-DoF proprio
  → Planner (NL → subgoals + preconditions + re-plan)
  → SmolVLA finetuned (language+vision → action chunks); router → ACT grasp ckpt if grasp_conf low
  → Coordinator (roles, 0.05 m guard, mutex, clamps, e-stop; 20 Hz Sync → Async if infer > 33 ms)
  → MuJoCo dual-SO-101 dinner-table scene
  → OpenVINO 2026.3 IR (FP16 + NNCF INT8) via device-abstracted inference (CPU default, AUTO override)
```

- Heterogeneous intent (measure on Core Ultra, never claim early): CPU = physics/coordinator (deterministic); iGPU = policy decode; NPU = vision encode. Validated per-device with `benchmark_app`, not assumed splits (`HETERO` does not support NPU).
- Video pattern follows Physical AI Studio + `physicalai` runtime conventions (unified capture, protocol robots, auto-backend `InferenceModel`, pluggable `PolicySource`, `InferenceLatencyBenchmark`).

## 2. V3 file list (relative paths, repo root is this directory)

- `env/bimanual_so101_env.py` — Gymnasium-style reset/step, headless `mujoco.Renderer → NumPy`, `use_coupling` flag (1 = oracle/data collection, 0 = policy eval), `_handle_grasps` + friction pinch.
- `env/kinematics.py` — bounded SLSQP IK (targets 0.16–0.42 m, max reach 0.53 m), `check_bimanual_collision_risk(min_dist=0.05)`.
- `env/domain_randomizer.py` — nominal-friction cache (never `*=`), six axes: pose ±5 cm+, mass ×0.5–2.0, friction ±20%+, shape variants, light angle/intensity ±25%+, background textures, placement shuffles.
- `assets/scene/dinner_table_scene.xml` — `integrator="implicitfast" timestep="0.002" frame_skip=25` (20 Hz control / 500 Hz physics), `Kp 15–55 Kv 1.5–5.5`, links 1–5 `contype=0 conaffinity=0` + gripper pads `1/1`, `plate_target_site / mug_rim_site / handoff_zone`, drawer slide joint, gravity `0 0 -9.81`.
- `data/scripted_expert.py` — 6 phases (open drawer → retrieve fork → coordinated handoff → place plate → hold mug → pour), phase mutex (B home while A works), atomic transfer (`attached_b=fork; attached_a=None` iff `grip>0.3 and dist<0.15 m`), pour 65° tilt above mug rim with B stabilizing.
- `data/generate_demonstrations.py` — 50–100 episodes, LeRobot v2.0 (`meta/info.json`, `meta/episodes.jsonl`, `data/chunk-000/episode_*.npz`), paraphrased NL templates, ≥1 handoff + hold+pour.
- `policy/planner.py` — NL → subgoal state machine (APPROACH → OPEN → RETRIEVE → HANDOFF → PLACE → POUR) + precondition validation + re-plan hooks. Not labeled VLM.
- `policy/smolvla/` — finetune configs (`lerobot/smolvla_base`, frozen vision encoder, action expert trained).
- `policy/act_bimanual_policy.py` — ResNet18 multi-camera tokens + transformer chunk decoder (50×14-DoF) + temporal ensembling; grasp-fallback role.
- `policy/train.py` — fallback-only custom trainer (L1 + temporal consistency); primary path is `training/train_smolvla.sh`.
- `training/train_smolvla.sh` — `lerobot-train --policy.path=lerobot/smolvla_base`, 20–50k steps (batch 16 on 4090 ~5 h / 32–64 on A40/A100 ~2–3.5 h) → `HF_USER/so101-bimanual-table-smolvla`; cloud recipe (ffmpeg, HF token).
- `openvino_optimization/export_and_convert.py` — `torch.jit.trace → ov.convert_model` (no ONNX hop on WSL), FP32 + FP16 (`compress_to_fp16=True`); pinned `openvino==2026.3` (convert == runtime).
- `openvino_optimization/quantize_nncf.py` — NNCF INT8 post-training quantization, 40 calibration samples, `compress_weights INT8_ASYM all_layers`.
- `openvino_optimization/ov_bimanual_infer.py` — `available_devices` query, default CPU, `--devices CPU GPU NPU AUTO`, try/except CPU fallback, `LATENCY` (control) / `THROUGHPUT` (bench) hints, `MSE < 1e-3` parity check pattern.
- `benchmark/benchmark_intel.py → output/benchmark_report.json` — measured-only (Torch CPU / OV FP32/FP16/INT8 ms, p50/p90/p99, FPS, speedup, device, precision); NPU/iGPU = `TBD_CoreUltra`.
- `benchmark/evaluate_seeds.py --seeds 10 --use_coupling {0,1} → output/10_seed_evaluation_report.json` — split scorecard (oracle coupling=1 vs autonomous policy coupling=0) per phase (drawer, retrieve, handoff, place, pour) + coordination score.
- `scripts/install_linux.sh` — OSMesa/EGL (`libosmesa6-dev libgl1-mesa-glx libglfw3`), `MUJOCO_GL=egl|osmesa`, `OCL_ICD_VENDORS=/etc/OpenCL/vendors`.
- `scripts/verify_stack.py` — 4 isolated checks (python/torch → headless MuJoCo step → OV compile → policy tensor shapes) with actionable errors.
- `scripts/record_demo_video.py --output output/demo_10_seeds.mp4 --seeds 2 --steps 210` — headless 4-view composite (overhead, front 3D, both wrists) + HUD (phase, seed, coordination, OV status, instruction subtitle) via `VideoWriter(mp4v)`, no `imshow`.
- `tests/test_{domain_randomizer,expert,kinematics,openvino,task_planner}.py` — headless, CPU-runnable pytest suite.
- `docs/INTEL_RUNBOOK.md` — judges on Ubuntu 24.04 Core Ultra: NPU (`/dev/accel/accel0`, `dmesg | grep intel_vpu`) → iGPU (`clinfo`, `vainfo`) → `verify_stack.py` → `benchmark_app` + `benchmark_intel.py` → video; plus `docs/ARCHITECTURE.md`, `docs/SETUP_WSL2.md`, root `README.md` (rubric-mapped).

## 3. Phased build with gates (tracer-bullet order; gates block next phase)

- Phase 0 — Env + scaffold: WSL2 Ubuntu 24.04 HWE + Miniforge `intel_dev_env` Py3.11 + MuJoCo + LeRobot `[smolvla]` + OV CPU; cloud GPU for training. Gate: `mujoco demo + ov CPU infer + lerobot-train --help` green; `available_devices=['CPU']` logged preliminary.
- Phase 1 — Scene + stability: build XML + env + kinematics per §2. Gate: oracle waypoints 10/10 fixed seed, no QACC explosion, tracking err ~0.002 rad.
- Phase 2 — Oracle → dataset: run `scripted_expert` (`--use_coupling=1`), generate 50–100 LeRobot episodes. Gate: replays cleanly in LeRobot viewer.
- Phase 3 — Train: finetune SmolVLA + ACT fallback ckpt + router. Gate: single-seed policy rollout >70%; torch-vs-OV MSE <1e-3.
- Phase 4 — Planner + coordinator: wire NL→subgoals + roles/mutex/clamps/e-stop + Sync→Async. Gate: novel phrasing executes without code change.
- Phase 5 — Optimize: export FP16 + NNCF INT8, bench matrix on CPU. Gate: latency drop with no task-success regression on fixed seed.
- Phase 6 — Randomize: enable full six-axis randomizer. Gate: oracle 10/10; policy shows honest variance (flat 100% = overfit signal, add data).
- Phase 7 — Eval + deliverables: freeze only if policy ≥7/10 full-task; produce video + bench JSON + README/arch/runbook. Gate: clean-clone reproduction of video + bench; Core Ultra runbook passes unmodified with all three devices visible.

## 4. Judging map (100 pts)

- End-to-End + Bimanual 30 → 6-stage sequence, handoff, coordination, accuracy (§2 scene/oracle/coordinator + split eval).
- VLA Reasoning 20 → SmolVLA language conditioning + planner context + re-plan + adaptation.
- Robustness 15 → six-axis randomizer + 10-seed split report.
- OpenVINO + Core Ultra 20 → FP16/INT8 + per-device latency/throughput/precision + preserved quality + runbook.
- Reproducibility 10 → relative paths + install/verify scripts + tests + LeRobot-standard data + clean-clone gate.
- Innovation 5 → chunking + temporal ensembling + mutex/ownership + zero-drift randomizer + HUD composite video.

## 5. Risks → mitigations

1. No Core Ultra → AUTO + CPU fallback, measured-only JSON, TBD columns, convert==runtime pin; verify CPU now, full trio later, same code.
2. WSL/display → headless Renderer, MUJOCO_GL, OSMesa/EGL, VideoWriter, OCL/mesa fixes; isolated verify checks.
3. Bimanual instability → implicitfast@0.002×25, Kp/Kv, link mask, in-reach IK, 0.05 m guard + mutex + atomic transfer + clamps/e-stop; oracle before learned.
4. Grasp fail/overfit → coupling oracle-only; SmolVLA + ACT fallback + paraphrased NL; broad randomizer + nominal-cache fix; MSE + 7/10 gates; split report.

## 6. Next (tickets carry execution; each `ready-for-agent` with blocking edges as text)

1. Scaffold skeleton + verify/install scripts (unblocks all).
2. Scene + kinematics + headless env.
3. Oracle + demonstration generation (LeRobot format).
4. SmolVLA finetune + ACT fallback + router.
5. Planner + coordinator wiring.
6. OV export + NNCF + bench harness.
7. Randomizer broadening + drift tests.
8. 10-seed eval + video + README/arch/runbook + Core Ultra pass.
