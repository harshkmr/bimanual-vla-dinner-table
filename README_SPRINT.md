# Bimanual VLA Dinner-Table — Dual SO-101 in MuJoCo, OpenVINO on Intel

> Intel Physical AI Online Challenge · Setting Up a Dinner Table · simulation-first.
> **Hardware (stated exactly):** dev + demo on Intel i7-9750H (CPU) + Intel UHD Graphics 630 (iGPU) via
> OpenVINO 2026.3.1, plus NVIDIA GTX 1650 for optional training. No Core Ultra on site; per organizer
> guidance (AmitB, Discord) execution on Intel CPU+iGPU XPUs is accepted and all numbers below are measured
> on this host. NPU is not present on this host — no NPU numbers are claimed anywhere.

## Measured results (from `output/*.json` — nothing hand-typed)

- Task (oracle baseline, coupling-assisted): **8/10 seeds full** (`10_seed_evaluation_report.json`);
  all non-pour phases 10/10; pour 8/10.
- Autonomous BC-MLP policy (no coupling, honest partial): **0/10 full**, drawer phase reached on 1/10
  (`10_seed_evaluation_report_autonomous.json`). See "Honest limits" — this is reported, not hidden.
- Inference (BC-MLP 39→13, OpenVINO IR): CPU FP32 p50 **0.090 ms**, CPU FP16 **0.081 ms**,
  iGPU (UHD 630) FP16 **0.438 ms** (`benchmark_report.json`); FP16 parity MSE vs torch **6.8e-9**
  on in-distribution samples. GPU.1 device unstable on this host (native plugin abort) — excluded.
- Video: `output/demo_10_seeds.mp4` (152 s, 10 seeds, command + outcome + phase HUD).

## Architecture (Observe → Understand → Plan → Act → Optimize)

- **Observe:** headless MuJoCo renderer (overhead + 2 wrist cams, 640×480) + sim state.
- **Understand/Plan:** NL command → 6-phase planner (OPEN → RETRIEVE → HANDOFF → PLACE → HOLD → POUR)
  with phase one-hot conditioning; re-approach retries on pour miss.
- **Act:** two SO-101 (6-DoF + gripper each, `SO-ARM100` public model, mirrored B arm) with static roles,
  shared-zone sequencing, atomic single-owner handoff transfer, joint/velocity clamps, step cap + watchdog.
- **Policy:** behavior-cloning MLP trained on oracle demonstrations (partial — see below); oracle
  baseline (IK waypoints for data generation only, gated behind `--use_coupling`) proves scene feasibility.
- **Optimize:** torch → `jit.trace` → OpenVINO IR FP32/FP16 (`ov.convert_model`, no ONNX hop);
  INT8 listed as stretch (not run in sprint window). Deployment: CPU default, iGPU via `AUTO`/explicit.

## Bimanual coordination

Arm A: drawer, fork retrieve, bottle pour. Arm B: fork receive (mid-air handoff at zone), place by plate,
mug hold at pour station while A pours. Exactly one owner per object at any time (atomic transfer);
phase mutex keeps arms out of the shared zone except the handoff window.

## Training (honest)

BC-MLP (33→39D state+phase in, 13D ctrl out, 256×2) on 9,184 oracle steps (8 full episodes, 10 seeds),
300 epochs CPU-minutes, train MSE 3.6e-5, input-noise augmentation. Result: memorizes trajectories but
fails closed-loop (0/10) — distribution shift without coupling/contact learning. Next: DAgger/ACT or
LeRobot SmolVLA finetune with GPU time (documented, not claimed).

## Robustness

Seeded pose jitter on all props (±1.5–2 cm) + 10-seed dual scorecard (oracle vs autonomous).
Contact lessons baked into scene: zero-penetration spawns, separated layout, full arm collision masking
(base interpenetration locked shoulder_pan; jaw clips ejected the bottle) — all grasps are coupling-zone
based under the disclosed oracle flag; props still collide with table/floor/drawer/each other.

## Reproduce

```bash
pip install mujoco torch openvino opencv-python numpy  # + scipy
python scripts/build_scene.py
python data/scripted_expert.py --seed 0
python benchmark/evaluate_seeds.py --seeds 0-9
python policy/train_mlp.py --data data/demos10.npz --epochs 300
python policy/export_mlp_ov.py && python benchmark/benchmark_intel.py
python scripts/assemble_video.py
pytest tests/ -v
```

## Honesty rules (grep-checkable seeds)

- Liquid display is a fill meter: `kinematic proxy, not fluid sim`.
- Adaptation tests: `perturbation is eval-only`. Scene variation: `randomizer is eval-only unless tagged augmented`.
- IK waypoints generate oracle data only; the submitted controller is the learned policy + planner.
- No projected/foreign numbers: every figure above comes from `output/*.json` in this repo.

## File map

`env/` sim+IK+scorer · `assets/robots/so101/` vendored arm + `assets/scene/` built XML ·
`data/` oracle+demos · `policy/` planner/router/MLP/export · `openvino_models/` IR ·
`benchmark/` eval+bench → `output/*.json` · `scripts/` build/verify/video · `weights/` checkpoints ·
`tickets/`, `spec/`, plan docs at root.
