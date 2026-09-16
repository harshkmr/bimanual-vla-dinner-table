# IMPLEMENTATION PLAN V3.1 — Bimanual VLA Dinner-Table (V3 + audit deltas merged)

> Intel Physical AI Online Challenge — dinner-table option. Merges `IMPLEMENTATION_PLAN_V3.md` (base) with the audit report (F1–F11).
> Rule: V3 architecture, phases, gates, and rubric map are unchanged except where a delta explicitly tightens them. Every delta below traces to its audit ID.
> Robot: Dual simulated SO-101 | Sim: MuJoCo | Policy: SmolVLA finetuned + ACT grasp fallback | Inference: OpenVINO (version per §0) | Deploy: Core Ultra Series 2/3
> Spec: `spec/SPEC-bimanual-vla-dinner-table.md` | Tickets: `tickets/` | Status: No Core Ultra yet — CPU-preliminary, NPU/iGPU TBD

## 0. Delta summary (what changed vs V3)

- [F1 MUST] Version ground truth: hackathon snapshot page is the single source of truth for the OpenVINO/NNCF/benchmark_app pin (not a guessed `2026.3`). One literal version string, checked by `verify_stack.py`, captured into bench JSON. Re-pin before any export work.
- [F2 MUST] Measurable pour: coded `pour_success` predicate + deterministic kinematic liquid proxy (`bottle.liquid_level` → `mug.fill_ratio`) + HUD fill % — tilt alone is not success.
- [F3] One shared `env/eval_scoring.py` (per-phase predicates + AND-aggregate) consumed by eval script, video HUD, and freeze gate.
- [F4] Camera/data contract locked before data gen (per-camera resolution, dataset meta shapes, verify asserts).
- [F5] NNCF calibration 40 → 100+ stratified (≥12/phase), accuracy-aware if available; INT8 runs the fixed-seed task eval, not just MSE.
- [F6] Bench hygiene: version/CPU/governor metadata, warmup ≥3, p50/p90/p99, within-machine-only comparisons.
- [F7] Episode floor 50 with escalation (double episodes on <70% autonomous; data-quality check under 50%).
- [F8] Hard `max_steps` (~300) + stuck-detector scored as failure, never a hang.
- [F9] Eval-only `--perturb` adaptation clip (2–3 seeds) for the reasoning story.
- [F10] Video step count recomputed from the real episode budget.
- [F11] Canonical README file map for clean-clone optics.

## 1. Constraints & ground truth [F1]

- Dev box (verified): non-Ultra Windows + WSL2 Ubuntu 24.04 HWE workflow + cloud GPU for training. Final demo box: Ubuntu 24.04 on Core Ultra 2/3 (`/dev/accel/accel0`, `clinfo`, `vainfo`, `intel_dev_env` Py3.11, Physical AI Studio, Anomalib, LeRobot PyTorch-XPU, `verify_stack.py` green).
- [F1] Before Phase 5: run `scripts/check_ov_version.sh` on a disposable box against the hackathon resource snapshot; freeze the newest installable 2026.x/2025.x across OpenVINO + NNCF + `benchmark_app`; store the single literal in the constraints artifact; `verify_stack.py:check_openvino_version()` enforces it and stamps it into every bench JSON. `convert == runtime` holds within that frozen env only.
- Everything stays device-abstracted (`AUTO`, CPU default + fallback); NPU/iGPU columns stay TBD until measured on Core Ultra. No projected numbers in user docs.

## 2. Architecture (Observe → Understand → Plan → Act → Optimize) — V3 + deltas

```text
NL instruction + 3xRGB (contract per F4) + 14-DoF proprio
  → Planner (NL → APPROACH/OPEN/RETRIEVE/HANDOFF/PLACE/POUR + preconditions + re-plan; eval-only --perturb per F9)
  → SmolVLA finetuned (language+vision → chunks); router → ACT grasp ckpt if grasp_conf low
  → Coordinator (roles, 0.05 m guard, mutex, clamps, e-stop; 20 Hz Sync → Async; max_steps ~300 + watchdog per F8)
  → MuJoCo dual-SO-101 scene (implicitfast@0.002×25, Kp15–55/Kv1.5–5.5, link mask, use_coupling gate, liquid proxy channel per F2)
  → Shared eval_scoring.py (F3) → split 10-seed scorecard + HUD
  → OpenVINO IR (FP16 + NNCF INT8, 100+ stratified cal per F5) via abstracted inference (CPU default, AUTO override)
```

- Liquid proxy honesty rule: deterministic kinematic channel only (MuJoCo has no particles); README + video caption label it a proxy, never "fluid simulated".
- Perturbation honesty rule: `--perturb` is eval-only evidence of re-planning, never training data.

## 3. File list (V3 §2 + audit additions marked [F#])

- `env/bimanual_so101_env.py` — reset/step, headless renderer, `use_coupling` flag, `_handle_grasps`; adds [F2] liquid channel (`liquid_level`/`fill_ratio` update while pour condition holds), [F8] `max_steps` + stuck-detector, [F9] `--perturb` hook (or `env/perturb.py`).
- `env/kinematics.py` — SLSQP IK (0.16–0.42 m, reach 0.53 m), `check_bimanual_collision_risk(0.05)`. Unchanged.
- `env/domain_randomizer.py` — nominal-cache fix + six axes (pose ±5 cm+, mass ×0.5–2.0, friction ±20%+, shape variants, light ±25%+, backgrounds, placement). Unchanged.
- `env/eval_scoring.py` [F3] + `tests/test_eval_scoring.py` — per-phase predicates (drawer_open, retrieved_fork, handoff via atomic-transfer flag, plate_at_target, pour_success [F2], mug_stable_final), AND-aggregate + weights, single JSON for eval/HUD/gate.
- Version constraint artifact [F1] — single literal OV version string (checked by verify, stamped into bench JSON) + `scripts/check_ov_version.sh`.
- `assets/scene/dinner_table_scene.xml` — V3 physics + [F2] liquid state channel (bottle/mug).
- `data/scripted_expert.py` + `data/generate_demonstrations.py` — 6 phases + atomic transfer; 50–100 LeRobot v2.0 episodes [F7 floor/escalation] with [F4] locked per-camera resolutions recorded in `meta/info.json`.
- `policy/planner.py`, `policy/smolvla/`, `policy/act_bimanual_policy.py`, `policy/train.py` (fallback-only), `training/train_smolvla.sh` — V3 unchanged + [F7] escalation rule before touching architecture.
- `openvino_optimization/export_and_convert.py` (trace→IR, FP32/FP16; version = constraints literal), `quantize_nncf.py` ([F5] 100+ stratified, ≥12/phase, accuracy-aware if available), `ov_bimanual_infer.py` (AUTO/fallback/hints + parity).
- `benchmark/benchmark_intel.py → output/benchmark_report.json` — V3 fields + [F1] `openvino_version` + [F6] CPU/governor/Turbo metadata, warmup ≥3, p50/p90/p99; within-machine-only rule.
- `benchmark/evaluate_seeds.py --seeds 10 --use_coupling {0,1} [--perturb step]` [F9] → split scorecard + [F5] INT8 fixed-seed task-success column.
- `scripts/install_linux.sh`, `scripts/verify_stack.py` (+ [F1] version check, [F4] model-input shape asserts), `scripts/record_demo_video.py` ([F2] fill %/tilt HUD, [F10] steps recomputed from budget, 4-view composite, no imshow).
- `eval/eval_10_seeds.py` (thin wrapper), `tests/test_{domain_randomizer,expert,kinematics,openvino,task_planner}.py` + new `test_eval_scoring.py`, `docs/INTEL_RUNBOOK.md` (+ [F6] same-machine-class contract), `ARCHITECTURE.md`, `SETUP_WSL2.md`, `README.md` (+ [F11] canonical file map).

## 4. Phased build with gates (V3 order; deltas bolded)

- Phase 0 — Env + scaffold. **Gate: [F1] snapshot version frozen + `check_ov_version` green; [F4] camera contract locked;** mujoco demo + OV CPU infer + `lerobot-train --help` green; `available_devices=['CPU']` preliminary.
- Phase 1 — Scene + stability (+ [F2] liquid channel, [F8] step cap/watchdog plumbed). Gate: oracle waypoints 10/10 fixed, no QACC, err ~milliradian.
- Phase 2 — Oracle → dataset (coupling=1; [F4] locked resolutions; [F7] 50-episode floor). Gate: LeRobot viewer replay clean.
- Phase 3 — Train (SmolVLA + ACT fallback + router). **Gate: [F7] autonomous (coupling=0) single-seed >70% else double episodes (+paraphrases) first, <50% → data-quality replay;** MSE <1e-3.
- Phase 4 — Planner + coordinator (+ [F9] perturb hook). Gate: novel phrasing executes; perturbation triggers visible re-sequence.
- Phase 5 — Optimize (**[F5]** 100+ stratified cal; FP16 + INT8; bench matrix on CPU). **Gate: latency drop + [F5] INT8 fixed-seed task success ≈ FP32 (preserved quality measured on task, not just MSE).**
- Phase 6 — Randomize (six axes). Gate: oracle 10/10; policy honest variance.
- Phase 7 — Eval + deliverables (**[F3]** shared scorer; **[F10]** recomputed video steps; freeze iff policy ≥7/10 coupling=0; **[F6]** metadata in JSON; **[F11]** README file map). Gate: clean-clone video + bench reproduction; Core Ultra runbook unmodified, trio visible.

## 5. Judging map deltas (base: V3 §4)

- 30 task/bimanual: +F2 real pour metric, +F8 scored-not-hung failures, +F10 complete video.
- 20 reasoning: planner hooks +F9 perturbation clip as verifiable adaptation evidence.
- 15 robustness: 6-axis + 10 seeds +F2/F8 comparability.
- 20 optimization: measured-only TBD +F1 version truth, +F5 INT8 task-success proof, +F6 fair per-machine hygiene.
- 10 reproducibility: relative paths + clean-clone +F4 contracts, +F7 floor/escalation, +F11 map, +F3 single scorer.
- 5 innovation: unchanged.

## 6. Risks → mitigations (V3 §5 + deltas)

1. No Core Ultra → AUTO/CPU fallback, TBD columns, [F1] version freeze + [F6] same-class contract.
2. WSL/display → headless renderer, MUJOCO_GL, OSMesa/EGL, VideoWriter, OCL/mesa; isolated verify (+ version + shape asserts).
3. Bimanual instability → implicitfast@0.002×25, gains, link mask, in-reach IK, guard + mutex + atomic transfer + clamps/e-stop + [F8] step cap; oracle before learned.
4. Grasp fail/overfit → coupling oracle-only; SmolVLA + ACT + paraphrases; broad randomizer + nominal-cache; MSE + [F5] INT8 task eval + 7/10 gate; split report; [F7] escalation before architecture changes.
5. Unjudgeable pour → [F2] coded predicate + proxy channel + HUD (labeled proxy).

## 7. Open questions (from audit — resolve in order)

1. Which snapshot OV version installs cleanly on a disposable box (WSL2 vs bare Linux)? [F1 — blocks Phase 5]
2. Camera IDs + resolutions per camera — lock before data gen. [F4 — blocks Phase 2]
3. GPU-hour budget (20–50k steps vs LoRA/targeted-layer fallback)? [affects Phase 3 wall time]
4. Core Ultra arrival date? [determines TBD vs measured bench columns only]
