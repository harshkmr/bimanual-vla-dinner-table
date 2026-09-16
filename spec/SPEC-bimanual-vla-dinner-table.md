# Spec — Bimanual VLA Dinner-Table (Dual SO-101 / MuJoCo / OpenVINO / Core Ultra)

Status: draft spec synthesized from conversation (no interview per skill). Tracker: none configured — published as local file; triage label `ready-for-agent` recorded in frontmatter below.
Triage: ready-for-agent
Domain vocabulary: Observe / Understand / Plan / Act / Optimize; bimanual coordination; shared workspace; hand-off; complementary actions; Vision-Language-Action (VLA); domain randomization; OpenVINO IR; heterogeneous mapping (CPU / iGPU / NPU); Intel Core Ultra Series 2/3.
ADRs: none found in repo — no prior decisions to respect.

## Problem Statement

As an entrant in the Intel Physical AI Online Challenge (dinner-table option), I need an end-to-end Physical AI solution in simulation where two simulated SO-101 arms interpret natural-language instructions, reason over camera observations, coordinate both manipulators, and complete a multi-step table-setting task (open drawer, retrieve cutlery, place plate and mug, pour), robust across randomized scenes, optimized with OpenVINO for Intel Core Ultra Series 2/3 — but today the repo is empty (only a plan document, no code, no verified runs), the dev box is a non-Ultra Windows machine, and prior drafts either scripted the behavior without a learned VLA policy or claimed unmeasured accelerator numbers, so nothing yet satisfies the 100-point rubric or the five required deliverables.

## Solution

A hierarchical VLA system built from the merged v3 plan: a natural-language task planner decomposes dinner-table commands into subgoals with state tracking and re-planning; a finetuned SmolVLA policy (language + multi-view vision to action chunks) executes them with an ACT grasp fallback and router; a bimanual coordinator enforces roles, shared-workspace mutex, collision guards, clamps and e-stop; a headless MuJoCo dual-arm dinner-table simulation with physics stabilization and full six-axis domain randomization provides oracle data collection, LeRobot-standard datasets, 10-seed evaluation with an oracle-vs-policy split scorecard, and a composite demo video; the policy exports to OpenVINO IR (FP16 + NNCF INT8) behind a device-abstracted inference seam reporting measured-only latency/throughput/device/precision, with NPU/iGPU numbers left TBD until a Core Ultra runbook is executed unmodified on real hardware.

## User Stories

1. As a challenge entrant, I want headless MuJoCo observations from overhead plus two wrist cameras, so that policies train and evaluate without display servers on WSL2/CI.
2. As a challenge entrant, I want a stable dual-arm dinner-table scene with drawer, plate, mug, bottle and cutlery, so that the full task sequence is physically executable.
3. As a challenge entrant, I want implicit high-frequency physics with tuned gains and self-collision masking, so that arms track waypoints without explosions or locking.
4. As a challenge entrant, I want SLSQP-calibrated IK waypoints inside each arm's reach envelope, so that drawer, utensil, handoff, plate, mug and bottle targets are reachable.
5. As a challenge entrant, I want phase-mutexed arm roles with an atomic handoff ownership transfer, so that exactly one arm owns an object at a time and arms never fight.
6. As a challenge entrant, I want an end-effector separation guard outside the handoff zone, so that shared-workspace collisions are penalized and avoided.
7. As a challenge entrant, I want kinematic grasp coupling gated behind an oracle-only flag, so that demonstration collection is reliable while policy evaluation uses pure contact physics.
8. As a challenge entrant, I want a scripted oracle completing open-drawer, retrieve, handoff, place and pour, so that scene viability is proven before any learning.
9. As a challenge entrant, I want 50–100 demonstration episodes in LeRobot standard format with paraphrased language templates, so that finetuning starts from spec-compliant data.
10. As a challenge entrant, I want a finetuned SmolVLA policy conditioned on language plus multi-view vision, so that VLA reasoning points are defended with a true VLA.
11. As a challenge entrant, I want an ACT grasp fallback with a confidence router, so that fine manipulation degrades gracefully instead of failing the episode.
12. As a challenge entrant, I want a planner mapping free-form commands to subgoal sequences with precondition checks, so that novel phrasings execute without code changes.
13. As a challenge entrant, I want re-planning on subtask failure, so that multi-step context is maintained as scene state changes.
14. As a challenge entrant, I want joint, velocity, workspace and e-stop safety guards, so that learned actions cannot damage the simulated setup or violate limits.
15. As a challenge entrant, I want fixed-rate control decoupled from slower inference, so that control stays real-time when inference lags.
16. As a challenge entrant, I want domain randomization over poses, mass, friction, shape, lighting, background and placement with zero-drift nominal caching, so that robustness points survive all six spec axes.
17. As a challenge entrant, I want FP16 and NNCF INT8 OpenVINO exports with output-parity validation, so that optimization preserves task quality.
18. As a challenge entrant, I want device-abstracted inference defaulting to CPU with AUTO and per-device override plus fallback, so that the same code runs on WSL2 now and Core Ultra later.
19. As a challenge entrant, I want a benchmark report with measured-only latency percentiles, throughput, speedup, device and precision, so that judges can verify every number locally.
20. As a challenge entrant, I want a 10-seed evaluation with an oracle-vs-policy split scorecard per phase plus coordination score, so that learned-policy variance is reported honestly.
21. As a challenge entrant, I want a composite demo video with command overlay, seed variation, inference HUD and handoff close-up, so that the recommended demonstration sequence is easy to verify.
22. As a challenge entrant, I want a turnkey Core Ultra runbook (drivers, reboot checks, stack install, verify, bench, video), so that judges reproduce the demo unmodified on real hardware.
23. As a challenge entrant, I want isolated stack verification with actionable errors per subsystem, so that environment failures are diagnosable in minutes.
24. As a challenge entrant, I want a pytest suite covering randomizer drift, oracle trajectories, kinematics, OpenVINO compile and planner transitions, so that regressions are caught before video freeze.
25. As a challenge entrant, I want a concise architecture README mapping choices to rubric points, so that technical quality and innovation are communicated clearly.

## Implementation Decisions

- Primary policy is finetuned SmolVLA (frozen vision encoder, trained action expert) with ACT as grasp-only fallback behind a confidence router; custom from-scratch training is fallback-only, and all training goes through LeRobot tooling with Hub publication.
- Planner is a subgoal state machine (approach, open, retrieve, handoff, place, pour) with precondition validation and re-plan hooks; it is not labeled a VLM unless a real vision-language backbone is added.
- Simulation exposes a Gymnasium-style reset/step seam with headless RGB observations and joint proprioception; physics uses an implicit solver at a small timestep with a control decimation factor, tuned actuator gains, structural-link collision masking with live gripper pads, and calibrated reach-envelope geometry.
- Bimanual coordination uses static arm roles, a shared-zone mutex, a minimum-separation guard, interpolated IK waypoints, and an atomic single-owner handoff transfer.
- Grasp coupling is oracle-only behind an explicit flag: enabled for demonstration recording and the oracle baseline column, disabled for autonomous policy evaluation and the frozen demo.
- Randomizer caches nominal physics values at initialization and perturbs against them (never compounds), covering all six challenge axes with ranges at least as wide as pose ±5 cm, mass ×0.5–2.0, friction ±20%, plus shape variants, lighting variation, backgrounds and placement shuffles.
- Optimization converts via JIT trace directly to OpenVINO IR (no ONNX hop on WSL), exports FP32/FP16, applies NNCF INT8 post-training quantization with a fixed calibration set, and validates output parity within a tight tolerance before benchmarking.
- Inference seam auto-discovers devices, defaults to CPU, accepts explicit device selection including AUTO, falls back to CPU on compile failure, and separates latency-hint control from throughput-hint benchmarking; conversion and runtime OpenVINO versions are pinned identical.
- Benchmark and evaluation seams emit JSON only from locally measured runs; accelerator columns stay TBD until measured on Core Ultra, and no projected numbers appear in user-facing docs.
- Video seam composites overhead, front and both wrist views headlessly with task phase, seed, coordination, inference status and instruction subtitle, without any interactive display calls.
- Proposed test seams (highest possible, fewest across codebase — please confirm these match expectations): one top-level episode seam running a full instruction plus seed to a phase-by-phase report headlessly, underpinned by exactly three sub-seams (simulation reset/step, policy action-chunk, inference compile/infer); new seams only if the episode seam cannot reach a behavior.

## Testing Decisions

- A good test asserts externally observable behavior (episode phase success, report JSON shape and values, video file produced, inference latency measured, parity tolerance met) and never implementation details (weights internals, solver iterations, trace graph names).
- Modules under test: simulation stepping and reset, domain randomizer drift-freeness, oracle trajectory collision-freeness, kinematics round-trip and guards, OpenVINO compile and quantized inference, planner parsing and transitions, benchmark and evaluation report generation, video recording headlessly.
- Prior art: none in repo (no tests exist); follow the v2-proposed pytest layout as the first suite and keep every test headless and CPU-runnable so WSL2 and CI can run them identically.

## Out of Scope

- Physical SO-101 hardware (simulation-first challenge; hardware deployment only via the existing runtime protocol, no hardware purchase or bring-up).
- Pi0.5 as primary policy (tracked as stretch only due to conversion surgery and inference cost).
- Claimed NPU/iGPU performance numbers before a real Core Ultra measurement (placeholders stay TBD).
- Production hardening beyond the challenge (multi-table catering scenarios, fleet management, cloud serving, mobile UI).
- Rewriting the filed v1 plan document (it stays as history; v3 scaffold and tickets carry the build forward).

## Further Notes

- Repo state at spec time: single plan file plus empty git history (no commits); no glossary, ADRs, tests, or tracker config found, so challenge-document vocabulary is used throughout and local per-ticket files act as the tracker with `ready-for-agent` recorded.
- Current dev box is non-Ultra Windows (WSL2 + cloud GPU workflow); all performance claims stay CPU-preliminary until the runbook passes on Core Ultra Series 2/3 with all three devices visible.
- Video reference follows the workshop collect-to-deploy pattern; transcript was unavailable at plan time, so the pipeline mirrors Physical AI Studio plus the physicalai runtime conventions (unified capture, protocol robots, auto-backend inference, pluggable runtime sources).
- Ticket breakdown follows as tracer-bullet slices in execution order; each ticket declares its blocking edges as text and carries the `ready-for-agent` label.
