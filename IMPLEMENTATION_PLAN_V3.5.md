# IMPLEMENTATION PLAN V3.5 — Bimanual VLA Dinner-Table (best-of-all freeze, regenerated)

> Merges: ours `IMPLEMENTATION_PLAN_V3.md` + `IMPLEMENTATION_PLAN_V3.1.md` (F1–F11) + `v3.2.md` (G1–G6 canonical freeze)
> + user `V3.3` (H1–H6 empirical) + user `V3.4` (K1–K6 claim integrity).
> Rule: V3, V3.1, v3.2, V3.3, V3.4 stay on disk unchanged; this file is the execution freeze.
> Traceability: **F** = audit base · **G** = V3.2 follow-up audit · **H** = V3.3 empirical · **K** = V3.4 claim freeze · **M** = V3.5 resolutions (new only here).
> Robot: Dual simulated SO-101 (6-DoF) in MuJoCo | Hierarchical VLA: Semantic Planner + SmolVLA finetune (primary) + Bimanual ACT (verified workhorse/fallback)
> Edge: OpenVINO (snapshot-pinned per F1/K6) + NNCF INT8 | Target: Core Ultra Series 2/3 (CPU measured; iGPU/NPU TBD)
> Spec: `spec/SPEC-bimanual-vla-dinner-table.md` | Tickets: `tickets/` | Status: scaffold + specs on disk, no logic yet, no commits

## 0. Lineage verdict (why this freeze, in one paragraph)

V3.2 is the correct base (our line, SmolVLA-primary, G1–G6 fully specified); V3.3's value is execution detail (dual JSON filenames, static scorer shape, fill-% HUD, git case fix, native convert restated) but its ACT-primary demotion and narrow widths are rejected; V3.4's K1–K6 integrity governance is adopted whole. V3.5 = V3.2 + H-details + K-governance + five M-resolutions below. Anything V3.3 contradicts on policy direction, widths, versions, paths, or scores loses to the M-rule cited.

## 1. M-resolutions (the only new content vs V3.4)

- **[M1] SmolVLA-primary locked (settles H1 vs K2).** V3.2 and the brief center SmolVLA; V3.3 demoted it to "compatible", V3.4 left K2 open. V3.5 ships the SmolVLA finetune as primary with the verified ACT policy as fallback/workhorse behind the confidence router. If organizers confirm ACT-only suffices, delete the SmolVLA row from `rubric_trace.md` — zero rework. Never claim a SmolVLA artifact before its checkpoint file exists.
- **[M2] Widths take V3.1 maxima as the gate, V3.3 minima as the floor.** Gate: pose ±5 cm+, mass ×0.5–2.0, friction ±20%+, shape variants, light ±25%+, backgrounds, placement. V3.3's ±3.5 cm / ±15% is the documented minimum any seed must at least cover. Full randomizer stays eval-only; any training augmentation is the explicit G2 tagged set only.
- **[M3] Camera contract reconciled per-model.** V3.3's fixed 128×128×3 is adopted as the recording floor, not the ceiling: record at the higher of (model preprocessing needs), store exact shapes in dataset meta (F4), downsample per-model at train/infer, assert shapes in `verify_stack.py`. No data regen when switching heads.
- **[M4] Foreign numbers inadmissible.** V3.3's 8.51 ms / 14 MB / 100% / 8-8-tests figures were measured in `c:/infra summit`, not this repo — they may not appear here until regenerated from this repo's own JSON artifacts via K1 `verify_claims.sh`.
- **[M5] Single-CLI enforcement scheduled.** G5 removes `eval/eval_10_seeds.py` (current scaffold stub becomes a forwarder with interface test, then deleted). Canonical CLI is `benchmark/evaluate_seeds.py` only.

## 2. Constraints & ground truth (F1/K6/G1 + M4)

| Parameter | V3.5 specification | Verification |
|---|---|---|
| Dev host | Windows + WSL2 Ubuntu 24.04 LTS | `verify_stack.py` |
| Target | Core Ultra Series 2/3, Ubuntu 24.04 | `/dev/accel/accel0`, `clinfo`, `vainfo` |
| OpenVINO/NNCF/`benchmark_app` | Single literal from hackathon snapshot page after disposable-box install (V3.3's `2026.3.1` is a target until installed) | `check_openvino_version()` + `check_ov_version.sh`, stamped into bench JSON |
| MuJoCo/torch/lerobot | Pins live only in `requirements-cpu.lock` + `requirements-cuda.lock` (`pip show` sourced; `3.13.0` never hand-typed) | lockfile hash asserts + `output/env_snapshot.json` |
| Policy shipped | SmolVLA checkpoint (primary, M1) + ACT checkpoint (fallback) + router | checkpoint files + gates |
| Devices | `AUTO`, CPU default, fallback | `available_devices` |
| Reporting | CPU measured only; iGPU/NPU TBD; every judge-visible number from `output/*.json` via `verify_claims.sh` or labeled target-not-measured | `rubric_trace.md` at freeze |

## 3. Architecture (V3.2 Tiers + M1)

```text
NL instruction + 3xRGB (M3 contract) + 14-DoF proprio
  → Tier 1 Planner (NL → APPROACH/OPEN/RETRIEVE/HANDOFF/PLACE/POUR + preconditions + re-plan; --item fork|spoon|both slot (G3); --perturb eval-only (F9))
  → Tier 2 SmolVLA finetuned (primary, M1); router → ACT chunk policy on low grasp confidence
  → Tier 3 Coordinator + MuJoCo env (implicitfast@0.002×25, Kp15–55/Kv1.5–5.5, link mask, 0.05 m guard, mutex, atomic handoff, use_coupling gate, liquid proxy labeled proxy, max_steps ~300 + watchdog (F8))
  → Tier 4 Scoring + OV + video (static EvaluationScorer (H5) → dual JSONs (H3) + HUD fill % (H6); FP16 + NNCF INT8 100+ stratified (F5); 4-view composite + snapshots; run_dev/run_judge (G4); verify_claims (K1); rubric_trace (K5))
```

Honesty rules (all in README next to the claims they qualify): liquid proxy is a proxy, never fluid; `--perturb` inference-only; randomizer eval-only unless episode tagged augmented (G2); NPU/iGPU TBD, never projected.

## 4. File inventory (`<repo>/` relative; K4 — no machine paths)

- `env/` — `bimanual_so101_env.py` (coupling gate, liquid channel F2, watchdog F8, perturb hook F9), `kinematics.py`, `domain_randomizer.py` (nominal-cache, M2 widths, eval-only), `eval_scoring.py` (static scorer H5/F3), `dinner_table_tasks.py`, `__init__.py` (H2 case-fix respected), constraints literal artifact.
- `assets/scene/dinner_table_scene.xml` (+ `assets/robots/so101.xml` when vendored) — physics + sites + liquid channel.
- `data/` — `scripted_expert.py` (`--item`, G3), `generate_demonstrations.py` (nominal bootstrap G2, 50+ floor F7, M3 shapes, G2 tags), `dataset_inspect.py`, `lerobot_dataset/`.
- `policy/` — `planner.py` (not VLM), `vision_encoder.py`, `act_bimanual_policy.py`, `smolvla/` configs + harness, `train.py` (fallback-only).
- `training/train_smolvla.sh` — primary finetune (M1), cloud GPU, Hub push; LoRA/targeted-layer fallback if budget binds.
- `openvino_optimization/` — `export_and_convert.py` (native convert H4, version = constraints literal), `quantize_nncf.py` (F5), `ov_bimanual_infer.py` (AUTO/fallback/hints/parity).
- `weights/` + `openvino_models/` — gitignored artifact families, delivered by scripts (K4).
- `benchmark/` — `benchmark_intel.py` (F6 hygiene), `evaluate_seeds.py` (sole CLI G5/M5; `--use_coupling`, `--perturb`, `--item`, `--seeds`).
- `scripts/` — `run_dev.sh` (full pipeline), `run_judge.sh` (verify → fetch prebuilt → benchmark_app → bench → eval coupling 0 → video; zero training), `verify_stack.py` (version + shape asserts), `verify_claims.sh` (K1), `record_demo_video.py` (H6, steps recomputed F10), `install_linux.sh` (from lockfiles G1), `install_windows.ps1` (convenience only), `check_ov_version.sh` (F1).
- `tests/` — `test_{domain_randomizer,expert,kinematics,openvino,task_planner,eval_scoring}.py` + CLI interface guard (G5); headless CPU-runnable.
- `docs/` — `INTEL_RUNBOOK.md` (one-command judge path G4, same-class contract F6), `ARCHITECTURE.md`, `SETUP_WSL2.md`; root `README.md` (auto Measured-results block only + honesty rules + two run scripts + canonical file map F11).
- `output/` — `env_snapshot.json` (G1), `benchmark_report.json` (sole latency source), `10_seed_evaluation_report.json` + `..._autonomous.json` (H3), `rubric_trace.md` (K5), `demo_10_seeds.mp4` + snapshot PNGs.
- `requirements-cpu.lock` + `requirements-cuda.lock` committed; `.gitignore` covers all `__pycache__/`, `weights/`, `openvino_models/`.

## 5. Phases & gates (V3.2 flow + M-deltas bolded)

- Phase 0 — Lockdown: lockfiles + snapshot + run scaffolds + `verify_claims` scaffold + **M3 camera contract + F1 version freeze**. Gate: verify green, literals asserted, snapshot written.
- Phase 1 — Scene + liquid + scorer + watchdog. Gate: scoring + kinematics tests green; err ≤0.05 (G6, report nominal); QACC clean.
- Phase 2 — Nominal dual-cutlery dataset (randomizer OFF, coupling=1; 50+ eps; locked shapes; G2 tags). Gate: inspect clean. **`--item both` stability decided here (open Q5); single-item ships first.**
- Phase 3 — **M1 SmolVLA finetune (primary) + ACT train (fallback) + router.** Gate: autonomous single-seed >70% else F7 double-episodes first (<50% → data-quality replay; G2 augmented set documented, default off); MSE <1e-3.
- Phase 4 — Reasoner + perturb (eval-only). Gate: planner tests green incl. spoon variants; ~step-100 perturbation visibly replans.
- Phase 5 — Convert + INT8 (100+ stratified) + bench. Gate: openvino tests green; INT8 <⅓ FP32; fixed-seed task success preserved (F5); numbers JSON-only.
- Phase 6 — 10-seed eval (M2 widths, eval-only, seeds never overlap collected states). Gate: oracle JSON feasibility 100%; autonomous JSON honest variance, no hangs.
- Phase 7 — Video + judge pipeline + claim freeze. Gate: `run_judge.sh` unmodified on rehearsal Linux; `verify_claims.sh` green; `rubric_trace.md` generated; README == JSON; all tests green; freeze iff policy ≥7/10 coupling=0.

## 6. Rubric trace (artifacts, never self-scores — K5)

| Pts | Row | Artifact required at freeze |
|---|---|---|
| 30 | Bimanual + multi-step | scorer + autonomous report (≥7/10, no hangs) + handoff/pour HUD clips + spoon+fork coverage (G3) |
| 20 | VLA / reasoning | SmolVLA checkpoint (**M1**) + planner + perturbation clip + language-slot coverage |
| 15 | Robustness | 10 seeds × six axes (M2 widths, eval-only, G2 provenance tags) + dual JSONs |
| 20 | OpenVINO | bench JSON (FP16/INT8, p50–p99, within-machine, INT8 task column, version stamps) |
| 10 | Reproducibility | lockfiles, snapshot, single CLI (M5), run_judge, verify_claims, file map |
| 5 | Innovation | chunking + ensembling + zero-drift randomizer + HUD video |

## 7. Risks (V3.2 §6 + M1/M4)

1. No Core Ultra → AUTO/CPU, TBD columns, rehearsed run_judge (CPU-only first). 2. Judge stoppage → one command, prebuilt artifacts, zero training. 3. Drift → lockfiles + snapshot (MuJoCo pin fixed). 4. Contamination → nominal bootstrap + dual scorecards + labeled proxy + non-overlapping seeds. 5. SmolVLA load (**M1**) → primary staffed first; ACT workhorse guarantees submittable demo regardless; K2 question answered by shipping both. 6. Claim drift → verify_claims + run IDs (**M4** purges foreign numbers). 7. Hearsay versions → lock-verified literals only.

## 8. Immediate tasks (ordered)

1. Version freeze on disposable box + constraints literal + verify asserts (before code). 2. Record M1 in spec. 3. `eval_scoring.py` + liquid proxy + tests. 4. `--item` parameterization. 5. Lockfiles + snapshot. 6. `verify_claims.sh`; purge un-artifacted numbers. 7. Run scripts. 8. Purge machine paths; rubric rows → artifact language. 9. **M5:** collapse `eval/eval_10_seeds.py` stub into forwarder-then-delete with interface test.

## 9. Revision discipline

V3, V3.1, v3.2, V3.3, V3.4 remain on disk unchanged; this V3.5 supersedes them in review flow. Future constraint/rubric changes bump to V3.6 with the F/G/H/K/M chain intact — no silent edits.
