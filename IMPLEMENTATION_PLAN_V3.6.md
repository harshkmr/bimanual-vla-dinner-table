# IMPLEMENTATION PLAN V3.6 — Bimanual VLA Dinner-Table (claim-governed execution freeze)

> Merges: `IMPLEMENTATION_PLAN_V3.5.md` (M1–M5, kept on disk **unchanged**) + **N1–N6** (new in this file only).
> Rule: V3, V3.1, v3.2, V3.3 (text supplied externally), V3.4 (text supplied externally), V3.5 all remain on disk unchanged;
> this V3.6 supersedes them as the execution freeze. No silent edits.
> Traceability: **F** = V3 audit base · **G** = V3.2 follow-up audit · **H** = V3.3 empirical · **K** = V3.4 claim integrity ·
> **M** = V3.5 resolutions · **N** = V3.6 resolutions (new here only).
> Robot: Dual simulated SO-101 (6-DoF) in MuJoCo | Hierarchical VLA: Semantic Planner + SmolVLA finetune (primary) + Bimanual ACT (verified workhorse/fallback)
> Edge: OpenVINO (snapshot-pinned per F1/K6) + NNCF INT8 | Target: Core Ultra Series 2/3 (CPU measured; iGPU/NPU TBD)
> Spec: `spec/SPEC-bimanual-vla-dinner-table.md` | Tickets: `tickets/` (TICKET-09 owns the N-deltas) | Status: scaffold + specs on disk, no logic yet, no commits

---

## 0. Lineage verdict (why V3.6 exists)

V3.5 is adopted **whole** as the execution freeze — M1–M5 are not reversed or weakened. V3.6 exists to convert V3.5's
*statements* of governance into *machine-checkable practice*. Its six resolutions (N1–N6) close the exact gaps flagged
in review: the confidence router was undefined (reproducibility hole), the M1 kill-switch had no deadline, M4 was prose
not a gate, "freeze" was a word not a checklist, the M5 deletion had no audit trail, and the kinematic-pour honesty rule
lived only beside the risks instead of next to the claims it qualifies. Nothing in V3.5 is contradicted; every
N-resolution is an augmentation, tagged to the M/K/F rule it enforces.

---

## 1. N-resolutions (the only new content vs V3.5)

- **[N1] Router determinism closes M1's open loop.** The confidence router SHALL be a pure function of (state, chunk
  confidences, `config/router_config.json`): route to ACT only when SmolVLA success-probability < τ measured over the
  last N predicted chunks, τ and N being **single literal values read from that config file**, not heuristics. New test
  `tests/test_router_determinism.py` asserts: same seeds + same config → identical router decisions in 10/10 runs; the
  routing outcome per seed (smolvla|act|mixed, with counts) is written into the per-seed eval JSON (feeds H3 dual
  scorecards). Without this, the 10-seed eval is not reproducible and the router is a black box.
  Confidence-source rule (added at filing): SmolVLA/ACT emit no calibrated probabilities, so the "success-probability"
  input is the documented proxy `mean(last-N chunk confidences)` where each chunk confidence comes from the policy
  harness's configured scorer (ensemble variance or planner score — recorded in the config); the router itself stays pure.
- **[N2] M1 kill-switch gets a hard deadline.** At **freeze day −2** (calendar date recorded in
  `docs/FREEZE_CHANGELOG.md` at Phase 7): if no SmolVLA checkpoint file exists under `weights/smolvla/` (e.g.
  `.safetensors` or the pinned format), the freeze re-routes to ACT-primary via a **one-line revision** to the changelog
  and `rubric_trace.md` row update — never a silent edit (§10 remains binding). The decision cannot be left to the
  morning of submission. The code default is a `policy_selection` literal in `router_config.json`, so fallback is a flag
  flip, not a pressured edit.
- **[N3] M4 becomes machine-checkable, not policy.** `scripts/verify_claims.sh` gains a strict mode (default ON at
  freeze): **hard-fails on any numeric performance/success literal in `README.md`, `docs/*.md`, or benchmark JSONs that
  does not correspond to a value with provenance in `output/*.json` of this repo** (value + run-id + artifact file).
  Version literals, gate constants, dates, and run IDs are allowlisted (they are configuration, not claims).
  Foreign numbers (e.g. any "8.51 ms", "14 MB", "100% oracle", "8/8 tests" literal from other repos) can appear **only**
  after re-derivation in this repo, evidenced by a regeneration record in `output/claim_regeneration.json` with the exact
  command run. A judge-time smoke run (`--strict-numbers`) is part of `run_judge.sh` (G4).
- **[N4] The freeze becomes a go/no-go checklist.** Phase 7 closes only when `output/FREEZE_CHECKLIST.md` exists with
  **all boxes checked**: (1) prebuilt weights + OpenVINO FP16/INT8 models present; (2) SmolVLA checkpoint exists **or**
  the N2 kill-switch line is logged; (3) `run_judge.sh` dry-run green on a clean rehearsal Linux box; (4)
  `verify_claims.sh --strict-numbers` green; (5) README measured-claims block == report JSON; (6) all tests green
  (incl. N1 router test); (7) 10-seed autonomous ≥ 7/10 at coupling=0 (V3.5 gate) with no hangs; (8) HUD video +
  snapshots exist. Missing = no freeze; document what blocks and re-plan (V3.6.1 or V3.7).
- **[N5] M5's deletion gets a visible trail.** The `eval/eval_10_seeds.py` forwarder is deleted **in the freeze commit
  itself**, with the deletion listed in the commit message and in `docs/FREEZE_CHANGELOG.md` (auditor-visible). Its
  interface test (`tests/test_eval_cli_guard.py`) must pass before deletion, asserting the canonical CLI
  `benchmark/evaluate_seeds.py` produces identical exit codes and output names. The canonical CLI is the only eval entry
  point thereafter.
- **[N6] Honesty disclaimers move to the claim site.** The kinematic-proxy admission is printed *next to every
  judge-visible liquid claim*: the HUD overlay footer reads `liquid=FILL meter (kinematic proxy, not fluid sim)` and the
  README claims block carries the same suffix on both scorecards (H3); the `--perturb` and domain-randomizer eval-only
  notes appear beside the robustness rows of `rubric_trace.md` (K5), not only in §8 below. Verifiable by grep-checks
  asserting the seed strings exist in README + HUD renderer config. Seed strings (exact, shared):
  `kinematic proxy, not fluid sim`, `perturbation is eval-only`, `randomizer is eval-only unless tagged augmented`.

---

## 2. Constraints & ground truth (V3.5 §2, augmented by N-note per row)

| Parameter | V3.5 specification | V3.6 enforcement |
|---|---|---|
| Dev host | Windows + WSL2 Ubuntu 24.04 LTS | unchanged |
| Target | Core Ultra Series 2/3, Ubuntu 24.04 | unchanged |
| OpenVINO/NNCF/bench versions | Hackathon snapshot page, literal target until installed | `check_ov_version.sh` (F1) + pinned literal in one file |
| MuJoCo / torch / lerobot | lockfile-pinned (`requirements-cpu.lock`, `requirements-cuda.lock`) | lockfile hash asserts (G1) |
| Policy shipped | SmolVLA ckpt (primary, M1) + ACT ckpt (fallback) + router | + **N1** router config + `test_router_determinism` |
| Devices | AUTO default, CPU fallback | unchanged |
| Numbers | CPU-measured only; iGPU/NPU TBD | all performance/success numbers pass **N3** strict verify (versions/constants allowlisted) |
| Freeze readiness | — | **N4** checklist; **N2** day-−2 kill line; **N5** deletion proof |

---

## 3. Architecture (V3.5 §3 + **N1** dotted in)

```text
User instruction + 3xRGB (M3 contract) + 14-DoF proprio
  → Tier 1 Planner (NL → phases APPR/OPEN/RETRIEVE/HANDOFF/PLACE/POUR + preconditions, --item fork|spoon|both (G3), --perturb (F9))
  → Tier 2 SmolVLA finetuned (primary, M1); config-driven pure router (N1) → ACT chunk policy on low confidence
  → Tier 3 Coordinator + MuJoCo env (implicitfast dt=0.002×25; liquid proxy LABELED proxy (N6); 300-step watchdog F8; coupling gate; site-based scoring)
  → Tier 4 Scoring (static H5/F3) → dual JSONs (H3, router mix per seed N1) + HUD fill % (H6, proxy label N6) → OpenVINO FP16/NNCF INT8 (F5) → video + freeze gate (N4)
  Honesty rule: every claim carries its disclaimer where it renders (N6); all numbers verifiable via N3 strict mode.
```

---

## 4. File deltas vs V3.5 (additions in **bold**)

- **`config/router_config.json`** (**N1**: τ, N, chunk-length, `policy_selection`; single source of truth)
- **`policy/router.py`** (**N1**: pure `decide_route` + config loader; no heuristics)
- **`tests/test_router_determinism.py`** (**N1**)
- **`scripts/verify_claims.sh`** — strict mode w/ provenance map **N3** (performance/success claims; versions/constants allowlisted)
- **`output/claim_regeneration.json`** **N3** template (empty ledger; every entry needs command + run-id + artifact)
- **`output/FREEZE_CHECKLIST.md`** **N4** template (8 boxes, all unchecked until freeze)
- **`docs/FREEZE_CHANGELOG.md`** **N4+N5** (commit-trail; N2 kill-switch date placeholder)
- **HUD renderer + README claims block** — proxy disclaimers **N6** (exact seed strings)
- **`tests/test_eval_cli_guard.py`** (**N5**: canonical-CLI parity guard, passes before forwarder deletion)
- **`tickets/TICKET-09-v36-governance.md`** (owns N1–N6 implementation)
- (all other V3.5 files unchanged in purpose)

---

## 5. Phases & gates (V3.5 flow; **bold** = N additions)

- **Phase 0** — Toolchain + lockfiles + snapshot + `verify_claims.sh` strict scaffold. **Gate: verify green; literal-files-only version.** + create `FREEZE_CHANGELOG.md`, `FREEZE_CHECKLIST.md` templates, `router_config.json` initial values.
- **Phase 1** — Scene + liquid proxy + static scorer + watchdog. Gate: scoring/kinematics tests green; joint err ≤ 0.05 (G6); QACC clean.
- **Phase 2** — Nominal dual-cutlery dataset (randomizer OFF, coupling=1, ≥50 eps, G2 tags). Gate: `dataset_inspect` clean; single-item ships first; `--item both` decision logged.
- **Phase 3** — SmolVLA finetune (primary) + ACT train (fallback) + **router wired from config (N1)**. Gate: autonomous single-seed >70% (else F7 escalation), MSE <1e-3, **router unit + determinism tests pass (N1)**.
- **Phase 4** — Planner + `--perturb`. Gate: planner tests incl. spoon/fork slots; ~step-100 perturbation visibly replans.
- **Phase 5** — OpenVINO convert FP16/INT8 + NNCF 100+ stratified + benchmark. Gate: INT8 ≤ ≈⅓ FP32; task success preserved; **bench numbers enter the claim ledger (N3)**.
- **Phase 6** — 10-seed dual eval (widths M2; seeds non-overlapping; eval-only randomizer). Gate: oracle JSON feasibility 100%; autonomous honest variance; no hangs; **router mix per-seed in JSON (N1)**.
- **Phase 7** — Freeze: 4-view HUD video + judge pipeline. **Final gate = N4 checklist all-true** (incl. N2 kill line, N5 deletion proof, N3 strict verify green, `run_judge.sh` dry run on clean Linux) **+ rubric trace generated + README == JSON**.

---

## 6. Rubric trace (artifacts, never self-scores — K5 + N rows)

| Pts | Row | Artifact required at freeze |
|---|---|---|
| 30 | Bimanual+multi-step | scorer/autonomous report ≥7/10 no hangs; HUD handoff/pour clips **with proxy disclaimer (N6)**; spoon+fork rows (G3) |
| 20 | VLA / reasoning | SmolVLA checkpoint (M1) + planner + perturb clip + **router mix log (N1)** |
| 15 | Robustness | 10 seeds × six axes (M2, eval-only, G2 provenance) + dual JSONs |
| 20 | OpenVINO | bench JSON FP16/INT8 p50–p99 within-machine + **claim ledger provenance (N3)** |
| 10 | Reproducibility | lockfiles, snapshot, single CLI (M5+N5 proof), run_judge, strict verify, freeze checklist |
| 5 | Innovation | chunking + ensembling + zero-drift randomizer + HUD telemetry |

---

## 7. Risks (V3.5 §6 + N rows)

1. No Core Ultra → AUTO/CPU, TBD columns, rehearsed `run_judge`. 2. Judge stoppage → one command, prebuilt artifacts, zero training. 3. Drift/deps → lockfiles + snapshot. 4. Contamination → nominal bootstrap, dual scorecards, non-overlap seeds. 5. **Router variability (N1 migration)**: determinism test auto-fails on env-dependent decisions; config-only knobs. 6. **SmolVLA fail (N2)**: day-2 kill line makes ACT-primary a *documented route*, not last-hour chaos. 7. **Claim drift (N3)**: strict-numbers hard-fail in CI + `run_judge`. 8. Foreign-derived numbers (M4 enforced at N3). 9. Liquid-proxy underestimation (N6) — admission inline at each claim site.

---

## 8. Honesty rules (V3.5 + N6 placement)

1. Liquid proxy is a proxy; the HUD label is the fill meter + `kinematic proxy, not fluid sim` (N6). 2. `--perturb`
   is inference-time only — `perturbation is eval-only`, noted in the robustness row (F9). 3. Randomizer eval-only
   unless tagged augmented — `randomizer is eval-only unless tagged augmented` (G2). 4. Intel numbers from *this*
   repo's JSONs; iGPU/NPU TBD (M4/N3). 5. Every performance/success claim traceable via `verify_claims.sh
   --strict-numbers`.

---

## 9. Immediate tasks (ordered; V3.5 tasks assumed implicit)

1. Lock the literal version file + OpenVINO snapshot; add strict mode to `verify_claims.sh` (**N3**) scaffolding.
2. Create `config/router_config.json` + deterministic `policy/router.py` + `test_router_determinism.py` (**N1**).
3. Seed `output/FREEZE_CHANGELOG.md`, `output/FREEZE_CHECKLIST.md` templates (**N4/N5**).
4. `eval_scoring.py` + liquid channel + tests; 5. `--item`; 6. lockfiles + `env_snapshot.json`; 7. `run_dev/run_judge.sh`
   (judge smoke runs `--strict-numbers`); 8. purge machine-specific paths; 9. M5 forwarder → interface test
   (`test_eval_cli_guard.py`); 10. Add proxy disclaimers to HUD + README (**N6**); 11. Calendar the day-2 kill deadline
   in the docs (**N2**).

---

## 10. Revision discipline (unchanged, extended)

V3, V3.1, v3.2, V3.5 remain on disk unchanged (V3.3/V3.4 texts supplied externally and never edited here); this **V3.6**
supersedes them in review flow. Future constraint/rubric changes bump to **V3.6.x** for mechanical edits (row counts,
gate numbers) or **V3.7** for design changes, with the F/G/H/K/M/N chain intact — no silent edits, no foreign numbers
(N3), no uncharted deletions (N5).
