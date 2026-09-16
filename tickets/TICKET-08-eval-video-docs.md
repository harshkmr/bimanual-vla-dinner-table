# TICKET-08 — 10-seed eval + video + docs + Core Ultra runbook

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 7
Blocked-By: TICKET-01, TICKET-02, TICKET-03, TICKET-04, TICKET-05, TICKET-06, TICKET-07
Blocks: (none — terminal ticket)

## Scope

Ship the five required deliverables.

- `evaluate_seeds.py --seeds 10 --use_coupling {0,1} → output/10_seed_evaluation_report.json`: split oracle-vs-policy scorecard per phase + coordination score; freeze video only if policy ≥7/10 full-task.
- `record_demo_video.py → output/demo_10_seeds.mp4`: headless 4-view composite + HUD (phase, seed, coordination, inference status, instruction subtitle).
- `README.md` (rubric-mapped) + `ARCHITECTURE.md` + `SETUP_WSL2.md` + `INTEL_RUNBOOK.md` (drivers → reboot checks → stack install → verify → bench → video, unmodified on Core Ultra).
- Remaining pytest files (`test_expert/kinematics/openvino/task_planner`) green, CPU-runnable.

## Acceptance

- Clean-clone reproduction of video + bench JSON; runbook passes on Core Ultra with all three devices visible.
- Deliverable checklist: repo, sim, bench script, 10-seed video, readme/arch — all present.

## Notes

Terminal ticket — demo freeze lives here; no new behavior, only verification and packaging.
