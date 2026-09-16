# TICKET-03 — Oracle + LeRobot demonstration data

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 2
Blocked-By: TICKET-01, TICKET-02
Blocks: TICKET-04, TICKET-07, TICKET-08

## Scope

Prove scene viability with a scripted oracle, then bank spec-compliant training data.

- `scripted_expert.py`: 6 phases (open drawer → retrieve fork → coordinated handoff → place plate → hold mug → pour), phase mutex, atomic single-owner handoff transfer, pour tilt above mug rim with stabilizing arm.
- `_handle_grasps` coupling behind `--use_coupling=1` for recording/oracle baseline only; default 0 for eval.
- `generate_demonstrations.py`: 50–100 episodes in LeRobot v2.0 layout with paraphrased NL templates, ≥1 handoff + hold+pour.

## Acceptance

- Oracle completes full sequence on fixed seed; dataset replays cleanly in LeRobot viewer.
- Eval default (`use_coupling=0`) uses pure contact physics.

## Notes

Oracle is data infrastructure, not the submitted policy — split scorecard depends on this flag.
