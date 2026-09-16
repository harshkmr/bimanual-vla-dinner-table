# TICKET-05 — Planner + bimanual coordinator wiring

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 4
Blocked-By: TICKET-01, TICKET-04
Blocks: TICKET-07, TICKET-08

## Scope

Wire understanding to action with safety.

- `planner.py`: NL → subgoal state machine with precondition validation + re-plan hooks (named planner, not VLM).
- Coordinator: static arm roles, shared-zone mutex, joint/velocity/workspace clamps, e-stop hook, fixed-rate control decoupled from inference (sync → async when slow).

## Acceptance

- Novel paraphrase of the dinner command executes without code change; failed subtask triggers re-plan.
- No limit violations during 10-seed smoke run.

## Notes

Keeps multi-step context across the sequence — the observable VLA behavior judges test.
