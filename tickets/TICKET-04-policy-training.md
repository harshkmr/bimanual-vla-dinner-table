# TICKET-04 — SmolVLA finetune + ACT fallback + router

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 3
Blocked-By: TICKET-01, TICKET-03
Blocks: TICKET-05, TICKET-06, TICKET-07, TICKET-08

## Scope

Deliver the learned VLA core that defends the reasoning points.

- `train_smolvla.sh`: LeRobot training from the SmolVLA base (frozen vision encoder, trained action expert), 20–50k steps, Hub push.
- ACT grasp checkpoint (trace → OpenVINO convert path) as grasp-only fallback.
- Confidence router: low grasp confidence → ACT grasp, else SmolVLA chunk; custom `train.py` kept fallback-only.

## Acceptance

- Single-seed sim rollout >70% before randomization; checkpoint + Hub repo recorded.
- Torch-vs-OpenVINO parity within tolerance before benchmarking (measured in TICKET-06).

## Notes

This is the spec-critical ticket: language-conditioned policy must exist or VLA points fail.
