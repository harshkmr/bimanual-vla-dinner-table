# Bimanual VLA Dinner-Table (V3 scaffold)

Intel Physical AI Online Challenge — dual SO-101 / MuJoCo / OpenVINO / Core Ultra 2/3.
Plan: `IMPLEMENTATION_PLAN_V3.md` · Spec: `spec/SPEC-bimanual-vla-dinner-table.md` · Tickets: `tickets/` (blocking edges as text, `ready-for-agent`).

Status: TICKET-01 scaffold only — every module raises `NotImplementedError` until its ticket lands. Start with `scripts/install_linux.sh` + `scripts/verify_stack.py`, then TICKET-02.

## Measured results (K1/N3 — auto-generated block only; empty until real runs)

No measured numbers yet. Every performance/success figure here must come from `output/*.json` via `scripts/verify_claims.sh --strict-numbers`, or be labeled target-not-measured. NPU/iGPU columns are TBD until a Core Ultra run.

## Honesty rules (N6 seed strings — grep-checkable, kept verbatim)

- Liquid display is a fill meter: `kinematic proxy, not fluid sim`.
- Mid-episode adaptation tests: `perturbation is eval-only`.
- Scene variation: `randomizer is eval-only unless tagged augmented`.
