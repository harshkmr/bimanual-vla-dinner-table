# TICKET-07 — Broadened domain randomization + drift tests

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 6
Blocked-By: TICKET-01, TICKET-02
Blocks: TICKET-08

## Scope

Cover all six challenge axes without drift.

- `domain_randomizer.py`: nominal-value cache at init, perturbations against base (never compounding); poses, mass (wide), friction, shape variants, lighting, backgrounds, placement shuffles.
- `test_domain_randomizer.py`: zero-drift across seeds; range coverage per axis.

## Acceptance

- Oracle stays 10/10 across randomizer; policy shows honest variance (flat 100% rejected as overfit signal).

## Notes

Missing shape/background = lost robustness points; keep ranges at V3 widths or wider.
