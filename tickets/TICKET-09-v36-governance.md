# TICKET-09 — V3.6 governance implementation (N1–N6)

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.6.md §1, §9
Blocked-By: TICKET-01
Blocks: TICKET-08

## Scope

Machine-checkable governance on top of the V3.5 build. No architecture changes.

- N1: `config/router_config.json` literals + pure `policy/router.py` + `tests/test_router_determinism.py` (10/10 determinism, boundary + kill-switch, mix shape). Confidence input is the documented proxy (mean last-N chunk confidences).
- N2: day-−2 kill-switch date in `docs/FREEZE_CHANGELOG.md`; `policy_selection` flag flip (no code edit).
- N3: `scripts/verify_claims.sh` strict mode + `output/claim_regeneration.json` ledger; versions/constants/dates allowlisted.
- N4: `output/FREEZE_CHECKLIST.md` 8-box gate for Phase 7.
- N5: `tests/test_eval_cli_guard.py` parity guard; forwarder deletion at freeze commit with changelog trail.
- N6: exact seed strings (`kinematic proxy, not fluid sim`; `perturbation is eval-only`; `randomizer is eval-only unless tagged augmented`) in README claims block + HUD footer + rubric trace; grep-checkable.

## Acceptance

- Router tests + CLI guard pass; `verify_claims.sh` runs (ledger-empty pass with note pre-runs).
- Freeze checklist + changelog templates present; N2 date filled before Phase 7.
- README + HUD stub carry the N6 seed strings verbatim.

## Notes

Scaffold lands here; real logic per the referenced tickets/phases. N-items never weaken M1–M5.
