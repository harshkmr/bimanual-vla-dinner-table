# TICKET-01 — Scaffold skeleton + install/verify scripts

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 0
Blocked-By: (none — root ticket)
Blocks: TICKET-02, TICKET-03, TICKET-04, TICKET-05, TICKET-06, TICKET-07, TICKET-08

## Scope

Create the v3 directory skeleton with stub modules and the two environment scripts, so every later ticket has a place to land and a green headless baseline.

- Dirs: `env/ assets/scene/ data/ policy/smolvla/ training/ openvino_optimization/ benchmark/ scripts/ eval/ output/ tests/ docs/`
- Stubs (TODO + ticket ref, no logic): env modules, planner, policies, training script, OV export/quantize/infer, bench, eval, video recorder.
- `scripts/install_linux.sh`: OSMesa/EGL libs, `MUJOCO_GL`, `OCL_ICD_VENDORS`.
- `scripts/verify_stack.py`: 4 isolated checks (python/torch → headless MuJoCo step → OV compile CPU → policy tensor shapes) with actionable errors.

## Acceptance (externally observable)

- Fresh clone on WSL2 Ubuntu 24.04: install script runs, `verify_stack.py` passes on CPU with `available_devices=['CPU']` logged preliminary.
- `pytest tests/ -v` collects (stubs may skip, zero hard failures).
- No absolute local paths; no accelerator numbers claimed.

## Notes

Tracer-bullet root — keep thin; logic lands in later tickets.
