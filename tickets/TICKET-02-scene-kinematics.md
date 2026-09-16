# TICKET-02 — Scene + kinematics + headless env

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 1
Blocked-By: TICKET-01
Blocks: TICKET-03, TICKET-07, TICKET-08

## Scope

Build the stable dual-SO-101 dinner-table simulation: XML scene, IK, headless environment stepping.

- `dinner_table_scene.xml`: implicitfast solver, small timestep + control decimation (20 Hz control), tuned actuator gains, structural-link collision masking with live gripper pads, drawer slide joint, plate/mug/bottle/cutlery + target sites + handoff zone, calibrated reach-envelope geometry.
- `kinematics.py`: bounded IK solver, all targets inside reach, end-effector separation guard outside handoff zone.
- `bimanual_so101_env.py`: Gymnasium-style reset/step, headless renderer to NumPy RGB (overhead + 2 wrists), `use_coupling` flag plumbed (behavior lands in TICKET-03).

## Acceptance

- Scripted waypoint rollout 10/10 on fixed seed, no solver explosion, tracking error ~milliradian scale.
- No GLFW/GLX dependency; video path uses file writer only.

## Notes

Physics values per V3 §2; keep units and sites exact so later tickets align.
