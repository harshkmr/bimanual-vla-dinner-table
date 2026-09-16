# Freeze Checklist (N4) — Phase 7 closes only when ALL boxes are checked.

- [ ] (1) Prebuilt weights + OpenVINO FP16/INT8 models present (`weights/`, `openvino_models/`)
- [ ] (2) SmolVLA checkpoint exists under `weights/smolvla/` OR N2 kill-switch line logged below
- [ ] (3) `run_judge.sh` dry-run green on a clean rehearsal Linux box
- [ ] (4) `verify_claims.sh --strict-numbers` green
- [ ] (5) README measured-claims block == report JSON (no drift)
- [ ] (6) All tests green (incl. N1 router determinism + N5 CLI guard)
- [ ] (7) 10-seed autonomous ≥ 7/10 at coupling=0, no hangs
- [ ] (8) HUD video + snapshot frames exist (`output/demo_10_seeds.mp4`, `demo_snapshot_*.png`)

N2 kill-switch log (one-line revision if fired, else "not fired"):
- N2: not fired.
