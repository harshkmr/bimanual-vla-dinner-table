# TICKET-06 — OpenVINO export + NNCF + bench harness

Triage: ready-for-agent
Spec: spec/SPEC-bimanual-vla-dinner-table.md
Plan: IMPLEMENTATION_PLAN_V3.md §3 Phase 5
Blocked-By: TICKET-01, TICKET-04
Blocks: TICKET-07, TICKET-08

## Scope

Make optimization measurable and honest.

- `export_and_convert.py`: JIT trace → OpenVINO IR directly (no ONNX hop on WSL), FP32 + FP16; pinned OpenVINO version (convert == runtime).
- `quantize_nncf.py`: INT8 post-training quantization with fixed calibration set.
- `ov_bimanual_infer.py`: device auto-discovery, CPU default, explicit override incl. AUTO, CPU fallback on failure, latency vs throughput hints, parity check.
- `benchmark_intel.py → output/benchmark_report.json`: measured-only percentiles/throughput/speedup/device/precision; accelerator columns TBD until Core Ultra.

## Acceptance

- Latency drops vs Torch baseline with no task-success regression on fixed seed; JSON regenerates locally; no projected numbers in docs.

## Notes

Zero-fabrication rule lives here — projections are a spec violation, not a shortcut.
