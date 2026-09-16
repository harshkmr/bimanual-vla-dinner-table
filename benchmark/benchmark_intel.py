# Sprint: Intel inference benchmark -> output/benchmark_report.json (measured only).
"""Profiles the exported MLP policy on each available device (CPU/GPU.x):
latency p50/p90/p99, throughput, precision. Usage: python benchmark/benchmark_intel.py"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def bench(compiled, x, iters=200, warmup=10):
    for _ in range(warmup):
        compiled(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        compiled(x)
        ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3),
            "p90_ms": round(float(np.percentile(a, 90)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "fps": round(1000.0 / float(np.mean(a)), 1), "iters": iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "openvino_models", "mlp_bc_fp16.xml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "output", "benchmark_report.json"))
    ap.add_argument("--devices", default="")
    args = ap.parse_args()
    import openvino as ov
    import torch

    ck = torch.load(os.path.join(REPO_ROOT, "weights", "mlp_bc10.pt"), map_location="cpu", weights_only=False)
    x = (np.random.randn(1, int(ck["in_dim"])).astype(np.float32)
         - ck["mu"].numpy()) / ck["sd"].numpy()
    core = ov.Core()
    devices = args.devices.split(",") if args.devices else list(core.available_devices)
    report = {"openvino_version": ov.__version__, "host": platform.node(),
              "cpu": platform.processor(), "model": os.path.basename(args.model),
              "precision": "FP16" if "fp16" in args.model else "FP32",
              "input": [1, int(ck["in_dim"])], "results": {}}
    for dev in devices:
        try:
            comp = core.compile_model(args.model, dev)
            report["results"][dev] = bench(comp, x)
            print(dev, report["results"][dev])
        except Exception as e:  # noqa: BLE001
            report["results"][dev] = {"error": str(e)[:200]}
            print(dev, "FAILED:", str(e)[:120])
    json.dump(report, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
