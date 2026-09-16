# Sprint: BC-MLP -> OpenVINO IR (FP32 + FP16) via direct convert (no ONNX hop).
"""Usage: python policy/export_mlp_ov.py --weights weights/mlp_bc10.pt --outdir openvino_models
Needs: pip install openvino (present: 2026.3.1, devices CPU/GPU.0/GPU.1)."""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from policy.train_mlp import BCMLP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(REPO_ROOT, "weights", "mlp_bc10.pt"))
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "openvino_models"))
    args = ap.parse_args()
    import openvino as ov

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    net = BCMLP(in_dim=int(ck["in_dim"]), out_dim=int(ck["out_dim"])).eval()
    net.load_state_dict(ck["state"])
    ex = torch.randn(1, int(ck["in_dim"]))
    with torch.no_grad():
        ref = net(ex).numpy()
    traced = torch.jit.trace(net, ex)  # trace path per Intel ACT conversion guide
    ov_model = ov.convert_model(traced, input=[("x", [1, int(ck["in_dim"])], torch.float32)])
    os.makedirs(args.outdir, exist_ok=True)
    ov.save_model(ov_model, os.path.join(args.outdir, "mlp_bc_fp32.xml"))
    ov.save_model(ov_model, os.path.join(args.outdir, "mlp_bc_fp16.xml"), compress_to_fp16=True)
    # parity check on FP16 with in-distribution samples (random N(0,1) inputs are
    # far off-manifold and produce giant activations — never use them for parity)
    core = ov.Core()
    z = np.load(os.path.join(REPO_ROOT, "data", "demos10.npz"))
    xs = (z["X"][:32] - ck["mu"].numpy()) / ck["sd"].numpy()
    with torch.no_grad():
        ref = net(torch.from_numpy(xs)).numpy()
    comp = core.compile_model(os.path.join(args.outdir, "mlp_bc_fp16.xml"), "CPU")
    got = np.stack([comp(xs[i:i + 1])[0][0] for i in range(len(xs))])
    mse = float(np.mean((got - ref) ** 2))
    print(f"saved FP32+FP16 to {args.outdir}; FP16 parity MSE vs torch: {mse:.2e}")
    print("openvino:", ov.__version__, "devices:", core.available_devices)


if __name__ == "__main__":
    main()
