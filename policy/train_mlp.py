# Sprint: behavior-cloning MLP 33->13 (partial imitation policy, CPU-trainable).
"""Usage: python policy/train_mlp.py --data data/demos.npz --epochs 200 --out weights/mlp_bc.pt"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BCMLP(nn.Module):
    def __init__(self, in_dim=39, out_dim=13):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, out_dim))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "demos.npz"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.01,
                    help="input noise std (robustness augmentation for OOD jitter)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "weights", "mlp_bc.pt"))
    args = ap.parse_args()
    z = np.load(args.data)
    X = torch.from_numpy(z["X"])
    Y = torch.from_numpy(z["Y"])
    # normalize with training stats (saved alongside for eval/export)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    ds = torch.utils.data.TensorDataset(Xn, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    net = BCMLP().train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    for ep in range(args.epochs):
        tot = 0.0
        for xb, yb in dl:
            xb = xb + torch.randn_like(xb) * args.noise
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{args.epochs} mse={tot / len(ds):.6f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "in_dim": X.shape[1], "out_dim": 13}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
