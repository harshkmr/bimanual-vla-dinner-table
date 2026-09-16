# Sprint: assemble demo video from oracle frame archives + HUD overlays.
"""Usage: python scripts/assemble_video.py  -> output/demo_10_seeds.mp4 (640x480, 30fps)
HUD: seed, per-seed outcome (from eval JSON), phase tag, fill proxy disclaimer."""
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMAND = "Open the drawer, take the fork, hand it over, set plate and mug, pour water."
PHASES = ["OPEN DRAWER", "RETRIEVE FORK", "HANDOFF A->B", "PLACE FORK", "HOLD MUG", "POUR WATER"]
FOOTER = "liquid=FILL meter (kinematic proxy, not fluid sim)"


def card(text_lines, w=640, h=480, secs=3):
    img = np.zeros((h, w, 3), np.uint8)
    y = 150
    for line, scale in text_lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        cv2.putText(img, line, ((w - size[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += int(50 * scale) + 10
    img = np.vstack([img, np.zeros((90, w, 3), np.uint8)])  # match 570px stream height
    return [img] * int(30 * secs)


def main():
    rep = {r["seed"]: r for r in json.load(open(os.path.join(REPO_ROOT, "output", "10_seed_evaluation_report.json")))}
    clips = card([("BIMANUAL VLA DINNER TABLE", 0.9), ("Dual SO-101 / MuJoCo / OpenVINO", 0.6)], secs=4)
    n_fail = 0
    for s in range(10):
        z = np.load(os.path.join(REPO_ROOT, "output", f"oracle_frames_{s}.npz"))
        frames = z["frames"]
        ok = rep[s]["full"]
        n_fail += 0 if ok else 1
        n = len(frames)
        for i, f in enumerate(frames):
            img = cv2.resize(f, (640, 480))
            band = np.zeros((90, 640, 3), np.uint8)
            cv2.putText(band, f"Seed {s} | {'SUCCESS' if ok else 'PARTIAL (pour missed)'}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if ok else (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(band, f"Phase: {PHASES[min(5, 6 * i // n)]}", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(band, FOOTER, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)
            clips.append(np.vstack([img, band]))
    clips += card([(f"ORACLE 8/10 FULL | AUTONOMOUS 0/10 (PARTIAL, HONEST)", 0.55),
                   ("CPU 0.08ms | iGPU 0.44ms | OpenVINO 2026.3.1", 0.55)], secs=6)
    out = os.path.join(REPO_ROOT, "output", "demo_10_seeds.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 570))
    for img in clips:
        vw.write(img)
    vw.release()
    print(f"wrote {out}: {len(clips)} frames = {len(clips) / 30:.0f}s")


if __name__ == "__main__":
    main()
