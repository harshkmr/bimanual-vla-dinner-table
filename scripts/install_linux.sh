#!/usr/bin/env bash
# TICKET-01 — WSL2/Ubuntu dev-box installer (G1-aware: lockfiles win when present).
# Final Core Ultra box uses the hackathon 1_/2_ scripts; this prepares the dev box.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] System packages (mesa/osmesa, glfw, media,ocl)..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg \
  libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dri mesa-utils \
  clinfo vainfo curl git
export OCL_ICD_VENDORS=/etc/OpenCL/vendors || true

echo "[2/4] Headless GL default..."
if ! grep -q "MUJOCO_GL" ~/.bashrc 2>/dev/null; then
  echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
fi
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

echo "[3/4] Python environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "[4/4] Python deps (lockfile wins per G1; pins come from snapshot env, never hand-typed)..."
if [ -f "requirements-cpu.lock" ]; then
  pip install -r requirements-cpu.lock
else
  echo "NOTE: no requirements-cpu.lock yet (generated on the snapshot env, TICKET-01) — installing requirements.txt baseline."
  pip install -r requirements.txt
fi

echo "---"
echo "Next: python scripts/verify_stack.py"
echo "Full suite: pytest tests/ -v"
