import json
import os
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bimanual VLA Dinner-Table", layout="wide", page_icon="🤖")

REPO = Path(__file__).parent

st.title("🤖 Bimanual VLA Dinner-Table — Dual SO-101 / MuJoCo / OpenVINO on Intel")
st.caption("Intel Physical AI Online Challenge · Setting Up a Dinner Table · Simulation-first · Intel i7-9750H CPU + Intel UHD 630 iGPU (OpenVINO 2026.3.1) + GTX 1650")

# --- Cover ---
cover = REPO / "output" / "cover.png"
if cover.exists():
    st.image(str(cover), caption="Cover: dual SO-101 dinner-table scene (MuJoCo)", use_container_width=True)
else:
    st.info("Cover image will appear after scene build (output/cover.png).")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("🎥 Demo Video — 10 Randomized Seeds")
    video = REPO / "output" / "demo_10_seeds.mp4"
    if video.exists():
        st.video(str(video))
        st.caption("Command: “Open drawer, take fork, handoff, set plate/mug, pour water.” · HUD: seed, phase, outcome + “kinematic proxy, not fluid sim” disclaimer · 8/10 oracle full, autonomous honest 0/10")
    else:
        st.warning("Video not yet generated — run `python scripts/assemble_video.py` in sprint repo.")
    st.subheader("📄 Slides")
    slides = REPO / "docs" / "slides.pdf"
    if slides.exists():
        with open(slides, "rb") as f:
            st.download_button("Download slides.pdf", f, file_name="slides.pdf", mime="application/pdf")
    else:
        st.info("Slides PDF not found (docs/slides.pdf).")
with col2:
    st.subheader("📊 Benchmark — OpenVINO IR")
    bench = REPO / "output" / "benchmark_report.json"
    if bench.exists():
        data = json.loads(bench.read_text())
        st.json(data, expanded=False)
        # Table view
        rows = []
        for dev, v in data.get("results", {}).items():
            if "p50_ms" in v:
                rows.append({"device": dev, "p50 ms": v["p50_ms"], "p90 ms": v["p90_ms"], "fps": v["fps"]})
            else:
                rows.append({"device": dev, "status": v.get("status","")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Model: {data.get('model','')} · Parity MSE vs torch: {data.get('parity_mse_vs_torch','')} · {data.get('device_note','')}")
        st.caption("NPU not present on this host — no NPU numbers claimed.")
    else:
        st.warning("benchmark_report.json not found — run benchmark/benchmark_intel.py")

    st.subheader("🧪 10-Seed Scorecard")
    for name, label in [("10_seed_evaluation_report.json","Oracle (coupling=1)"), ("10_seed_evaluation_report_autonomous.json","Autonomous BC-MLP (coupling=0)")]:
        p = REPO / "output" / name
        if p.exists():
            arr = json.loads(p.read_text())
            df = pd.DataFrame([{"seed": r["seed"], "full": r["full"], "steps": r["steps"], "fill": r.get("mug_fill",0), **r["flags"]} for r in arr])
            st.markdown(f"**{label}** — {sum(r['full'] for r in arr)}/{len(arr)} full")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"{name} not found.")

st.divider()
st.subheader("🏗️ Architecture — Observe → Understand → Plan → Act → Optimize")
st.markdown("""
- **Observe:** headless MuJoCo (overhead + 2 wrist cams) + sim state
- **Understand/Plan:** 6-phase planner (OPEN→RETRIEVE→HANDOFF→PLACE→HOLD→POUR) + phase one-hot
- **Act:** dual SO-101 (6-DoF+gripper, SO-ARM100 model, mirrored B) · atomic handoff · clamps · watchdog
- **Policy:** BC-MLP on oracle demos (partial, honest 0/10) · oracle IK waypoints are *data-only* (gated flag)
- **Optimize:** torch → jit.trace → OpenVINO FP32/FP16 (no ONNX hop) · CPU 0.08ms / iGPU 0.44ms · deploy CPU default
""")

st.subheader("🔧 Reproduce")
st.code("""pip install mujoco torch openvino opencv-python numpy scipy
python scripts/build_scene.py
python data/scripted_expert.py --seed 0
python benchmark/evaluate_seeds.py --seeds 0-9
python policy/train_mlp.py --data data/demos10.npz --epochs 300
python policy/export_mlp_ov.py && python benchmark/benchmark_intel.py
python scripts/assemble_video.py""", language="bash")

st.caption("Honesty seeds (grep-checkable): `kinematic proxy, not fluid sim` · `perturbation is eval-only` · `randomizer is eval-only unless tagged augmented` · All numbers from output/*.json")
st.divider()
st.markdown("**Links:** [GitHub](https://github.com/harshkmr/bimanual-vla-dinner-table) · [Video](output/demo_10_seeds.mp4) · [Slides](docs/slides.pdf)")
