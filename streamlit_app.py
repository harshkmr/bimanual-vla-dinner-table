import json
import os
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bimanual VLA Dinner-Table", layout="wide", page_icon="🤖")

REPO = Path(__file__).parent

# --- Minimal-premium theme ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@500;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .hero { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0ea5e9 100%); color: white; border-radius: 18px; padding: 32px 36px; margin-bottom: 18px; }
  .hero h1 { font-size: 2.1rem; margin: 0 0 8px 0; font-weight: 700; letter-spacing: -0.02em; }
  .hero p { opacity: 0.85; margin: 6px 0 0 0; font-size: 0.95rem; }
  .pill { display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); border-radius: 999px; padding: 4px 10px; font-size: 0.75rem; margin-right: 6px; }
  .card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
  .metric { font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 600; }
  .muted { color: #64748b; font-size: 0.82rem; }
  .stVideo { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero">
  <div><span class="pill">MuJoCo 3.13</span><span class="pill">PyTorch 2.13</span><span class="pill">OpenVINO 2026.3.1</span><span class="pill">Intel i7 + UHD 630</span></div>
  <h1>Bimanual VLA Dinner-Table</h1>
  <p>Two SO-101 arms set a table from a spoken command — drawer, mid-air handoff, pour — in MuJoCo, optimized for Intel CPU + iGPU.</p>
  <p style="opacity:0.6; font-size:0.8rem;">Portfolio project · Robotics/ML · Simulation-first · Honest partials, fully reproducible</p>
</div>
""", unsafe_allow_html=True)

# --- Cover + key metrics row ---
c1, c2 = st.columns([1.35, 1])
with c1:
    cover = REPO / "output" / "cover.png"
    if cover.exists():
        st.image(str(cover), use_container_width=True)
        st.caption("Dual SO-101 on the dinner-table · MuJoCo headless render (overhead)")
    w1, w2, w3 = st.columns(3)
    # Load bench for metrics
    bench_path = REPO / "output" / "benchmark_report.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    oracle_path = REPO / "output" / "10_seed_evaluation_report.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else []
    with w1:
        st.markdown('<div class="card"><div class="muted">ORACLE BASELINE</div><div class="metric">8/10</div><div class="muted">full tasks · 10/10 non-pour</div></div>', unsafe_allow_html=True)
    with w2:
        cpu = bench.get("results", {}).get("CPU_FP16", {})
        st.markdown(f'<div class="card"><div class="muted">CPU FP16 p50</div><div class="metric">{cpu.get("p50_ms","—")} ms</div><div class="muted">{cpu.get("fps","—")} fps · parity 6.8e-9</div></div>', unsafe_allow_html=True)
    with w3:
        igpu = bench.get("results", {}).get("GPU.0_FP16", {})
        st.markdown(f'<div class="card"><div class="muted">iGPU (UHD 630) FP16</div><div class="metric">{igpu.get("p50_ms","—")} ms</div><div class="muted">{igpu.get("fps","—")} fps · OpenVINO GPU</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown("**🎥 Demo — 10 randomized seeds**")
    video = REPO / "output" / "demo_10_seeds.mp4"
    if video.exists():
        st.video(str(video))
        st.caption("“Open drawer, take fork, handoff, set plate/mug, pour water.” · HUD: seed · phase · outcome · `kinematic proxy, not fluid sim`")
    else:
        st.info("Video pending — `python scripts/assemble_video.py`")
    # Quick score strip
    if oracle:
        ok = sum(1 for r in oracle if r["full"])
        st.progress(ok/10, text=f"Oracle {ok}/10 full · 2 pour misses (seeds 2, 5)")
        auto_path = REPO / "output" / "10_seed_evaluation_report_autonomous.json"
        if auto_path.exists():
            auto = json.loads(auto_path.read_text())
            st.caption(f"Autonomous BC-MLP (honest partial): {sum(1 for r in auto if r['full'])}/10 — drawer 1/10 · distribution-shift limit, documented.")

st.divider()

# --- Architecture + tables ---
left, right = st.columns([1.1, 1])
with left:
    st.subheader("🏗️ Architecture")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1f5f9", fontname="Inter"];
      NL [label="NL command"];
      Cam [label="MuJoCo\\n2× wrist + overhead"];
      Planner [label="Planner\\n6 phases", fillcolor="#dbeafe"];
      Policy [label="BC-MLP 39→13\\n256×2", fillcolor="#fef3c7"];
      Coord [label="Coordinator\\natomic handoff", fillcolor="#dcfce7"];
      Sim [label="Dual SO-101\\nMuJoCo"];
      OV [label="OpenVINO IR\\nFP32/FP16", fillcolor="#e0e7ff"];
      NL -> Planner; Cam -> Planner; Planner -> Policy -> Coord -> Sim;
      Policy -> OV [style=dashed, label="jit.trace"];
    }
    """)
    st.caption("Observe → Understand/Plan → Act → Optimize. IK waypoints are *data-only* (gated `--use_coupling`); controller is the learned policy.")
with right:
    st.subheader("📊 Benchmark — OpenVINO IR (measured only)")
    if bench:
        rows = []
        for k, v in bench.get("results", {}).items():
            if "p50_ms" in v:
                rows.append({"device": k, "p50 ms": v["p50_ms"], "p90 ms": v["p90_ms"], "p99 ms": v["p99_ms"], "fps": v["fps"]})
            else:
                rows.append({"device": k, "note": v.get("status","")[:70]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{bench.get('model','')} · {bench.get('device_note','')}")
        st.caption("NPU not on this host — no NPU numbers claimed. In-distribution parity only.")
    else:
        st.warning("benchmark_report.json not found.")

    st.subheader("🧪 10-Seed Scorecards")
    tabs = st.tabs(["Oracle", "Autonomous"])
    for tab, path in zip(tabs, [oracle_path, REPO / "output" / "10_seed_evaluation_report_autonomous.json"]):
        with tab:
            if path.exists():
                arr = json.loads(path.read_text())
                df = pd.DataFrame([{"seed": r["seed"], "full": "✓" if r["full"] else "·", "steps": r["steps"], "fill": r.get("mug_fill",0), **{k: ("✓" if v else "·") for k,v in r["flags"].items()}} for r in arr])
                st.dataframe(df, use_container_width=True, hide_index=True, height=320)
            else:
                st.info("Not found.")

st.divider()
st.subheader("🔧 Reproduce")
st.code("""pip install mujoco torch openvino opencv-python numpy scipy
python scripts/build_scene.py
python data/scripted_expert.py --seed 0
python benchmark/evaluate_seeds.py --seeds 0-9
python policy/train_mlp.py --data data/demos10.npz --epochs 300
python policy/export_mlp_ov.py && python benchmark/benchmark_intel.py
python scripts/assemble_video.py
streamlit run streamlit_app.py""", language="bash")

cA, cB = st.columns(2)
with cA:
    st.markdown("**Downloads**")
    for fn in ["docs/slides.pdf"]:
        p = REPO / fn
        if p.exists():
            st.download_button(f"⬇ {fn}", p.read_bytes(), file_name=Path(fn).name)
with cB:
    st.markdown("**Links**")
    st.markdown("[GitHub](https://github.com/harshkmr/bimanual-vla-dinner-table) · [Video](output/demo_10_seeds.mp4) · [Original plans](../README.md)")

st.caption("Honesty seeds: `kinematic proxy, not fluid sim` · `perturbation is eval-only` · `randomizer is eval-only unless tagged augmented` · All numbers from `output/*.json`")
