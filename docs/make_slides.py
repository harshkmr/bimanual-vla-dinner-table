# Sprint: build slide deck PDF (reportlab) from measured repo artifacts.
"""Usage: python docs/make_slides.py -> docs/slides.pdf (10 slides, 2-3 sentences each)."""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLIDES = [
    ("Bimanual VLA Dinner-Table",
     ["Dual SO-101 arms set a dinner table in MuJoCo from a spoken command.",
      "Runs on Intel CPU+iGPU via OpenVINO. Simulation-first, fully reproducible."]),
    ("Problem & Solution",
     ["Table-setting needs two arms, language understanding, and edge deployment — rarely shown together.",
      "We combine a language planner, a learned policy, bimanual coordination, and OpenVINO optimization in one repo."]),
    ("How It Works",
     ["Planner splits the command into 6 phases; arms execute with atomic handoff ownership and safety clamps.",
      "Stack: MuJoCo, PyTorch BC policy (OpenVINO IR FP32/FP16), headless cameras, JSON scorecards."]),
    ("Demo: What You See",
     ["Screen recording: natural-language command, 10 randomized seeds, coordinated handoff and pour.",
      "HUD shows seed, outcome, phase, and the fill-meter proxy disclaimer on every clip."]),
    ("Measured Results",
     ["Oracle baseline 8/10 full tasks (all non-pour phases 10/10); learned policy honestly 0/10 partial.",
      "Inference: CPU 0.08ms, iGPU 0.44ms FP16; torch parity MSE 6.8e-9. All figures from output/*.json."]),
    ("Market Scope",
     ["TAM: home-assistive and hospitality robotics, a multi-billion-dollar emerging market.",
      "SAM: simulation-first manipulation R&D teams needing cheap bimanual benchmarks on Intel edge hardware."]),
    ("Revenue & Competition",
     ["Paths: edge-deployment tooling licenses, simulation content packs, integration services.",
      "Vs single-arm demos and cloud-GPU policies: ours is bimanual, language-driven, and runs on commodity Intel chips."]),
    ("Future Prospects",
     ["Scale to full cutlery sets, DAgger/ACT and SmolVLA finetuning, NPU targets on Core Ultra.",
      "Same pipeline ports to other tabletop tasks with new data only."]),
    ("Reproducibility",
     ["One repo: build script regenerates the scene, one command per phase, JSON-verified claims.",
      "Judges rerun sim, bench, and eval locally; no hidden steps, no projected numbers."]),
    ("Thank You",
     ["Code, video, benchmark, and architecture summary are in the submission.",
      "Honest partials today; the pipeline is built for full autonomy next."]),
]


def main():
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfgen import canvas

    W, H = landscape(A4)
    out = os.path.join(REPO_ROOT, "docs", "slides.pdf")
    c = canvas.Canvas(out, pagesize=landscape(A4))
    for i, (title, lines) in enumerate(SLIDES):
        c.setFillColorRGB(0.07, 0.2, 0.4)
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 30)
        c.drawString(60, H - 80, title)
        c.setFont("Helvetica", 20)
        y = H - 150
        for line in lines:
            for chunk in [line[j:j + 95] for j in range(0, len(line), 95)]:
                c.drawString(60, y, chunk)
                y -= 34
        c.setFont("Helvetica", 14)
        c.drawRightString(W - 60, 40, f"{i + 1}/{len(SLIDES)}")
        c.showPage()
    c.save()
    print("wrote", out)


if __name__ == "__main__":
    main()
