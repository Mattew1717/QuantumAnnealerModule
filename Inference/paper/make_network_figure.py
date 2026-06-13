import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT_DIR = Path(__file__).parent / "plots"
FILENAME = "multiising_network"

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

# Palette aligned with Inference/utils/plot.py.
MODULE_FACE = "#EAF0F8"
MODULE_EDGE = "#1f3b73"
SPIN_FACE = "#C9C9C9"   # grey spins, as in the standalone Ising-model figure
SPIN_EDGE = "#333333"
COMB_FACE = "#FCEAE0"  # light orange, paired with COMB_EDGE
COMB_EDGE = "#8a3d1a"
IO_FACE = "#F2F2F2"
IO_EDGE = "#555555"
EDGE_COLOR = "#555555"
COUPLING_COLOR = "#3a3a3a"


def draw_ising_glyph(ax, cx, cy, radius=0.46, n_spins=5):
    """A small fully connected spin graph (K5): grey nodes = spins and thin
    dark edges = pairwise couplings, matching the standalone Ising-model
    figure. The first vertex sits at the 12 o'clock position."""
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_spins, endpoint=False)
    pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=COUPLING_COLOR, linewidth=0.45, zorder=2)
    for (px, py) in pts:
        ax.add_patch(Circle((px, py), 0.078, facecolor=SPIN_FACE,
                            edgecolor=SPIN_EDGE, linewidth=0.7, zorder=3))


def arrow(ax, xy_from, xy_to, color=EDGE_COLOR, lw=0.9):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 8)
        ax.axis("off")

        # ----- input block -----
        in_cx, in_cy = 0.95, 4.0
        ax.add_patch(FancyBboxPatch(
            (in_cx - 0.55, in_cy - 1.0), 1.1, 2.0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=IO_FACE, edgecolor=IO_EDGE, linewidth=0.9, zorder=2))
        ax.text(in_cx, in_cy + 0.45, r"$\boldsymbol{\theta}$", ha="center",
                va="center", fontsize=13)
        ax.text(in_cx, in_cy - 0.5, "input\nbiases", ha="center", va="center",
                fontsize=7.5, color="#333333")

        # ----- five Ising modules -----
        n_modules = 5
        mod_cx = 4.0
        half_w, half_h = 1.55, 0.62
        mod_cys = np.linspace(7.0, 1.0, n_modules)
        comb_cx, comb_cy = 7.9, 4.0

        for m, cy in enumerate(mod_cys, start=1):
            ax.add_patch(FancyBboxPatch(
                (mod_cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=MODULE_FACE, edgecolor=MODULE_EDGE, linewidth=1.0,
                zorder=2))
            draw_ising_glyph(ax, mod_cx - half_w + 0.62, cy, radius=0.40)
            ax.text(mod_cx + 0.30, cy + 0.20, rf"Ising module {m}",
                    ha="center", va="center", fontsize=7.6)
            ax.text(mod_cx + 0.30, cy - 0.22,
                    rf"$F^{{({m})}}=\lambda^{{({m})}}E_0+\varepsilon^{{({m})}}$",
                    ha="center", va="center", fontsize=6.8, color="#333333")

            # input -> module (shared input replicated to every module)
            arrow(ax, (in_cx + 0.55, in_cy), (mod_cx - half_w, cy))
            # module -> combiner, labelled with the combiner weight w_m
            start = (mod_cx + half_w, cy)
            end = (comb_cx - 0.95, comb_cy + (cy - comb_cy) * 0.18)
            arrow(ax, start, end, color=COMB_EDGE, lw=1.0)
            lx = start[0] + 0.45 * (end[0] - start[0])
            ly = start[1] + 0.45 * (end[1] - start[1]) + 0.16
            ax.text(lx, ly, rf"$w_{{{m}}}$", ha="center", va="center",
                    fontsize=8, color=COMB_EDGE)

        # ----- linear combiner -----
        ax.add_patch(FancyBboxPatch(
            (comb_cx - 0.95, comb_cy - 1.25), 1.9, 2.5,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=COMB_FACE, edgecolor=COMB_EDGE, linewidth=1.1, zorder=2))
        ax.text(comb_cx, comb_cy + 0.78, "Linear", ha="center", va="center",
                fontsize=8.5)
        ax.text(comb_cx, comb_cy + 0.40, "combiner", ha="center", va="center",
                fontsize=8.5)
        ax.text(comb_cx, comb_cy - 0.32,
                r"$b+\sum_{m=1}^{M} w_m F^{(m)}$",
                ha="center", va="center", fontsize=8.2)

        # ----- output -----
        out_x = 9.9
        arrow(ax, (comb_cx + 0.95, comb_cy), (out_x - 0.05, comb_cy), lw=1.0)
        ax.text(out_x + 0.05, comb_cy + 0.32, r"$F(\boldsymbol{\theta})$",
                ha="center", va="center", fontsize=11)

        fig.tight_layout(pad=0.2)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(OUT_DIR / f"{FILENAME}.{ext}", format=ext,
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
    print(f"[ok] wrote {FILENAME}.svg/.pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
