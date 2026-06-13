"""
Render the standalone Ising-model figure (Fig. 1 of the paper) as a high
quality vector PDF. Three panels: (i) a fully connected K5 Ising graph with
spin variables z_i, local biases theta_i and a labelled coupling Gamma_12;
(ii) the same graph after minimization, with black/white nodes encoding the
ground-state assignment z*; (iii) the resulting spin vector z*, the minimal
energy E_0(theta, Gamma) and a colour legend.

Outputs SVG + PDF (for the manuscript) and a PNG (for quick visual checks)
in Inference/paper/plots/.

Run from the repository root:
    python -m Inference.paper.make_ising_model_figure
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT_DIR = Path(__file__).parent / "plots"
FILENAME = "ising_model"

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

NODE_GREY = "#BFBFBF"
NODE_BLACK = "#000000"
NODE_WHITE = "#FFFFFF"
EDGE_COLOR = "#1a1a1a"
NODE_EDGE = "#1a1a1a"

# Pentagon angles (degrees): z1 top, then clockwise z2, z3, z4, z5.
ANGLES_DEG = [90, 18, -54, -126, 162]
NODE_R = 0.34


def _pentagon(cx, cy, R):
    return [(cx + R * np.cos(np.deg2rad(a)), cy + R * np.sin(np.deg2rad(a)))
            for a in ANGLES_DEG]


def draw_graph(ax, cx, cy, R, faces, label_colors, star=False,
               show_theta=False, show_gamma=False):
    pts = _pentagon(cx, cy, R)
    # couplings: all pairs
    for i in range(5):
        for j in range(i + 1, 5):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=EDGE_COLOR, linewidth=1.0, zorder=1)
    # spins
    for k, (px, py) in enumerate(pts):
        ax.add_patch(Circle((px, py), NODE_R, facecolor=faces[k],
                            edgecolor=NODE_EDGE, linewidth=1.1, zorder=3))
        lbl = rf"$z_{{{k + 1}}}^*$" if star else rf"$z_{{{k + 1}}}$"
        ax.text(px, py, lbl, ha="center", va="center", fontsize=9.5,
                color=label_colors[k], zorder=4)
    # theta labels, placed radially outward from each spin
    if show_theta:
        for k, (px, py) in enumerate(pts):
            a = np.deg2rad(ANGLES_DEG[k])
            tx, ty = cx + (R + 0.62) * np.cos(a), cy + (R + 0.62) * np.sin(a)
            ax.text(tx, ty, rf"$\theta_{{{k + 1}}}$", ha="center",
                    va="center", fontsize=11)
    # one coupling label on the z1--z2 edge
    if show_gamma:
        mx = (pts[0][0] + pts[1][0]) / 2
        my = (pts[0][1] + pts[1][1]) / 2
        ax.text(mx + 0.30, my + 0.18, r"$\Gamma_{12}$", ha="left",
                va="center", fontsize=11)


def big_arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=22, color="#000000", linewidth=2.2,
                 shrinkA=0, shrinkB=0, zorder=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z_star = [+1, -1, -1, +1, -1]  # ground-state assignment shown in panel (ii)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(10.5, 3.4))
        ax.set_xlim(0, 17.5)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")

        cy = 3.1

        # ----- panel (i): parametric Ising graph -----
        draw_graph(ax, 2.9, cy, 1.6,
                   faces=[NODE_GREY] * 5,
                   label_colors=["#000000"] * 5,
                   show_theta=True, show_gamma=True)

        big_arrow(ax, 5.2, 6.3, cy)

        # ----- panel (ii): solved graph (black=+1, white=-1) -----
        faces = [NODE_BLACK if s > 0 else NODE_WHITE for s in z_star]
        lbl_c = ["#FFFFFF" if s > 0 else "#000000" for s in z_star]
        draw_graph(ax, 9.0, cy, 1.6, faces=faces, label_colors=lbl_c, star=True)

        big_arrow(ax, 11.3, 12.2, cy)

        # ----- panel (iii): output vector, energy and legend -----
        vx = 13.9
        ax.text(vx - 0.80, cy, r"$\mathbf{z}^* =$", ha="right", va="center",
                fontsize=12)
        spacing = 0.46
        ys = cy + spacing * np.array([2, 1, 0, -1, -2])
        for s, yy in zip(z_star, ys):
            ax.text(vx, yy, rf"${'+1' if s > 0 else '-1'}$", ha="center",
                    va="center", fontsize=11)
        # parentheses around the column
        ax.text(vx - 0.42, cy, "(", ha="center", va="center", fontsize=40)
        ax.text(vx + 0.42, cy, ")", ha="center", va="center", fontsize=40)
        ax.text(vx, cy - 3 * spacing - 0.15, r"$E_0(\theta,\Gamma)$",
                ha="center", va="top", fontsize=11)

        # legend
        lx = 15.4
        ax.add_patch(Circle((lx, cy + 0.45), 0.20, facecolor=NODE_BLACK,
                            edgecolor=NODE_EDGE, linewidth=1.0))
        ax.text(lx + 0.45, cy + 0.45, r"$=\,+1$", ha="left", va="center",
                fontsize=11)
        ax.add_patch(Circle((lx, cy - 0.45), 0.20, facecolor=NODE_WHITE,
                            edgecolor=NODE_EDGE, linewidth=1.0))
        ax.text(lx + 0.45, cy - 0.45, r"$=\,-1$", ha="left", va="center",
                fontsize=11)

        for ext in ("svg", "pdf", "png"):
            fig.savefig(OUT_DIR / f"{FILENAME}.{ext}", format=ext,
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
    print(f"[ok] wrote {FILENAME}.svg/.pdf/.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
