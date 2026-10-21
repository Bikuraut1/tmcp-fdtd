import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import numpy as np
from fig_style import plt, COL_W, C
from matplotlib.patches import Polygon, Rectangle, Circle

W = L = 15.9; n = 2.0; foff = 4.0; Gs = 40.0
fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.9), gridspec_kw=dict(width_ratios=[1.0, 1.05]))
for ax in (a, b):
    ax.grid(False); ax.set_aspect("equal"); ax.axis("off")
arrow = dict(arrowstyle="<->", lw=0.5, shrinkA=0, shrinkB=0)
# ---------------- (a) top view ----------------
a.add_patch(Rectangle((-Gs/2, -Gs/2), Gs, Gs, fc="#efe6cf", ec="0.45", lw=0.5))
x0, x1, y0, y1 = -W/2, W/2, -L/2, L/2
pts = [(x0, y0), (x1-n, y0), (x1-n, y0+n), (x1, y0+n), (x1, y1), (x0+n, y1), (x0+n, y1-n), (x0, y1-n)]
a.add_patch(Polygon(pts, fc="#c9a227", ec="k", lw=0.6))
a.add_patch(Circle((0, -foff), 0.55, fc="w", ec="k", lw=0.6, zorder=5))
a.plot([0], [0], "k+", ms=4, mew=0.5)
a.annotate("", xy=(x0, y1+2.4), xytext=(x1, y1+2.4), arrowprops=arrow)
a.text(0, y1+3.2, "$W=L$", ha="center", va="bottom", fontsize=7)
a.annotate("", xy=(x1+1.4, y0), xytext=(x1+1.4, y0+n), arrowprops=arrow)
a.text(x1+1.9, y0+n/2, "$c$", va="center", fontsize=7)
a.annotate("", xy=(2.2, 0), xytext=(2.2, -foff), arrowprops=arrow)
a.text(2.7, -foff/2, "$d$", va="center", fontsize=7)
a.text(-0.8, -foff-1.6, "probe", ha="center", va="top", fontsize=6)
ox, oy = -Gs/2+2.5, -Gs/2+2.5
a.annotate("", xy=(ox+6, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", lw=0.6))
a.annotate("", xy=(ox, oy+6), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", lw=0.6))
a.text(ox+6.6, oy, "$x$", va="center", fontsize=7); a.text(ox, oy+6.6, "$y$", ha="center", va="bottom", fontsize=7)
a.text(Gs/2-1.5, -Gs/2+1.5, "substrate", ha="right", va="bottom", fontsize=6, color="0.35")
a.set_xlim(-Gs/2-0.5, Gs/2+0.5); a.set_ylim(-Gs/2-0.5, Gs/2+0.5)
a.set_title("(a) top view", fontsize=7, pad=2)
# ---------------- (b) side view (thickness exaggerated) ----------------
T = 3.0        # drawn substrate thickness
b.add_patch(Rectangle((-14, 0), 28, T, fc="#efe6cf", ec="0.45", lw=0.5))
b.plot([-14, 14], [0, 0], color="k", lw=1.8, solid_capstyle="butt")
b.plot([-7, 7], [T, T], color="#c9a227", lw=2.4, solid_capstyle="butt")
b.plot([0, 0], [T/3, T], color="k", lw=0.9)
b.add_patch(Rectangle((-0.5, 0), 1.0, T/3, fc="w", ec=C["red"], lw=0.7))
b.annotate("gap source", xy=(0.6, T/6), xytext=(4.5, -2.2), fontsize=6, color=C["red"],
           arrowprops=dict(arrowstyle="-", lw=0.4, color=C["red"]), va="center")
b.text(-13.6, -1.0, "ground", fontsize=6, va="top")
b.text(0, T+0.9, "patch", ha="center", va="bottom", fontsize=6)
b.annotate("", xy=(12.6, 0), xytext=(12.6, T), arrowprops=arrow)
b.text(12.1, T/2, "$h$", ha="right", va="center", fontsize=7)
b.text(-7.5, T/2, r"$\varepsilon_r(t)$", ha="center", va="center", fontsize=7)
b.text(0, -4.6, r"$\varepsilon_r(t)=\varepsilon_r+\delta\varepsilon\,\sin(2\pi f_m t)$", ha="center", fontsize=6.5)
b.set_xlim(-15, 15); b.set_ylim(-6, 7)
b.set_title("(b) side view (not to scale)", fontsize=7, pad=2)
fig.tight_layout(w_pad=0.2)
fig.savefig(f"{FIG}/fig1_geometry.pdf"); fig.savefig(f"{FIG}/fig1_geometry.png", dpi=200)
