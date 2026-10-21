import sys, os; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tmcp_paths import *  # ROOT, DATA, R1, R2, R2C, R3, COMM, OUT, FIG, TAB
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-oriented style for a two-column letters journal (single column ~ 3.5 in)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "grid.color": "0.85",
    "axes.grid": True,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
COL_W = 3.45   # inches, single column
# colour-blind-safe, prints distinguishably in grey as well
C = dict(blue="#1f4e9c", orange="#d9711c", green="#2e8b57", red="#b2182b", grey="#555555", purple="#6a3d9a")
