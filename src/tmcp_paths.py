"""Central paths. Every script imports this module, so the repository can live anywhere.

TMCP_DATA  : root of the simulation data (default: <repo>/data, the data shipped with the paper).
             Point it to an empty directory (e.g. ./rerun) to re-create all data from scratch.
TMCP_OUT   : where figures and tables are written (default: <repo>/outputs).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TMCP_DATA", ROOT / "data")).resolve()
R1 = DATA / "round1"            # first submission: coarse static + time-modulated runs, refined static
R2 = DATA / "round2"            # first revision: benchmark, refined dynamic f_c/50, two-mode fit, analyses
R2C = R2 / "coarse"             # first revision: coarse static/dynamic checks
R3 = DATA / "round3"            # second revision: energy accounting, refined f_c/200, finite board, 50 ohm, masks
COMM = ROOT / "data" / "commercial"   # HFSS and CST exports (always read from the shipped data)
OUT = Path(os.environ.get("TMCP_OUT", ROOT / "outputs")).resolve()
FIG = OUT / "figures"
TAB = OUT / "tables"
for _p in (R1, R2, R2C, R3, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DATA", str(R1))
