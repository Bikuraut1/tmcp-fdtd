#!/usr/bin/env python3
"""
One entry point that reproduces every table and figure of the letter and its Supporting Information.

    python reproduce.py all              # analysis + tables + figures + verify   (about 1 minute, uses the shipped data)
    python reproduce.py tables           # LaTeX tables          -> outputs/tables/
    python reproduce.py figures          # PDF/PNG figures       -> outputs/figures/
    python reproduce.py analysis         # derived JSON files (two-mode fit, unequal tuning, revision analysis)
    python reproduce.py verify           # compare outputs/tables/*.tex with reference/tables/ (the manuscript)
    python reproduce.py simulate --list  # list every full-wave run behind the paper, with runtimes
    python reproduce.py simulate --job dense_base [--job ...]
    python reproduce.py simulate --group round3 | --all

Simulations write into $TMCP_DATA (default ./data, i.e. they overwrite the shipped results).
To re-create the data from scratch without touching the shipped copy:

    TMCP_DATA=./rerun python reproduce.py simulate --all
    TMCP_DATA=./rerun python reproduce.py all
"""
import argparse, os, subprocess, sys, time, filecmp
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# ---------------------------------------------------------------------------------------------------------------
# Every full-wave run behind the paper.  (name, group, script, argv, extra env, runtime [min], output, used in)
# Runtimes are wall-clock times of the original runs (numba, small cloud machine); a desktop is usually faster.
# ---------------------------------------------------------------------------------------------------------------
C = "simulations/"
FC = {"FC": "5.754e9"}
JOBS = [
    # --- round 1: static design and time-modulated runs, coarse grid (dx = 0.497 mm) ---
    ("r1_pulse", "round1", C + "run_1575.py", ["pulse:2.15", "pulse:2.2", "pulse:2.25"], FC, 24,
     "round1/pulse_eps*.npz", "Fig. 2, 3; Table 1; adiabatic model"),
    ("r1_harm", "round1", C + "run_1575.py",
     ["harm:200:0.05", "harm:200:0.2", "harm:50:0.05", "harm:50:0.2", "harm:20:0.05", "harm:20:0.2",
      "harm:10:0.05", "harm:10:0.2", "harm:20:0.0"], FC, 110, "round1/results.jsonl", "Fig. 4, 6; Tables 2, 3"),
    ("r1_refined", "round1", C + "conv_1575.py", [], {}, 75, "round1/refined.npz", "Fig. 2; Table 1 (refined grid)"),
    # --- round 2: benchmark, refined dynamic, coarse checks ---
    ("benchmark", "round2", C + "benchmark_cavity.py", [], {}, 2, "round2/bench_*.npz, benchmark.json",
     "Fig. S1 (cavity benchmark)"),
    ("r2_refined_dyn", "round2", C + "run_refined_dyn.py", ["50,0.05", "50,0.2"], {}, 330,
     "round2/refined_dyn.jsonl", "S3 (refined grid, f_m = f_c/50)"),
    ("r2_main", "round2", C + "run_rev.py", ["harm:200:0.05:tag=200_0.05", "harm:200:0.2:tag=200_0.2"], {}, 140,
     "round2/coarse/harm.jsonl", "Table 2, S2 (original box)"),
    ("r2_static", "round2", C + "run_rev.py",
     ["pulse:2.2:tag=static_base", "pulse:2.0", "pulse:2.1", "pulse:2.3", "pulse:2.4",
      "pulse:2.2:tand=0.01:tag=static_tand0.01", "pulse:2.2:src_R=50:tag=static_R50",
      "pulse:2.2:hb_lat=10:hb_top=30:tag=static_hb", "pulse:2.2:npml=16:tag=static_npml16",
      "pulse:2.2:tand=0:tag=static_lossless",
      "pulse:2.2:air_gap=10:below=10:closed=1:hb_lat=29:hb_bot=5:tag=static_finite"], {}, 105,
     "round2/coarse/*.npz", "Fig. S3; S2 static checks; scaling-law table"),
    ("r2_checks", "round2", C + "run_rev.py",
     ["harm:20:0.2:settle=8:nwin=1:tag=dt090_20_0.2", "harm:20:0.2:cfl=0.45:settle=8:nwin=1:tag=dt045_20_0.2",
      "harm:200:0.2:src_R=50:nwin=1:tag=R50_200_0.2", "harm:50:0.2:tand=0:tag=lossless_50_0.2",
      "harm:200:0.05:tand=0.01:nwin=1:tag=tand01_200_0.05"], {}, 85, "round2/coarse/harm.jsonl",
     "S3 (time step), S7 (loss), Table 5 (50 ohm, delta_eps = 0.2)"),
    ("r2_fc570", "round2", C + "run_rev.py", ["harm:100:0.05:nwin=1:tag=fc570_100_0.05"], {"FC": "5.70e9"}, 13,
     "round2/coarse/harm.jsonl", "S6 (carrier off the AR minimum)"),
    # --- round 3: power balance, Huygens box, finite board, 50 ohm, partial modulation, refined f_c/200 ---
    ("r3_energy", "round3", C + "run_r3.py",
     ["harm:200:0.2:tand=0:energy=1:hb_bot=0:tag=E_lossless_200_0.2", "harm:200:0.2:energy=1:hb_bot=0:tag=E_200_0.2",
      "harm:200:0.05:energy=1:hb_bot=0:tag=E_200_0.05"], {}, 60, "round3/harm.jsonl", "Table 4 (power balance), S4"),
    ("r3_dense", "round3", C + "run_r3.py",
     ["pulse:2.2:dense=1:tag=dense_base", "pulse:2.2:tand=0.01:dense=1:tag=dense_tand0.01"], {}, 17,
     "round3/dense_*.npz", "Table 1 (coarse AR band on a 2-MHz grid)"),
    ("r3_box", "round3", C + "run_r3.py",
     ["harm:200:0.2:hb_lat=10:hb_top=30:tag=bigbox_200_0.2", "harm:200:0.05:hb_lat=10:hb_top=30:tag=bigbox_200_0.05",
      "harm:200:0.05:air_gap=10:below=10:closed=1:hb_lat=29:hb_bot=5:tag=finite_200_0.05",
      "harm:200:0.2:air_gap=10:below=10:closed=1:hb_lat=29:hb_bot=5:tag=finite_200_0.2"], {}, 120,
     "round3/harm.jsonl", "S4 (Huygens surface, finite board)"),
    ("r3_source", "round3", C + "run_r3.py", ["harm:200:0.05:src_R=50:tag=R50_200_0.05"], {}, 20,
     "round3/harm.jsonl", "Table 5 (50 ohm source)"),
    ("r3_masks", "round3", C + "run_r3.py",
     ["pulse:2.2:mask=uquad:sdeps=0.05:tag=uquad_p", "pulse:2.2:mask=uquad:sdeps=-0.05:tag=uquad_m",
      "harm:200:0.05:mask=uquad:tag=uquad_200_0.05",
      "pulse:2.2:mask=patch:sdeps=0.05:tag=patch_p", "pulse:2.2:mask=patch:sdeps=-0.05:tag=patch_m",
      "harm:200:0.05:mask=patch:tag=patch_200_0.05"], {}, 60, "round3/{uquad,patch}_*", "Fig. 5 (unequal tuning)"),
    ("r3_refined", "round3", C + "run_r3.py",
     ["harm:200:0.05:grid=ref:nh=4:tag=ref_200_0.05", "harm:200:0.2:grid=ref:nh=4:tag=ref_200_0.2"], {}, 400,
     "round3/harm.jsonl", "S3 (refined grid, f_m = f_c/200)"),
]
JOB = {j[0]: j for j in JOBS}

ANALYSIS = ["src/analyze_mod.py", "analysis/analyze_rev.py", "analysis/twomode.py", "analysis/twomode_r3.py",
            "analysis/unequal_r3.py", "analysis/power_r3.py"]
TABLES = ["analysis/static_table.py", "analysis/gen_tables.py", "analysis/gen_power_table.py",
          "analysis/gen_source_table.py", "analysis/gen_s3_table.py", "analysis/si_tables.py"]
FIGURES = ["figures/fig1_geometry.py", "figures/fig2_validation_1575.py", "figures/fig3_eps_sensitivity.py",
           "figures/fig4_harmonics.py", "figures/fig5_sb_patterns.py", "figures/fig6_fm.py",
           "figures/fig7_unequal.py", "figures/figS1_benchmark.py", ("figures/fig2_validation_1575.py", {"CST": "1"})]


def run(script, args=(), env=None, quiet=True):
    e = dict(os.environ); e.setdefault("MPLBACKEND", "Agg"); e.update(env or {})
    t0 = time.time()
    p = subprocess.run([PY, str(ROOT / script), *args], cwd=ROOT, env=e,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None, text=True)
    tag = f"{script} {' '.join(args)}".strip() + ("".join(f"  [{k}={v}]" for k, v in (env or {}).items()))
    if p.returncode:
        print(f"  FAILED  {tag}\n{p.stdout if quiet else ''}"); sys.exit(p.returncode)
    print(f"  ok  {time.time() - t0:6.1f} s  {tag}")


def do_list(steps):
    for s in steps:
        run(*(s if isinstance(s, tuple) else (s, (), None))[:1], env=(s[1] if isinstance(s, tuple) else None))


def verify():
    ref, out = ROOT / "reference" / "tables", ROOT / "outputs" / "tables"
    bad = 0
    for f in sorted(ref.glob("*.tex")):
        o = out / f.name
        same = o.exists() and filecmp.cmp(f, o, shallow=False)
        bad += not same
        print(f"  {'IDENTICAL' if same else 'DIFFERENT'}  {f.name}")
    figs = sorted(p.name for p in (ROOT / "outputs" / "figures").glob("*.pdf"))
    print(f"  {len(figs)} figures in outputs/figures: {', '.join(figs)}")
    if bad:
        print(f"\n{bad} table(s) differ from the manuscript (expected only after re-running simulations, "
              "where last-digit changes can come from a different CPU/thread count)."); return 1
    print("\nAll tables are identical to the manuscript.")
    return 0


def simulate(a):
    if a.list:
        print(f"{'job':16s} {'group':7s} {'~min':>5s}  output  ->  used in")
        for n, g, s, argv, env, t, out, use in JOBS:
            print(f"{n:16s} {g:7s} {t:5d}  {out}  ->  {use}")
        print(f"\nTotal: about {sum(j[5] for j in JOBS) / 60:.0f} h (original machine). Data go to "
              f"{os.environ.get('TMCP_DATA', ROOT / 'data')}.")
        return
    sel = [j for j in JOBS if a.all or j[0] in (a.job or []) or j[1] in (a.group or [])]
    unknown = set(a.job or []) - set(JOB)
    if unknown or not sel:
        sys.exit(f"unknown/no job(s): {sorted(unknown)}; see  python reproduce.py simulate --list")
    for n, g, s, argv, env, t, out, use in sel:
        print(f"[{n}] ~{t} min -> {out}")
        if a.dry_run:
            print("   ", " ".join(f"{k}={v}" for k, v in env.items()), "python", s, " ".join(argv)); continue
        run(s, argv, env, quiet=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("all", "analysis", "tables", "figures", "verify"):
        sp.add_parser(c)
    s = sp.add_parser("simulate")
    s.add_argument("--list", action="store_true"); s.add_argument("--all", action="store_true")
    s.add_argument("--job", action="append"); s.add_argument("--group", action="append",
                                                              choices=["round1", "round2", "round3"])
    s.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    if a.cmd == "simulate":
        return simulate(a)
    if a.cmd in ("all", "analysis"):
        print("analysis:"); do_list(ANALYSIS)
    if a.cmd in ("all", "tables"):
        print("tables:"); do_list(TABLES)
    if a.cmd in ("all", "figures"):
        print("figures:"); do_list(FIGURES)
    if a.cmd in ("all", "verify"):
        print("verify:"); return verify()


if __name__ == "__main__":
    sys.exit(main() or 0)
