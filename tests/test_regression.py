"""Fast regression tests (about 2-4 minutes in total).

    python -m pytest -q tests          (or:  python tests/test_regression.py)

1. short_pulse  : 600 steps of the paper's coarse antenna model (static and time-modulated), compared with
                  stored reference values -> catches any change of the FDTD kernels, CPML, source or NTFF.
2. cavity       : the PEC-cavity benchmark of Fig. S1 (D-based update vs. exact Hill-equation solution).
3. tables       : every table regenerated from the shipped data is identical to the manuscript.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REF = Path(__file__).with_name("reference_short_pulse.json")


def short_runs():
    import fdtd_core as fc
    H = 1.575e-3; DX = 15.9e-3 / 32
    m = fc.build_model(L_cells=32, W_cells=32, notch=4, feed_off_cells=8, dx=DX, dy=DX, dz=H / 5, h=H, pcb_margin=24)
    f = np.array([5.6e9, 5.754e9, 5.9e9])
    out = {}
    for name, kw in (("static", {}), ("modulated", dict(delta_eps=0.2, f_mod=5.754e9 / 10))):
        r = fc.run(m, Nt=600, dft_freqs=f, verbose=False, **kw)
        E = np.array([fc.broadside(r, k) for k in range(len(f))])
        out[name] = dict(v_feed=[float(x) for x in r["v_feed"][::50]], i_feed=[float(x) for x in r["i_feed"][::50]],
                         energy=float(np.max(r["energy"])),
                         Ex_abs=np.abs(E[:, 0]).tolist(), Ey_abs=np.abs(E[:, 1]).tolist(),
                         flux=[fc.box_flux(r, k)[0] for k in range(len(f))])
    return out


def _close(a, b, path=""):
    if isinstance(a, dict):
        for k in a: _close(a[k], b[k], path + "/" + k)
    else:
        a, b = np.asarray(a, float), np.asarray(b, float)
        scale = max(np.max(np.abs(b)), 1e-300)
        assert np.allclose(a, b, rtol=1e-7, atol=1e-9 * scale), f"{path}: max dev {np.max(np.abs(a - b)) / scale:.2e}"


def test_short_pulse():
    _close(short_runs(), json.load(open(REF)))


def test_cavity_benchmark(tmp_path):
    env = dict(os.environ, TMCP_DATA=str(tmp_path), MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(ROOT / "simulations/benchmark_cavity.py")], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    new = json.load(open(tmp_path / "round2/benchmark.json"))
    ref = json.load(open(ROOT / "data/round2/benchmark.json"))
    for k, v in ref.items():
        if isinstance(v, dict):
            assert abs(new[k]["max_rel_err"] - v["max_rel_err"]) <= 1e-6 * max(1, v["max_rel_err"]) + 1e-12, k
    # the paper's claim: D-based update error < 1e-3, naive update > 1e-2
    assert all(v["max_rel_err"] < 1e-3 for k, v in new.items() if k.endswith(",D"))
    assert all(v["max_rel_err"] > 1e-2 for k, v in new.items() if k.endswith(",naive"))


def test_tables_identical(tmp_path):
    env = dict(os.environ, TMCP_OUT=str(tmp_path), MPLBACKEND="Agg")
    for s in ("static_table", "gen_tables", "gen_power_table", "gen_source_table", "gen_s3_table", "si_tables"):
        subprocess.run([sys.executable, str(ROOT / f"analysis/{s}.py")], check=True, env=env, stdout=subprocess.DEVNULL)
    for ref in (ROOT / "reference/tables").glob("*.tex"):
        assert (tmp_path / "tables" / ref.name).read_text() == ref.read_text(), ref.name


if __name__ == "__main__":
    if "--make-reference" in sys.argv:          # only for maintainers, after an intentional change of the solver
        json.dump(short_runs(), open(REF, "w"), indent=1); print("wrote", REF); sys.exit()
    import tempfile
    test_short_pulse(); print("short_pulse ok")
    with tempfile.TemporaryDirectory() as d: test_cavity_benchmark(Path(d)); print("cavity ok")
    with tempfile.TemporaryDirectory() as d: test_tables_identical(Path(d)); print("tables ok")
