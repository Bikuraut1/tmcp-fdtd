# tmcp-fdtd

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![reproduce](https://github.com/USERNAME/tmcp-fdtd/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/tmcp-fdtd/actions/workflows/ci.yml)

Code and data for

> B. Raut, D. K. Naik and D. C. Panda, "FDTD Analysis of Sideband Generation and Polarization Conversion in a
> Circularly Polarized Patch Antenna on a Time-Modulated Substrate," *Microwave and Optical Technology Letters*
> (submitted).

This repository holds a three-dimensional FDTD solver (Python + numba) for media with time-varying permittivity,
all the simulation data behind the letter, and one script that rebuilds **every table and figure** of the letter
and of its Supporting Information from those data.

## Quick start (about 1 minute)

```bash
git clone https://github.com/USERNAME/tmcp-fdtd.git
cd tmcp-fdtd
python -m pip install -r requirements.txt        # numpy, scipy, matplotlib, numba, pytest
python reproduce.py all
```

`reproduce.py all` recomputes the derived analyses, writes the LaTeX tables to `outputs/tables/` and the figures to
`outputs/figures/`, then compares each table with the version in the manuscript (`reference/tables/`). It ends with:

```
All tables are identical to the manuscript.
```

With conda, run `conda env create -f environment.yml && conda activate tmcp-fdtd` instead of the pip step.
With make: `make install`, `make all`, `make test`.

Fast regression tests (short FDTD runs of the antenna, the cavity benchmark and the tables), about 2 minutes:

```bash
python -m pytest -q tests
```

## Where each result comes from

| Manuscript item | Script | Data |
|---|---|---|
| Fig. 1 (geometry) | `figures/fig1_geometry.py` | none |
| Fig. 2, Table 1 (static antenna vs HFSS) | `figures/fig2_validation_1575.py`, `analysis/static_table.py` | `round1/pulse_eps2.200.npz`, `round1/refined.npz`, `round3/dense_base.npz`, `commercial/hfss/` |
| Table 2 (polarization of the sidebands) | `analysis/gen_tables.py` | `round1/results.jsonl`, `round2/twomode.json`, `round2/analysis_rev.json` |
| Fig. 3, Table 3 (harmonics, f_m = f_c/200) | `figures/fig4_harmonics.py`, `analysis/gen_tables.py` | `round2/coarse/harm.jsonl`, `round1/pulse_eps*.npz` (adiabatic model) |
| Fig. 4 (sideband patterns) | `figures/fig5_sb_patterns.py` | `round2/coarse/harm_200_0.05.npz` |
| Table 4 (harmonic-resolved power balance) | `analysis/gen_power_table.py` | `round3/harm.jsonl` (tags `E_*`) |
| Table 5 (50 Ω source) | `analysis/gen_source_table.py` | `round3/harm.jsonl`, `round2/coarse/harm.jsonl` (tags `R50_*`) |
| Fig. 5 (unequal tuning of the two modes) | `figures/fig7_unequal.py`, `analysis/unequal_r3.py` | `round3/{uquad,patch}_*`, `round2/unequal_r3.json` |
| Fig. 6 (modulation frequency) | `figures/fig6_fm.py` | `round1/results.jsonl`, `round2/refined_dyn.jsonl` |
| Fig. S1 (cavity benchmark) | `figures/figS1_benchmark.py` | `round2/bench_*.npz`, `round2/benchmark.json` |
| Fig. S2 (with CST) | `CST=1 figures/fig2_validation_1575.py` | `commercial/cst/` |
| Table S1 (coarse vs refined grid) | `analysis/gen_s3_table.py` | `round3/harm.jsonl` (tags `ref_*`) |
| Table S2 (Huygens surface, finite board) | `analysis/si_tables.py` | `round3/harm.jsonl` (tags `E_*`, `bigbox_*`, `finite_*`), `round2/coarse/harm.jsonl` |
| Table S3 (two-mode sensitivity) | `analysis/twomode_r3.py`, `analysis/si_tables.py` | `round2/twomode_r3.json` |
| S6 scaling-law table, Fig. S3 | `analysis/si_tables.py`, `figures/fig3_eps_sensitivity.py` | `round2/coarse/pulse_eps*.npz`, `round2/analysis_rev.json` |
| Other numbers in the text (Q, energy decay, checks) | `src/analyze_mod.py`, `analysis/analyze_rev.py`, `analysis/power_r3.py` | `round1/numbers.json`, `round2/analysis_rev.json` |

All data paths are relative to `data/`. `outputs/tables/*.tex` are the exact files `\input` by the manuscript.

## Re-running the simulations

The shipped data were produced by the full-wave runs listed by

```bash
python reproduce.py simulate --list
```

Each job is one line of `reproduce.py` (script, arguments, output file and where it is used). The whole set takes
about a day (the refined-grid dynamic runs dominate); single jobs take 2 minutes to about 4 hours. To
re-create everything in a separate folder, leaving the shipped data untouched:

```bash
TMCP_DATA=./rerun python reproduce.py simulate --all         # or --group round3, or --job r3_energy
TMCP_DATA=./rerun TMCP_OUT=./outputs_rerun python reproduce.py all
```

(or `make simulate`). The solver is deterministic and does not depend on the number of threads: re-runs of the cavity
benchmark and of the dense static run reproduce the shipped files bit for bit. A different CPU or library version can
change the last digit of a few table entries, which `verify` then reports.

The HFSS and CST results in `data/commercial/` are exports from the commercial solvers and are not re-created.

## Repository layout

```
src/            fdtd_core.py   Yee FDTD + CPML, D-based time-varying update, soft/resistive gap source,
                               NTFF (5- or 6-faced Huygens box), per-harmonic Poynting flux, energy accounting
                mod_study.py   CW + modulation runs and one-period harmonic extraction
                adiabatic.py   adiabatic (quasi-static) model and scaling law
                analyze_mod.py numbers of the first submission
                tmcp_paths.py  all paths (TMCP_DATA, TMCP_OUT)
simulations/    the full-wave runs (see `reproduce.py simulate --list`)
analysis/       post-processing and LaTeX table generators
figures/        one script per figure
data/           round1/ round2/ round3/  simulation results (npz, jsonl, json); commercial/  HFSS and CST exports
reference/      tables exactly as in the manuscript (for `reproduce.py verify`)
tests/          fast regression tests
```

The data are grouped by the round in which they were produced (`round1` first submission, `round2` first revision,
`round3` second revision); the `simulate --list` table gives the job behind every file.

## Model in brief

Corner-truncated square patch (W = L = 15.9 mm, truncation c = 2 mm, probe offset 4 mm) on RT/duroid 5880
(h = 1.575 mm, ε_r = 2.2, tan δ = 0.0009). Coarse grid Δx = Δy = 0.497 mm, Δz = h/5; refined grid Δx/2, Δz/2.
The substrate permittivity is ε_r(t) = 2.2 + δε sin(2π f_m t) with f_m = f_c/N. Field update:
E^{n+1} = [(ε^n/Δt − σ^n/2) E^n + ∇×H] / (ε^{n+1}/Δt + σ^{n+1}/2).
Harmonics are extracted with a rectangular one-period DFT in the periodic steady state.

## Citation and licence

Please cite the article and this repository (see `CITATION.cff`; GitHub's "Cite this repository" button uses it).
Code: MIT licence. Data: CC BY 4.0. See `LICENSE`.
