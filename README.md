# TrackArt (Python)

A from-scratch **Python port of TrackArt v2.6.3** — a MATLAB application for
**single particle tracking (SPT) data analysis and simulation** — built on
`numpy` / `scipy` / `matplotlib` with a **tkinter** GUI.

The port keeps the original's five-module workflow, the same data layouts and
the same numerical methods (Michalet 2010 MSD error analysis, Qian 1991 D
errors, multi-exponential CPD fitting, permutation-based error estimation),
so results are directly comparable with the original software and with the
original `documentation.pdf`.
https://sourceforge.net/projects/trackart/

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (used for the environment and running)

## Install & run

```bash
cd trackart-py
uv sync
uv run trackart
```

That opens the main window with the tab bar: **SIM · Import · MSD · CPD · FIT**.
You can also run `uv run python -m trackart`.

## Modules

### SIM — trajectory simulation
Monte Carlo simulation of particle trajectories:

- **1particle 1D** — each particle diffuses with one of two diffusion
  coefficients (`D1`, `D2`) according to the population fraction.
- **Switch** — every particle switches between two states with rate
  constants `k21`, `k12`; fractions are estimated from the simulation.
- **Domains** — diffusion on a binary map of (simulated or imported) domains;
  state and switch probability depend on the underlying phase. Pair
  correlation G(r) can be computed for the domain map.

`Microstep` increases the temporal resolution of the simulation (mostly
important in Domains mode). `Stat. loc. uncert.` adds Gaussian
localisation noise. `Simulate` generates coordinates, `Fig` opens a
publication figure, `Export coordinates` writes them to CSV (the same
6-column format the Import module reads).

### Import — data import, inspection and filtering
- Reads **Mosaic ImageJ Particle Tracker** output (`*.xls` / `*.txt`,
  tab-separated, 1 header row) and **TrackArt simulation exports** (`*.csv`).
- Multiple files can be imported at once (track ids are made unique).
- `data in um` toggles pixel → micrometer calibration (`Px size`).
- `Calculate` computes per-trajectory SDs, MSDs and linear-fit diffusion
  coefficients (tracks shorter than 10 frames are dropped, as in the
  original).
- `Exclusion` filters trajectories by length, D range, MSD fit R² and mean
  intensity, plus an optional random sub-sampling. Plots: length
  distribution, all trajectories, single trajectory, length vs intensity,
  intensity histogram, D histogram, all MSDs (x/y log and D-heatmap
  options).
- `Export` writes the filtered coordinates back to CSV.

### MSD — mean square displacement analysis
Overall MSD is calculated for `Show (lags)` time lags and fit with one of
four models:

| model | equation |
|---|---|
| Normal | `<r²> = a + b·Δt`, `b = 4D` |
| Anomalous | `<r²> = a + b·Δt^c` |
| Confined | `<r²> = rc²[1 − A1·exp(−4A2·D·Δt/rc²)] + c` |
| Active | `<r²> = v²·Δt² + 4D·Δt + c` |

For the normal model the full Michalet 2010 error analysis is performed:
MSD standard deviation (from the analytic covariance), errors of the fit
parameters `a` and `b` for weighted and unweighted fits, dynamic and static
localisation uncertainty, reduced localisation error `x`, and the optimal
number of lags `pMin`. The Qian 1991 relative D error is also reported.
`Fit (lags)` + `weighted fit` control the fit range/weights; `intercept`
can be forced.

### CPD — cumulative probability distribution
For each time lag, the CPD of squared displacements is calculated (with
`Min P` / `Min r²` cutoffs), binned into `Bin size` bins and fit with
**1p / 2p / 3p** exponential models:

```
1 − Σ Fᵢ·exp(−r²/aᵢ),   aᵢ = 4·Dᵢ·t
```

Initial guesses (and optional *fixed* parameters) for `D`, `F` and the
`intercept` are set in the panel. The 3p fit constrains `D1 ≤ D2 ≤ D3` and
`ΣFᵢ = 1`. A residuals panel and a preview of per-population MSDs and
fractions help select the right model; an F-test between models is printed
to the console. `Export CPD` / `Export fit` save the binned CPD / fitted
curves.

### FIT — final fit of populations
Fits the per-population MSD curves extracted by the CPD module (each with
its own lags, model and forced intercept) and computes the fractions as the
mean (and std) over the selected fraction lags. `Permute` estimates the D
error by re-running the CPD analysis on random subsets of the trajectories
(Thompson-Tau outlier removal, extrapolation to the full dataset), exactly
as described in the original manual §6.4. `Copy` writes the results back
into the CPD guess fields.

## Test data

The `TestData/` folder contains real DOPC/SM single-molecule data
(Mosaic Particle Tracker `.xls` output). In the Import tab select several
files, set `Exposure 10 ms`, `Px size 0.16 um`, `Lags to use 5`, hit
*Calculate*, then experiment with the exclusion criteria (the manual's
worked example uses `Min frames 30`, `Min D 0.05`, `Max D 15`,
`Min R 0.9`, intensity `0.5–1.5`).

## Automated tests

```bash
cd trackart-py
uv run python tests/test_numerics.py   # simulation + MSD + CPD/FIT recovery
uv run python tests/test_3pop.py       # 3-population CPD + FIT recovery
uv run python tests/test_domains.py    # domains simulation
uv run python tests/test_import.py     # import of the shipped TestData
uv run python tests/smoke_gui.py       # builds the GUI, runs the full pipeline
```

The numerical tests reproduce the tutorial results of the original manual
(e.g. recovering D ≈ 2 / 0.2 µm²/s with fractions ≈ 50 % for the two-
population example, and D ≈ 2 / 0.2 / 0.02 for the three-population one).

## Notes on faithfulness

The port follows the original MATLAB code line by line. Known, deliberate
differences:

- `calculateMSD` uses the corrected `K = Σ max(len − Δt, 0)` (the original
  could produce negative K for trajectories shorter than the lag).
- The Domains-mode seeding/move loops have retry guards so a degenerate
  parameter set (e.g. `k21 = k12 = 0`) cannot freeze the GUI; the original
  would loop forever.
- The CPD binning keeps the original's `(bin centre)²` convention, which the
  exponential fit absorbs (results match the binned-on-r² variant to within
  fit noise).
- `nanstd`/`nanvar` use the sample (N−1) definition, as in MATLAB.

## License

GPL v3, matching the original TrackArt license (`TrackArt - current/license.txt`).
