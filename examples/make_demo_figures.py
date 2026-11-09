"""Generate example figures for every module (no GUI needed).

Usage:  uv run python examples/make_demo_figures.py
Output: PNG files in examples/out/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trackart.core.coordinates import calculateMSD, calculateSD
from trackart.core.cpd import cpd_calculateCPD
from trackart.core.fit import fit_fitMSD
from trackart.core.msd import msd_calculateDcoeff
from trackart.core.simulate import sim_simulateCoordinates, sim_simulateDomains
from trackart.gui import plots
from trackart.gui.state import State

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)


def demo_sim():
    state = State()
    domains = sim_simulateDomains(256, 30, 0.3, 0.0)
    state.SIM.domains = domains
    state.SIM.param = {"domainsResolution": 256, "simRange": 25.6,
                       "method": "Domains", "particleCount": 8}
    state.SIM.plot.showDomains = 1
    state.SIM.plot.showTrajectories = 0

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plots.sim_plots(state.SIM, ax)
    fig.savefig(os.path.join(OUT, "sim_domains.png"), dpi=110)
    plt.close(fig)

    state.SIM.plot.showDomains = 0
    state.SIM.plot.showTrajectories = 1
    coords, stats = sim_simulateCoordinates(
        8, 1000, 0.01, [3.0, 0.3], 0.5, 2, 2, 0.02, 25.6, True,
        "Domains", domains=domains)
    state.SIM.coordinates = coords
    state.SIM.param["method"] = "Domains"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plots.sim_plots(state.SIM, ax)
    fig.savefig(os.path.join(OUT, "sim_trajectories_domains.png"), dpi=110)
    plt.close(fig)

    # pair correlation
    state.SIM.plot.showDomains = 0
    state.SIM.plot.showTrajectories = 0
    state.SIM.plot.showPairCorr = 1
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plots.sim_plots(state.SIM, ax)
    fig.savefig(os.path.join(OUT, "sim_pair_correlation.png"), dpi=110)
    plt.close(fig)
    state.SIM.plot.showPairCorr = 0
    print("sim figures OK")


def demo_msd():
    state = State()
    coords, _ = sim_simulateCoordinates(
        1, 500, 0.01, [3.0, 0.0], 1.0, 0, 0, 0.0, 10.0, False, "1particle1D")
    state.SIM.coordinates = coords
    SD, MSDall = calculateSD(coords, 20, 1)
    MSD = calculateMSD(SD, 0.01, 1)
    state.SIM.MSD = MSD
    state.SIM.MSDall = MSDall
    param = {"lagsToCalculate": 20, "lagsToFit": 10, "weightedFit": True,
             "model": "normal", "s0": 0.1, "te": 0, "tau": 0.01,
             "intercept": [0, 0], "dataSource": "simulated"}
    results, table = msd_calculateDcoeff(MSD, param)
    state.MSD.results = results
    state.MSD.param = param
    state.MSD.plot.msd = 1
    state.MSD.plot.msdSDV = 1
    state.MSD.plot.diffCoeffSlope = 1

    fig, ax = plt.subplots(figsize=(6, 5))
    plots.msd_plots(state, ax)
    ax.set_title("MSD fit, normal diffusion")
    fig.savefig(os.path.join(OUT, "msd_normal.png"), dpi=110)
    plt.close(fig)
    print("MSD figure OK, D =", round(results["D"], 4))


def demo_cpd_fit():
    state = State()
    coords1, _ = sim_simulateCoordinates(
        50, 200, 0.01, [2.0, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords2, _ = sim_simulateCoordinates(
        50, 200, 0.01, [0.2, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords2[:, 0] += 50
    coords2[:, 2] += 50
    coords = np.vstack([coords1, coords2])
    state.SIM.coordinates = coords

    lags = 8
    SD, MSDall = calculateSD(coords, lags, 1)
    MSD = calculateMSD(SD, 1, 1)
    state.SIM.SD = SD
    state.SIM.MSD = MSD
    tau = 0.01
    Dguess = np.zeros((6, 3))
    Dguess[0, 0] = 1.0 * 4 * tau
    Dguess[1, 0] = 0.1 * 4 * tau
    Dguess[2, 0] = 0.01 * 4 * tau
    Dguess[3, 0] = 0.33
    Dguess[4, 0] = 0.33
    Dguess[5, 0] = 0.34

    cpd_bin, results = cpd_calculateCPD(SD, lags, Dguess, 500, True, 0, 0,
                                        tau, MSD[:lags, 3])
    state.CPD.cpd_bin = cpd_bin
    state.CPD.results = results
    state.CPD.param = {"showLag": 5, "plotModel": 2}

    fig = plt.figure(figsize=(6, 7))
    ax_top = fig.add_axes([0.12, 0.42, 0.8, 0.5])
    ax_res = fig.add_axes([0.12, 0.1, 0.8, 0.22])
    preview = [fig.add_axes([0.05 + 0.23 * k, 0.02, 0.2, 0.05]) for k in range(4)]
    plots.cpd_plots(state, ax_top, ax_res, preview)
    fig.savefig(os.path.join(OUT, "cpd_2p.png"), dpi=110)
    plt.close(fig)

    # FIT
    params = {"model": 2,
              "p1lags": np.arange(1, 6), "p2lags": np.arange(1, 6),
              "p1model": 1, "p2model": 1,
              "p1intercept": [0, 1], "p2intercept": [0, 1],
              "fractionLags": np.arange(1, 6)}
    fr = fit_fitMSD(results["model_2p"], params)
    state.FIT.results = fr
    state.FIT.param = params
    state.FIT.data = results["model_2p"]

    fig = plt.figure(figsize=(8, 6))
    axes = [fig.add_axes([0.1, 0.55, 0.35, 0.38]),
            fig.add_axes([0.6, 0.55, 0.35, 0.38]),
            fig.add_axes([0.1, 0.1, 0.35, 0.38]),
            fig.add_axes([0.6, 0.1, 0.35, 0.38])]
    plots.fit_plots(state, axes)
    fig.savefig(os.path.join(OUT, "fit_2p.png"), dpi=110)
    plt.close(fig)
    print("CPD/FIT figures OK, D1 =", round(fr["p1"]["D"], 3),
          "D2 =", round(fr["p2"]["D"], 3), "F1 =", round(fr["fraction"][0, 0], 3))


if __name__ == "__main__":
    demo_sim()
    demo_msd()
    demo_cpd_fit()
    print("figures written to", OUT)
