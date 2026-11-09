"""Numeric verification of the core modules against TrackArt's documented
results (tutorial sections 7.1.7 / 7.4.1 / 7.3.1)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.core.coordinates import calculateMSD, calculateSD
from trackart.core.cpd import cpd_calculateCPD
from trackart.core.fit import fit_fitMSD
from trackart.core.msd import msd_calculateDcoeff
from trackart.core.simulate import sim_simulateCoordinates


def test_roundsd_scalar():
    """roundsd must work on scalars (used by sim_fracprob after Switch
    simulations) and on arrays with zeros (regression for the
    'numpy.float64' item-assignment crash)."""
    from trackart.core.util import roundsd
    assert roundsd(0.5, 2) == 0.5
    assert roundsd(0.012345, 3) == 0.0123
    assert roundsd(12345, 2) == 12000
    assert roundsd(0.0, 2) == 0.0
    assert np.allclose(roundsd([0.5, 0.0, 1.234], 2), [0.5, 0.0, 1.2])

    # sim_fracprob path (scalar k's and tau)
    from trackart.core.simulate import sim_fracprob
    F1, F2, p21, p12 = sim_fracprob(1.0, 1.0, 0.01)
    print(f"[fracprob] F1={F1} F2={F2} p21={p21} p12={p12}")
    assert F1 == 0.5 and F2 == 0.5
    assert p12 == 0.01 and p21 == 0.01


def test_simulation_1p1D():
    rng = np.random.default_rng(42)
    # tutorial 7.1.2: 1 particle, D = 3 um^2/s, 500 frames, 10 ms
    coords, stats = sim_simulateCoordinates(
        1, 500, 0.01, [3.0, 0.0], 1.0, 0, 0, 0.0, 10.0, False, "1particle1D")
    SD, MSDall = calculateSD(coords, 20, 1)
    MSD = calculateMSD(SD, 0.01, 1)
    param = {
        "lagsToCalculate": 20, "lagsToFit": 10, "weightedFit": True,
        "model": "normal", "s0": 0, "te": 0, "tau": 0.01,
        "intercept": [0, 0],
    }
    results, table = msd_calculateDcoeff(MSD, param)
    print(f"[1p1D sim] D = {results['D']:.4f} (expected ~3)")
    assert abs(results["D"] - 3.0) < 0.6


def test_msd_weighted_fit():
    rng = np.random.default_rng(7)
    coords, stats = sim_simulateCoordinates(
        1, 500, 0.01, [3.0, 0.0], 1.0, 0, 0, 0.0, 10.0, False, "1particle1D")
    SD, MSDall = calculateSD(coords, 20, 1)
    MSD = calculateMSD(SD, 0.01, 1)
    param = {
        "lagsToCalculate": 20, "lagsToFit": 10, "weightedFit": True,
        "model": "normal", "s0": 0, "te": 0, "tau": 0.01,
        "intercept": [0, 0],
    }
    results, table = msd_calculateDcoeff(MSD, param)
    print(f"[MSD weighted] D = {results['D']:.4f}, table rows: {len(table)}")
    assert abs(results["D"] - 3.0) < 0.8


def test_cpd_2p_recovery():
    """Tutorial 7.4.1: 100 particles, D1=2 / D2=0.2, 50/50, 100 frames."""
    rng = np.random.default_rng(123)
    coords1, _ = sim_simulateCoordinates(
        50, 100, 0.01, [2.0, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords2, _ = sim_simulateCoordinates(
        50, 100, 0.01, [0.2, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords = np.vstack([coords1, coords2])
    # renumber tracks of the second population (ids 1..50 -> 51..100)
    coords2[:, 0] += 50
    coords2[:, 2] += 50
    coords = np.vstack([coords1, coords2])

    lags = 10
    SD, MSDall = calculateSD(coords, lags, 1)
    MSD = calculateMSD(SD, 1, 1)  # tau=1, timescale irrelevant for K
    N = MSD[:lags, 3]

    tau = 0.01
    Dguess = np.zeros((6, 3))
    Dguess[0, 0] = 1.0 * 4 * tau   # D1 guess = 1 um^2/s
    Dguess[1, 0] = 0.1 * 4 * tau   # D2 guess = 0.1 um^2/s
    Dguess[2, 0] = 0.01 * 4 * tau  # D3 guess = 0.01 um^2/s
    Dguess[3, 0] = 0.33
    Dguess[4, 0] = 0.33
    Dguess[5, 0] = 0.34

    for square_x in (True, False):
        cpd_bin, results = cpd_calculateCPD(
            SD, lags, Dguess, 1000, True, 0, 0, tau, N, square_x=square_x)
        a2 = results["model_2p"]["a"]
        D1 = a2[0, -1] / (4 * tau * lags)
        D2 = a2[1, -1] / (4 * tau * lags)
        F1 = a2[2, -1]
        print(f"[CPD 2p square_x={square_x}] D1={D1:.3f} D2={D2:.3f} F1={F1:.3f}")

    # the faithful (original) behaviour
    cpd_bin, results = cpd_calculateCPD(
        SD, lags, Dguess, 1000, True, 0, 0, tau, N, square_x=True)
    a2 = results["model_2p"]["a"]
    D1 = a2[0, -1] / (4 * tau * lags)
    D2 = a2[1, -1] / (4 * tau * lags)
    F1 = a2[2, -1]
    print(f"[CPD 2p faithful] last-lag D1={D1:.3f} D2={D2:.3f} F1={F1:.3f}")


def test_fit_module():
    """Run the full CPD + FIT pipeline (tutorial 7.4.1) with the corrected
    binning and check that D and fractions are recovered."""
    rng = np.random.default_rng(123)
    coords1, _ = sim_simulateCoordinates(
        50, 100, 0.01, [2.0, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords2, _ = sim_simulateCoordinates(
        50, 100, 0.01, [0.2, 0.0], 1.0, 0, 0, 0.0, 20.0, True, "1particle1D")
    coords2[:, 0] += 50
    coords2[:, 2] += 50
    coords = np.vstack([coords1, coords2])

    lags = 10
    SD, MSDall = calculateSD(coords, lags, 1)
    MSD = calculateMSD(SD, 1, 1)
    N = MSD[:lags, 3]
    tau = 0.01

    Dguess = np.zeros((6, 3))
    Dguess[0, 0] = 1.0 * 4 * tau
    Dguess[1, 0] = 0.1 * 4 * tau
    Dguess[2, 0] = 0.01 * 4 * tau
    Dguess[3, 0] = 0.33
    Dguess[4, 0] = 0.33
    Dguess[5, 0] = 0.34

    cpd_bin, results = cpd_calculateCPD(
        SD, lags, Dguess, 1000, True, 0, 0, tau, N, square_x=True)

    params = {
        "model": 2,
        "p1lags": np.arange(1, 6),
        "p2lags": np.arange(1, 6),
        "p3lags": np.arange(1, 6),
        "p1model": 1, "p2model": 1, "p3model": 1,
        "p1intercept": [0, 1], "p2intercept": [0, 1], "p3intercept": [0, 1],
        "fractionLags": np.arange(1, 6),
    }
    fit_results = fit_fitMSD(results["model_2p"], params)
    D1 = fit_results["p1"]["D"]
    D2 = fit_results["p2"]["D"]
    F1 = fit_results["fraction"][0, 0]
    print(f"[FIT 2p] D1={D1:.3f} (expected ~2) D2={D2:.3f} (expected ~0.2) F1={F1:.3f} (expected ~0.5)")
    assert abs(D1 - 2.0) < 0.5
    assert abs(D2 - 0.2) < 0.1
    assert abs(F1 - 0.5) < 0.15


if __name__ == "__main__":
    test_roundsd_scalar()
    test_simulation_1p1D()
    test_msd_weighted_fit()
    test_cpd_2p_recovery()
    test_fit_module()
    print("ALL TESTS PASSED")
