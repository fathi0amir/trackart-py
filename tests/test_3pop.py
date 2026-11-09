import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.core.coordinates import calculateMSD, calculateSD
from trackart.core.cpd import cpd_calculateCPD
from trackart.core.fit import fit_fitMSD
from trackart.core.simulate import sim_simulateCoordinates


def test_3p():
    np.random.seed(5)
    Ds = [2.0, 0.2, 0.02]
    coords_parts = []
    for k, D in enumerate(Ds):
        c, _ = sim_simulateCoordinates(20, 300, 0.01, [D, 0.0], 1.0, 0, 0,
                                       0.0, 20.0, True, "1particle1D")
        c = c.copy()
        c[:, 0] += k * 100
        c[:, 2] += k * 100
        coords_parts.append(c)
    coords = np.vstack(coords_parts)

    lags = 8
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

    cpd_bin, results = cpd_calculateCPD(SD, lags, Dguess, 500, True, 0, 0,
                                        tau, N, square_x=True)
    a3 = results["model_3p"]["a"]
    D1 = a3[0, -1] / (4 * tau * lags)
    D2 = a3[1, -1] / (4 * tau * lags)
    D3 = a3[2, -1] / (4 * tau * lags)
    print(f"[CPD 3p] last lag D1={D1:.3f} D2={D2:.3f} D3={D3:.3f} "
          f"F={np.round(a3[3:6, -1], 3)}")

    params = {
        "model": 3,
        "p1lags": np.arange(1, 6), "p2lags": np.arange(1, 6),
        "p3lags": np.arange(1, 6),
        "p1model": 1, "p2model": 1, "p3model": 1,
        "p1intercept": [0, 1], "p2intercept": [0, 1], "p3intercept": [0, 1],
        "fractionLags": np.arange(1, 6),
    }
    fr = fit_fitMSD(results["model_3p"], params)
    print(f"[FIT 3p] D1={fr['p1']['D']:.3f} D2={fr['p2']['D']:.3f} "
          f"D3={fr['p3']['D']:.3f}")
    print(f"[FIT 3p] F={np.round(fr['fraction'][:, 0], 3)} "
          f"(expected ~[0.33, 0.33, 0.33])")
    assert abs(fr["p1"]["D"] - 2.0) < 0.6, "D1 recovery failed"
    assert abs(fr["p2"]["D"] - 0.2) < 0.1, "D2 recovery failed"
    assert abs(fr["p3"]["D"] - 0.02) < 0.02, "D3 recovery failed"
    assert np.all(np.abs(fr["fraction"][:, 0] - 1 / 3) < 0.12), "fractions failed"


if __name__ == "__main__":
    test_3p()
    print("3-POP TEST PASSED")
