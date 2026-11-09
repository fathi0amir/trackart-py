import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.core.simulate import (sim_reduceCoordinates, sim_simulateCoordinates,
                                    sim_simulateDomains)


def test_domains_map():
    domains = sim_simulateDomains(256, 20, 0.3, 0.0)
    conc = np.sum(domains) / domains.size
    print(f"[domains map] concentration={conc:.3f} (target 0.3)")
    assert 0.25 < conc < 0.35
    return domains


def test_domains_sim():
    domains = sim_simulateDomains(256, 20, 0.3, 0.0)
    # tutorial-style: D1=5 (outside), D2=10 (inside), k21=100, k12=0.1
    tau = 0.01
    coords, stats = sim_simulateCoordinates(
        5, 1000, tau, [5.0, 10.0], 1.0, 100, 0.1, 0.0,
        25.6, True, "Domains", domains=domains)
    print(f"[domains sim] shape={coords.shape} actualF1={stats['actualF1']:.3f}")
    print("  x range:", coords[:, 4].min(), "-", coords[:, 4].max())
    # coordinates should be in um within the range
    assert coords[:, 4].max() <= 26.0
    assert coords[:, 5].max() <= 26.0


def test_reduce():
    domains = sim_simulateDomains(128, 10, 0.2, 0.0)
    tau = 0.01 / 10
    coords, stats = sim_simulateCoordinates(
        2, 1000, tau, [2.0, 0.0], 1.0, 1, 1, 0.0,
        12.8, False, "Domains", domains=domains)
    red = sim_reduceCoordinates(coords, 10)
    frames = np.unique(red[:, 3])
    print(f"[reduce] {len(np.unique(coords[:, 3]))} -> {len(frames)} frames")
    assert len(frames) == 100


if __name__ == "__main__":
    test_domains_map()
    test_domains_sim()
    test_reduce()
    print("DOMAINS TESTS PASSED")
