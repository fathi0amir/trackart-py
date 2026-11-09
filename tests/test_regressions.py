"""Deterministic numerical regressions for MATLAB parity and corrected modes."""

import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.core.coordinates import calculateMSD, calculateSD
from trackart.core.cpd import (cpd_3p_model, cpd_calculateCPD, cpd_fit_3p,
                               _vartest2)
from trackart.core.fit import fit_fitMSD
from trackart.core.fitting import fmincon_3p
from trackart.core.import_data import import_trackImport
from trackart.core.msd import msd_activeDiffusion, msd_confinedDiffusion, msd_MSDlinearFit
from trackart.core.util import ecdf
from trackart.gui.plots import _plot_pair_correlation


class NumericalRegressions(unittest.TestCase):
    def test_fraction_sample_std_and_single_lag(self):
        msd = np.column_stack((np.arange(1, 6), np.arange(1, 6),
                               np.zeros(5), np.full(5, 100)))
        data = {"MSD1": msd, "MSD2": msd,
                "a": np.array([np.ones(5), np.ones(5), np.arange(1, 6) / 10])}
        params = {"model": 2, "p1lags": [1, 2], "p2lags": [1, 2],
                  "p1model": 1, "p2model": 1, "p1intercept": [0, 0],
                  "p2intercept": [0, 0], "fractionLags": [1, 2, 3, 4, 5]}
        self.assertAlmostEqual(fit_fitMSD(data, params)["fraction"][0, 1],
                               0.15811388300841897)
        params["fractionLags"] = [1]
        self.assertEqual(fit_fitMSD(data, params)["fraction"][0, 1], 0)

    def test_ecdf_matches_original_for_ties(self):
        x, y = ecdf([0, 0, 1, 2])
        assert_allclose(x, [0, 0, 1, 2])
        assert_allclose(y, [0, 0.5, 0.75, 1])

    def test_three_population_constraints_and_fixed_values(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0.001, 2, 200)
        truth = np.array([1.2, 0.25, 0.03, 0.2, 0.3, 0.5])
        y = cpd_3p_model(truth, x) + rng.normal(0, 0.002, len(x))
        for fixed in ([], [3], [3, 4], [3, 4, 5], [0, 2], list(range(6))):
            with self.subTest(fixed=fixed):
                guess = np.zeros((6, 3))
                guess[:, 0] = truth
                guess[fixed, 1] = 1
                result = cpd_fit_3p(x, y, 1, guess)
                self.assertAlmostEqual(result[3:].sum(), 1, places=7)
                self.assertTrue(np.all(np.diff(result[:3]) <= 1e-7))
                assert_allclose(result[fixed], truth[fixed], atol=1e-12)
        guess[3:, 0] = [0.8, 0.8, 0.2]
        with self.assertRaises(ValueError):
            cpd_fit_3p(x, y, 1, guess)

    def test_corrected_cpd_binning_and_legacy_mode(self):
        rng = np.random.default_rng(123)
        samples = rng.exponential(0.02, 1500)
        sd = np.column_stack((np.ones(len(samples)), samples))
        guess = np.zeros((6, 3))
        guess[:3, 0] = [0.02, 0.006, 0.001]
        guess[3:, 0] = [0.33, 0.33, 0.34]
        legacy, _ = cpd_calculateCPD(sd, 1, guess, 100, True, 0, 0, 0.01,
                                      [len(samples)], square_x=True)
        corrected, result = cpd_calculateCPD(sd, 1, guess, 100, True, 0, 0, 0.01,
                                              [len(samples)], square_x=False)
        self.assertAlmostEqual(legacy[-1, 0, 0], samples.max() ** 2)
        self.assertAlmostEqual(corrected[-1, 0, 0], samples.max())
        self.assertAlmostEqual(result["model_1p"]["a"][0, 0], 0.02, delta=0.003)

    def test_solver_failure_is_reported(self):
        failed = SimpleNamespace(success=False, message="test failure")
        with patch("trackart.core.fitting.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                fmincon_3p(lambda x: x[0] ** 2, [1], [0], [2],
                           None, None, None, None)

    def test_active_and_confined_corrected_and_legacy(self):
        time = np.arange(1, 21) * 0.1
        msd = np.column_stack((time, 2 * time**2 + 0.8 * time + 0.3,
                               np.zeros(20), np.full(20, 1000)))
        corrected = msd_activeDiffusion(msd, np.arange(1, 21), [0.3, 1], legacy=False)
        legacy = msd_activeDiffusion(msd, np.arange(1, 21), [0.3, 1], legacy=True)
        assert_allclose([corrected["intercept"], corrected["D"]], [0.3, 0.2], atol=1e-7)
        self.assertNotAlmostEqual(legacy["intercept"], 0.3)
        msd[:, 1] = 2 * (1 - 0.65 * np.exp(-4 * 0.35 * 0.8 * time / 2)) + 0.15
        corrected = msd_confinedDiffusion(msd, np.arange(1, 21), [0.15, 1], legacy=False)
        legacy = msd_confinedDiffusion(msd, np.arange(1, 21), [0.15, 1], legacy=True)
        self.assertEqual(corrected["A2"], corrected["fit"][2])
        self.assertEqual(legacy["A2"], legacy["A1"])

    def test_variance_test_unequal_sample_counts(self):
        rng = np.random.default_rng(42)
        from scipy.stats import f
        for _ in range(100):
            x, y = rng.normal(size=7), 1.6 * rng.normal(size=29)
            ratio = np.var(x, ddof=1) / np.var(y, ddof=1)
            p = 2 * min(f.cdf(ratio, 6, 28), f.sf(ratio, 6, 28))
            self.assertEqual(_vartest2(x, y), int(p < 0.05))

    def test_pair_correlation_uses_both_axes(self):
        from matplotlib.figure import Figure
        domains = (np.random.default_rng(2).random((24, 24)) < 0.4).astype(float)
        ax = Figure().subplots()
        _plot_pair_correlation(domains, 24, ax)
        fft = np.fft.fft2(domains)
        gr = 1 / (np.fft.ifft2(fft * fft.conj()).real / (domains.mean()**2 * domains.size))
        n = 12
        a = np.mean([gr[:n, 0], gr[0, :n], gr[-1, -n:][::-1],
                     gr[-n:, -1][::-1]], axis=0)
        assert_allclose(ax.lines[0].get_ydata(), a[0] + 1 - a)

    @unittest.skipUnless(os.environ.get("TRACKART_TEST_DATA"), "original data not provided")
    def test_original_real_data_against_octave(self):
        root = os.environ["TRACKART_TEST_DATA"]
        names = [f"DOPC_SM_ATTO647N_25C_{i:03}.xls" for i in (1, 2)]
        coords = import_trackImport([os.path.join(root, name) for name in names])
        self.assertEqual(coords.shape, (21568, 13))
        ids, counts = np.unique(coords[:, 0], return_counts=True)
        data = coords[np.isin(coords[:, 0], ids[counts >= 30][:10])].copy()
        data[:, 4:6] *= 0.16
        sd, _ = calculateSD(data, 5)
        msd = calculateMSD(sd, 0.01)
        self.assertAlmostEqual(msd[0, 1], 0.0458118435812, places=11)
        self.assertAlmostEqual(msd_MSDlinearFit(msd, 5, [0, 0])["D"],
                               1.2859382231978607, places=10)


if __name__ == "__main__":
    unittest.main()
