"""GUI integration regressions; run under a display or xvfb-run."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.gui.app import TrackArtApp


def set_entry(entry, value):
    entry.delete(0, "end")
    entry.insert(0, str(value))


class GUIRegressions(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.app = TrackArtApp()
        self.messages = []
        self.app.message = lambda *args: self.messages.append(args)
        self.app.update()

    def tearDown(self):
        self.app.update()
        self.app.destroy()

    def test_resimulation_and_cpd_before_msd(self):
        sim = self.app.tabs["SIM"]
        set_entry(sim.e_particles, 20)
        set_entry(sim.e_N, 200)
        set_entry(sim.e_D_1, 2)
        sim.on_simulate()
        self.app._show_module("MSD")
        msd = self.app.tabs["MSD"]
        set_entry(msd.e_show_lags, 5)
        set_entry(msd.e_fit_lags, 3)
        msd.on_fit()
        self.assertAlmostEqual(self.app.state.MSD.results["D"], 2, delta=0.4)
        set_entry(sim.e_D_1, 0.2)
        sim.on_simulate()
        self.assertIsNone(self.app.state.MSD.results)
        msd.on_fit()
        self.assertAlmostEqual(self.app.state.MSD.results["D"], 0.2, delta=0.05)

        self.app.state.SIM.MSDall = None
        self.app._show_module("CPD")
        cpd = self.app.tabs["CPD"]
        cpd.var_data.set("simulation")
        set_entry(cpd.e_lags, 5)
        set_entry(cpd.e_bin, 100)
        cpd.on_calculate()
        self.assertIsNotNone(self.app.state.CPD.results, self.messages)
        msd.on_fit()
        self.assertAlmostEqual(self.app.state.SIM.MSD[0, 0], 0.01)
        self.assertAlmostEqual(self.app.state.MSD.results["D"], 0.2, delta=0.05)

        self.app._show_module("FIT")
        fit = self.app.tabs["FIT"]
        fit.var_model.set("1p")
        fit.on_fit()
        self.assertIsNotNone(self.app.state.FIT.results, self.messages)
        set_entry(fit.e_repeats, 3)
        set_entry(fit.e_perm_pct, 50)
        self.app.state.CPD.param["Dguess"][:, 1] = 1
        fit.on_permute()
        self.assertEqual(self.app.state.FIT.results["permError"].shape, (1,))
        self.assertTrue(np.all(np.isfinite(self.app.state.FIT.results["permError"])))
        old_msd = self.app.state.MSD.results
        cpd_bin, cpd_results = self.app.state.CPD.cpd_bin, self.app.state.CPD.results
        with patch("trackart.gui.cpd_tab.cpd_calculateCPD",
                   return_value=(cpd_bin, cpd_results)):
            cpd.on_calculate()
        self.assertIsNone(self.app.state.FIT.results)
        self.assertFalse(fit.results_table.get_children())
        self.assertTrue(all(not ax.lines for ax in fit.plot_panel.axes))
        self.assertIs(self.app.state.MSD.results, old_msd)
        self.assertIs(self.app.state.CPD.results, cpd_results)

        set_entry(sim.e_tau, 20)
        msd.on_fit()
        self.assertAlmostEqual(self.app.state.SIM.MSD[0, 0], 0.01)
        self.assertFalse(self.messages)

    def test_switch_rates_and_scalar_lags(self):
        sim = self.app.tabs["SIM"]
        sim.var_method.set("Switch")
        sim._on_method_change()
        set_entry(sim.e_particles, 2)
        set_entry(sim.e_N, 30)
        set_entry(sim.e_D_2, 0.2)
        set_entry(sim.e_microsteps, 10)
        set_entry(sim.e_k_1, 1)
        set_entry(sim.e_k_2, 1)
        from trackart.gui import sim_tab
        with patch.object(sim_tab, "sim_simulateCoordinates",
                          wraps=sim_tab.sim_simulateCoordinates) as simulate:
            sim.on_simulate()
            self.assertEqual(simulate.call_args.args[5], 10)
            self.app.legacy_numerics.set(False)
            self.app.invalidate_analysis()
            sim.on_simulate()
            self.assertEqual(simulate.call_args.args[5], 1)
            self.assertAlmostEqual(simulate.call_args.args[2], 0.001)
        self.app._show_module("FIT")
        fit = self.app.tabs["FIT"]
        set_entry(fit.p_rows["p1"][0], 5)
        set_entry(fit.e_fraction_lags, 5)
        params = fit._read_param()
        assert_allclose(params["p1lags"], [1, 2, 3, 4, 5])
        assert_allclose(params["fractionLags"], [1, 2, 3, 4, 5])
        self.assertFalse(params["legacy_models"])
        set_entry(fit.p_rows["p2"][0], "invalid")
        set_entry(fit.e_fraction_lags, "invalid")
        fit.var_use_all.set(True)
        params = fit._read_param()
        assert_allclose(params["p2lags"], [1, 2, 3, 4, 5])
        assert_allclose(params["fractionLags"], [1, 2, 3, 4, 5])
        self.assertFalse(self.messages)

    def test_permutation_all_models_and_subset_counts(self):
        self.app._show_module("FIT")
        fit = self.app.tabs["FIT"]
        set_entry(fit.e_repeats, 4)
        set_entry(fit.e_perm_pct, 50)
        sd = np.column_stack((np.repeat(np.arange(1, 7), 6),
                              np.ones((36, 3))))
        self.app.state.SIM.SD = sd
        cpd_param = {"dataSource": "simulated", "lagsToCalculate": 3,
                     "Dguess": np.zeros((6, 3)), "bin": 10, "fitToBin": True,
                     "minP": 0, "minR": 0, "tau": 0.01, "N": [30, 24, 18],
                     "square_x": False}
        self.app.state.CPD.param = cpd_param
        for model in (1, 2, 3):
            with self.subTest(model=model):
                self.app.state.FIT.param = {"model": model}
                self.app.state.FIT.results = {
                    **{f"p{i}": {"D": float(i)} for i in range(1, model + 1)},
                    "resultsTable": [("D permut:", "", "", "")]}
                counter = iter(range(4))

                def fake_fit(data, params):
                    i = next(counter)
                    return {**{f"p{k}": {"D": k + i / 10}
                               for k in range(1, model + 1)},
                            "fraction": np.array([[0.3, 0], [0.4, 0], [0.3, 0]])}

                fake_cpd = {"model_1p": {}, "model_2p": {}, "model_3p": {}}
                with patch("trackart.gui.fit_tab.cpd_calculateCPD", return_value=(None, fake_cpd)) as call:
                    with patch("trackart.gui.fit_tab.fit_fitMSD", side_effect=fake_fit):
                        fit.on_permute()
                self.assertFalse(self.messages)
                self.assertEqual(len(self.app.state.FIT.results["permError"]), model)
                self.assertTrue(np.all(np.isfinite(self.app.state.FIT.results["permError"])))
                assert_allclose(call.call_args.args[8], [15, 12, 9])
                self.assertFalse(call.call_args.kwargs["square_x"])


if __name__ == "__main__":
    unittest.main()
