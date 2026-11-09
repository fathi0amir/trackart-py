"""CPD module tab: cumulative probability distribution analysis."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np

from ..core.coordinates import calculateMSD, calculateSD
from ..core.cpd import cpd_calculateCPD
from . import plots, widgets as W


class CPDTab(tk.Frame):
    def __init__(self, app, state):
        super().__init__(app, bg=W.BG)
        self.app = app
        self.state = state

        left = tk.Frame(self, bg=W.BG, width=360)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)

        right = tk.Frame(self, bg=W.BG)
        right.pack(side="left", fill="both", expand=True, padx=4, pady=6)

        self._build_data_source(left)
        self._build_calc_panel(left)
        self._build_plot_panel(left)

        # right side: CPD plot + residuals + preview
        top = W.box(right, "Cumulative Probability Distribution and fits")
        top.pack(fill="both", expand=True, pady=(0, 4))
        self.plot_panel = W.PlotPanel(top, nrows=2, figsize=(6, 6),
                                      sharex=False)
        self.plot_panel.axes[0].set_position([0.12, 0.40, 0.83, 0.52])
        self.plot_panel.axes[1].set_position([0.12, 0.08, 0.83, 0.22])

        bottom = W.box(right, "MSD and Fraction preview")
        bottom.pack(fill="both", expand=True)
        self.preview_panel = W.PlotPanel(bottom, nrows=1, ncols=4,
                                         figsize=(7, 1.8))
        for k, ax in enumerate(self.preview_panel.axes):
            ax.set_position([0.03 + 0.24 * k, 0.25, 0.2, 0.6])

        self.plot_panel.draw()
        self.preview_panel.draw()

    # ------------------------------------------------------------------
    def _build_data_source(self, parent):
        box = W.box(parent, "Data source")
        box.pack(fill="x", pady=3)
        self.var_data = tk.StringVar(value="import")
        f = tk.Frame(box, bg=W.BG)
        f.pack(fill="x", padx=6, pady=2)
        W.radio(f, "simulation", self.var_data, "simulation").pack(side="left", padx=8)
        W.radio(f, "import", self.var_data, "import").pack(side="left", padx=8)

    def _build_calc_panel(self, parent):
        box = W.box(parent, "Calculate and fit CDF")
        box.pack(fill="x", pady=3)

        f, self.e_lags = W.lab_entry(box, "Lags to calculate:", 5)
        f.pack(fill="x", padx=6, pady=1)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=1)
        W.label(row, "Bin size:", width=14, anchor="w").pack(side="left")
        self.e_bin = W.Entry(row, default=1000, width=7)
        self.e_bin.pack(side="left")
        self.var_fit_to_bin = tk.BooleanVar(value=True)
        W.check(row, "fix", self.var_fit_to_bin).pack(side="left", padx=(8, 0))
        f, self.e_minP = W.lab_entry(box, "Min P:", 0)
        f.pack(fill="x", padx=6, pady=1)
        f, self.e_minR = W.lab_entry(box, "Min r^2 [\u00b5m\u00b2]:", 0.000)
        f.pack(fill="x", padx=6, pady=1)

        self.button_calc = W.button(box, "CPD calculate & fit", self.on_calculate,
                                    fg="blue")
        self.button_calc.pack(fill="x", padx=6, pady=4)

        W.label(box, "Initial fit (or fix) parameters", font=("TkDefaultFont", 8, "bold"),
                anchor="w").pack(fill="x", padx=6, pady=(2, 0))

        grid = tk.Frame(box, bg=W.BG)
        grid.pack(fill="x", padx=4, pady=2)
        W.label(grid, "").grid(row=0, column=0, sticky="w")
        W.label(grid, "D [\u00b5m\u00b2/s]").grid(row=0, column=1, padx=2)
        W.label(grid, "fix").grid(row=0, column=2, padx=2)
        W.label(grid, "F").grid(row=0, column=3, padx=2)
        W.label(grid, "fix").grid(row=0, column=4, padx=2)
        W.label(grid, "intercept").grid(row=0, column=5, padx=2)

        self.guess_widgets = []
        defaults = [("fraction 1", 2, 0.33, 0), ("fraction 2", 0.2, 0.33, 0),
                    ("fraction 3", 0.02, 0.34, 0)]
        for r, (name, d1, f1, ic) in enumerate(defaults, start=1):
            W.label(grid, name, width=10, anchor="w").grid(row=r, column=0, sticky="w")
            e_d = W.Entry(grid, default=d1, width=7)
            e_d.grid(row=r, column=1, padx=2)
            v_d = tk.BooleanVar(value=False)
            W.check(grid, "", v_d).grid(row=r, column=2, padx=2)
            e_f = W.Entry(grid, default=f1, width=7)
            e_f.grid(row=r, column=3, padx=2)
            v_f = tk.BooleanVar(value=False)
            W.check(grid, "", v_f).grid(row=r, column=4, padx=2)
            e_i = W.Entry(grid, default=ic, width=7)
            e_i.grid(row=r, column=5, padx=2)
            self.guess_widgets.append((e_d, v_d, e_f, v_f, e_i))

    def _build_plot_panel(self, parent):
        box = W.box(parent, "Plot")
        box.pack(fill="x", pady=3)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=1)
        W.label(row, "CDF lag:", width=10, anchor="w").pack(side="left")
        self.e_show_lag = W.Entry(row, default=5, width=7)
        self.e_show_lag.pack(side="left")

        W.label(box, "Plot for:", anchor="w").pack(fill="x", padx=6)
        self.var_model = tk.StringVar(value="1p")
        f = tk.Frame(box, bg=W.BG)
        f.pack(fill="x", padx=6)
        for name in ("1p", "2p", "3p"):
            W.radio(f, name, self.var_model, name, self.on_plot_change).pack(side="left", padx=8)

        btns = tk.Frame(box, bg=W.BG)
        btns.pack(fill="x", padx=6, pady=4)
        self.button_export_cpd = W.button(btns, "Export CPD", self.on_export_cpd)
        self.button_export_cpd.pack(side="left", expand=True, fill="x", padx=2)
        self.button_export_fit = W.button(btns, "Export fit", self.on_export_fit)
        self.button_export_fit.pack(side="left", expand=True, fill="x", padx=2)
        self.button_fig = W.button(btns, "Fig", self.on_fig)
        self.button_fig.pack(side="left", expand=True, fill="x", padx=2)

    # ------------------------------------------------------------------
    def _build_dguess(self, tau):
        Dguess = np.zeros((6, 3))
        for i, (e_d, v_d, e_f, v_f, e_i) in enumerate(self.guess_widgets):
            Dguess[i, 0] = e_d.get_num()
            Dguess[i, 1] = int(v_d.get())
            Dguess[3 + i, 0] = e_f.get_num()
            Dguess[3 + i, 1] = int(v_f.get())
            Dguess[i, 2] = e_i.get_num()
        Dguess[0:3, 0] = Dguess[0:3, 0] * 4 * tau
        return Dguess

    def on_calculate(self):
        state = self.state
        param = {}
        if self.var_data.get() == "simulation":
            if state.SIM.coordinates is None:
                self.app.message("Nothing to fit. Simulate / Import or select right data source.")
                return
            param["tau"] = self.app.sim_tau_ms() / 1000
            param["dataSource"] = "simulated"
        else:
            if state.import_data.dataFiltered is None:
                self.app.message("Nothing to fit. Simulate / Import or select right data source.")
                return
            param["tau"] = self.app.import_tau_ms() / 1000
            param["dataSource"] = "imported"

        param["lagsToCalculate"] = self.e_lags.get_int()
        param["bin"] = self.e_bin.get_int()
        param["fitToBin"] = bool(self.var_fit_to_bin.get())
        param["minP"] = self.e_minP.get_num()
        param["minR"] = self.e_minR.get_num()
        param["square_x"] = self.app.legacy_numerics.get()
        tau = param["tau"]
        param["Dguess"] = self._build_dguess(tau)
        param["intercept"] = [self.guess_widgets[i][4].get_num() for i in range(3)]

        try:
            self.app.busy(True)
            self.app.update()
            lags = param["lagsToCalculate"]
            if param["dataSource"] == "simulated":
                need = (state.SIM.MSDall is None or
                        state.SIM.MSDall.shape[1] - 1 < lags)
                if need:
                    SD, MSDall = calculateSD(state.SIM.coordinates, lags, 1)
                    state.SIM.SD = SD
                    state.SIM.MSDall = MSDall
                SD = state.SIM.SD
                state.SIM.MSD = calculateMSD(SD, tau)
            else:
                need = (state.import_data.SDfiltered is None or
                        state.import_data.SDfiltered.shape[1] - 1 < lags)
                if need:
                    SD, MSDall = calculateSD(state.import_data.dataFiltered, lags, 1)
                    state.import_data.SDfiltered = SD
                    state.import_data.MSDallFiltered = MSDall
                SD = state.import_data.SDfiltered
                state.import_data.MSD = calculateMSD(SD, tau)

            N = state.SIM.MSD if param["dataSource"] == "simulated" else state.import_data.MSD
            N = N[0:lags, 3]
            param["N"] = N

            cpd_bin, results = cpd_calculateCPD(
                SD, lags, param["Dguess"], param["bin"], param["fitToBin"],
                param["minP"], param["minR"], tau, N,
                square_x=param["square_x"])
            self.app.invalidate_fit()
            state.CPD.cpd_bin = cpd_bin
            state.CPD.results = results
            state.CPD.param = param
            param["showLag"] = self.e_show_lag.get_int()
            param["plotModel"] = int(self.var_model.get().replace("p", ""))
            self._plot()
            self._log_ftest(results)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.app.message(f"CPD calculation failed:\n{exc}", "error")
        finally:
            self.app.busy(False)

    def _log_ftest(self, results):
        ftest = results.get("ftest")
        if ftest is None:
            return
        print("F-Test (residuals). Hypothesis: variances are the same.")
        print("0 - hypothesis cannot be rejected at the 5% significance level")
        print("1 - hypothesis can be rejected at the 5% significance level")
        print("1st row: compare 1p and 2p, 2nd row: 2p and 3p, 3rd row: 1p and 3p")
        print(np.round(ftest, 2))

    # ------------------------------------------------------------------
    def on_plot_change(self):
        if self.state.CPD.param is None:
            return
        self.state.CPD.param["showLag"] = self.e_show_lag.get_int()
        self.state.CPD.param["plotModel"] = int(self.var_model.get().replace("p", ""))
        self._plot()

    def _plot(self):
        if self.state.CPD.results is None:
            return
        plots.cpd_plots(self.state, self.plot_panel.axes[0],
                        self.plot_panel.axes[1], self.preview_panel.axes)
        self.plot_panel.draw()
        self.preview_panel.draw()

    def on_fig(self):
        if self.state.CPD.results is None:
            return
        import matplotlib.pyplot as plt
        self.state.CPD.showFigure = True
        fig = plt.figure(figsize=(6, 7))
        ax_top = fig.add_axes([0.12, 0.42, 0.8, 0.5])
        ax_res = fig.add_axes([0.12, 0.1, 0.8, 0.22])
        preview = [fig.add_axes([0.05 + 0.23 * k, 0.02, 0.2, 0.05]) for k in range(4)]
        plots.cpd_plots(self.state, ax_top, ax_res, preview, show_figure=True)
        self.state.CPD.showFigure = False
        plt.show()

    def on_export_cpd(self):
        if self.state.CPD.cpd_bin is None:
            return
        i = self.e_show_lag.get_int() - 1
        if 0 <= i < self.state.CPD.cpd_bin.shape[2]:
            self._save_array(self.state.CPD.cpd_bin[:, :, i], "Export CPD")

    def on_export_fit(self):
        if self.state.CPD.results is None:
            return
        i = self.e_show_lag.get_int() - 1
        fit = self.state.CPD.results.get("cpd_binFit")
        if fit is not None and 0 <= i < fit.shape[2]:
            self._save_array(fit[:, :, i], "Export fit")

    def _save_array(self, arr, title):
        path = filedialog.asksaveasfilename(
            parent=self, title=title, defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            np.savetxt(path, arr, delimiter=",")
