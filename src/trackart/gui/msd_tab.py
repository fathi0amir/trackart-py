"""MSD module tab: MSD calculation and diffusion model fitting."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from ..core.coordinates import calculateMSD, calculateSD
from ..core.msd import msd_calculateDcoeff
from . import plots, widgets as W

MODELS = ["Normal", "Confined/corraled", "Anomalous", "Active"]
MODEL_KEY = {"Normal": "normal", "Confined/corraled": "confined",
             "Anomalous": "anomalous", "Active": "active"}

# Equations shown in the "Fit to" box (mathtext), matching the MATLAB GUI.
MODEL_EQUATIONS = {
    "normal": [r"$\langle r^2 \rangle = a + b\,\Delta t$",
                r"$b = 4D$"],
    "anomalous": [r"$\langle r^2 \rangle = a + b\,\Delta t^{\alpha}$",
                   r"$\alpha < 1, \quad b = 4D$"],
    "confined": [r"$\langle r^2 \rangle = a + \langle r_c^2 \rangle "
                  r"[1 - A_1 \mathrm{exp}(-A_2 b\,\Delta t / "
                  r"\langle r_c^2 \rangle)]$"],
    "active": [r"$\langle r^2 \rangle = a + v^2 \Delta t^2 + b\,\Delta t$",
                r"$b = 4D$"],
}


class MSDTab(tk.Frame):
    def __init__(self, app, state):
        super().__init__(app, bg=W.BG)
        self.app = app
        self.state = state

        left = tk.Frame(self, bg=W.BG, width=330)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)

        right = tk.Frame(self, bg=W.BG)
        right.pack(side="left", fill="both", expand=True, padx=4, pady=6)

        self._build_params_panel(left)
        self._build_results_panel(left)

        self.plot_panel = W.PlotPanel(right, figsize=(6, 5))
        self.plot_panel.pack(fill="both", expand=True)

        bottom = tk.Frame(right, bg=W.BG)
        bottom.pack(fill="x", pady=4)
        self._build_plot_options(bottom)
        self._build_model_panel(bottom)
        self._build_fit_to_panel(bottom)

        self.plot_panel.draw()

    # ------------------------------------------------------------------
    def _build_params_panel(self, parent):
        box = W.box(parent, "MSD calculations")
        box.pack(fill="x", pady=3)

        src = W.box(box, "Data source")
        src.pack(fill="x", padx=6, pady=3)
        self.var_data = tk.StringVar(value="simulation")
        f = tk.Frame(src, bg=W.BG)
        f.pack(fill="x", padx=4, pady=1)
        W.radio(f, "simulation", self.var_data, "simulation").pack(side="left", padx=6)
        W.radio(f, "import", self.var_data, "import").pack(side="left", padx=6)

        f, self.e_show_lags = W.lab_entry(box, "Show (lags):", 20)
        f.pack(fill="x", padx=6, pady=1)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=1)
        W.label(row, "Fit (lags):", width=12, anchor="w").pack(side="left")
        self.e_fit_lags = W.Entry(row, default=10, width=7)
        self.e_fit_lags.pack(side="left")
        self.var_weighted = tk.BooleanVar(value=True)
        self._weighted_cb = W.check(row, "weighted fit", self.var_weighted)
        self._weighted_cb.pack(side="left", padx=(8, 0))

        f, self.e_sigma = W.lab_entry(box, "sigma [\u00b5m]:", 0.1)
        f.pack(fill="x", padx=6, pady=1)
        f, self.e_psf = W.lab_entry(box, "PSF size [\u00b5m]:", 0)
        f.pack(fill="x", padx=6, pady=1)
        f, self.e_te = W.lab_entry(box, "t exp [ms]:", 0)
        f.pack(fill="x", padx=6, pady=1)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=1)
        W.label(row, "intercept:", width=12, anchor="w").pack(side="left")
        self.e_intercept = W.Entry(row, default=0, width=7)
        self.e_intercept.pack(side="left")
        self.var_force_intercept = tk.BooleanVar(value=False)
        W.check(row, "force", self.var_force_intercept).pack(side="left", padx=(8, 0))

        self.button_fit = W.button(box, "Fit", self.on_fit, width=10, fg="blue")
        self.button_fit.pack(pady=4)

    def _build_results_panel(self, parent):
        box = W.box(parent, "Results")
        box.pack(fill="x", pady=3)
        columns = ("Parameter", "Value")
        tree = ttk.Treeview(box, columns=columns, show="headings", height=13)
        tree.heading("Parameter", text="Parameter")
        tree.heading("Value", text="Value")
        tree.column("Parameter", width=190, anchor="w")
        tree.column("Value", width=100, anchor="e")
        tree.pack(fill="x", padx=6, pady=4)
        self.results_table = tree

    def _build_plot_options(self, parent):
        box = W.box(parent, "Plots")
        box.pack(side="left", fill="y", padx=(0, 6))
        self.var_plot_msd = tk.BooleanVar(value=True)
        self.var_plot_slope = tk.BooleanVar(value=False)
        self.var_plot_sdv = tk.BooleanVar(value=False)
        self.var_plot_all = tk.BooleanVar(value=False)
        for text, var in (("MSD", self.var_plot_msd),
                          ("D slope", self.var_plot_slope),
                          ("MSD SDV", self.var_plot_sdv),
                          ("MSD all", self.var_plot_all)):
            W.check(box, text, var, self.on_plot_change).pack(anchor="w", padx=6)
        self.button_figure = W.button(box, "Figure", self.on_figure)
        self.button_figure.pack(fill="x", padx=6, pady=(4, 2))
        self.button_export = W.button(box, "Export MSD", self.on_export)
        self.button_export.pack(fill="x", padx=6, pady=2)

    def _build_model_panel(self, parent):
        box = W.box(parent, "Diffusion model:")
        box.pack(side="left", fill="y", padx=(0, 6))
        self.var_model = tk.StringVar(value="Normal")
        grid = tk.Frame(box, bg=W.BG)
        grid.pack(padx=4, pady=2)
        for i, name in enumerate(MODELS):
            W.radio(grid, name, self.var_model, name, self.on_model_change,
                    anchor="w").grid(row=i // 2, column=i % 2, sticky="w",
                                     padx=4, pady=2)

    def _build_fit_to_panel(self, parent):
        box = W.box(parent, "Fit to")
        box.pack(side="left", fill="y")
        inner = tk.Frame(box, bg="white", bd=1, relief="solid")
        inner.pack(padx=6, pady=6)
        self._eq_frame = inner
        self._update_equations()

    def _update_equations(self):
        """Re-render the equations for the currently selected model."""
        for child in self._eq_frame.winfo_children():
            child.destroy()
        model = MODEL_KEY[self.var_model.get()]
        for tex in MODEL_EQUATIONS[model]:
            W.math_label(self._eq_frame, tex, max_width=400).pack(padx=8, pady=4)

    # ------------------------------------------------------------------
    def on_model_change(self):
        model = MODEL_KEY[self.var_model.get()]
        normal = model == "normal"
        state = "normal" if normal else "disabled"
        for w in (self.e_sigma, self.e_psf, self.e_te, self._weighted_cb):
            w.configure(state=state)
        self._update_equations()

    def on_plot_change(self):
        p = self.state.MSD.plot
        p.msd = int(self.var_plot_msd.get())
        p.diffCoeffSlope = int(self.var_plot_slope.get())
        p.msdSDV = int(self.var_plot_sdv.get())
        p.msdAll = int(self.var_plot_all.get())
        self._plot()

    def on_fit(self):
        state = self.state
        var_data = self.var_data.get()

        param = {
            "lagsToCalculate": self.e_show_lags.get_int(),
            "lagsToFit": self.e_fit_lags.get_int(),
            "weightedFit": bool(self.var_weighted.get()),
            "s0": self.e_sigma.get_num(),
            "sig": self.e_sigma.get_num(),
            "te": self.e_te.get_num() / 1000,
            "intercept": [self.e_intercept.get_num(),
                          int(self.var_force_intercept.get())],
            "model": MODEL_KEY[self.var_model.get()],
        }
        state.MSD.plot.showFigure = 0

        if var_data == "simulation":
            if state.SIM.coordinates is None:
                self.app.message("Nothing to fit. Simulate / Import or select right data source.")
                return
            param["tau"] = self.app.sim_tau_ms() / 1000
            param["dataSource"] = "simulated"
            frames_max = int(np.max(state.SIM.coordinates[:, 3]))
        else:
            if state.import_data.dataFiltered is None:
                self.app.message("Nothing to fit. Simulate / Import or select right data source.")
                return
            param["tau"] = self.app.import_tau_ms() / 1000
            param["dataSource"] = "imported"
            frames_max = int(np.max(state.import_data.dataFiltered[:, 3]))

        if param["lagsToFit"] > param["lagsToCalculate"]:
            self.app.message("You cannot fit more lags than calculated...")
            return
        if param["lagsToFit"] < 2:
            self.app.message("You need minimum 2 points...")
            return
        if param["lagsToCalculate"] > frames_max:
            self.app.message("Cannot calculate so many lags")
            self.e_show_lags.delete(0, "end")
            self.e_show_lags.insert(0, str(frames_max))
            return

        # calculate MSDs (cached like the original)
        if var_data == "simulation":
            need = (state.SIM.MSDall is None or
                    state.SIM.MSDall.shape[1] - 1 < param["lagsToCalculate"])
            if need:
                SD, MSDall = calculateSD(state.SIM.coordinates,
                                         param["lagsToCalculate"], 1)
                MSD = calculateMSD(SD, param["tau"], 1)
                state.SIM.SD = SD
                state.SIM.MSDall = MSDall
                state.SIM.MSD = MSD
            MSD = state.SIM.MSD
        else:
            need = (state.import_data.MSDallFiltered is None or
                    state.import_data.MSDallFiltered.shape[1] - 1 < param["lagsToCalculate"])
            if need:
                SD, MSDall = calculateSD(state.import_data.dataFiltered,
                                         param["lagsToCalculate"], 1)
                MSD = calculateMSD(SD, param["tau"], 1)
                state.import_data.SDfiltered = SD
                state.import_data.MSDallFiltered = MSDall
                state.import_data.MSD = MSD
            MSD = state.import_data.MSD

        results, table = msd_calculateDcoeff(MSD, param)
        state.MSD.results = results
        state.MSD.param = param
        self._fill_results(table)
        self._plot()

    def _fill_results(self, table):
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        for name, value in table:
            if isinstance(value, float):
                value = f"{value:.6g}"
            elif isinstance(value, np.floating):
                value = f"{float(value):.6g}"
            self.results_table.insert("", "end", values=(name, str(value)))

    def _plot(self):
        p = self.state.MSD.plot
        # keep checkbox state in sync
        p.msd = int(self.var_plot_msd.get())
        p.diffCoeffSlope = int(self.var_plot_slope.get())
        p.msdSDV = int(self.var_plot_sdv.get())
        p.msdAll = int(self.var_plot_all.get())
        plots.msd_plots(self.state, self.plot_panel.axes[0])
        self.plot_panel.draw()

    def on_figure(self):
        import matplotlib.pyplot as plt
        self.state.MSD.plot.showFigure = 1
        fig, ax = plt.subplots(figsize=(6, 6))
        plots.msd_plots(self.state, ax, show_figure=True)
        self.state.MSD.plot.showFigure = 0
        plt.show()

    def on_export(self):
        results = self.state.MSD.results
        if results is None or "MSD" not in results:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self, title="Save as", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            msd = results["MSD"]
            header = "Time [s],MSD [um^2],STDev,K"
            np.savetxt(path, msd, delimiter=",", header=header, comments="")
