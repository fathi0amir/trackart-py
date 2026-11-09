"""FIT module tab: fit per-population MSDs and fractions from the CPD
analysis, with permutation-based error estimation."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from ..core.cpd import cpd_calculateCPD
from ..core.fit import fit_fitMSD
from ..core.util import nanmean, nanstd, outliers, parse_lags
from . import plots, widgets as W


class FITTab(tk.Frame):
    def __init__(self, app, state):
        super().__init__(app, bg=W.BG)
        self.app = app
        self.state = state

        left = tk.Frame(self, bg=W.BG, width=430)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)

        right = tk.Frame(self, bg=W.BG)
        right.pack(side="left", fill="both", expand=True, padx=4, pady=6)

        self._build_params_panel(left)
        self._build_results_panel(left)

        right_box = W.box(right, "MSD & fraction")
        right_box.pack(fill="both", expand=True)
        self.plot_panel = W.PlotPanel(right_box, nrows=2, ncols=2,
                                      figsize=(7, 5))
        for i, ax in enumerate(self.plot_panel.axes):
            pos = [(0.1, 0.55, 0.35, 0.38), (0.6, 0.55, 0.35, 0.38),
                   (0.1, 0.1, 0.35, 0.38), (0.6, 0.1, 0.35, 0.38)][i]
            ax.set_position(pos)
        self.plot_panel.draw()

    # ------------------------------------------------------------------
    def _build_params_panel(self, parent):
        box = W.box(parent, "Calculate and fit CDF")
        box.pack(fill="x", pady=3)

        head = tk.Frame(box, bg=W.BG)
        head.pack(fill="x", padx=6)
        W.label(head, "", width=8, anchor="w").pack(side="left")
        W.label(head, "Lags", width=8, anchor="w").pack(side="left")
        W.label(head, "Model", width=8, anchor="w").pack(side="left")
        W.label(head, "Intercept", width=10, anchor="w").pack(side="left")

        self.p_rows = {}
        for r, key in enumerate(("p1", "p2", "p3")):
            row = tk.Frame(box, bg=W.BG)
            row.pack(fill="x", padx=6, pady=1)
            W.label(row, key, width=8, anchor="w").pack(side="left")
            e_lags = W.Entry(row, default="1:5", width=8)
            e_lags.pack(side="left")
            e_model = W.Entry(row, default=1, width=6)
            e_model.pack(side="left", padx=(4, 0))
            e_intercept = W.Entry(row, default=0, width=7)
            e_intercept.pack(side="left", padx=(4, 0))
            v_force = tk.BooleanVar(value=False)
            W.check(row, "force", v_force).pack(side="left", padx=(6, 0))
            if key == "p1":
                self.var_use_all = tk.BooleanVar(value=False)
                W.check(row, "all", self.var_use_all, self.on_use_all).pack(side="left", padx=(6, 0))
            if key == "p3":
                self.button_fit = W.button(row, "Fit", self.on_fit, width=6, fg="blue")
                self.button_fit.pack(side="left", padx=(8, 0))
            self.p_rows[key] = (e_lags, e_model, e_intercept, v_force)

        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=1)
        W.label(row, "F", width=8, anchor="w").pack(side="left")
        self.e_fraction_lags = W.Entry(row, default="1:5", width=8)
        self.e_fraction_lags.pack(side="left")

        btns = tk.Frame(box, bg=W.BG)
        btns.pack(fill="x", padx=6, pady=3)
        self.button_copy = W.button(btns, "Copy", self.on_copy, width=7)
        self.button_copy.pack(side="left", expand=True, fill="x", padx=2)
        self.button_fig = W.button(btns, "Fig", self.on_fig, width=7)
        self.button_fig.pack(side="left", expand=True, fill="x", padx=2)
        self.button_table = W.button(btns, "Table", self.on_table, width=7)
        self.button_table.pack(side="left", expand=True, fill="x", padx=2)

        # model selection
        mod = W.box(box, "Model:")
        mod.pack(fill="x", padx=6, pady=3)
        self.var_model = tk.StringVar(value="3p")
        f = tk.Frame(mod, bg=W.BG)
        f.pack(fill="x", padx=4, pady=1)
        for name in ("1p", "2p", "3p"):
            W.radio(f, name, self.var_model, name).pack(side="left", padx=8)

        row = tk.Frame(mod, bg=W.BG)
        row.pack(fill="x", padx=4, pady=2)
        W.label(row, "Perm", width=6, anchor="w").pack(side="left")
        self.e_perm_pct = W.Entry(row, default=50, width=6)
        self.e_perm_pct.pack(side="left")
        W.label(row, "%", width=3).pack(side="left")
        W.label(row, "repeats:", width=9, anchor="e").pack(side="left")
        self.e_repeats = W.Entry(row, default=10, width=6)
        self.e_repeats.pack(side="left")
        self.button_permute = W.button(row, "Permute", self.on_permute, width=8)
        self.button_permute.pack(side="left", padx=(8, 0))

    def _build_results_panel(self, parent):
        box = W.box(parent, "Fit:")
        box.pack(fill="x", pady=3)
        columns = ("", "P1", "P2", "P3")
        tree = ttk.Treeview(box, columns=columns, show="headings", height=12)
        tree.heading("", text="")
        tree.heading("P1", text="P1")
        tree.heading("P2", text="P2")
        tree.heading("P3", text="P3")
        tree.column("", width=110, anchor="w")
        tree.column("P1", width=90, anchor="e")
        tree.column("P2", width=90, anchor="e")
        tree.column("P3", width=90, anchor="e")
        tree.pack(fill="x", padx=6, pady=4)
        self.results_table = tree

    # ------------------------------------------------------------------
    def on_use_all(self):
        st = "disabled" if self.var_use_all.get() else "normal"
        for key in ("p2", "p3"):
            e_lags, e_model, e_intercept, v_force = self.p_rows[key]
            for w in (e_lags, e_model, e_intercept):
                w.configure(state=st)
        self.e_fraction_lags.configure(state=st)

    def _read_param(self):
        param = {}
        for key in ("p1", "p2", "p3"):
            e_lags, e_model, e_intercept, v_force = self.p_rows[key]
            lags = parse_lags(e_lags.get()) or [1]
            param[f"{key}lags"] = np.asarray(lags, dtype=int)
            param[f"{key}model"] = int(float(e_model.get() or 1))
            param[f"{key}intercept"] = [float(e_intercept.get() or 0),
                                        int(v_force.get())]
        param["fractionLags"] = np.asarray(parse_lags(self.e_fraction_lags.get()) or [1],
                                           dtype=int)
        use_all = self.var_use_all.get()
        if use_all:
            param["p2lags"] = param["p1lags"]
            param["p3lags"] = param["p1lags"]
            param["p2model"] = param["p1model"]
            param["p3model"] = param["p1model"]
            param["p2intercept"] = param["p1intercept"]
            param["p3intercept"] = param["p1intercept"]
            param["fractionLags"] = param["p1lags"]
        param["fitSource"] = "CPD"
        param["model"] = int(self.var_model.get().replace("p", ""))
        return param

    def on_fit(self):
        state = self.state
        if state.CPD.results is None:
            self.app.message("Run the CPD calculation first.")
            return
        param = self._read_param()
        model = param["model"]
        if model == 1:
            data = state.CPD.results["model_1p"]
        elif model == 2:
            data = state.CPD.results["model_2p"]
        else:
            data = state.CPD.results["model_3p"]

        try:
            results = fit_fitMSD(data, param)
        except Exception as exc:
            self.app.message(f"Fit failed:\n{exc}", "error")
            return

        table = self._build_results_table(results, param)
        results["resultsTable"] = table

        state.FIT.results = results
        state.FIT.param = param
        state.FIT.data = data

        self._fill_results(table)
        self._plot()
        for b in (self.button_fig, self.button_copy,
                  self.button_table, self.button_permute):
            b.configure(state="normal")

    def _build_results_table(self, results, param):
        p1 = results["p1"]
        rows = {
            "D [\u00b5m\u00b2/s]:": {},
            "D err:": {},
            "D permut:": {},
            "F:": {},
            "F std:": {},
            "a:": {},
            "b:": {},
            "c:": {},
            "A1:": {},
            "A2:": {},
            "rc:": {},
        }

        def fill(row_name, col, value):
            rows[row_name][col] = "" if value is None else f"{value:.6g}"

        fill("D [\u00b5m\u00b2/s]:", 1, p1["D"])
        if param["p1model"] == 1:
            fill("D err:", 1, p1.get("qianErr"))
            fill("a:", 1, p1.get("a"))
            fill("b:", 1, p1.get("b"))
        elif param["p1model"] == 2:
            fill("c:", 1, p1.get("c"))
            fill("A1:", 1, p1.get("A1"))
            fill("A2:", 1, p1.get("A2"))
            fill("rc:", 1, p1.get("rc"))
        elif param["p1model"] == 3:
            fill("a:", 1, p1.get("a"))
            fill("b:", 1, p1.get("b"))
            fill("c:", 1, p1.get("c"))

        if param["model"] > 1:
            p2 = results["p2"]
            fill("D [\u00b5m\u00b2/s]:", 2, p2["D"])
            fill("F:", 1, results["fraction"][0, 0])
            fill("F:", 2, results["fraction"][1, 0])
            fill("F std:", 1, results["fraction"][0, 1])
            fill("F std:", 2, results["fraction"][1, 1])
            if param["p2model"] == 1:
                fill("D err:", 2, p2.get("qianErr"))
                fill("a:", 2, p2.get("a"))
                fill("b:", 2, p2.get("b"))
            elif param["p2model"] == 2:
                fill("c:", 2, p2.get("c"))
                fill("A1:", 2, p2.get("A1"))
                fill("A2:", 2, p2.get("A2"))
                fill("rc:", 2, p2.get("rc"))
            elif param["p2model"] == 3:
                fill("a:", 2, p2.get("a"))
                fill("b:", 2, p2.get("b"))
                fill("c:", 2, p2.get("c"))

        if param["model"] == 3:
            p3 = results["p3"]
            fill("D [\u00b5m\u00b2/s]:", 3, p3["D"])
            fill("F:", 3, results["fraction"][2, 0])
            fill("F std:", 3, results["fraction"][2, 1])
            if param["p3model"] == 1:
                fill("D err:", 3, p3.get("qianErr"))
                fill("a:", 3, p3.get("a"))
                fill("b:", 3, p3.get("b"))
            elif param["p3model"] == 2:
                fill("c:", 3, p3.get("c"))
                fill("A1:", 3, p3.get("A1"))
                fill("A2:", 3, p3.get("A2"))
                fill("rc:", 3, p3.get("rc"))
            elif param["p3model"] == 3:
                fill("a:", 3, p3.get("a"))
                fill("b:", 3, p3.get("b"))
                fill("c:", 3, p3.get("c"))

        return [(k, v.get(1, ""), v.get(2, ""), v.get(3, ""))
                for k, v in rows.items()]

    def _fill_results(self, table):
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        for row in table:
            self.results_table.insert("", "end", values=row)

    def _plot(self):
        if self.state.FIT.results is None:
            return
        plots.fit_plots(self.state, self.plot_panel.axes)
        self.plot_panel.draw()

    # ------------------------------------------------------------------
    def on_permute(self):
        state = self.state
        if state.FIT.results is None or state.CPD.param is None:
            self.app.message("Run Fit first.")
            return
        try:
            repeats = max(1, self.e_repeats.get_int())
            percentage = max(0.01, min(1.0, self.e_perm_pct.get_num() / 100))
            model = state.FIT.param["model"]
            cpd_param = state.CPD.param

            if cpd_param["dataSource"] == "simulated":
                SD = state.SIM.SD
            else:
                SD = state.import_data.SDfiltered
            tracks = np.unique(SD[:, 0])
            size = int(round(len(tracks) * percentage))
            if size < 2:
                self.app.message("Too few tracks for permutation.")
                return

            rng = np.random.default_rng()
            perm_results = np.zeros((repeats, 6))
            for i in range(repeats):
                chosen = rng.permutation(len(tracks))[:size]
                tracks_tmp = np.sort(tracks[chosen])
                SD_tmp = SD[np.isin(SD[:, 0], tracks_tmp)]
                _, results_tmp = cpd_calculateCPD(
                    SD_tmp, cpd_param["lagsToCalculate"], cpd_param["Dguess"],
                    cpd_param["bin"], cpd_param["fitToBin"], cpd_param["minP"],
                    cpd_param["minR"], cpd_param["tau"], cpd_param["N"])
                if model == 1:
                    data = results_tmp["model_1p"]
                elif model == 2:
                    data = results_tmp["model_2p"]
                else:
                    data = results_tmp["model_3p"]
                fr = fit_fitMSD(data, state.FIT.param)
                perm_results[i, 0] = fr["p1"]["D"]
                if model == 2:
                    perm_results[i, 1] = fr["p2"]["D"]
                    perm_results[i, 2] = fr["fraction"][0, 0]
                    perm_results[i, 3] = fr["fraction"][1, 0]
                elif model == 3:
                    perm_results[i, 1] = fr["p2"]["D"]
                    perm_results[i, 2] = fr["p3"]["D"]
                    perm_results[i, 3] = fr["fraction"][0, 0]
                    perm_results[i, 4] = fr["fraction"][1, 0]
                    perm_results[i, 5] = fr["fraction"][2, 0]

            perm_results = perm_results[:, : (3 if model == 1 else (4 if model == 2 else 6))]
            clean, _ = outliers(perm_results)
            perm_std = np.array([nanstd(clean[:, k]) for k in range(clean.shape[1])])
            perm_mean = np.array([nanmean(clean[:, k]) for k in range(clean.shape[1])])
            rel_err = perm_std / perm_mean
            rel_err_full = np.sqrt(rel_err ** 2 * percentage)

            table = list(state.FIT.results["resultsTable"])
            perm_errs = rel_err_full * np.array([state.FIT.results["p1"]["D"],
                                                 state.FIT.results["p2"]["D"] if model > 1 else 0,
                                                 state.FIT.results["p3"]["D"] if model > 2 else 0])
            new_table = []
            for row in table:
                if row[0] == "D permut:":
                    vals = list(row)
                    for k in range(1, model + 1):
                        vals[k] = f"{perm_errs[k - 1]:.6g}"
                    row = tuple(vals)
                new_table.append(row)
            state.FIT.results["resultsTable"] = new_table
            state.FIT.results["permResults"] = perm_results
            state.FIT.results["permError"] = perm_errs
            self._fill_results(new_table)
        except Exception as exc:
            self.app.message(f"Permutation failed:\n{exc}", "error")

    def on_copy(self):
        state = self.state
        if state.FIT.results is None:
            return
        cpd_tab = self.app.tabs.get("CPD")
        if cpd_tab is None:
            self.app.message("Open the CPD tab first.")
            return
        model = state.FIT.param["model"]
        res = state.FIT.results
        guess = cpd_tab.guess_widgets
        if model == 1:
            guess[0][0].delete(0, "end"); guess[0][0].insert(0, f"{res['p1']['D']:.6g}")
        elif model == 2:
            guess[0][0].delete(0, "end"); guess[0][0].insert(0, f"{res['p1']['D']:.6g}")
            guess[1][0].delete(0, "end"); guess[1][0].insert(0, f"{res['p2']['D']:.6g}")
            guess[2][0].delete(0, "end"); guess[2][0].insert(0, "0")
            guess[0][2].delete(0, "end"); guess[0][2].insert(0, f"{res['fraction'][0, 0]:.6g}")
            guess[1][2].delete(0, "end"); guess[1][2].insert(0, f"{res['fraction'][1, 0]:.6g}")
            guess[2][2].delete(0, "end"); guess[2][2].insert(0, "0")
        else:
            for k in range(3):
                guess[k][0].delete(0, "end")
                guess[k][0].insert(0, f"{res[f'p{k + 1}']['D']:.6g}")
                guess[k][2].delete(0, "end")
                guess[k][2].insert(0, f"{res['fraction'][k, 0]:.6g}")

    def on_fig(self):
        if self.state.FIT.results is None:
            return
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        axes = [fig.add_axes([0.1, 0.55, 0.35, 0.38]),
                fig.add_axes([0.6, 0.55, 0.35, 0.38]),
                fig.add_axes([0.1, 0.1, 0.35, 0.38]),
                fig.add_axes([0.6, 0.1, 0.35, 0.38])]
        plots.fit_plots(self.state, axes, show_figure=True)
        plt.show()

    def on_table(self):
        state = self.state
        if state.FIT.results is None or state.CPD.results is None:
            return
        lags = state.CPD.param["lagsToCalculate"]
        model = state.FIT.param["model"]
        results = state.CPD.results
        win = tk.Toplevel(self)
        win.title(f"{model}p model")
        cols = {"1p": ["Time [s]", "MSD [\u00b5m\u00b2]", "STDev"],
                "2p": ["Time [s]", "MSD1 [\u00b5m\u00b2]", "STDev", "MSD2 [\u00b5m\u00b2]", "STDev", "F1"],
                "3p": ["Time [s]", "MSD1", "STDev", "MSD2", "STDev", "MSD3", "STDev", "F1", "F2", "F3"]}
        tree = ttk.Treeview(win, columns=cols[f"{model}p"], show="headings", height=lags + 1)
        for c in cols[f"{model}p"]:
            tree.heading(c, text=c)
            tree.column(c, width=80, anchor="e")
        tree.pack(fill="both", expand=True, padx=4, pady=4)
        if model == 1:
            data = results["model_1p"]["MSD1"][:, :3]
        elif model == 2:
            m1 = results["model_2p"]["MSD1"][:, :3]
            m2 = results["model_2p"]["MSD2"][:, 1:3]
            f1 = results["model_2p"]["a"][2, :]
            data = np.column_stack([m1, m2, f1])
        else:
            m1 = results["model_3p"]["MSD1"][:, :3]
            m2 = results["model_3p"]["MSD2"][:, 1:3]
            m3 = results["model_3p"]["MSD3"][:, 1:3]
            f = results["model_3p"]["a"][3:6, :].T
            data = np.column_stack([m1, m2, m3, f])
        for row in data:
            tree.insert("", "end", values=[f"{v:.6g}" for v in row])
