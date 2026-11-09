"""SIM module tab: Monte Carlo trajectory simulation."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np

from ..core.simulate import (sim_fracprob, sim_reduceCoordinates,
                             sim_simulateCoordinates, sim_simulateDomains)
from . import plots, widgets as W


class SIMTab(tk.Frame):
    def __init__(self, app, state):
        super().__init__(app, bg=W.BG)
        self.app = app
        self.state = state

        # the left panel scrolls vertically so no parameter/button is ever
        # clipped when the window is shorter than the content
        left = W.ScrollFrame(self)
        left.pack(side="left", fill="y", padx=6, pady=6)
        L = left.inner
        right = tk.Frame(self, bg=W.BG)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self._build_simulate_panel(L)
        self._build_method_panel(L)
        self._build_domains_panel(L)
        self._build_plot_panel(L)

        tk.Frame(L, bg=W.BG, height=4).pack()

        self.button_simulate = W.button(L, "Simulate", self.on_simulate,
                                        width=22, fg="blue", font=("TkDefaultFont", 10, "bold"))
        self.button_simulate.pack(pady=4)

        btns = tk.Frame(L, bg=W.BG)
        btns.pack(fill="x")
        self.button_fig = W.button(btns, "Fig", self.on_fig, width=10)
        self.button_fig.pack(side="left", expand=True, fill="x", padx=2)
        self.button_export = W.button(btns, "Export coordinates", self.on_export, width=16)
        self.button_export.pack(side="left", expand=True, fill="x", padx=2)
        self.button_export.configure(state="disabled")

        # plot area
        title = W.label(right, "Particle Trajectories", font=("TkDefaultFont", 10, "bold"))
        title.pack()
        self.plot_panel = W.PlotPanel(right, figsize=(6.5, 6))
        self.plot_panel.pack(fill="both", expand=True)

        self._method = "1particle1D"
        self._sync_controls()
        self.plot_panel.draw()

    # ------------------------------------------------------------------
    def _build_simulate_panel(self, parent):
        box = W.box(parent, "Simulate trajectories")
        box.pack(fill="x", pady=3)

        # particles + random origin
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=1)
        W.label(row, "# particles", width=18, anchor="w").pack(side="left")
        self.e_particles = W.Entry(row, default=1, width=7)
        self.e_particles.pack(side="left")
        self.var_random_start = tk.BooleanVar(value=True)
        self._random_start_cb = W.check(row, "Random origin", self.var_random_start)
        self._random_start_cb.pack(side="left", padx=(8, 0))

        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=1)
        W.label(row, "# frames", width=18, anchor="w").pack(side="left")
        self.e_N = W.Entry(row, default=1000, width=7)
        self.e_N.pack(side="left")

        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=1)
        W.label(row, "Step [ms]", width=18, anchor="w").pack(side="left")
        self.e_tau = W.Entry(row, default=10, width=7)
        self.e_tau.pack(side="left")

        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=1)
        W.label(row, "Range [\u00b5m]", width=18, anchor="w").pack(side="left")
        self.e_range = W.Entry(row, default=10, width=7)
        self.e_range.pack(side="left")

        # static loc uncert + microsteps (stacked like the original)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=1)
        W.label(row, "Stat. loc. uncert. [\u00b5m]", width=18, anchor="w").pack(side="left")
        sub = tk.Frame(row, bg=W.BG)
        sub.pack(side="left")
        self.e_sig0 = W.Entry(sub, default=0, width=7)
        self.e_sig0.pack()
        sub2 = tk.Frame(row, bg=W.BG)
        sub2.pack(side="left", padx=(8, 0))
        W.label(sub2, "microsteps", anchor="w").pack(side="left")
        self.e_microsteps = W.Entry(sub2, default=1, width=7)
        self.e_microsteps.pack(side="left")

        # p1 / p2 grid
        grid = tk.Frame(box, bg=W.BG)
        grid.pack(fill="x", padx=4, pady=4)
        W.label(grid, "").grid(row=0, column=0, sticky="w")
        W.label(grid, "p1").grid(row=0, column=1, sticky="e", padx=4)
        W.label(grid, "p2").grid(row=0, column=2, sticky="e", padx=4)

        def grid_row(r, text, d1, d2, name):
            W.label(grid, text, anchor="w").grid(row=r, column=0, sticky="w")
            e1 = W.Entry(grid, default=d1, width=7)
            e1.grid(row=r, column=1, sticky="e", padx=4)
            e2 = W.Entry(grid, default=d2, width=7)
            e2.grid(row=r, column=2, sticky="e", padx=4)
            setattr(self, name + "_1", e1)
            setattr(self, name + "_2", e2)

        grid_row(1, "Diffusion coeff [\u00b5m\u00b2/s]", 2, 0, "e_D")
        grid_row(2, "Fraction", 0.5, 0.5, "e_F")
        grid_row(3, "Rate constant", 1, 1, "e_k")
        grid_row(4, "Prob. of switch", 0, 0, "e_prob")

        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=4, pady=(2, 4))
        W.label(row, "Actual F1:", width=18, anchor="w").pack(side="left")
        self.l_actual_f1 = W.label(row, "1", anchor="w")
        self.l_actual_f1.pack(side="left")

    def _build_method_panel(self, parent):
        box = W.box(parent, "Simulation method")
        box.pack(fill="x", pady=3)
        self.var_method = tk.StringVar(value="1particle1D")
        for label, value in (("1particle 1D", "1particle1D"),
                             ("Switch", "Switch"),
                             ("Domains", "Domains")):
            W.radio(box, label, self.var_method, value,
                    command=self._on_method_change).pack(anchor="w", padx=6, pady=1)

    def _on_method_change(self):
        self._method = self.var_method.get()
        self._sync_controls()

    def _sync_controls(self):
        m = self._method
        is_1p1d = m == "1particle1D"
        is_domains = m == "Domains"
        for e in (self.e_F_1, self.e_F_2):
            e.configure(state="normal" if is_1p1d else "disabled")
        for e in (self.e_k_1, self.e_k_2, self.e_prob_1, self.e_prob_2):
            e.configure(state="normal" if not is_1p1d else "disabled")
        self._random_start_cb.configure(
            state="disabled" if is_domains else "normal")
        self.check_plot_domains.configure(
            state="normal" if is_domains else "disabled")
        self.check_plot_paircorr.configure(
            state="normal" if is_domains else "disabled")
        self.button_sim_domains.configure(
            state="normal" if is_domains else "disabled")
        self.button_import_domains.configure(
            state="normal" if is_domains else "disabled")

    def _build_domains_panel(self, parent):
        box = W.box(parent, "Domains")
        box.pack(fill="x", pady=3)

        def row(text, name, default):
            r = tk.Frame(box, bg=W.BG)
            r.pack(fill="x", padx=4, pady=1)
            W.label(r, text, width=18, anchor="w").pack(side="left")
            e = W.Entry(r, default=default, width=7)
            e.pack(side="left")
            setattr(self, name, e)

        row("Concentration", "e_domains_conc", 0.5)
        row("Domain diameter [\u00b5m]", "e_domains_diameter", 1)
        row("Variation", "e_domains_variation", 0)
        row("Image size [px]", "e_domains_resolution", 512)
        row("pX size [\u00b5m]", "e_px_size", 0.16)

        btns = tk.Frame(box, bg=W.BG)
        btns.pack(fill="x", padx=4, pady=3)
        self.button_sim_domains = W.button(btns, "Sim domains", self.on_sim_domains, width=10)
        self.button_sim_domains.pack(side="left", expand=True, fill="x", padx=2)
        self.button_import_domains = W.button(btns, "Import", self.on_import_domains, width=8)
        self.button_import_domains.pack(side="left", expand=True, fill="x", padx=2)

    def _build_plot_panel(self, parent):
        box = W.box(parent, "Plot")
        box.pack(fill="x", pady=3)
        self.var_plot_domains = tk.BooleanVar(value=False)
        self.var_plot_micro = tk.BooleanVar(value=False)
        self.var_plot_paircorr = tk.BooleanVar(value=False)

        self.check_plot_domains = W.check(box, "Domains", self.var_plot_domains,
                                          self.on_plot_change)
        self.check_plot_domains.pack(anchor="w", padx=6)
        self.check_plot_micro = W.check(box, "Microsteps", self.var_plot_micro,
                                        self.on_plot_change)
        self.check_plot_micro.pack(anchor="w", padx=6)
        self.check_plot_paircorr = W.check(box, "Pair correlation",
                                           self.var_plot_paircorr,
                                           self.on_plot_change)
        self.check_plot_paircorr.pack(anchor="w", padx=6)

    # ------------------------------------------------------------------
    def on_plot_change(self):
        p = self.state.SIM.plot
        p.showDomains = int(self.var_plot_domains.get())
        p.showMicroSteps = int(self.var_plot_micro.get())
        p.showPairCorr = int(self.var_plot_paircorr.get())
        if p.showPairCorr:
            self.check_plot_domains.configure(state="disabled")
            self.check_plot_micro.configure(state="disabled")
        else:
            self.check_plot_domains.configure(
                state="normal" if self._method == "Domains" else "disabled")
            self.check_plot_micro.configure(state="normal")
        self._plot()

    def on_sim_domains(self):
        self.app.busy(True)
        try:
            sz = self.e_domains_resolution.get_int()
            diameter_um = self.e_domains_diameter.get_num()
            sim_range = self.e_range.get_num()
            conc = self.e_domains_conc.get_num()
            variation = self.e_domains_variation.get_num()
            diameter_px = diameter_um * sz / sim_range if sim_range else 0
            domains = sim_simulateDomains(sz, diameter_px, conc, variation)

            param = self.state.SIM.param
            param["domainsCrowding"] = conc
            param["domainsDiameter"] = diameter_px
            param["domainsResolution"] = sz
            param["domainsVariation"] = variation
            param["method"] = "Domains"
            param["simRange"] = sim_range
            self.state.SIM.param = param
            self.state.SIM.domains = domains
            self.state.SIM.coordinates = None
            self.state.SIM.plot.showDomains = 1
            self.state.SIM.plot.showTrajectories = 0
            self.var_plot_domains.set(True)
            self._plot()
            self.button_simulate.configure(state="normal")
        except Exception as exc:  # pragma: no cover
            self.app.message(f"Domain simulation failed:\n{exc}", "error")
        finally:
            self.app.busy(False)

    def on_import_domains(self):
        files = filedialog.askopenfilenames(
            parent=self, title="Select file",
            filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")])
        if not files:
            return
        try:
            from PIL import Image
            img = np.asarray(Image.open(files[0]).convert("L"))
            domains = (img > 0).astype(np.uint8)
            domains = np.flipud(domains)
            sim_range = self.e_range.get_num()
            res = domains.shape[1]
            param = self.state.SIM.param
            param["domainsResolution"] = res
            param["simRange"] = sim_range
            param["method"] = "Domains"
            self.state.SIM.param = param
            self.state.SIM.domains = domains
            self.state.SIM.coordinates = None
            self.state.SIM.plot.showTrajectories = 0
            self.state.SIM.plot.showDomains = 1
            self.var_plot_domains.set(True)
            self.e_domains_resolution.delete(0, "end")
            self.e_domains_resolution.insert(0, str(res))
            self.e_px_size.delete(0, "end")
            self.e_px_size.insert(0, f"{sim_range / res:.4g}")
            self._plot()
        except Exception as exc:  # pragma: no cover
            self.app.message(f"Could not import image:\n{exc}", "error")

    def on_simulate(self):
        try:
            particle_count = self.e_particles.get_int()
            N = self.e_N.get_int()
            tau_ms = self.e_tau.get_num()
            D = [self.e_D_1.get_num(), self.e_D_2.get_num()]
            F = [self.e_F_1.get_num(), self.e_F_2.get_num()]
            k21 = self.e_k_1.get_num()
            k12 = self.e_k_2.get_num()
            sig0 = self.e_sig0.get_num()
            microsteps = max(1, self.e_microsteps.get_int())
            sim_range = self.e_range.get_num()
            random_start = self.var_random_start.get()
            domains = self.state.SIM.domains
            method = self._method

            tau = tau_ms / 1000 / microsteps
            N = N * microsteps
            k21 = k21 * microsteps
            k12 = k12 * microsteps

            param = {
                "N": N, "tau": tau * microsteps, "microsteps": microsteps,
                "k21": k21, "k12": k12, "sig0": sig0, "method": method,
                "simRange": sim_range, "particleCount": particle_count,
                "domainsResolution": self.state.SIM.param.get("domainsResolution", 512),
            }
            self.state.SIM.param = param

            something = (D[0] * F[0] + D[1] * F[1] != 0) or (method == "Domains" and domains is not None)
            if not something:
                self.app.message("Nothing to simulate, check parameters.")
                return

            self.app.busy(True)
            self.app.update()
            coords, stats = sim_simulateCoordinates(
                particle_count, N, tau, D, F[0], k21, k12, sig0, sim_range,
                random_start, method, domains)
            self.state.SIM.stats = stats
            self.state.SIM.microCoordinates = coords
            if microsteps > 1:
                coords = sim_reduceCoordinates(coords, microsteps)
            self.state.SIM.coordinates = coords
            self.state.SIM.plot.showTrajectories = 1
            self.state.SIM.plot.showFigure = 0
            self._plot()

            if method == "1particle1D":
                self.e_prob_1.delete(0, "end"); self.e_prob_1.insert(0, "0")
                self.e_prob_2.delete(0, "end"); self.e_prob_2.insert(0, "0")
            elif method == "Switch":
                F1, F2, p21, p12 = sim_fracprob(k21, k12, tau)
                self.e_F_1.delete(0, "end"); self.e_F_1.insert(0, f"{F1:.3g}")
                self.e_F_2.delete(0, "end"); self.e_F_2.insert(0, f"{F2:.3g}")
                self.e_prob_1.delete(0, "end"); self.e_prob_1.insert(0, f"{p12:.3g}")
                self.e_prob_2.delete(0, "end"); self.e_prob_2.insert(0, f"{p21:.3g}")

            actual_f1 = np.mean(stats.get("actualF1", 0))
            self.l_actual_f1.configure(text=f"{actual_f1:.3g}")
            self.button_export.configure(state="normal")
        except Exception as exc:  # pragma: no cover
            self.app.message(f"Simulation failed:\n{exc}", "error")
        finally:
            self.app.busy(False)

    def _plot(self):
        plots.sim_plots(self.state.SIM, self.plot_panel.axes[0])
        self.plot_panel.draw()

    def on_fig(self):
        import matplotlib.pyplot as plt
        self.state.SIM.plot.showFigure = 1
        fig, ax = plt.subplots(figsize=(6, 6))
        plots.sim_plots(self.state.SIM, ax, show_figure=True)
        self.state.SIM.plot.showFigure = 0
        plt.show()

    def on_export(self):
        if self.state.SIM.coordinates is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            np.savetxt(path, self.state.SIM.coordinates, delimiter=",")
