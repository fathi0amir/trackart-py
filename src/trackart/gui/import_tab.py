"""Import module tab: import, inspect and filter trajectories."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from ..core.coordinates import calculateMSD, calculateSD
from ..core.import_data import (import_calculateDall, import_trackImport,
                                import_tracksExcl)
from . import plots, widgets as W
from .state import ImportState

PLOT_TYPES = [
    ("Tracks lengths distribution", "trackLengthDistribution"),
    ("All trajectories", "allTrajectories"),
    ("Single trajectory", "singleTrajectory"),
    ("Length vs intensity", "lengthVsIntensity"),
    ("Intensities histogram", "intensitiesHistogram"),
    ("D histogram", "dHistogram"),
    ("All MSDs", "allMSD"),
]


class ImportTab(tk.Frame):
    def __init__(self, app, state):
        super().__init__(app, bg=W.BG)
        self.app = app
        self.state = state

        left = tk.Frame(self, bg=W.BG, width=300)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)

        center = tk.Frame(self, bg=W.BG)
        center.pack(side="left", fill="both", expand=True, padx=4, pady=6)

        right = tk.Frame(self, bg=W.BG, width=320)
        right.pack(side="left", fill="y", padx=6, pady=6)
        right.pack_propagate(False)

        self._build_dataset_panel(left)
        self._build_exclusion_panel(left)
        self._build_stats_panel(left)

        self.plot_panel = W.PlotPanel(center, figsize=(6, 5))
        self.plot_panel.pack(fill="both", expand=True)
        self._build_plot_controls(center)

        self._build_table(right)
        self._build_help(right)

        self._sync_plot_flags()
        self.plot_panel.draw()

    # ------------------------------------------------------------------
    def _build_dataset_panel(self, parent):
        box = W.box(parent, "Dataset parameters")
        box.pack(fill="x", pady=3)
        self.button_select = W.button(box, "Select file(s)", self.on_select_files)
        self.button_select.pack(fill="x", padx=6, pady=4)
        self.var_calibrated = tk.BooleanVar(value=False)
        W.check(box, "data in \u00b5m", self.var_calibrated).pack(anchor="w", padx=6)

        f, self.e_tau = W.lab_entry(box, "Exposure [ms]", 10)
        f.pack(fill="x", padx=6, pady=1)
        f, self.e_px_size = W.lab_entry(box, "Px size [\u00b5m]", 0.16)
        f.pack(fill="x", padx=6, pady=1)
        row = tk.Frame(box, bg=W.BG)
        row.pack(fill="x", padx=6, pady=(1, 4))
        W.label(row, "Lags to use", width=14, anchor="w").pack(side="left")
        self.e_lags = W.Entry(row, default=5, width=7)
        self.e_lags.pack(side="left")
        self.button_calculate = W.button(row, "Calculate", self.on_calculate, width=8)
        self.button_calculate.pack(side="left", padx=(6, 0))

    def _build_exclusion_panel(self, parent):
        box = W.box(parent, "Exclusion")
        box.pack(fill="x", pady=3)

        def row(text, name, default):
            f, e = W.lab_entry(box, text, default)
            f.pack(fill="x", padx=6, pady=1)
            setattr(self, name, e)

        row("Min frames", "e_min_frames", 30)
        row("Min D", "e_minD", 0)
        row("Max D", "e_maxD", 10)
        row("Min R", "e_minR", 0.9)
        row("Min avg intensity", "e_min_int", 0)
        row("Max avg intensity", "e_max_int", 0)
        row("Random percentage", "e_random_pct", 100)

        btns = tk.Frame(box, bg=W.BG)
        btns.pack(fill="x", padx=6, pady=4)
        self.button_exclude = W.button(btns, "Exclude", self.on_exclude, width=9)
        self.button_exclude.pack(side="left", expand=True, fill="x", padx=2)
        self.button_reset = W.button(btns, "Reset", self.on_reset, width=9)
        self.button_reset.pack(side="left", expand=True, fill="x", padx=2)

    def _build_stats_panel(self, parent):
        box = W.box(parent, "Statistics")
        box.pack(fill="x", pady=3)
        head = tk.Frame(box, bg=W.BG)
        head.pack(fill="x", padx=6, pady=2)
        W.label(head, "", width=14, anchor="w").pack(side="left")
        W.label(head, "Imported", width=8, anchor="e").pack(side="left", expand=True)
        W.label(head, "Filtered", width=8, anchor="e").pack(side="left", expand=True)

        self.stats_vars = {}
        for text in ("Coordinates", "Trajectories", "Avg D", "Avg traj. size"):
            r = tk.Frame(box, bg=W.BG)
            r.pack(fill="x", padx=6)
            W.label(r, text, width=14, anchor="w").pack(side="left")
            v1 = tk.StringVar(value="0")
            v2 = tk.StringVar(value="0")
            W.label(r, textvariable=v1, width=8, anchor="e").pack(side="left", expand=True)
            W.label(r, textvariable=v2, width=8, anchor="e").pack(side="left", expand=True)
            self.stats_vars[text] = (v1, v2)

    def _build_plot_controls(self, parent):
        box = W.box(parent, "Plots")
        box.pack(fill="x", pady=(4, 0))

        inner = tk.Frame(box, bg=W.BG)
        inner.pack(fill="x", padx=4, pady=2)
        col1 = tk.Frame(inner, bg=W.BG)
        col1.pack(side="left", fill="y")
        col2 = tk.Frame(inner, bg=W.BG)
        col2.pack(side="left", padx=8)

        self.var_plot = tk.StringVar(value="trackLengthDistribution")
        for text, key in PLOT_TYPES[:5]:
            W.radio(col1, text, self.var_plot, key, self.on_plot_select,
                    anchor="w").pack(anchor="w", padx=2)
        for text, key in PLOT_TYPES[5:]:
            W.radio(col2, text, self.var_plot, key, self.on_plot_select,
                    anchor="w").pack(anchor="w", padx=2)

        f = tk.Frame(col2, bg=W.BG)
        f.pack(anchor="w", pady=1)
        self.e_trajectory_number = W.Entry(f, default=1, width=6)
        self.e_trajectory_number.pack(side="left")
        self.var_xlog = tk.BooleanVar(value=False)
        self.var_ylog = tk.BooleanVar(value=False)
        self.var_scatter = tk.BooleanVar(value=False)
        W.check(col2, "x-log", self.var_xlog, self.on_plot_select).pack(anchor="w")
        W.check(col2, "y-log", self.var_ylog, self.on_plot_select).pack(anchor="w")
        W.check(col2, "scat heatmap", self.var_scatter, self.on_plot_select).pack(anchor="w")

        btns = tk.Frame(box, bg=W.BG)
        btns.pack(fill="x", padx=4, pady=3)
        self.button_fig = W.button(btns, "Fig", self.on_fig, width=8)
        self.button_fig.pack(side="left", expand=True, fill="x", padx=2)
        self.button_export = W.button(btns, "Export", self.on_export, width=8)
        self.button_export.pack(side="left", expand=True, fill="x", padx=2)

    def _build_table(self, parent):
        box = W.box(parent, "Data")
        box.pack(fill="both", expand=True, pady=3)
        columns = ("#", "D", "R^2", "Length", "Intensity")
        tree = ttk.Treeview(box, columns=columns, show="headings", height=12)
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=55, anchor="e", stretch=True)
        sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.table = tree

    def _build_help(self, parent):
        box = W.box(parent, "Help")
        box.pack(fill="x", pady=3)
        text = ("1. Select data file(s) - hold ctrl for multiple selection\n"
                "2. Adjust Exposure, Pixel Size, Lags to use\n"
                "3. Hit 'Calculate'\n"
                "4. Inspect data and exclude unwanted trajectories\n\n"
                "Reset - removes filtering\n"
                "Export - saves trajectories back into file")
        W.label(box, text, justify="left", anchor="w",
                font=("TkDefaultFont", 8)).pack(fill="x", padx=6, pady=4)

    # ------------------------------------------------------------------
    def on_select_files(self):
        files = filedialog.askopenfilenames(
            parent=self, title="Select file(s)",
            filetypes=[("All files", "*.*"), ("Excel", "*.xls"),
                       ("Text", "*.txt"), ("CSV", "*.csv")])
        if not files:
            return
        try:
            coords = import_trackImport(list(files))
        except Exception as exc:
            self.app.message(str(exc), "error")
            return
        self.app.invalidate_analysis()
        self.state.import_data = ImportState()
        imp = self.state.import_data
        imp.coordinates = coords
        imp.files = list(files)
        imp.stats["pathName"] = os.path.dirname(files[0])
        imp.stats["fileName"] = [os.path.basename(f) for f in files]

        ext = os.path.splitext(files[0])[1].lower()
        self.var_calibrated.set(ext == ".csv")
        for v1, v2 in self.stats_vars.values():
            v1.set("0")
            v2.set("0")
        self._clear_table()
        self.button_calculate.configure(state="normal")
        self.button_exclude.configure(state="disabled")
        self.button_reset.configure(state="disabled")
        self.button_fig.configure(state="disabled")
        self.button_export.configure(state="disabled")

    def on_calculate(self):
        imp = self.state.import_data
        if imp.coordinates is None:
            return
        param = {
            "isCalibrated": bool(self.var_calibrated.get()),
            "pxSize": self.e_px_size.get_num(),
            "lagsToCalculate": self.e_lags.get_int(),
            "tau": self.e_tau.get_num() / 1000,
        }
        data = imp.coordinates.copy()
        if not param["isCalibrated"]:
            data[:, 4:6] = data[:, 4:6] * param["pxSize"]

        # track lengths
        track_ids, lengths = np.unique(data[:, 0], return_counts=True)
        keep_ids = track_ids[lengths > 10]
        data = data[np.isin(data[:, 0], keep_ids)]

        # per-track statistics, indexed by track id (like the original):
        # col 0 = track id, col 1 = length, col 2 = mean intensity,
        # col 3 = flag for kept (length > 10) tracks
        tracks = np.zeros((int(np.max(data[:, 0])) + 1 if len(data) else 0, 4))
        for tr in keep_ids:
            sel = data[:, 0] == tr
            tracks[int(tr), 0] = tr
            tracks[int(tr), 1] = np.sum(sel)
            mean_val = np.mean(data[sel, 7])
            tracks[int(tr), 2] = mean_val
            tracks[int(tr), 3] = 1
            data[sel, 6] = mean_val
        tracks[tracks[:, 2] == 0, 2] = np.nan

        SD, MSDall = calculateSD(data, param["lagsToCalculate"], 1)
        MSD = calculateMSD(SD, param["tau"])
        Dall = import_calculateDall(MSDall, tracks, param["lagsToCalculate"],
                                    param["tau"])

        self.app.invalidate_analysis()
        imp.data = data
        imp.dataFiltered = data
        imp.Dall = Dall
        imp.DallFiltered = Dall
        imp.SD = SD
        imp.SDfiltered = SD
        imp.MSD = MSD
        imp.MSDall = MSDall
        imp.MSDallFiltered = MSDall
        imp.tracksExcl = keep_ids
        imp.tracks = tracks
        imp.param = param
        imp.plot.showFigure = 0

        stat_tracks = len(np.unique(data[:, 0]))
        stat_coord = len(data)
        avgD = np.nanmean(Dall[:, 1])
        self.stats_vars["Coordinates"][0].set(str(stat_coord))
        self.stats_vars["Trajectories"][0].set(str(stat_tracks))
        self.stats_vars["Avg D"][0].set(f"{avgD:.2f}")
        self.stats_vars["Avg traj. size"][0].set(f"{stat_coord / stat_tracks:.0f}")

        imp.stats["raw"] = [stat_coord, stat_tracks, stat_coord / stat_tracks, avgD]

        self._update_table(Dall)
        self._plot()
        for b in (self.button_exclude, self.button_reset,
                  self.button_fig, self.button_export):
            b.configure(state="normal")

    def on_exclude(self):
        imp = self.state.import_data
        if imp.SD is None:
            return
        param = imp.param
        param["minFrames"] = self.e_min_frames.get_int()
        param["minD"] = self.e_minD.get_num()
        param["maxD"] = self.e_maxD.get_num()
        param["minR"] = self.e_minR.get_num()
        param["minInt"] = self.e_min_int.get_num()
        param["maxInt"] = self.e_max_int.get_num()
        param["randomPercentage"] = self.e_random_pct.get_num() / 100
        imp.param = param

        tracks_excl = import_tracksExcl(
            imp.Dall, param["minFrames"], param["minD"], param["maxD"],
            param["minR"], param["minInt"], param["maxInt"])
        if len(tracks_excl) == 0:
            self.app.message("No trajectories meet the criteria.")
            return

        size = int(round(len(tracks_excl) * param["randomPercentage"]))
        if size < 1:
            self.app.message("Random percentage too small.")
            return
        self.app.invalidate_analysis()
        random_tracks = np.random.permutation(len(tracks_excl))[:size]
        tracks_excl = np.sort(tracks_excl[random_tracks])

        mask = np.isin(imp.data[:, 0], tracks_excl)
        imp.dataFiltered = imp.data[mask]
        imp.DallFiltered = imp.Dall[np.isin(imp.Dall[:, 0], tracks_excl)]
        imp.MSDallFiltered = imp.MSDall[np.isin(imp.MSDall[:, 0], tracks_excl)]
        imp.SDfiltered = imp.SD[np.isin(imp.SD[:, 0], tracks_excl)]
        imp.MSD = calculateMSD(imp.SDfiltered, param["tau"])
        imp.tracksExcl = tracks_excl

        stat_tracks = len(np.unique(imp.dataFiltered[:, 0]))
        stat_coord = len(imp.dataFiltered)
        avgD = np.mean(imp.DallFiltered[:, 1])
        imp.stats["filtered"] = [stat_coord, stat_tracks, stat_coord / stat_tracks, avgD]
        self.stats_vars["Coordinates"][1].set(str(stat_coord))
        self.stats_vars["Trajectories"][1].set(str(stat_tracks))
        self.stats_vars["Avg D"][1].set(f"{avgD:.2f}")
        self.stats_vars["Avg traj. size"][1].set(f"{stat_coord / stat_tracks:.0f}")
        self._update_table(imp.DallFiltered)
        self._plot()

    def on_reset(self):
        imp = self.state.import_data
        if imp.SD is None:
            return
        self.app.invalidate_analysis()
        tau = imp.param["tau"]
        imp.dataFiltered = imp.data
        imp.DallFiltered = imp.Dall
        imp.MSDallFiltered = imp.MSDall
        imp.MSD = calculateMSD(imp.SD, tau)
        imp.SDfiltered = imp.SD
        imp.tracksExcl = imp.Dall[:, 0]
        for v1, v2 in self.stats_vars.values():
            v2.set("0")
        self._update_table(imp.Dall)
        self._plot()

    # ------------------------------------------------------------------
    def on_plot_select(self):
        self._sync_plot_flags()
        self._plot()

    def _sync_plot_flags(self):
        p = self.state.import_data.plot
        key = self.var_plot.get()
        p.trackLengthDistribution = int(key == "trackLengthDistribution")
        p.allTrajectories = int(key == "allTrajectories")
        p.singleTrajectory = int(key == "singleTrajectory")
        p.lengthVsIntensity = int(key == "lengthVsIntensity")
        p.intensitiesHistogram = int(key == "intensitiesHistogram")
        p.dHistogram = int(key == "dHistogram")
        p.allMSD = int(key == "allMSD")
        p.xLog = int(self.var_xlog.get())
        p.yLog = int(self.var_ylog.get())
        p.scatter = int(self.var_scatter.get())
        p.trajectoryNumber = self.e_trajectory_number.get_int()

    def _plot(self):
        plots.import_plots(self.state.import_data, self.plot_panel.axes[0])
        self.plot_panel.draw()

    def on_fig(self):
        import matplotlib.pyplot as plt
        self.state.import_data.plot.showFigure = 1
        fig, ax = plt.subplots(figsize=(6, 6))
        plots.import_plots(self.state.import_data, ax, show_figure=True)
        self.state.import_data.plot.showFigure = 0
        plt.show()

    def on_export(self):
        imp = self.state.import_data
        if imp.dataFiltered is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save as", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            np.savetxt(path, imp.dataFiltered[:, :6], delimiter=",")

    def _update_table(self, data):
        self._clear_table()
        for row in data:
            self.table.insert("", "end", values=[f"{v:.4g}" if isinstance(v, (int, float)) else v
                                                 for v in row[:5]])

    def _clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
