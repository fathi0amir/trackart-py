"""Main TrackArt window: a tkinter app with the five modules as tabs.

The tab bar is a row of radio-style buttons (SIM, Import, MSD, CPD, FIT)
matching the original MATLAB GUI.
"""

from __future__ import annotations

import tkinter as tk

from . import widgets as W
from .state import State

MODULES = ["SIM", "Import", "MSD", "CPD", "FIT"]


class TrackArtApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("trackart")
        self.configure(bg=W.BG)
        self.geometry("1280x920")
        self.minsize(1000, 720)

        self.state = State()
        self.tabs = {}
        self._current_tab = None
        self._tab_var = tk.StringVar(value="SIM")

        self._build_tab_bar()
        self._show_module("SIM")

    # ------------------------------------------------------------------
    def _build_tab_bar(self):
        bar = tk.Frame(self, bg=W.BG)
        bar.pack(side="top", fill="x", padx=6, pady=(6, 2))
        for name in MODULES:
            rb = tk.Radiobutton(
                bar, text=name, variable=self._tab_var, value=name,
                command=lambda n=name: self._show_module(n),
                indicatoron=False, relief="raised", bd=1, bg=W.BG,
                selectcolor="#d9d9d9", font=("TkDefaultFont", 9, "bold"),
                padx=14, pady=2,
            )
            rb.pack(side="left", padx=2)

    def _show_module(self, name):
        self._tab_var.set(name)
        if self._current_tab is not None:
            self._current_tab.pack_forget()
        if name not in self.tabs:
            self.tabs[name] = self._create_tab(name)
        self.tabs[name].pack(fill="both", expand=True)
        self._current_tab = self.tabs[name]

    def _create_tab(self, name):
        from .cpd_tab import CPDTab
        from .fit_tab import FITTab
        from .import_tab import ImportTab
        from .msd_tab import MSDTab
        from .sim_tab import SIMTab

        cls = {"SIM": SIMTab, "Import": ImportTab, "MSD": MSDTab,
               "CPD": CPDTab, "FIT": FITTab}[name]
        return cls(self, self.state)

    # ------------------------------------------------------------------
    def sim_tau_ms(self):
        """Step [ms] from the SIM tab (default 10 when not yet opened)."""
        tab = self.tabs.get("SIM")
        if tab is not None:
            try:
                return tab.e_tau.get_num()
            except AttributeError:
                pass
        return 10.0

    def import_tau_ms(self):
        """Exposure [ms] from the Import tab (default 10 when not yet opened)."""
        tab = self.tabs.get("Import")
        if tab is not None:
            try:
                return tab.e_tau.get_num()
            except AttributeError:
                pass
        return 10.0

    # ------------------------------------------------------------------
    def message(self, text, kind="info"):
        from tkinter import messagebox
        if kind == "error":
            messagebox.showerror("TrackArt", text, parent=self)
        else:
            messagebox.showwarning("TrackArt", text, parent=self)

    def busy(self, flag=True):
        self.configure(cursor="watch" if flag else "")
        self.update_idletasks()


def main():
    import matplotlib
    matplotlib.use("TkAgg")
    app = TrackArtApp()
    app.mainloop()


if __name__ == "__main__":
    main()
