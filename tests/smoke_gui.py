"""GUI smoke test: instantiate the app, walk all tabs, run a quick SIM
simulation and a CPD calculation, then close.  Needs a display (Tk)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("TkAgg")


def main():
    from trackart.gui.app import TrackArtApp

    app = TrackArtApp()
    app.update()

    # walk through all tabs; each must REPLACE the previous view (only one
    # tab frame mapped at a time) and fill the window
    for name in ("SIM", "Import", "MSD", "CPD", "FIT"):
        app._show_module(name)
        app.update()
        app.update_idletasks()
        mapped = [t for t in app.tabs.values() if t.winfo_ismapped()]
        assert [t.__class__.__name__ for t in mapped] == [
            name + "Tab"], f"{name}: expected only {name}Tab mapped, got {mapped}"
        assert app.tabs[name].winfo_width() > 500 and \
            app.tabs[name].winfo_height() > 300, \
            f"{name} tab does not fill the window"
    print("tabs OK (one view at a time, fills window)")

    # run a quick simulation from the SIM tab, for every method radio
    app._show_module("SIM")
    app.update()
    sim = app.tabs["SIM"]
    sim.e_N.delete(0, "end"); sim.e_N.insert(0, "200")
    for method in ("1particle1D", "Switch"):
        sim.var_method.set(method)
        sim._on_method_change()
        app.update()
        sim.on_simulate()
        app.update()
        assert app.state.SIM.coordinates is not None, \
            f"simulation failed for method {method}"
        print(f"SIM simulate ({method}) OK, coords:",
              app.state.SIM.coordinates.shape)

    # back to 1particle1D for the rest of the pipeline
    sim.var_method.set("1particle1D")
    sim._on_method_change()
    sim.on_simulate()
    app.update()

    # MSD fit on simulated data
    app._show_module("MSD")
    app.update()
    msd = app.tabs["MSD"]

    # the "Fit to" equations must render for every diffusion model
    from trackart.gui.msd_tab import MODELS
    for model in MODELS:
        msd.var_model.set(model)
        msd.on_model_change()
        app.update()
        imgs = [k for k in msd._eq_frame.winfo_children()
                if getattr(k, "image", None) is not None]
        assert imgs, f"no equation rendered for model {model}"
    print("MSD model equations OK")

    msd.on_fit()
    app.update()
    print("MSD fit OK")

    # CPD on simulated data
    app._show_module("CPD")
    app.update()
    cpd = app.tabs["CPD"]
    cpd.var_data.set("simulation")
    cpd.e_lags.delete(0, "end"); cpd.e_lags.insert(0, "5")
    cpd.e_bin.delete(0, "end"); cpd.e_bin.insert(0, "200")
    cpd.on_calculate()
    app.update()
    assert app.state.CPD.results is not None, "CPD failed"
    print("CPD calculate OK")

    # FIT on CPD results
    app._show_module("FIT")
    app.update()
    fit = app.tabs["FIT"]
    fit.on_fit()
    app.update()
    assert app.state.FIT.results is not None, "FIT failed"
    print("FIT OK")

    app.destroy()
    print("GUI SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
