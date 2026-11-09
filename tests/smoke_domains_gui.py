"""GUI test for the Domains simulation mode."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("TkAgg")


def main():
    from trackart.gui.app import TrackArtApp

    app = TrackArtApp()
    app.update()
    app._show_module("SIM")
    app.update()
    sim = app.tabs["SIM"]

    # switch to Domains method
    sim.var_method.set("Domains")
    sim._on_method_change()
    app.update()

    # simulate a small domain map (fast)
    sim.e_domains_resolution.delete(0, "end")
    sim.e_domains_resolution.insert(0, "128")
    sim.e_range.delete(0, "end")
    sim.e_range.insert(0, "10")
    sim.on_sim_domains()
    app.update()
    assert app.state.SIM.domains is not None, "domain map failed"
    print("domains map OK:", app.state.SIM.domains.shape)

    # simulate trajectories through the domains
    sim.e_N.delete(0, "end"); sim.e_N.insert(0, "300")
    sim.e_D_1.delete(0, "end"); sim.e_D_1.insert(0, "2")
    sim.e_D_2.delete(0, "end"); sim.e_D_2.insert(0, "5")
    sim.e_k_1.delete(0, "end"); sim.e_k_1.insert(0, "10")
    sim.e_k_2.delete(0, "end"); sim.e_k_2.insert(0, "10")
    sim.on_simulate()
    app.update()
    assert app.state.SIM.coordinates is not None, "domains simulation failed"
    print("domains sim OK:", app.state.SIM.coordinates.shape,
          "actualF1:", round(app.state.SIM.stats.get("actualF1", 0), 3))

    # the trajectory must be drawn ON TOP of the domains (not hidden behind
    # the early-return): axes must hold both the imshow image and line plots
    ax = sim.plot_panel.axes[0]
    images = [a for a in ax.get_children()
              if getattr(a, "get_array", None) is not None]
    lines = [a for a in ax.get_lines() if a.get_xdata().size]
    assert images, "domains image not drawn"
    assert lines, "trajectory not drawn on top of domains"
    print("domains + trajectory overlay OK")

    # pair correlation plot
    sim.var_plot_paircorr.set(True)
    sim.on_plot_change()
    app.update()
    print("pair correlation plot OK")

    # go back to 1particle 1D: the lingering domain map must no longer be
    # drawn (method no longer 'Domains') and the trajectory must appear
    sim.var_plot_paircorr.set(False)
    sim.on_plot_change()
    sim.var_method.set("1particle1D")
    sim._on_method_change()
    sim.on_simulate()
    app.update()
    assert app.state.SIM.coordinates is not None, "1p1d sim after domains failed"
    ax = sim.plot_panel.axes[0]
    images = [a for a in ax.get_children()
              if getattr(a, "get_array", None) is not None]
    lines = [a for a in ax.get_lines() if a.get_xdata().size]
    assert not images, "domains still drawn after switching back to 1p1d"
    assert lines, "1p1d trajectory not drawn after domains"
    print("back to 1particle1D OK")

    app.destroy()
    print("DOMAINS GUI SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
