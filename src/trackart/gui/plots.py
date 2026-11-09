"""Plotting routines (ports of sim_plots.m, import_plots.m, msd_plots.m,
cpd_plots.m and fit_plots.m) drawing into matplotlib Axes."""

from __future__ import annotations

import numpy as np

from ..core import cpd as cpd_core
from ..core.msd import (msd_anomalousDiffusionModel, msd_confinedDiffusionModel,
                        msd_activeDiffusionModel)
from ..core.util import colorrgb


def set_box(ax):
    """Draw a full box around the axes (MATLAB 'box on')."""
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)

# ---------------------------------------------------------------------------
# SIM
# ---------------------------------------------------------------------------

def sim_plots(state, ax, show_figure=False):
    if state.domains is None and state.coordinates is None:
        return
    show_domains = state.plot.showDomains
    show_micro = state.plot.showMicroSteps
    show_pair_corr = state.plot.showPairCorr
    show_trajectories = state.plot.showTrajectories

    param = state.param or {}
    domains_res = param.get("domainsResolution", 512)
    sim_range = param.get("simRange", 10.0)
    method = param.get("method", "1particle1D")

    ax.clear()

    if show_pair_corr and state.domains is not None:
        _plot_pair_correlation(state.domains, sim_range, ax)
        ax.set_title("Pair correlation curve G(r)")
        return

    # Domains are drawn only while the Domains method is selected (the
    # original checked the sim_methodDomains radio value), so switching back
    # to 1particle1D/Switch hides them.  Like the original, trajectories are
    # then drawn on top of the domains (hold-on), not instead of them.
    if show_domains and state.domains is not None and method == "Domains":
        ax.set_box_aspect(1.0)
        domains = state.domains
        res = domains.shape[1]
        scale = sim_range / res
        ax.imshow(domains, cmap="copper", origin="lower",
                  extent=[1, res, 1, res], interpolation="nearest")
        ax.set_title("Domains visualisation")
        ax.set_xlabel("X coordinate [\u00b5m]")
        ax.set_ylabel("Y coordinate [\u00b5m]")
        _label_ticks_um(ax, scale)

    if show_trajectories and state.coordinates is not None:
        ax.set_box_aspect(1.0)
        coords = state.microCoordinates if show_micro else state.coordinates
        if method == "Domains":
            coords = coords.copy()
            coords[:, 4:6] = coords[:, 4:6] * (domains_res / sim_range)
        tracks = np.unique(coords[:, 0])
        colors = colorrgb(len(tracks))
        for k, tr in enumerate(tracks):
            sel = coords[:, 0] == tr
            ax.plot(coords[sel, 4], coords[sel, 5], color=colors[k],
                    linewidth=1)
        if method == "Domains":
            scale = sim_range / domains_res
            _label_ticks_um(ax, scale)
            ax.set_xlim(1, domains_res)
            ax.set_ylim(1, domains_res)
        else:
            ax.set_aspect("equal")
        ax.set_xlabel("X coordinate [\u00b5m]")
        ax.set_ylabel("Y coordinate [\u00b5m]")
        ax.set_title("Particle Trajectories")
    set_box(ax)


def _label_ticks_um(ax, scale):
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * scale:.2g}" for t in ticks])
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t * scale:.2g}" for t in ticks])


def _plot_pair_correlation(domains, sim_range, ax):
    res = domains.shape[1]
    scale = sim_range / res
    dim_b = domains.shape[1]
    dim_a = dim_b // 2
    fft = np.fft.fft2(domains.astype(float))
    autocorr = np.real(np.fft.ifft2(fft * np.conj(fft)))
    gr = 1.0 / (autocorr / (domains.mean() ** 2 * domains.shape[0] * domains.shape[1]))

    a = np.zeros((4, dim_a))
    a[0, :] = gr[0:dim_a, 0]
    a[1, :] = gr[0, 0:dim_a]
    a[2, :] = gr[dim_b - 1, dim_b - dim_a:dim_b][::-1]
    a[3, :] = gr[dim_b - dim_a:dim_b, dim_b - 1][::-1]
    a = np.mean(a, axis=0)

    x = (np.arange(1, dim_a + 1)) * scale
    c = np.sum(domains) / domains.shape[0] ** 2

    ax.plot(x, a[0] + 1 - a, "-r", linewidth=3)
    ax.axhline(c, color="k")
    ax.text(0.05 * max(x), c + 0.02, f"C = {c:.3g}", fontsize=12, fontweight="bold",
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0] * 0.5, x[-1])
    ax.set_xlabel("Radius [\u00b5m]")
    ax.set_ylabel("G(r)")


# ---------------------------------------------------------------------------
# IMPORT
# ---------------------------------------------------------------------------

def import_plots(state, ax, show_figure=False):
    if state.dataFiltered is None:
        return
    data = state.dataFiltered
    D = state.DallFiltered
    MSDall = state.MSDallFiltered
    param = state.param

    p = state.plot
    ax.clear()

    if p.intensitiesHistogram:
        ax.hist(D[:, 4], bins=50)
        ax.set_xlabel("relative intensity [au]")
        ax.set_ylabel("count")
        ax.set_title("Intensity distribution")
        ax.set_xlim(0, 2)

    if p.trackLengthDistribution:
        ax.hist(D[:, 3], bins=50)
        ax.set_xlabel("trajectory length [frames]")
        ax.set_ylabel("count")
        ax.set_title("Length distribution")

    if p.lengthVsIntensity:
        ax.scatter(D[:, 4], D[:, 3], s=4, c="r")
        ax.set_ylabel("trajectory length [frames]")
        ax.set_xlabel("avg trajectory intensity")
        ax.set_title("Length vs. intensity")
        ax.set_xlim(0, 2)

    if p.dHistogram:
        ax.hist(D[:, 1], bins=50)
        ax.set_xlabel("Diff Coeff [\u00b5m\u00b2/s]")
        ax.set_ylabel("count")
        ax.set_title("D distribution")

    if p.allTrajectories:
        if p.scatter:
            cvals = np.zeros(len(data))
            for tr in np.unique(D[:, 0]):
                cvals[data[:, 0] == tr] = D[D[:, 0] == tr, 1]
            sc = ax.scatter(data[:, 4], data[:, 5], s=4, c=cvals, cmap="jet")
            fig = ax.figure
            fig.colorbar(sc, ax=ax)
        else:
            tracks = np.unique(data[:, 0])
            colors = colorrgb(len(tracks))
            for k, tr in enumerate(tracks):
                sel = data[:, 0] == tr
                ax.plot(data[sel, 4], data[sel, 5], color=colors[k], linewidth=1)
        xmin, xmax = data[:, 4].min(), data[:, 4].max()
        ymin, ymax = data[:, 5].min(), data[:, 5].max()
        rng = max(xmax - xmin, ymax - ymin)
        ax.set_xlim(xmin, xmin + rng)
        ax.set_ylim(ymin, ymin + rng)
        ax.set_xlabel("X position [\u00b5m]")
        ax.set_ylabel("Y position [\u00b5m]")
        ax.set_title("Combined particle trajectories")

    if p.singleTrajectory:
        tr = int(p.trajectoryNumber)
        sel = data[:, 0] == tr
        if np.any(sel):
            ax.plot(data[sel, 4], data[sel, 5], color="b")
            xmin, xmax = data[sel, 4].min(), data[sel, 4].max()
            ymin, ymax = data[sel, 5].min(), data[sel, 5].max()
            rng = max(xmax - xmin, ymax - ymin) or 1.0
            ax.set_xlim(xmin, xmin + rng)
            ax.set_ylim(ymin, ymin + rng)
            ax.set_aspect("equal")
            ax.set_xlabel("X position [\u00b5m]")
            ax.set_ylabel("Y position [\u00b5m]")
            ax.set_title("Single trajectory")

    if p.allMSD:
        n = len(MSDall)
        if n > 0:
            for i in range(min(n, 500)):
                x = (np.arange(1, param.get("lagsToCalculate", 5) + 1)) * param.get("tau", 0.01)
                y = MSDall[i, 1:param.get("lagsToCalculate", 5) + 1]
                ax.plot(x, y, "-", linewidth=1)
            ax.set_xlim(param.get("tau", 0.01), param.get("lagsToCalculate", 5) * param.get("tau", 0.01))
            ax.set_ylim(0, np.nanmax(MSDall[:, 1:]))
            ax.set_xlabel("time [s]")
            ax.set_ylabel("MSD [\u00b5m\u00b2]")
            ax.set_title("MSD curves")

    if p.yLog and not p.allTrajectories and not p.singleTrajectory:
        ax.set_yscale("log")
    if p.xLog and not p.allTrajectories and not p.singleTrajectory:
        ax.set_xscale("log")
    set_box(ax)


# ---------------------------------------------------------------------------
# MSD
# ---------------------------------------------------------------------------

def msd_plots(state, ax, show_figure=False):
    results = state.MSD.results
    if results is None:
        return
    param = state.MSD.param
    lags = int(param["lagsToCalculate"])
    tau = param["tau"]
    if state.MSD.param.get("dataSource") == "simulated":
        MSD = state.SIM.MSD
        MSDall = state.SIM.MSDall
    else:
        MSD = state.import_data.MSD
        MSDall = state.import_data.MSDallFiltered

    p = state.MSD.plot
    ax.clear()

    x = np.concatenate([[0], np.arange(1, lags + 1) * tau])

    n_pts = min(lags, len(MSD))
    y = MSD[:n_pts, 1]
    xv = x[1:n_pts + 1]

    sdv_as_errorbars = False
    if p.msd:
        if n_pts > 100:
            ax.plot(xv, y, "-r")
        else:
            if p.msdSDV and results["model"] == "normal":
                err = results.get("MSD", MSD)[:n_pts, 2]
                ax.errorbar(xv, y, yerr=err, fmt="none", ecolor="black",
                            capsize=0, linewidth=1)
                sdv_as_errorbars = True
            ax.plot(xv, y, ".r", markersize=8)

    if p.msdAll and MSDall is not None and len(MSDall):
        for i in range(min(len(MSDall), 500)):
            yy = MSDall[i, 1:n_pts + 1]
            ax.plot(xv, yy, "-", color=(0.5, 0.5, 0.5), linewidth=1)

    if p.msdSDV and not sdv_as_errorbars and results["model"] == "normal":
        sdv = results.get("MSD", MSD)[:n_pts, 2]
        y1 = y + sdv
        y2 = np.maximum(y - sdv, 0)
        ax.plot(xv, y1, "-b", xv, y2, "-b")

    if p.diffCoeffSlope:
        lags_fit = np.asarray(param["lagsToFit"])
        if lags_fit.ndim == 0 or len(lags_fit) == 1:
            lags_fit = np.arange(1, int(np.max(lags_fit)) + 2)
        else:
            lags_fit = lags_fit + 1
        model = results["model"]
        if model == "normal":
            yy = results["a"] + results["b"] * x
        elif model == "anomalous":
            yy = msd_anomalousDiffusionModel(results["fit"], x)
        elif model == "confined":
            yy = msd_confinedDiffusionModel(results["fit"], x)
        elif model == "active":
            yy = msd_activeDiffusionModel(results["fit"], x)
        else:
            yy = results["a"] + results["b"] * x
        ax.plot(x, yy, "--", color=(0.8, 0.8, 0.8))
        idx = np.clip(lags_fit - 1, 0, len(x) - 1)
        ax.plot(x[idx], yy[idx], "-k", linewidth=2)

    ax.set_xlim(0, 1.03 * max(x))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSD [\u00b5m\u00b2]")
    set_box(ax)


# ---------------------------------------------------------------------------
# CPD
# ---------------------------------------------------------------------------

def cpd_plots(state, ax_top, ax_resid, preview_axes, show_figure=False):
    results = state.CPD.results
    if results is None:
        return
    cpd_bin = state.CPD.cpd_bin
    show_lags = state.CPD.param.get("showLag", 5)
    if np.isscalar(show_lags):
        show_lags = [int(show_lags)]
    model = state.CPD.param.get("plotModel", 2)

    for ax in [ax_top, ax_resid] + list(preview_axes):
        ax.clear()

    lag = show_lags[-1]
    xd = cpd_bin[:, 0, lag - 1]
    yd = cpd_bin[:, 1, lag - 1]

    if model == 1:
        ypred = cpd_core.cpd_1p_model(results["model_1p"]["a"][:, lag - 1], xd)
        a = results["model_1p"]["a"]
    elif model == 2:
        ypred = cpd_core.cpd_2p_model(results["model_2p"]["a"][:, lag - 1], xd)
        a = results["model_2p"]["a"]
    else:
        ypred = cpd_core.cpd_3p_model(results["model_3p"]["a"][:, lag - 1], xd)
        a = results["model_3p"]["a"]

    ax_top.plot(xd, ypred, "-k", linewidth=1, label="Fit")
    ax_top.plot(xd, yd, "--r", linewidth=1, label="cpd")
    ax_top.legend(loc="lower right")
    ax_top.set_xlabel("r^2 [\u00b5m\u00b2]")
    ax_top.set_ylabel("Probability")

    lag_str = f"{show_lags[0]}-{show_lags[-1]}" if len(show_lags) > 1 else str(lag)
    ax_top.set_title(f"Cumulative probability distribution for lag {lag_str}")

    finite = np.isfinite(yd) & np.isfinite(ypred)
    if np.any(finite):
        idx = np.where(cpd_bin[:, 1, lag - 1] > 0.98)[0]
        x_max = cpd_bin[idx[0], 0, lag - 1] if len(idx) else np.nanmax(yd)
        ax_top.set_xlim(0, x_max if np.isfinite(x_max) else np.nanmax(xd))
        ax_top.set_ylim(0, 1)

    # residuals (only for the last lag)
    res = yd - ypred
    ax_resid.plot(xd, res, "-r")
    ax_resid.axhline(0, color="k")
    ax_resid.set_xlim(ax_top.get_xlim())
    ax_resid.set_ylabel(f"Residuals for lag {lag}")

    # preview panels
    n_panels = len(preview_axes)
    titles = ["Population 1", "Population 2", "Population 3", "Fraction"]
    if model == 1:
        _preview_axes(preview_axes[0], results["model_1p"]["a"][0, :], "Population 1")
    elif model == 2:
        if n_panels >= 3:
            _preview_axes(preview_axes[0], results["model_2p"]["a"][0, :], "Population 1")
            _preview_axes(preview_axes[1], results["model_2p"]["a"][1, :], "Population 2")
            _preview_axes(preview_axes[2], results["model_2p"]["a"][2, :], "Fraction")
    else:
        for k in range(3):
            _preview_axes(preview_axes[k], results["model_3p"]["a"][k, :], titles[k])
        if n_panels >= 4:
            ax6 = preview_axes[3]
            ax6.plot(results["model_3p"]["a"][3, :], ".r", markersize=6)
            ax6.plot(results["model_3p"]["a"][4, :], ".g", markersize=6)
            ax6.plot(results["model_3p"]["a"][5, :], ".b", markersize=6)
            ax6.set_ylim(0, 1)
            ax6.set_title("Fraction", fontsize=8)
            ax6.tick_params(labelsize=6)
            set_box(ax6)


def _preview_axes(target, values, title):
    target.plot(values, ".r", markersize=6)
    target.set_title(title, fontsize=8)
    target.tick_params(labelsize=6)
    set_box(target)


# ---------------------------------------------------------------------------
# FIT
# ---------------------------------------------------------------------------

def fit_plots(state, axes, show_figure=False):
    results = state.FIT.results
    param = state.FIT.param
    data = state.FIT.data
    if results is None or data is None:
        return
    lags_calculated = len(data["MSD1"])

    x1 = np.concatenate([[0], data["MSD1"][:, 0]])
    y1 = data["MSD1"][:, 1]
    y_err1 = data["MSD1"][:, 2]
    y_fit1 = _model_fit(results["p1"], param["p1model"], x1)

    model = param["model"]
    for ax in axes:
        ax.clear()

    if model == 1:
        ax = axes[0]
        ax.plot(x1, y_fit1, "--", color=(0.7, 0.7, 0.7))
        lags = np.asarray(param["p1lags"], dtype=int) + 1
        ax.plot(x1[np.clip(lags, 0, len(x1) - 1)], y_fit1[np.clip(lags, 0, len(x1) - 1)], "-k", linewidth=2)
        ax.errorbar(x1[1:], y1, yerr=y_err1, fmt=".r", markersize=6)
        ax.set_ylim(0, max(np.nanmax(y1), np.nanmax(y_fit1)))
        ax.set_xlim(0, np.nanmax(x1))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("MSD [\u00b5m\u00b2]")
        ax.set_title("MSD fit, 1p model")
        set_box(ax)
        return

    if model == 2:
        x2 = np.concatenate([[0], data["MSD2"][:, 0]])
        y2 = data["MSD2"][:, 1]
        y_err2 = data["MSD2"][:, 2]
        y_fit2 = _model_fit(results["p2"], param["p2model"], x2)
        ax1, ax2, ax4 = axes[0], axes[1], axes[2]
        _fit_panel(ax1, x1, y1, y_err1, y_fit1, param["p1lags"],
                   "r1^2, 2p model")
        _fit_panel(ax2, x2, y2, y_err2, y_fit2, param["p2lags"],
                   "r2^2, 2p model", marker="b")
        _fraction_panel_2p(ax4, data["a"][2, :], results["fraction"][0, 0],
                           param["fractionLags"], lags_calculated, "F1")
        return

    if model == 3:
        x2 = np.concatenate([[0], data["MSD2"][:, 0]])
        y2 = data["MSD2"][:, 1]
        y_err2 = data["MSD2"][:, 2]
        y_fit2 = _model_fit(results["p2"], param["p2model"], x2)
        x3 = np.concatenate([[0], data["MSD3"][:, 0]])
        y3 = data["MSD3"][:, 1]
        y_err3 = data["MSD3"][:, 2]
        y_fit3 = _model_fit(results["p3"], param["p3model"], x3)
        ax1, ax2, ax3, ax4 = axes
        _fit_panel(ax1, x1, y1, y_err1, y_fit1, param["p1lags"], "r1^2, 3p model")
        _fit_panel(ax2, x2, y2, y_err2, y_fit2, param["p2lags"], "r2^2, 3p model", marker="b")
        _fit_panel(ax3, x3, y3, y_err3, y_fit3, param["p3lags"], "r3^2, 3p model", marker="g")
        _fraction_panel_3p(ax4, data, results, param["fractionLags"],
                           lags_calculated)


def _model_fit(res, model, x):
    if model == 1:
        return res["a"] + res["b"] * x
    elif model == 2:
        return msd_confinedDiffusionModel(res["fit"], x)
    elif model == 3:
        return msd_anomalousDiffusionModel(res["fit"], x)
    return res["a"] + res["b"] * x


def _fit_panel(ax, x, y, yerr, yfit, lags, title, marker="r"):
    ax.plot(x, yfit, "--", color=(0.7, 0.7, 0.7))
    lags = np.asarray(lags, dtype=int) + 1
    ax.plot(x[np.clip(lags, 0, len(x) - 1)], yfit[np.clip(lags, 0, len(x) - 1)],
            "-k", linewidth=2)
    ax.errorbar(x[1:], y, yerr=yerr, fmt=f".{marker}", markersize=6)
    ax.set_ylim(0, max(np.nanmax(y), np.nanmax(yfit)))
    ax.set_xlim(0, np.nanmax(x))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("r^2 [\u00b5m\u00b2]")
    ax.set_title(title)
    set_box(ax)


def _fraction_panel_2p(ax, values, frac, fraction_lags, lags_calculated, title):
    ax.axhline(frac, color=(0.7, 0.7, 0.7), linestyle=":")
    ax.plot(np.arange(1, lags_calculated + 1), values, ".r", markersize=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, lags_calculated)
    ax.set_xlabel("time lag")
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    set_box(ax)


def _fraction_panel_3p(ax, data, results, fraction_lags, lags_calculated):
    cols = ["r", "b", "g"]
    for k in range(3):
        ax.plot(np.arange(1, lags_calculated + 1), data["a"][3 + k, :],
                f".{cols[k]}", markersize=6)
        ax.axhline(results["fraction"][k, 0], color=cols[k], linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, lags_calculated)
    ax.set_xlabel("time lag")
    ax.set_ylabel("Fraction")
    ax.set_title("Fraction")
    set_box(ax)
