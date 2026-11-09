"""Cumulative probability distribution (CPD) calculation and fitting.

Ports of TrackArt's ``cpd/`` MATLAB module.  CPDs of squared displacements
are calculated for a number of time lags and fit with mono-, bi- and
tri-exponential models following the approach of Schuetz et al. 1997.
"""

from __future__ import annotations

import numpy as np

from .fitting import fmincon_3p, interlace, lsqcurvefit, lsqcurvefit_fixed
from .util import ecdf, nanmean

# Faithful port: the original squares the bin centres of the CPD histogram
# (``cpd_bin(:, 1, i) = xout.^2``).  Set to False to bin on r^2 directly.
CPD_SQUARE_X = True


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def cpd_1p_model(a, data):
    """1 - exp(-|x|/a) ; a = mean squared displacement of the population."""
    data = np.asarray(data, dtype=float)
    return 1 - np.exp(-np.abs(data) / a[0])


def cpd_2p_model(x, xdata):
    """1 - (F1 exp(-|x|/a1) + F2 exp(-|x|/a2)) ; x = [a1, a2, F1]."""
    xdata = np.asarray(xdata, dtype=float)
    return 1 - (x[2] * np.exp(-np.abs(xdata) / x[0])
                + (1 - x[2]) * np.exp(-np.abs(xdata) / x[1]))


def cpd_3p_model(x, xdata):
    """Triple exponential; x = [a1, a2, a3, F1, F2, F3]."""
    xdata = np.asarray(xdata, dtype=float)
    f1, f2, f3 = x[3], x[4], x[5]
    return 1 - (f1 * np.exp(-np.abs(xdata) / x[0])
                + f2 * np.exp(-np.abs(xdata) / x[1])
                + f3 * np.exp(-np.abs(xdata) / x[2]))


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def cpd_fit_1p(x, y, i, Dguess):
    """Fit the 1-population model.  Dguess: (6, 3) array of guesses."""
    a0 = np.array([Dguess[0, 0] * i])
    lb = np.array([0.0])
    ub = np.array([np.inf])
    a = lsqcurvefit(cpd_1p_model, x, y, a0, lb, ub, max_nfev=10000)
    return a


def cpd_fit_2p(x, y, i, Dguess):
    """Fit the 2-population model; Dguess (6, 3): [D1, D2, D3, F1, F2, F3]."""
    a0 = np.array([
        Dguess[0, 0] * i + Dguess[0, 2],
        Dguess[1, 0] * i + Dguess[0, 2],
        Dguess[3, 0],
    ])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([np.inf, np.inf, 1.0])
    fix = np.array([Dguess[0, 1] == 1, Dguess[1, 1] == 1, Dguess[3, 1] == 1])
    a = lsqcurvefit_fixed(cpd_2p_model, x, y, a0, fix, lb, ub, max_nfev=100000)
    return a


def cpd_fit_3p(x, y, i, Dguess):
    """Fit the 3-population model with constraints (D1 <= D2 <= D3,
    fractions sum to 1).  Mirrors the fmincon-based fit."""
    x0 = np.array([
        Dguess[0, 0] * i + Dguess[0, 2],
        Dguess[1, 0] * i + Dguess[1, 2],
        Dguess[2, 0] * i + Dguess[2, 2],
        Dguess[3, 0],
        Dguess[4, 0],
        Dguess[5, 0],
    ])
    fix = np.array([Dguess[k, 1] == 1 for k in range(6)])

    free = ~fix
    lb = np.zeros(np.sum(free))
    ub = np.concatenate([np.full(np.sum(free[:3]), np.inf),
                         np.ones(np.sum(free[3:]))])

    A = -np.ones((1, np.sum(free)))
    b = np.array([0.0])

    Aeq = np.zeros((1, np.sum(free)))
    Aeq[0, np.where(free)[0] >= 3] = 1.0
    tmp = x0.copy()
    tmp[~fix] = 0.0
    tmp[:3] = 0.0
    beq = 1.0 - np.sum(tmp)

    def mycon(xx, fix_=fix, x0_=x0):
        xfull = x0_.copy()
        xfull[~fix_] = xx
        c = np.array([xfull[1] - xfull[0], xfull[2] - xfull[1]])
        ceq = np.array([1 - xfull[3] - xfull[4] - xfull[5]])
        return c, ceq

    # first refine the guess with an unconstrained lsqcurvefit
    x1 = lsqcurvefit(cpd_3p_model, x, y, x0,
                     np.zeros(6), np.concatenate([np.full(3, np.inf), np.ones(3)]),
                     max_nfev=100000)
    x0 = x0.copy()
    x0[~fix] = x1[~fix]

    a_tmp = fmincon_3p(
        lambda xx: np.sum((cpd_3p_model(interlace(x0, xx, fix), x) - y) ** 2),
        x0[free], lb, ub, A, b, Aeq, beq, mycon)

    a = x0.copy()
    a[~fix] = a_tmp
    return a


# ---------------------------------------------------------------------------
# CPD calculation
# ---------------------------------------------------------------------------

def _calculate_cpd(data, minP, minR):
    """Cumulative probability distribution of the given displacements.

    Returns (cpd_x, cpd_y) with points below ``minP`` probability and below
    ``minR`` squared distance excluded.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = np.sort(data)[::-1]
    if data.size == 0:
        return np.array([]), np.array([])
    cpd_x, cpd_y = ecdf(data)
    keep = cpd_y > minP
    cpd_x = cpd_x[keep]
    cpd_y = cpd_y[keep]
    keep2 = cpd_x >= minR
    cpd_x = cpd_x[keep2]
    cpd_y = cpd_y[keep2]
    return cpd_x, cpd_y


def _hist_centers(data, nbins):
    """MATLAB ``hist`` bin centres: nbins equally spaced centres spanning
    min..max of the data."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.linspace(0, 1, nbins)
    lo, hi = np.min(data), np.max(data)
    if lo == hi:
        return np.full(nbins, lo)
    return np.linspace(lo, hi, nbins)


def cpd_calculateCPD(SD, lags_to_calculate, Dguess, bin_size, fit_to_bin,
                     minP, minR, tau, N, square_x=None):
    """Calculate CPDs for every lag and fit 1p/2p/3p models.

    Parameters
    ----------
    SD : array
        squared displacements, column 0 = track id, columns 1..lags = SD
    lags_to_calculate : int
    Dguess : (6, 3) array
        column 1: guess D in um^2/s (already scaled by 4*tau by the caller)
        column 2: fixed flags
        column 3: intercept per population
    bin_size : int
    fit_to_bin : bool
        fit to binned data (True) or to all data (False)
    minP, minR : float
    tau : float
    N : array of K values (number of SDs per lag)

    Returns (cpd_bin, results).
    """
    if square_x is None:
        square_x = CPD_SQUARE_X

    lags = int(lags_to_calculate)
    time = tau * np.arange(1, lags + 1)

    results = {"model_1p": {}, "model_2p": {}, "model_3p": {}}
    # cpd_bin: [x, y] per lag (like the original, which stores only the binned
    # x values and the mean CPD per bin)
    cpd_bin = np.full((bin_size, 2, lags), np.nan)

    for i in range(1, lags + 1):
        cpd_x, cpd_y = _calculate_cpd(SD[:, i], minP, minR)

        xout = _hist_centers(cpd_x, bin_size)
        x_bin = xout ** 2 if square_x else xout
        cpd_bin[:, 0, i - 1] = x_bin

        for j in range(bin_size):
            if j == 0:
                sel = cpd_x < x_bin[j]
            else:
                sel = (cpd_x < x_bin[j]) & (cpd_x >= x_bin[j - 1])
            if np.any(sel):
                cpd_bin[j, 1, i - 1] = nanmean(cpd_y[sel])

        if not fit_to_bin:
            a = cpd_fit_3p(cpd_x, cpd_y, i, Dguess)
            results["model_3p"].setdefault("a", np.zeros((6, lags)))
            results["model_3p"]["a"][:, i - 1] = a

            a = cpd_fit_2p(cpd_x, cpd_y, i, Dguess)
            results["model_2p"].setdefault("a", np.zeros((3, lags)))
            results["model_2p"]["a"][:, i - 1] = a

            a = cpd_fit_1p(cpd_x, cpd_y, i, Dguess)
            results["model_1p"].setdefault("a", np.zeros((1, lags)))
            results["model_1p"]["a"][0, i - 1] = a[0]
        else:
            tmp = cpd_bin[:, :, i - 1]
            tmp = tmp[~np.any(np.isnan(tmp), axis=1), :]

            a = cpd_fit_3p(tmp[:, 0], tmp[:, 1], i, Dguess)
            results["model_3p"].setdefault("a", np.zeros((6, lags)))
            results["model_3p"]["a"][:, i - 1] = a

            a = cpd_fit_2p(tmp[:, 0], tmp[:, 1], i, Dguess)
            results["model_2p"].setdefault("a", np.zeros((3, lags)))
            results["model_2p"]["a"][:, i - 1] = a

            a = cpd_fit_1p(tmp[:, 0], tmp[:, 1], i, Dguess)
            results["model_1p"].setdefault("a", np.zeros((1, lags)))
            results["model_1p"]["a"][0, i - 1] = a[0]

    N = np.asarray(N, dtype=float)
    results["model_3p"]["MSD1"] = _msd_table(time, results["model_3p"]["a"][0, :], N)
    results["model_3p"]["MSD2"] = _msd_table(time, results["model_3p"]["a"][1, :], N)
    results["model_3p"]["MSD3"] = _msd_table(time, results["model_3p"]["a"][2, :], N)
    results["model_2p"]["MSD1"] = _msd_table(time, results["model_2p"]["a"][0, :], N)
    results["model_2p"]["MSD2"] = _msd_table(time, results["model_2p"]["a"][1, :], N)
    results["model_1p"]["MSD1"] = _msd_table(time, results["model_1p"]["a"][0, :], N)

    # fitted curves on the binned x values: cpd_binFit = [x, fit1p, fit2p, fit3p]
    cpd_bin_fit = np.full((bin_size, 4, lags), np.nan)
    for i in range(lags):
        cpd_bin_fit[:, 0, i] = cpd_bin[:, 0, i]
        cpd_bin_fit[:, 1, i] = cpd_1p_model(results["model_1p"]["a"][:, i],
                                            cpd_bin[:, 0, i])
        cpd_bin_fit[:, 2, i] = cpd_2p_model(results["model_2p"]["a"][:, i],
                                            cpd_bin[:, 0, i])
        cpd_bin_fit[:, 3, i] = cpd_3p_model(results["model_3p"]["a"][:, i],
                                            cpd_bin[:, 0, i])

    # residuals and F-test between models
    ftest = np.zeros((3, lags))
    for i in range(lags):
        res1 = cpd_bin_fit[:, 1, i] - cpd_bin[:, 1, i]
        res2 = cpd_bin_fit[:, 2, i] - cpd_bin[:, 1, i]
        res3 = cpd_bin_fit[:, 3, i] - cpd_bin[:, 1, i]
        finite = np.isfinite(res1) & np.isfinite(res2) & np.isfinite(res3)
        if np.sum(finite) >= 2:
            ftest[0, i] = _vartest2(res1[finite], res2[finite])
            ftest[1, i] = _vartest2(res2[finite], res3[finite])
            ftest[2, i] = _vartest2(res1[finite], res3[finite])

    results["cpd_binFit"] = cpd_bin_fit
    results["ftest"] = ftest
    return cpd_bin, results


def _msd_table(time, a, N):
    """Build the [time, MSD, stdev, K] table for one population."""
    lags = len(time)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = a / np.sqrt(N)
    out = np.column_stack([time, a, err, N])
    return out[:lags, :]


def _vartest2(x, y):
    """Two-sample F-test for equal variances (MATLAB ``vartest2``).

    Returns 1 when the hypothesis of equal variances can be rejected at the
    5% significance level, 0 otherwise.
    """
    from scipy import stats

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return 0
    if np.var(x, ddof=1) == 0 and np.var(y, ddof=1) == 0:
        return 0
    res = stats.levene(x, y, center="mean")
    # MATLAB vartest2 rejects when F exceeds the critical value; use the
    # two-sided F-test p-value via scipy
    f = np.var(x, ddof=1) / np.var(y, ddof=1)
    if f < 1:
        f = 1.0 / f
    p = 2 * (1 - stats.f.cdf(f, len(x) - 1, len(y) - 1))
    # fall back to the levene p-value when the F ratio is degenerate
    if not np.isfinite(p):
        p = res.pvalue
    return 1 if p < 0.05 else 0
