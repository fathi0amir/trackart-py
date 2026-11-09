"""MSD analysis and diffusion model fitting.

Direct ports of TrackArt's ``msd/`` MATLAB module.  Implements:

* linear MSD fit with the Michalet 2010 error analysis (MSD standard
  deviation, covariance, errors of fit parameters a and b for weighted and
  unweighted fits, localisation uncertainty, pMin)
* anomalous, confined and active diffusion models
"""

from __future__ import annotations

import numpy as np

from .fitting import lsqcurvefit, lsqcurvefit_fixed, polyfit, polyfit0
from .util import nanmean, nanstd


# ---------------------------------------------------------------------------
# diffusion models
# ---------------------------------------------------------------------------

def msd_normalDiffusionModel(param, data, weights=None):
    """<r^2> = a + b*dt ; param = [b, a]."""
    y = param[1] + param[0] * np.asarray(data, dtype=float)
    if weights is not None:
        y = y * np.asarray(weights, dtype=float)
    return y


def msd_anomalousDiffusionModel(param, data):
    """<r^2> = a + b*dt^c ; param = [b, a, c]."""
    return param[1] + param[0] * np.asarray(data, dtype=float) ** param[2]


def msd_confinedDiffusionModel(param, data):
    """<r^2> = rc^2 [1 - A1 exp(-4 A2 D dt / rc^2)] + c.

    param = [rc^2, A1, A2, D, intercept]
    """
    data = np.asarray(data, dtype=float)
    return param[0] * (1 - param[1] * np.exp(-4 * param[2] * param[3] * data / param[0])) + param[4]


def msd_activeDiffusionModel(param, data):
    """<r^2> = v^2 dt^2 + 4 D dt + c ; param = [v^2, 4D, c]."""
    data = np.asarray(data, dtype=float)
    return param[0] * data ** 2 + param[1] * data + param[2]


# ---------------------------------------------------------------------------
# Michalet 2010 error analysis
# ---------------------------------------------------------------------------

def msd_locUncert(a, b, te, s0):
    """Dynamic and static localisation uncertainty (Michalet 2010, eq. 18/19).

    NOTE: as in the original there is a typo in the paper - s0 is added to the
    numerator of the static formula.
    """
    sigma = 0.5 * np.sqrt(a + (b * te) / 3)
    if s0 == 0:
        sigma_zero = sigma
    else:
        sigma_zero = (s0 * sigma) / np.sqrt(s0 ** 2 + (b * te / 4))
    return sigma, sigma_zero


def msd_SDV_calc(n, x, MSD):
    """f(n, N, x) value, Michalet appendix D.15.  K = N - n."""
    K = MSD[int(n) - 1, 3]  # 4th column
    n = float(n)
    if n > K:
        f = (6 * n ** 2 * K - 4 * n * K ** 2 + 4 * n + K ** 3 - K) / 6 / K + (2 * n * x + x ** 2) / K
        return 1.0 / f
    else:
        f = n * (4 * n ** 2 * K + 2 * K - n ** 3 + n) / 6 / K ** 2 + (2 * n * x + (1 + (1 - n / K) / 2) * x ** 2) / K
        return 1.0 / f


def msd_SDV_zero(MSD, lags_to_fit):
    """MSD standard deviation for x = 0 (weights for the weighted fit)."""
    MSD = np.asarray(MSD, dtype=float)
    out = np.zeros(int(lags_to_fit))
    for n in range(1, int(lags_to_fit) + 1):
        K = MSD[n - 1, 3]
        if n > K:
            out[n - 1] = (6 * n ** 2 * K - 4 * n * K ** 2 + 4 * n + K ** 3 - K) / 6 / K
        else:
            out[n - 1] = n * (4 * n ** 2 * K + 2 * K - n ** 3 + n) / 6 / K ** 2
    return 1.0 / out


def msd_COV_calc(n, m, x, MSD):
    """Covariance g(n, m), Michalet appendix F.12."""
    MSD = np.asarray(MSD, dtype=float)
    if m < n:
        m, n = n, m
    K = MSD[int(n) - 1, 3]
    P = MSD[int(m) - 1, 3]
    N = MSD[0, 3] + 1
    n = float(n)
    m = float(m)
    if (m + n) <= N:
        g = (n * (4 * n ** 2 * K + 2 * K - n ** 3 + n + (m - n) * (6 * n * P - 4 * n ** 2 - 2)) / 6 / K / P
             + (2 * n * x + (1 - n / 2 / P) * x ** 2) / K)
    else:
        g = ((6 * n ** 2 * K - 4 * n * K ** 2 + K ** 3 + 4 * n - K
              + (m - n) * ((n + m) * (2 * K + P) + 2 * n * P - 3 * K ** 2 + 1)) / 6 / K
             + (2 * n * x + x ** 2 / 2) / K)
    return g


def sigmaa(x, P, MSD):
    """Normalised error of fit parameter a (unweighted), Michalet supplement."""
    sum0 = 0.0
    sum1 = 0.0
    for i in range(1, P + 1):
        fi = msd_SDV_calc(i, x, MSD)
        sum0 += ((2 * P + 1) / 3 - i) ** 2 / fi
        psum_i = 0.0
        for j in range(1, i):
            gij = msd_COV_calc(i, j, x, MSD)
            psum_i += ((2 * P + 1) / 3 - j) * gij
        sum1 += psum_i * ((2 * P + 1) / 3 - i)
    temp = sum0 + 2 * sum1
    return np.sqrt(temp) * 6 / P / (P - 1) / x


def sigmab(x, P, MSD):
    """Normalised error of fit parameter b (unweighted)."""
    sum0 = 0.0
    sum1 = 0.0
    for i in range(1, P + 1):
        fi = msd_SDV_calc(i, x, MSD)
        sum0 += ((P + 1) / 2 - i) ** 2 / fi
        psum_i = 0.0
        for j in range(1, i):
            gij = msd_COV_calc(i, j, x, MSD)
            psum_i += ((P + 1) / 2 - j) * gij
        sum1 += psum_i * ((P + 1) / 2 - i)
    temp = sum0 + 2 * sum1
    return np.sqrt(temp) * 12 / P / (P - 1) / (P + 1)


def sigmaaW(x, P, MSD):
    """Normalised error of fit parameter a (weighted)."""
    sum0 = sum1 = sum2 = sumh0 = sumh1 = sumh2 = 0.0
    for i in range(1, P + 1):
        fi = msd_SDV_calc(i, x, MSD)
        sum0 += fi
        sum1 += i * fi
        sum2 += i ** 2 * fi
        h0 = h1 = 0.0
        for j in range(1, i):
            fj = msd_SDV_calc(j, x, MSD)
            gij = msd_COV_calc(i, j, x, MSD)
            hij = fi * fj * gij
            h0 += hij
            h1 += j * hij
        sumh0 += h0
        sumh1 += i * h0 + h1
        sumh2 += i * h1
    delta = sum0 * sum2 - sum1 ** 2
    temp = (sum2 + 2 * (sum2 ** 2 * sumh0 - sum1 * sum2 * sumh1 + sum1 ** 2 * sumh2) / delta) / delta
    return np.sqrt(temp) / x


def sigmabW(x, P, MSD):
    """Normalised error of fit parameter b (weighted)."""
    sum0 = sum1 = sum2 = sumh0 = sumh1 = sumh2 = 0.0
    for i in range(1, P + 1):
        fi = msd_SDV_calc(i, x, MSD)
        sum0 += fi
        sum1 += i * fi
        sum2 += i ** 2 * fi
        h0 = h1 = 0.0
        for j in range(1, i):
            fj = msd_SDV_calc(j, x, MSD)
            gij = msd_COV_calc(i, j, x, MSD)
            hij = fi * fj * gij
            h0 += hij
            h1 += j * hij
        sumh0 += h0
        sumh1 += i * h0 + h1
        sumh2 += i * h1
    delta = sum0 * sum2 - sum1 ** 2
    temp = (sum0 + 2 * (sum1 ** 2 * sumh0 - sum0 * sum1 * sumh1 + sum0 ** 2 * sumh2) / delta) / delta
    return np.sqrt(temp)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def msd_MSDlinearFit(MSD, lags_to_fit, intercept, weights=None, weighted_fit=False,
                     te=0, s0=0, tau=None):
    """Linear MSD fit ``<r^2> = a + b*t`` (msd_MSDlinearFit.m).

    ``lags_to_fit`` can be a scalar or a list of lags (1-based).
    """
    MSD = np.asarray(MSD, dtype=float)
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)
    coor_x = MSD[lags_to_fit - 1, 0]
    coor_y = MSD[lags_to_fit - 1, 1]
    lags_to_calculate = len(MSD)

    m = int(np.max(lags_to_fit))
    K = MSD[m - 1, 3]
    relative_err_qian = np.sqrt((2 * m) / (3 * K)) if K > 0 else np.nan

    results = {}

    if weights is None and not weighted_fit:
        # simple (unweighted) linear fit
        if intercept[1] == 0:
            p = polyfit(coor_x, coor_y, 1)
        else:
            p, _ = polyfit0(coor_x, coor_y - intercept[0], 1)
            p = p.copy()
            p[1] = intercept[0]
        results["D"] = p[0] / 4
        results["a"] = p[1]
        results["b"] = p[0]
        results["qianErr"] = results["D"] * relative_err_qian
        results["relativeErrQian"] = relative_err_qian
        results["fit"] = p
        return results

    # ------------------------------------------------------------------
    # full weighted / error-analysis fit
    # ------------------------------------------------------------------
    p0 = np.array([coor_y[1] / coor_x[1] if coor_x[1] != 0 else 0.0, 0.0])
    if weights is None:
        weights = np.ones(len(coor_y))
    ydata = coor_y * weights

    def linfit(p, xdata):
        return (p[1] + p[0] * xdata) * weights

    p = lsqcurvefit(linfit, coor_x, ydata, p0)

    D = p[0] / 4
    sigma, sigma_zero = msd_locUncert(p[1], p[0], te, s0)
    epsilon = p[1] + (4 * D * te) / 3
    alpha = p[0] * tau
    x = epsilon / alpha if alpha != 0 else np.inf

    if weighted_fit:
        norm_sigmaa = sigmaaW(x, int(np.max(lags_to_fit)), MSD)
        norm_sigmab = sigmabW(x, int(np.max(lags_to_fit)), MSD)
    else:
        norm_sigmaa = sigmaa(x, int(np.max(lags_to_fit)), MSD)
        norm_sigmab = sigmab(x, int(np.max(lags_to_fit)), MSD)

    # MSD standard deviation
    MSD_stdev = np.zeros(lags_to_calculate)
    for n in range(1, lags_to_calculate + 1):
        f = msd_SDV_calc(n, x, MSD)
        MSD_stdev[n - 1] = np.sqrt(1 / f) * alpha

    p_min = int(np.floor(2 + 2.7 * np.sqrt(x))) if x >= 0 else 0

    results["D"] = D
    results["sigmaDynamic"] = sigma
    results["sigmaStatic"] = sigma_zero
    results["qianErr"] = D * relative_err_qian
    results["relativeErrQian"] = relative_err_qian
    results["b"] = p[0]
    results["a"] = p[1]
    results["sigmab"] = p[0] * norm_sigmab
    results["sigmaa"] = p[1] * norm_sigmaa
    results["x"] = x
    results["epsilon"] = epsilon
    results["alpha"] = alpha
    results["MSD"] = MSD.copy()
    results["MSD"][:, 2] = MSD_stdev
    results["fit"] = p
    results["pMin"] = p_min
    return results


def msd_normalDiffusion(MSD, lags_to_fit, intercept, weights, weighted_fit, te, s0, tau):
    """Normal (Brownian) diffusion fit with Michalet error analysis."""
    MSD = np.asarray(MSD, dtype=float)
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)

    x = MSD[lags_to_fit - 1, 0]
    y = MSD[lags_to_fit - 1, 1]
    lags_to_calculate = len(MSD)

    m = int(np.max(lags_to_fit))
    K = MSD[m - 1, 3]
    relative_err_qian = np.sqrt((2 * m) / (3 * K)) if K > 0 else np.nan

    p0_guess = polyfit(x, y, 1)
    param0 = np.array([p0_guess[0], intercept[0]])
    lb = [0, 0]
    ub = [np.inf, np.inf]
    fix = np.array([0, int(intercept[1])], dtype=bool)

    weights_arr = np.asarray(weights, dtype=float).ravel()
    ydata = y * weights_arr

    p = lsqcurvefit_fixed(
        lambda p_, xd: msd_normalDiffusionModel(p_, xd, weights_arr),
        x, ydata, param0, fix, lb, ub)

    results = {
        "D": p[0] / 4,
        "a": p[1],
        "b": p[0],
        "qianErr": (p[0] / 4) * relative_err_qian,
        "relativeErrQian": relative_err_qian,
        "fit": p,
    }

    sigma, sigma_zero = msd_locUncert(p[1], p[0], te, s0)
    epsilon = p[1] + (4 * results["D"] * te) / 3
    alpha = p[0] * tau
    xx = epsilon / alpha if alpha != 0 else np.inf

    if weighted_fit:
        norm_sigmaa = sigmaaW(xx, int(np.max(lags_to_fit)), MSD)
        norm_sigmab = sigmabW(xx, int(np.max(lags_to_fit)), MSD)
    else:
        norm_sigmaa = sigmaa(xx, int(np.max(lags_to_fit)), MSD)
        norm_sigmab = sigmab(xx, int(np.max(lags_to_fit)), MSD)

    if intercept[1] == 0:
        results["sigmaa"] = p[1] * norm_sigmaa
    else:
        results["sigmaa"] = 0.0
    results["sigmab"] = p[0] * norm_sigmab
    results["x"] = xx
    results["epsilon"] = epsilon
    results["alpha"] = alpha
    results["sigmaDynamic"] = sigma
    results["sigmaStatic"] = sigma_zero

    MSD_stdev = np.zeros(lags_to_calculate)
    for n in range(1, lags_to_calculate + 1):
        f = msd_SDV_calc(n, xx, MSD)
        MSD_stdev[n - 1] = np.sqrt(1 / f) * alpha
    results["MSD"] = MSD.copy()
    results["MSD"][:, 2] = MSD_stdev
    results["pMin"] = int(np.floor(2 + 2.7 * np.sqrt(xx))) if xx >= 0 else 0
    return results


def msd_anomalousDiffusion(MSD, lags_to_fit, intercept):
    """Anomalous diffusion fit ``<r^2> = a + b t^c``."""
    MSD = np.asarray(MSD, dtype=float)
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)
    x = MSD[lags_to_fit - 1, 0]
    y = MSD[lags_to_fit - 1, 1]

    if intercept[1] == 0:
        intercept = [0, intercept[1]]

    p_guess = polyfit(x, y, 1)
    param0 = np.array([p_guess[0], intercept[0], 1.0])
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 1]
    fix = np.array([0, int(intercept[1]), 0], dtype=bool)

    p = lsqcurvefit_fixed(msd_anomalousDiffusionModel, x, y, param0, fix, lb, ub)

    return {
        "D": p[0] / 4,
        "a": p[1],
        "b": p[0],
        "c": p[2],
        "fit": p,
    }


def msd_confinedDiffusion(MSD, lags_to_fit, intercept):
    """Confined / corralled diffusion fit."""
    MSD = np.asarray(MSD, dtype=float)
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)
    x = MSD[lags_to_fit - 1, 0]
    y = MSD[lags_to_fit - 1, 1]

    if intercept[1] == 0:
        intercept = [0, intercept[1]]

    p_guess = polyfit(x[:3], y[:3], 1) if len(x) >= 3 else polyfit(x, y, 1)
    param0 = np.array([y[int(np.max(lags_to_fit)) - 1], 1.0, 1.0, p_guess[0], intercept[0]])
    lb = [0, 0, 0, 0, 0]
    ub = [np.inf] * 5
    fix = np.array([0, 0, 0, 0, int(intercept[1])], dtype=bool)

    p = lsqcurvefit_fixed(msd_confinedDiffusionModel, x, y, param0, fix, lb, ub)

    return {
        "D": p[3],
        "A1": p[1],
        "A2": p[1],
        "rc": p[0],
        "c": p[4],
        "fit": p,
    }


def msd_activeDiffusion(MSD, lags_to_fit, intercept):
    """Active (flow) diffusion fit ``<r^2> = v^2 t^2 + 4D t + c``."""
    MSD = np.asarray(MSD, dtype=float)
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)
    x = MSD[lags_to_fit - 1, 0]
    y = MSD[lags_to_fit - 1, 1]

    if intercept[1] == 0:
        intercept = [0, intercept[1]]

    p_guess = polyfit(x, y, 1)
    param0 = np.array([0.0, p_guess[0], intercept[0]])
    lb = [0, 0, 0]
    ub = [np.inf] * 3
    fix = np.array([0, int(intercept[1]), 0], dtype=bool)

    p = lsqcurvefit_fixed(msd_activeDiffusionModel, x, y, param0, fix, lb, ub)

    return {
        "D": p[1] / 4,
        "vsq": p[0],
        "intercept": p[2],
        "fit": p,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def msd_calculatePmin(MSD, param):
    """Optimal number of lags for the linear fit (Michalet 2010).

    Returns a (k, 4) array: columns [a, b, x, pMin].
    """
    MSD = np.asarray(MSD, dtype=float)
    sigma_zero = param.get("sig", 0)
    lags_to_calculate = int(param["lagsToCalculate"])
    tau = param["tau"]

    p = msd_MSDlinearFit(MSD, np.arange(1, lags_to_calculate + 1), [0, 0])

    x_vals = []
    pmin_vals = []
    rows = []

    x1 = sigma_zero ** 2 / (p["b"] * tau / 4) if p["b"] != 0 else 0.0
    pm = int(np.floor(2 + 2.7 * np.sqrt(x1)))
    rows.append([p["a"], p["b"], x1, pm])
    x_vals.append(x1)
    pmin_vals.append(pm)
    i = 1

    flag = 0
    if pm < lags_to_calculate:
        while not flag:
            p = msd_MSDlinearFit(MSD, np.arange(1, pmin_vals[i - 1] + 1), [0, 0])
            if p["a"] < 0:
                p["a"] = 0
            if p["b"] < 0:
                p["b"] = 0
            xi = p["a"] / (p["b"] * tau) if p["b"] != 0 else 0.0
            pmi = int(np.ceil(2 + 2.7 * np.sqrt(xi)))
            rows.append([p["a"], p["b"], xi, pmi])
            x_vals.append(xi)
            pmin_vals.append(pmi)
            if pmi >= pmin_vals[i - 1]:
                flag = 1
            i += 1
    return np.array(rows)


def msd_calculateDcoeff(MSD, param):
    """Calculate diffusion coefficients for the selected model.

    Returns (results, results_table) where results_table is a list of
    [name, value] rows.
    """
    MSD = np.asarray(MSD, dtype=float)
    lags_to_calculate = int(param["lagsToCalculate"])
    MSD = MSD[:lags_to_calculate, :]
    tau = param["tau"]
    lags_to_fit = param["lagsToFit"]
    weighted_fit = param["weightedFit"]
    model = param["model"]
    s0 = param.get("s0", 0)
    te = param.get("te", 0)
    intercept = param["intercept"]

    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)

    results = {}
    if weighted_fit:
        msd_weights = msd_SDV_zero(MSD, int(np.max(lags_to_fit)))
    else:
        msd_weights = np.ones(len(lags_to_fit))

    results_table = []

    if model == "normal":
        results = msd_normalDiffusion(MSD, lags_to_fit, intercept, msd_weights,
                                      weighted_fit, te, s0, tau)
        results["pMin"] = msd_calculatePmin(MSD, param)
        rows = [
            ("Diffusion coefficient [um^2/s]:", results["D"]),
            ("Error D:", results["sigmab"] / 4),
            ("Error D (Qian):", results["qianErr"]),
            ("a (intercept):", results["a"]),
            ("sigma a:", results["sigmaa"]),
            ("b:", results["b"]),
            ("sigma b:", results["sigmab"]),
            ("x (reduced localization error):", results["x"]),
            ("alpha:", results["alpha"]),
            ("epsilon:", results["epsilon"]),
            ("Est static localization error:", results["sigmaStatic"]),
            ("Est dynamic localization error:", results["sigmaDynamic"]),
        ]
        p_min_arr = np.atleast_2d(results["pMin"])
        if p_min_arr[-1, 3] < lags_to_calculate:
            rows.append(("p Min:", p_min_arr[-1, 3]))
        else:
            rows.append(("p Min:", "Not Enough"))
        results_table = rows

    elif model == "anomalous":
        results = msd_anomalousDiffusion(MSD, lags_to_fit, intercept)
        results_table = [
            ("Transport coefficient [um^2/s]:", results["D"]),
            ("a (intercept):", results["a"]),
            ("b:", results["b"]),
            ("alpha:", results["c"]),
        ]

    elif model == "confined":
        results = msd_confinedDiffusion(MSD, lags_to_fit, intercept)
        results_table = [
            ("Diffusion coefficient [um^2/s]:", results["D"]),
            ("A1:", results["A1"]),
            ("A2:", results["A2"]),
            ("rc^2:", results["rc"]),
            ("intercept:", results["c"]),
        ]

    elif model == "active":
        results = msd_activeDiffusion(MSD, lags_to_fit, intercept)
        results_table = [
            ("Diffusion coefficient [um^2/s]:", results["D"]),
            ("v:", np.sqrt(results["vsq"])),
            ("intercept:", results["intercept"]),
        ]

    results["model"] = model
    return results, results_table
