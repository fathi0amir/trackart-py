"""Numerical utilities ported from TrackArt's MATLAB tools.

* roundsd   - round to N significant digits (roundsd.m)
* outliers  - Thompson-Tau outlier removal (outliers.m)
* nanmean / nanstd wrappers matching MATLAB behaviour
* colorrgb  - distinct colour table for trajectories
"""

from __future__ import annotations

import numpy as np


def roundsd(x, n: int = 2):
    """Round ``x`` to ``n`` significant digits (MATLAB roundsd)."""
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        og = 10.0 ** (np.floor(np.log10(np.abs(x)) - n + 1))
    y = np.round(x / og) * og
    y[x == 0] = 0.0
    if scalar:
        return float(y[0])
    return y


def parse_number(s) -> float | None:
    """MATLAB ``str2num`` equivalent: returns None when input is not a number."""
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def parse_lags(s) -> list[int] | None:
    """Parse a MATLAB style lag spec: '5', '3:10', '1, 3, 5, 7', '5:7' -> list of ints."""
    s = str(s).strip()
    if not s:
        return None
    out: list[int] = []
    try:
        for part in s.split(","):
            part = part.strip()
            if ":" in part:
                a, b = part.split(":", 1)
                out.extend(range(int(float(a)), int(float(b)) + 1))
            else:
                out.append(int(float(part)))
    except ValueError:
        return None
    return out


def nanmean(a, axis=None):
    """MATLAB nanmean (ignores NaN, returns NaN for all-NaN slices)."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=axis)


def nanstd(a, axis=None):
    """MATLAB nanstd - sample standard deviation (N-1), ignoring NaN."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        return np.nanstd(a, axis=axis, ddof=1)


def nanvar(a, axis=None):
    """MATLAB nanvar - sample variance (N-1), ignoring NaN."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        return np.nanvar(a, axis=axis, ddof=1)


# Thompson's tau values for samples 3..41 (from outliers.m)
_THOMPSON_TAU = np.array(
    [
        1.150, 1.393, 1.572, 1.656, 1.711, 1.749, 1.777, 1.798, 1.815,
        1.829, 1.840, 1.849, 1.858, 1.865, 1.871, 1.876, 1.881, 1.885, 1.889,
        1.893, 1.896, 1.899, 1.902, 1.904, 1.906, 1.908, 1.910, 1.911, 1.913,
        1.914, 1.916, 1.917, 1.919, 1.920, 1.921, 1.922, 1.923, 1.924,
    ]
)


def outliers(X0, num_outliers: int = 1):
    """Remove up to ``num_outliers`` outliers using the Thompson-Tau method.

    Returns a copy of X0 (same shape) with detected outliers set to NaN, and
    the (flattened) indices of the removed values. Mirrors TrackArt's
    tools/outliers/outliers.m.
    """
    X0 = np.asarray(X0, dtype=float)
    shape = X0.shape
    flat = X0.ravel().copy()
    n1 = flat.size
    if n1 < 3:
        return X0.copy(), np.array([], dtype=int)

    # sort by absolute value, keeping NaN at the end
    order = np.argsort(np.abs(flat), kind="stable")
    X = flat[order]
    nan_mask = np.isnan(X)
    X = X[~nan_mask]
    n = X.size
    nns = int(nan_mask.sum())

    outliers_idx: list[int] = []

    def tau_sd():
        stdev = X.std(ddof=1)
        if n > len(_THOMPSON_TAU):
            return 1.960 * stdev
        return _THOMPSON_TAU[n - 1] * stdev

    i = 0
    while num_outliers > 0 and n > 0:
        xbar = X.mean()
        tauS = tau_sd()
        if abs(abs(X[-1]) - xbar) > tauS:
            X = X[:-1]
            n = len(X)
            # index into the original flattened array
            outliers_idx.append(order[-1 - i - nns])
            flat[order[-1 - i - nns]] = np.nan
            i += 1
        num_outliers -= 1

    out = flat.reshape(shape)
    return out, np.array(outliers_idx, dtype=int)


# ---------------------------------------------------------------------------
# colour tables
# ---------------------------------------------------------------------------

def colorrgb(n: int) -> np.ndarray:
    """Distinct colour table for ``n`` trajectories (like colorrgb.m).

    Uses a fixed golden-angle hue sequence so plots are reproducible.
    """
    if n <= 1:
        return np.array([[0.0, 0.4470, 0.7410]])
    h = (np.arange(n) * 0.6180339887498949) % 1.0
    s = 0.75 * np.ones(n)
    v = 0.85 * np.ones(n)
    hsv = np.stack([h, s, v], axis=1)
    from matplotlib.colors import hsv_to_rgb

    return hsv_to_rgb(hsv)


def ecdf(data):
    """Empirical CDF like MATLAB's ``ecdf``: returns (x, y) with y = CDF.

    Input is expected to be a 1-D array of finite values.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = np.sort(data)
    n = data.size
    if n == 0:
        return np.array([]), np.array([])
    values, counts = np.unique(data, return_counts=True)
    return np.concatenate((values[:1], values)), np.concatenate(
        ([0.0], np.cumsum(counts) / n))
