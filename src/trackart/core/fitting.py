"""Least-squares fitting helpers that mirror MATLAB's ``lsqcurvefit`` and
``fmincon`` as used throughout the original TrackArt code.

MATLAB ``lsqcurvefit(fun, x0, xdata, ydata, lb, ub, options, extra...)``
minimizes ``sum((fun(x, xdata, extra...) - ydata).^2)`` subject to
``lb <= x <= ub``.  We use :func:`scipy.optimize.least_squares` with the
trust-region reflective algorithm (the same one MATLAB uses by default).

Fixed parameters: TrackArt repeatedly uses the pattern::

    p_free = lsqcurvefit(@(x) myfun(interlace(p0, x, fix), xdata), p0(~fix), ...)
    p = interlace(p0, p_free, fix)

i.e. some parameters are fixed at their initial value and only the free ones
are optimised.  ``lsqcurvefit_fixed`` implements this in one call.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize


def _as_bounds(lb, ub, n):
    if lb is None:
        lb = -np.inf * np.ones(n)
    else:
        lb = np.asarray(lb, dtype=float).ravel()
        if lb.size == 1 and n > 1:
            lb = np.full(n, lb[0])
    if ub is None:
        ub = np.inf * np.ones(n)
    else:
        ub = np.asarray(ub, dtype=float).ravel()
        if ub.size == 1 and n > 1:
            ub = np.full(n, ub[0])
    return lb, ub


def lsqcurvefit(model, xdata, ydata, p0, lb=None, ub=None, weights=None,
                max_nfev=1_000_000, extra_args=()):
    """Fit ``model(p, xdata, *extra_args)`` to ``ydata`` (optionally weighted).

    When ``weights`` is given the residuals are ``(model - ydata) * weights``,
    which matches TrackArt's ``(fun(...).*weights)`` vs ``ydata.*weights``
    convention.
    """
    p0 = np.asarray(p0, dtype=float).ravel()
    ydata = np.asarray(ydata, dtype=float).ravel()
    xdata = np.asarray(xdata, dtype=float).ravel()
    n = p0.size
    lb, ub = _as_bounds(lb, ub, n)

    if weights is None:
        weights = np.ones_like(ydata)
    else:
        weights = np.asarray(weights, dtype=float).ravel()

    def fun(p):
        return ((model(p, xdata, *extra_args) - ydata) * weights).ravel()

    res = least_squares(fun, p0, bounds=(lb, ub), method="trf",
                        max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return res.x


def interlace(p_full, x_free, fix):
    """MATLAB ``interlace(p, x, fix)``: p(fix) keeps its value, p(~fix) = x."""
    p = np.asarray(p_full, dtype=float).copy()
    fix = np.asarray(fix, dtype=bool).ravel()
    p[~fix] = x_free
    return p


def lsqcurvefit_fixed(model, xdata, ydata, p0, fix, lb=None, ub=None,
                      weights=None, max_nfev=1_000_000, extra_args=()):
    """Like :func:`lsqcurvefit` but keeps parameters where ``fix`` is True at
    their initial value and only fits the free parameters."""
    p0 = np.asarray(p0, dtype=float).ravel()
    fix = np.asarray(fix, dtype=bool).ravel()
    n = p0.size
    lb, ub = _as_bounds(lb, ub, n)
    free = ~fix
    if not np.any(free):
        return p0.copy()
    x_free = lsqcurvefit(
        lambda p, xd: model(interlace(p0, p, fix), xd, *extra_args),
        xdata, ydata, p0[free], lb[free], ub[free], weights=weights,
        max_nfev=max_nfev,
    )
    return interlace(p0, x_free, fix)


def fmincon_3p(objective, x0, lb, ub, A, b, Aeq, beq, mycon=None,
               max_iter=5000, max_fun=1_000_000):
    """Constrained minimisation mirroring MATLAB's ``fmincon`` usage in
    ``cpd_fit_3p``.

    Parameters
    ----------
    objective : callable(x) -> float
    A, b : linear inequality constraints ``A @ x <= b``
    Aeq, beq : linear equality constraints ``Aeq @ x == beq``
    mycon : callable(x) -> (c, ceq) nonlinear inequality/equality constraints
    """
    x0 = np.asarray(x0, dtype=float).ravel()

    def _fun(x):
        return objective(x)

    cons = []
    if A is not None and len(A):
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float).ravel()
        cons.append({"type": "ineq", "fun": lambda x: b - A @ x})
    if Aeq is not None and len(Aeq):
        Aeq = np.atleast_2d(np.asarray(Aeq, dtype=float))
        beq = np.asarray(beq, dtype=float).ravel()
        cons.append({"type": "eq", "fun": lambda x: Aeq @ x - beq})
    if mycon is not None:
        def _ineq(x):
            c, ceq = mycon(x)
            return -np.asarray(c, dtype=float).ravel()  # c <= 0 -> -c >= 0

        def _eq(x):
            c, ceq = mycon(x)
            return np.asarray(ceq, dtype=float).ravel()

        cons.append({"type": "ineq", "fun": _ineq})
        cons.append({"type": "eq", "fun": _eq})

    res = minimize(_fun, x0, method="SLSQP", bounds=list(zip(lb, ub)),
                   constraints=cons, options={"maxiter": max_iter,
                                              "ftol": 1e-12})
    return res.x


def polyfit(x, y, deg):
    """MATLAB ``polyfit``: coefficients ordered from highest to lowest degree."""
    c = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), deg)
    return c


def polyfit0(x, y, deg):
    """Polynomial fit without constant term (polyfit0.m).

    Returns (p, S) where p has the constant term as the LAST element of the
    returned vector (zero), matching the MATLAB call convention
    ``p = polyfit0(x, y, 1); p(2) = intercept``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    deg = int(deg)
    # Build Vandermonde without the constant column
    V = np.vander(x, deg + 1)[:, :-1]
    coeffs, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    p = np.zeros(deg + 1)
    p[:deg] = coeffs  # highest-degree first, constant term zero
    # S structure with normr
    r = y - V @ coeffs

    class S:
        pass

    s = S()
    s.normr = float(np.linalg.norm(r))
    s.df = len(y) - deg
    return p, s
