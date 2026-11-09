"""FIT module: fit MSDs of each CPD population and the fractions.

Port of TrackArt's ``fit/fit_fitMSD.m``.
"""

from __future__ import annotations

import numpy as np

from .msd import (msd_MSDlinearFit, msd_anomalousDiffusion,
                  msd_confinedDiffusion)


def fit_fitMSD(data, parameters):
    """Fit the per-population MSD curves extracted from the CPD analysis.

    Parameters
    ----------
    data : dict with keys MSD1 [, MSD2 [, MSD3]] and a (fit coefficients)
    parameters : dict with keys:
        model           - 1, 2 or 3 populations
        p1lags/p2lags/p3lags - lags to fit (1-based lists)
        p1model/p2model/p3model - 1 normal, 2 confined, 3 anomalous
        p1intercept/p2intercept/p3intercept - [value, force_flag]
        fractionLags    - lags for the fraction average

    Returns a results dict with ``p1``, ``p2``, ``p3`` fit results and
    ``fraction`` (n_pop, 2): mean and std of each fraction.
    """
    results = {"p1": _fit_model(data["MSD1"], parameters["p1lags"],
                                parameters["p1model"], parameters["p1intercept"],
                                parameters.get("legacy_models", True))}

    if parameters["model"] == 2:
        results["p2"] = _fit_model(data["MSD2"], parameters["p2lags"],
                                   parameters["p2model"], parameters["p2intercept"],
                                   parameters.get("legacy_models", True))
        frac_lags = np.asarray(parameters["fractionLags"], dtype=int) - 1
        f1 = np.mean(data["a"][2, frac_lags])
        f1_std = np.std(data["a"][2, frac_lags], ddof=1 if len(frac_lags) > 1 else 0)
        results["fraction"] = np.array([
            [f1, f1_std],
            [1 - f1, f1_std],
        ])

    elif parameters["model"] == 3:
        results["p2"] = _fit_model(data["MSD2"], parameters["p2lags"],
                                   parameters["p2model"], parameters["p2intercept"],
                                   parameters.get("legacy_models", True))
        results["p3"] = _fit_model(data["MSD3"], parameters["p3lags"],
                                   parameters["p3model"], parameters["p3intercept"],
                                   parameters.get("legacy_models", True))
        frac_lags = np.asarray(parameters["fractionLags"], dtype=int) - 1
        means = np.array([np.mean(data["a"][3 + k, frac_lags]) for k in range(3)])
        stds = np.array([np.std(data["a"][3 + k, frac_lags],
                                ddof=1 if len(frac_lags) > 1 else 0)
                         for k in range(3)])
        results["fraction"] = np.column_stack([means, stds])

    else:  # 1 population
        results["fraction"] = np.array([[1.0, 0.0]])

    return results


def _fit_model(MSD, lags_to_fit, model, intercept, legacy_models=True):
    """Fit a single MSD curve with the selected model.

    model: 1 = normal (linear), 2 = confined, 3 = anomalous
    """
    if np.isscalar(lags_to_fit):
        lags_to_fit = np.arange(1, int(lags_to_fit) + 1)
    lags_to_fit = np.asarray(lags_to_fit, dtype=int)

    if model == 1:
        return msd_MSDlinearFit(MSD, lags_to_fit, intercept)
    elif model == 2:
        return msd_confinedDiffusion(MSD, lags_to_fit, intercept, legacy=legacy_models)
    elif model == 3:
        return msd_anomalousDiffusion(MSD, lags_to_fit, intercept)
    raise ValueError(f"Unknown fit model: {model}")
