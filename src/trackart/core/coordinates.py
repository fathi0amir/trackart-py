"""Square displacement (SD) and mean square displacement (MSD) calculations.

Direct ports of TrackArt's ``tools/calculateSD.m`` and
``tools/calculateMSD.m``.

Data format used across TrackArt (``data`` / ``coordinates``): a 2-D array
with one row per detection and columns::

    1  track (trajectory) id
    2  dataset id
    3  (unused / track number)
    4  frame number
    5  x coordinate [um]
    6  y coordinate [um]
    7  z coordinate
    8  intensity (m0)

``SD`` has shape (n_rows, lags+1): column 0 holds the track id, columns
1..lags hold the squared displacement for each time lag (NaN where not
defined). ``MSDall`` has shape (n_tracks, lags+1): column 0 holds the track
id, columns 1..lags the mean squared displacement per track.
"""

from __future__ import annotations

import numpy as np

from .util import nanmean, nanstd


def calculateSD(data, lags_to_calculate, step=1):
    """Calculate squared displacements for every trajectory and time lag.

    Returns (SD, MSDall).  ``data[:, 0]`` must be the track id, ``data[:, 4:6]``
    the x/y coordinates.
    """
    data = np.asarray(data, dtype=float)
    n_data = data.shape[0]
    lags = int(lags_to_calculate)

    SD = np.full((n_data, lags + 1), np.nan)
    SD[:, 0] = data[:, 0]

    track_list = np.unique(data[:, 0])
    MSDall_rows = []
    for tr in track_list:
        idx = np.where(data[:, 0] == tr)[0]
        track_start = idx[0]
        track_end = idx[-1]
        for dt in range(1, lags + 1):
            # dt steps (step is always 1 in TrackArt usage)
            start2 = track_start + dt
            end2 = track_end
            start1 = track_start
            end1 = track_end - dt
            if start2 <= end2:
                delta = data[start2:end2 + 1, 4:6] - data[start1:end1 + 1, 4:6]
                sd_vals = np.sum(delta ** 2, axis=1)
                if dt <= (track_end - track_start):
                    SD[track_start + dt:track_end + 1, dt] = sd_vals
            if dt - 1 < (track_end - track_start):
                SD[track_start:track_start + dt, dt] = np.nan
            else:
                SD[track_start:track_end + 1, dt] = np.nan

        # per-track MSD row
        row = np.full(lags + 1, np.nan)
        row[0] = tr
        for dt in range(1, lags + 1):
            col = SD[SD[:, 0] == tr, dt * step]
            row[dt] = nanmean(col)
        MSDall_rows.append(row)

    MSDall = np.array(MSDall_rows) if MSDall_rows else np.zeros((0, lags + 1))
    return SD, MSDall


def calculateMSD(SD, tau, step=1):
    """Calculate the overall (ensemble averaged) MSD from per-trajectory SD.

    Returns an array of shape (lags, 4)::

        column 1  time [s]
        column 2  mean MSD
        column 3  std of MSD
        column 4  K = number of squared displacements contributing to the lag

    NOTE: the original MATLAB code computes ``K = sum(count - dt)`` which
    yields *negative* contributions for trajectories shorter than the lag.
    We use the corrected ``sum(max(count - dt, 0))`` (the actual number of
    squared displacements), which is what the formulas assume.
    """
    SD = np.asarray(SD, dtype=float)
    n_cols = SD.shape[1]
    lags = (n_cols - 1) // step
    count = np.bincount(SD[:, 0].astype(int))
    count = count[count > 0].astype(float)

    MSD = np.zeros((lags, 4))
    MSD[:, 0] = tau * np.arange(1, lags + 1)
    for dt in range(1, lags + 1):
        col = SD[:, dt * step]
        MSD[dt - 1, 1] = nanmean(col)
        MSD[dt - 1, 2] = nanstd(col)
        MSD[dt - 1, 3] = np.sum(np.maximum(count - dt, 0))
    return MSD
