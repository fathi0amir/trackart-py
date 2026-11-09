"""Data import, calibration and trajectory exclusion.

Ports of TrackArt's ``import/`` MATLAB module:

* ``import_trackImport`` - read *.csv (TrackArt simulation export) or *.xls
  (Mosaic ImageJ Particle Tracker output, actually tab-separated text)
* ``import_calculateDall`` - per-trajectory D from a linear MSD fit
* ``import_tracksExcl``   - apply exclusion criteria
"""

from __future__ import annotations

import os

import numpy as np

from .coordinates import calculateMSD, calculateSD
from .fitting import polyfit
from .util import nanmean


def import_trackImport(files):
    """Import one or more trajectory files.

    Parameters
    ----------
    files : list[str]
        paths to the files to import (they must all be of the same type)

    Returns a ``coordinates`` array with the TrackArt layout::

        col 1  track id
        col 2  dataset id
        col 3  track number (original id within its file)
        col 4  frame
        col 5  x
        col 6  y
        col 7  z
        col 8  intensity (m0)
        col 9-13  m1..m4, NPscore
    """
    if not files:
        return None

    exts = {os.path.splitext(f)[1].lower() for f in files}
    if len(exts) != 1:
        raise ValueError("Files must be the same type or file not supported")

    ext = exts.pop()
    coordinates = None

    if ext == ".csv":
        for k, path in enumerate(files):
            tmp = np.loadtxt(path, delimiter=",", ndmin=2)
            if k == 0:
                coordinates = tmp
            else:
                tmp = tmp.copy()
                tmp[:, 1] = k + 1
                tmp[:, 0] = tmp[:, 2] + np.max(coordinates[:, 0])
                coordinates = np.vstack([coordinates, tmp])
        if coordinates is not None:
            # intensity column = 1 for simulated data
            coordinates = np.pad(coordinates, ((0, 0), (0, 13 - coordinates.shape[1])))
            coordinates[:, 7] = 1.0
            return coordinates

    elif ext in (".xls", ".xlsx", ".txt"):
        for k, path in enumerate(files):
            tmp = _read_tab_separated(path)
            if k == 0:
                coordinates = np.zeros((tmp.shape[0], 13))
                coordinates[:, 1:13] = tmp[:, :12]
                coordinates[:, 1] = 1  # dataset
                coordinates[:, 0] = coordinates[:, 2]  # track id
            else:
                tmp2 = np.zeros((tmp.shape[0], 13))
                tmp2[:, 1:13] = tmp[:, :12]
                tmp2[:, 1] = k + 1
                tmp2[:, 0] = tmp2[:, 2] + np.max(coordinates[:, 0])
                coordinates = np.vstack([coordinates, tmp2])
        return coordinates

    raise ValueError(f"Files must be the same type or file not supported ({ext})")


def _read_tab_separated(path):
    """Read an ImageJ Particle Tracker *.xls file (tab separated, 1 header
    row).  Returns the numeric data matrix."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    # skip the header row; parse everything as floats
    rows = []
    for line in lines[1:]:
        parts = line.rstrip("\r\n").split("\t")
        try:
            vals = [float(p.strip()) for p in parts if p.strip() != ""]
        except ValueError:
            continue
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    return np.array(rows)


def import_calculateDall(MSDall, tracks, lags_to_calculate, tau):
    """Calculate per-trajectory diffusion coefficients from linear MSD fits.

    Returns a (n_tracks, 5) array: [track id, D, R^2, length, mean intensity].
    """
    MSDall = np.asarray(MSDall, dtype=float)
    tracks = np.asarray(tracks, dtype=float)
    tracks = tracks[tracks[:, 3] == 1, :3]
    tracks_number = len(tracks)
    time = np.arange(1, lags_to_calculate + 1) * tau

    Dall = np.zeros((tracks_number, 5))
    for i in range(tracks_number):
        Dall[i, 0] = tracks[i, 0]
        y = MSDall[i, 1:lags_to_calculate + 1]
        p = polyfit(time[:lags_to_calculate], y, 1)
        Dall[i, 1] = p[0] / 4
        yhat = np.polyval(p, time[:lags_to_calculate])
        normr = np.linalg.norm(y - yhat)
        ss = np.sum((y - np.mean(y)) ** 2)
        Dall[i, 2] = 1 - normr ** 2 / ss if ss > 0 else np.nan
    Dall[:, 3] = tracks[:, 1]
    Dall[:, 4] = tracks[:, 2]
    return Dall


def import_tracksExcl(data, min_frames, minD, maxD, minR, min_int, max_int):
    """Return the track ids meeting the exclusion criteria.

    ``data`` columns: [track, D, R^2, length, mean intensity].
    """
    data = np.asarray(data, dtype=float)
    a = data[data[:, 3] >= min_frames, 0]
    b = data[data[:, 1] >= minD, 0]
    c = data[data[:, 1] <= maxD, 0]
    d = data[data[:, 2] >= minR, 0]
    e = data[data[:, 4] >= min_int, 0]
    f = data[data[:, 4] < max_int, 0]

    tracks_excl = np.intersect1d(np.intersect1d(np.intersect1d(a, b), np.intersect1d(c, d)), e)
    if max_int > min_int:
        tracks_excl = np.intersect1d(tracks_excl, f)
    return tracks_excl
