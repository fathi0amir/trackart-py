import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trackart.core.import_data import (import_calculateDall,
                                       import_trackImport, import_tracksExcl)
from trackart.core.coordinates import calculateMSD, calculateSD

DATA_DIR = os.environ.get(
    "TRACKART_TEST_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "TestData",
                 "SLB-on-mica_DOPC-SpM-ATTO677N", "25C"))


def test_xls_import():
    files = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR))][:3]
    coords = import_trackImport(files)
    print(f"[import] files={len(files)} rows={coords.shape} "
          f"tracks={len(np.unique(coords[:, 0]))}")
    assert coords.shape[1] == 13
    # dataset id = file index (1 for the first file, like the original)
    assert sorted(np.unique(coords[:, 1])) == [1, 2, 3]
    # track ids must be unique across files
    assert len(np.unique(coords[:, 0])) == len(np.unique(coords[:, 0]))
    assert len(np.unique(coords[:, 0])) >= 1000
    print("  x range:", coords[:, 4].min(), "-", coords[:, 4].max())
    print("  y range:", coords[:, 5].min(), "-", coords[:, 5].max())
    return coords


def test_calculate_pipeline():
    files = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR))][:3]
    coords = import_trackImport(files)

    # uncalibrated: multiply by pxSize
    px_size = 0.16
    data = coords.copy()
    data[:, 4:6] = data[:, 4:6] * px_size

    # drop tracks <= 10 frames (like the original)
    from collections import Counter
    counts = Counter(data[:, 0])
    keep = np.array([t for t, c in counts.items() if c > 10])
    data = data[np.isin(data[:, 0], keep)]

    # mean intensity per track into column 7 (TrackArt stores it there)
    for tr in np.unique(data[:, 0]):
        data[data[:, 0] == tr, 7] = np.mean(data[data[:, 0] == tr, 7])

    lags = 5
    SD, MSDall = calculateSD(data, lags, 1)
    MSD = calculateMSD(SD, 0.01)
    print(f"[pipeline] tracks={len(np.unique(data[:, 0]))} "
          f"MSD lag1={MSD[0, 1]:.4f} um^2")

    # Dall
    tracks = np.zeros((len(np.unique(data[:, 0])), 4))
    for k, tr in enumerate(np.unique(data[:, 0])):
        idx = data[:, 0] == tr
        tracks[k, 0] = tr
        tracks[k, 1] = np.sum(idx)
        tracks[k, 2] = np.mean(data[idx, 7])
        tracks[k, 3] = 1
    Dall = import_calculateDall(MSDall, tracks, lags, 0.01)
    print(f"[Dall] n={len(Dall)} D median={np.nanmedian(Dall[:, 1]):.3f} "
          f"R2 median={np.nanmedian(Dall[:, 2]):.3f}")

    # exclusion
    tracks_excl = import_tracksExcl(Dall, 30, 0.05, 15, 0.9, 0.5, 1.5)
    print(f"[excl] kept {len(tracks_excl)} of {len(Dall)} tracks")


if __name__ == "__main__":
    test_xls_import()
    test_calculate_pipeline()
    print("IMPORT TESTS PASSED")
