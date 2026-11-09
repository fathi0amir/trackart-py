"""End-to-end GUI test using the shipped TestData .xls files."""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("TkAgg")

DATA_DIR = os.environ.get(
    "TRACKART_TEST_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "TestData",
                 "SLB-on-mica_DOPC-SpM-ATTO677N", "25C"))


def main():
    from trackart.gui.app import TrackArtApp

    app = TrackArtApp()
    app.update()
    app._show_module("Import")
    app.update()

    imp = app.tabs["Import"]
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xls")))[:2]
    print("importing:", [os.path.basename(f) for f in files])

    # bypass the file dialog: fill state directly like on_select_files does
    from trackart.core.import_data import import_trackImport
    coords = import_trackImport(files)
    imp.state.import_data.coordinates = coords
    imp.state.import_data.files = files
    imp.on_calculate()
    app.update()
    st = imp.state.import_data
    print("after calculate: tracks =", len(st.Dall),
          "D median =", round(float(np.nanmedian(st.Dall[:, 1])), 3))
    assert len(st.Dall) > 100

    # exclusion with the tutorial criteria
    imp.e_min_frames.delete(0, "end"); imp.e_min_frames.insert(0, "30")
    imp.e_minD.delete(0, "end"); imp.e_minD.insert(0, "0.05")
    imp.e_maxD.delete(0, "end"); imp.e_maxD.insert(0, "15")
    imp.e_minR.delete(0, "end"); imp.e_minR.insert(0, "0.9")
    imp.e_min_int.delete(0, "end"); imp.e_min_int.insert(0, "0.5")
    imp.e_max_int.delete(0, "end"); imp.e_max_int.insert(0, "1.5")
    imp.on_exclude()
    app.update()
    print("after exclude: filtered tracks =", len(st.DallFiltered))
    assert len(st.DallFiltered) < len(st.Dall)

    # MSD on imported data
    app._show_module("MSD")
    app.update()
    msd = app.tabs["MSD"]
    msd.var_data.set("import")
    msd.on_fit()
    app.update()
    print("MSD on imported data OK, D =",
          round(app.state.MSD.results["D"], 4))

    app.destroy()
    print("IMPORT GUI SMOKE TEST PASSED")


if __name__ == "__main__":
    import numpy as np
    main()
