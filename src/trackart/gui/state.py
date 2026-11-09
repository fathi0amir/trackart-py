"""Shared application state, mirroring the MATLAB ``handles`` structure.

The five modules (SIM, Import, MSD, CPD, FIT) exchange data through this
object exactly like the original GUI did through ``handles.SIM``,
``handles.import`` etc.
"""

from __future__ import annotations

import numpy as np


class State:
    def __init__(self):
        self.SIM = SIMState()
        self.import_data = ImportState()
        self.MSD = MSDState()
        self.CPD = CPDState()
        self.FIT = FITState()


class SIMState:
    def __init__(self):
        self.coordinates = None      # simulated coordinates (6 cols)
        self.microCoordinates = None  # before decimation
        self.domains = None          # binary domain map
        self.param = {}
        self.stats = {}
        self.SD = None
        self.MSDall = None
        self.MSD = None
        self.plot = PlotFlags()


class ImportState:
    def __init__(self):
        self.coordinates = None      # raw imported coordinates
        self.data = None             # calibrated, short tracks removed
        self.dataFiltered = None
        self.Dall = None
        self.DallFiltered = None
        self.SD = None
        self.SDfiltered = None
        self.MSD = None
        self.MSDall = None
        self.MSDallFiltered = None
        self.tracksExcl = None
        self.tracks = None
        self.param = {}
        self.stats = {}
        self.files = []
        self.plot = PlotFlags()


class MSDState:
    def __init__(self):
        self.results = None
        self.param = None
        self.plot = PlotFlags()


class CPDState:
    def __init__(self):
        self.cpd_bin = None
        self.results = None
        self.param = None
        self.showFigure = False


class FITState:
    def __init__(self):
        self.results = None
        self.param = None
        self.data = None


class PlotFlags:
    """Which plots to show (per-module flags, like the MATLAB handles)."""

    def __init__(self):
        self.showFigure = 0
        self.showDomains = 0
        self.showMicroSteps = 0
        self.showPairCorr = 0
        self.showTrajectories = 1
        # import plot selection
        self.trackLengthDistribution = 1
        self.allTrajectories = 0
        self.singleTrajectory = 0
        self.lengthVsIntensity = 0
        self.intensitiesHistogram = 0
        self.dHistogram = 0
        self.allMSD = 0
        self.xLog = 0
        self.yLog = 0
        self.scatter = 0
        self.trajectoryNumber = 1
        # MSD plot selection
        self.msd = 1
        self.msdAll = 0
        self.msdSDV = 0
        self.diffCoeffSlope = 0
