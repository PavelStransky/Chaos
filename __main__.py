import matplotlib.pyplot as plt
import numpy as np

import Box2D, Box3D, GOE

from Statistics import *

def NNSD2D():
    a = 1
    b = np.sqrt(np.pi / 3)

    parameters = Box2D.Parameters(a, b)
    title = "2D box\n" + str(parameters)

    spectrum = Box2D.CalculateSpectrum(parameters, 1000000)
    unfolded = StretchSpectrum(spectrum)
    spacings = LevelSpacing(unfolded)

    PlotLevelDensity(spectrum, lambda x: Box2D.LevelDensity(parameters, x), extraTitle=title)
    PlotCummulativeLevelDensity(spectrum, lambda x: Box2D.CummulativeLevelDensity(parameters, x), extraTitle=title)

    PlotNNSD(spacings, Poisson, extraTitle=title)

def NNSD3D():
    a = 1
    b = np.sqrt(np.pi / 3)
    c = np.sqrt(np.exp(3) / 20)

    parameters = Box3D.Parameters(a, b, c)
    title = "3D box\n" + str(parameters)

    spectrum = Box3D.CalculateSpectrum(parameters, 100000)
    PlotLevelDensity(spectrum, lambda x: Box3D.LevelDensity(parameters, x), extraTitle=title)
    PlotCummulativeLevelDensity(spectrum, lambda x: Box3D.CummulativeLevelDensity(parameters, x), extraTitle=title)
    PlotCummulativeLevelDensity(spectrum, lambda x: Box3D.CummulativeLevelDensity(parameters, x) * len(spectrum) / Box3D.CummulativeLevelDensity(parameters, spectrum[-1]), extraTitle="Rescaled" + title)

    unfolded = Box3D.Unfolding(parameters, spectrum)
    PlotLevelDensity(unfolded, extraTitle="Unfolded" + title)

    unfolded = StretchSpectrum(unfolded)
    PlotLevelDensity(unfolded, extraTitle="Unfolded + Stretched" + title)

    spacings = LevelSpacing(unfolded)

    PlotNNSD(spacings, Poisson, extraTitle=title)

    # Polynomial unfolding
    unfolded = PolynomialUnfolding(spectrum, 30)
    PlotLevelDensity(unfolded, extraTitle="Unfolded" + title)

    unfolded = StretchSpectrum(unfolded)
    PlotLevelDensity(unfolded, extraTitle="Unfolded + Stretched" + title)

    spacings = LevelSpacing(unfolded)
    PlotNNSD(spacings, Poisson, extraTitle=title)

def NNSDGOE():
    N = 1000
    title = f"GOE N = {N}"

    spectrum = GOE.Ensemble(1000, 1, sigma=1)
    PlotLevelDensity(spectrum, extraTitle=title)

    unfolded = PolynomialUnfolding(spectrum, 11)
    PlotLevelDensity(unfolded, extraTitle="Unfolded" + title)

    unfolded = StretchSpectrum(unfolded)
    PlotLevelDensity(unfolded, extraTitle="Unfolded + Stretched" + title)

    spacings = LevelSpacing(unfolded)
    PlotNNSD(spacings, Wigner, extraTitle=title)

NNSDGOE()