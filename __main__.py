import matplotlib.pyplot as plt
import numpy as np

from Statistics import *
from Box2D import *

a = 1
b = np.sqrt(np.pi / 3)
b = 1.01

parameters = Parameters(a, b)
title = str(parameters)

spectrum = CalculateSpectrum(parameters, 100000)
unfolded = Unfolding(spectrum)
spacings = LevelSpacing(unfolded)

PlotLevelDensity(spectrum, lambda x: LevelDensity(parameters, x), extraTitle=title)
PlotCummulativeLevelDensity(spectrum, lambda x: CummulativeLevelDensity(parameters, x), extraTitle=title)

PlotNNSD(spacings, Poisson, extraTitle=title)
