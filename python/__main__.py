import matplotlib.pyplot as plt
import numpy as np

from Box2D import Box2D
from Box3D import Box3D
from GOE import GOE
import Stadium

from Statistics import *
from Plots import *

def NNSD2D(a=1, b=np.sqrt(np.pi / 3), num_states=100000):
    box2D = Box2D(a, b)
    spectrum = box2D.calculate_spectrum(num_states)
    title = str(box2D)

    unfolded = strech_spectrum(spectrum)
    spacings = level_spacing(unfolded)

    plot_level_density(spectrum, box2D.level_density, extraTitle=title)
    plot_cummulative_level_density(spectrum, box2D.cummulative_level_density, extraTitle=title)

    plot_nnsd(spacings, poisson, extraTitle=title)


def NNSD3D(a=1, b=np.sqrt(np.pi / 3), c = np.sqrt(np.exp(3) / 20), num_states=100000, polynomial_order=8):
    box3D = Box3D(a, b, c)
    spectrum = box3D.calculate_spectrum(num_states)
    title = str(box3D)

    plot_level_density(spectrum, box3D.level_density, extraTitle=title)
    plot_cummulative_level_density(spectrum, box3D.cummulative_level_density, extraTitle=title)
    plot_cummulative_level_density(spectrum, lambda x: box3D.cummulative_level_density(x) * len(spectrum) / box3D.cummulative_level_density(spectrum[-1]), extraTitle="Rescaled" + title)

    unfolded = box3D.cummulative_level_density(spectrum)
    plot_level_density(unfolded, extraTitle="Unfolded (exact LD)" + title)

    unfolded = strech_spectrum(unfolded)
    plot_level_density(unfolded, extraTitle="Unfolded (exact LD) + Stretched" + title)

    spacings = level_spacing(unfolded)

    plot_nnsd(spacings, poisson, extraTitle=title)

    # Polynomial unfolding
    unfolded = polynomial_unfolding(spectrum, polynomial_order)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial)" + title)

    unfolded = strech_spectrum(unfolded)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial) + Stretched" + title)

    spacings = level_spacing(unfolded)
    plot_nnsd(spacings, poisson, extraTitle=title)


def NNSDGOE(size=1000, polynomial_order=11):
    goe = GOE(size, -1)
    title = str(goe)

    spectrum = goe.calculate_spectrum()
    plot_level_density(spectrum, extraTitle=title)

    unfolded = polynomial_unfolding(spectrum, polynomial_order)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial)" + title)

    unfolded = strech_spectrum(unfolded)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial) + Stretched" + title)

    spacings = level_spacing(unfolded)
    plot_nnsd(spacings, wigner, extraTitle=title)

def NNSDStadium(size=1000, polynomial_order=7):
    spectrum = _Stadium.stadium_energies(Ne=size, full=False)
    title = "Stadium"

    plot_level_density(spectrum, extraTitle=title)

    unfolded = polynomial_unfolding(spectrum, polynomial_order)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial)" + title)

    spacings = level_spacing(unfolded)
    plot_nnsd(spacings, wigner, extraTitle=title)


#NNSD2D()
#NNSD2D(a=1, b=1.5)
#NNSD3D()
#NNSDGOE()

NNSDStadium(size=1000)