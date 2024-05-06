import matplotlib.pyplot as plt
import numpy as np

import box2D, box3D, goe
import statistics

# from Box3D import Box3D
# from GOE import GOE
# import Stadium

import statistics
import plots

def NNSD2D(num_states=100000, a=1, b=np.sqrt(np.pi / 3)):
    spectrum = box2D.spectrum(num_states, a, b)

    unfolded = box2D.cummulative_level_density(spectrum, a, b)
    spacings = statistics.level_spacing(unfolded)

    title = f"2D box (a, b) = ({a}, {b})"

    plots.level_density(spectrum, lambda x: box2D.level_density(x, a, b), title=title)
    plots.cummulative_level_density(spectrum, lambda x: box2D.cummulative_level_density(x, a, b), title=title)
    plots.nnsd(spacings, statistics.poisson, title=title)


def NNSD3D(num_states=100000, a=1, b=np.sqrt(np.pi / 3), c = np.sqrt(np.exp(3) / 20), polynomial_order=8):
    spectrum = box3D.spectrum(num_states)

    title = f"3D box (a, b, c) = ({a}, {b}, {c})"

    plots.level_density(spectrum, lambda x: box3D.level_density(x, a, b, c), title=title)
    plots.cummulative_level_density(spectrum, lambda x: box3D.cummulative_level_density(x, a, b, c), title=title)

    # Unfolding using the Weyl formula
    unfolded = box3D.cummulative_level_density(spectrum, a, b, c)
    plots.level_density(unfolded, title="Unfolded (exact LD)" + title)

    spacings = statistics.level_spacing(unfolded)
    plots.nnsd(spacings, statistics.poisson, title=title)

    # Polynomial unfolding
    unfolded = statistics.polynomial_unfolding(spectrum, polynomial_order)
    plots.level_density(unfolded, title=f"Unfolded ({polynomial_order} order polynomial)" + title)

    spacings = statistics.level_spacing(unfolded)
    plots.nnsd(spacings, statistics.poisson, title=title)


def NNSDGOE(size=1000):
    spectrum = goe.calculate_spectrum(size)

    title = f"GOE (size = {size})"

    plots.level_density(spectrum, title=title)

    for polynomial_order in range(1, 20, 2):
        unfolded = statistics.polynomial_unfolding(spectrum, polynomial_order)
        plots.level_density(unfolded, title=f"Unfolded ({polynomial_order} order polynomial)" + title)

        spacings = statistics.level_spacing(unfolded)
        plots.nnsd(spacings, statistics.wigner, title=title)


def NNSDStadium(size=1000, polynomial_order=7):
    spectrum = Stadium.stadium_energies(Ny=100, Ne=size, full=False)
    title = "Stadium"

    plot_level_density(spectrum, extraTitle=title)

    unfolded = polynomial_unfolding(spectrum, polynomial_order)
    plot_level_density(unfolded, extraTitle=f"Unfolded ({polynomial_order} order polynomial)" + title)

    spacings = level_spacing(unfolded)
    plot_nnsd(spacings, wigner, extraTitle=title)

if __name__ == "__main__":
    NNSD2D()
    NNSD2D(a=1, b=7/4)
    NNSD3D()
    NNSDGOE(size=10000)

    # NNSDStadium(size=500)