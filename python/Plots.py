import numpy as np
import matplotlib.pyplot as plt

from Statistics import wigner

def plot_level_density(spectrum, ExactLevelDensity=None, bins=50, extraTitle=""):
    """ Plots a graph of the level density (histogram) and theoretical prediciton (if available) """
    
    weight = bins / (spectrum[-1] - spectrum[0])            # Histogram bins normalization (to get an approximation) of the real level density
    weights = np.linspace(weight, weight, len(spectrum))

    plt.hist(spectrum, bins=bins, rwidth=0.8, weights=weights, label=f"Histogram ({len(spectrum)} levels, {bins} bins)")

    if ExactLevelDensity != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, ExactLevelDensity(x), label="Smooth part (formula)")
        plt.legend()

    plt.title("Level Density " + extraTitle)
    plt.xlabel("$E$")
    plt.ylabel(r"$\rho(E)$")

    plt.show()


def plot_cummulative_level_density(spectrum, ExactLevelDensity=None, extraTitle=""):
    """ Plots a graph of the cummulative level density (histogram) and theoretical prediciton (if available) """
    plt.plot(spectrum, range(len(spectrum)), label="Data")

    if ExactLevelDensity != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, ExactLevelDensity(x), label="Smooth part (formula)")
        plt.legend()

    plt.title("Cummulative LD " + extraTitle)
    plt.xlabel("$E$")
    plt.ylabel(r"$\rho(E)$")

    plt.show()


def plot_nnsd(spacings, Theoretical=wigner, bins=50, extraTitle=""):
    """ Plots a graph of the nearest neigbour spacing distribution (histogram) and theoretical prediciton (if available) """
    plt.hist(spacings, density=True, bins=bins, range=(0,5), rwidth=0.8, label="Numerical NNSD")

    x = np.linspace(0,5,100)
    plt.plot(x, Theoretical(x), label=Theoretical.__name__)
    plt.xlabel("$s$")
    plt.ylabel("$p(s)$")
    plt.title("NNSD " + extraTitle)
    plt.legend()
    plt.ylim(0)
    plt.show()
