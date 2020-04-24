import numpy as np
import matplotlib.pyplot as plt


def StretchSpectrum(spectrum):
    """ Norm the spectrum to get whole average level density 1 
    """
    return (spectrum - spectrum[0]) / (spectrum[-1] - spectrum[0]) * (len(spectrum) - 1)


def LevelSpacing(spectrum, shift=1):
    """ Calculates the spacing between nearest levels.
    
        Arguments:
        shift -- 1 for the nearest neigbour spacing, 2 for the next-to-nearest neigbour spacing etc.
     """
    result = np.roll(spectrum, -shift) - spectrum
    return result[:-shift]


def PlotLevelDensity(spectrum, ExactLevelDensity=None, bins=50, extraTitle=""):
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


def PlotCummulativeLevelDensity(spectrum, ExactLevelDensity=None, extraTitle=""):
    """ Plots a graph of the cummulative level density (histogram) and theoretical prediciton (if available) """
    plt.plot(spectrum, range(len(spectrum)), label="Data")

    if ExactLevelDensity != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, ExactLevelDensity(x), label="Smooth part (formula)")
        plt.legend()

    plt.title("Cummulative Level Density " + extraTitle)
    plt.xlabel("$E$")
    plt.ylabel(r"$\rho(E)$")

    plt.show()


def Poisson(s):
    """ Poisson NNSD """
    return np.exp(-s)


def Wigner(s):
    """ Wigner NNSD """
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s**2)


def PolynomialUnfolding(spectrum, degree):
    p = np.polynomial.Polynomial.fit(spectrum, range(len(spectrum)), degree)

    plt.plot(spectrum, range(len(spectrum)), label="Data")
    plt.plot(*p.linspace(), label=f"Polynomial of degree {degree}")
    plt.title("Polynomial Unfolding")
    plt.legend()
    plt.show()

    return p(spectrum)


def PlotNNSD(spacings, Theoretical=Wigner, bins=50, extraTitle=""):
    """ Plots a graph of the nearest neigbour spacing distribution (histogram) and theoretical prediciton (if available) """
    plt.hist(spacings, density=True, bins=bins, range=(0,5), rwidth=0.8, label="Numerical NNSD")

    x = np.linspace(0,5,100)
    plt.plot(x, Theoretical(x), label=Theoretical.__name__)
    plt.xlabel("$s$")
    plt.ylabel("$p(s)$")
    plt.title("Nearest-Neigbour Spacing Distribution " + extraTitle)
    plt.legend()
    plt.ylim(0)
    plt.show()

