import numpy as np
import matplotlib.pyplot as plt

import rmt_statistics

def level_density(spectrum, smooth_level_density=None, bins=50, title="", file=None):
    """ Plots a graph of the level density (histogram).

        Parameters:
        spectrum (array): energy levels
        smooth_level_density (function): function that calculates the smooth part of the level density (default: None)
        bins (int): number of bins (default: 50)
        title (str): title of the plot (default: "")
    """

    if spectrum.ndim > 1:
        spectrum = spectrum.flatten()

    # Histogram bins normalization (to get an approximation) of the real level density    
    weight = bins / (spectrum[-1] - spectrum[0])
    weights = np.linspace(weight, weight, len(spectrum))

    plt.hist(spectrum, bins=bins, rwidth=0.8, weights=weights, label=f"Histogram ({len(spectrum)} levels, {bins} bins)")

    if smooth_level_density != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, smooth_level_density(x), label="Smooth part (formula)")
        plt.legend()

    plt.title("Level Density " + title)
    plt.xlabel("$E$")
    plt.ylabel(r"$\rho(E)$")

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


def cummulative_level_density(spectrum, smooth_cummulative_level_density=None, title="", file=None):
    """ Plots a graph of the cummulative level density

        Parameters:
        spectrum (array): energy levels
        smooth_cummulative_level_density (function): function that calculates the smooth part of the cummulative level density (default: None)
        title (str): title of the plot (default: "")
      
    """

    plt.plot(spectrum, range(len(spectrum)), label="Step function from the data")

    if smooth_cummulative_level_density != None:
        x = np.linspace(spectrum[0], spectrum[-1], 100)
        plt.plot(x, smooth_cummulative_level_density(x), label="Smooth part (formula)")
        plt.legend()

    plt.title("Cummulative LD " + title)
    plt.xlabel("$E$")
    plt.ylabel(r"$\rho(E)$")

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


def nnsd(spacings, theory=rmt_statistics.wigner, bins=50, title="", file=None):
    """ Plots a graph of the nearest neigbour spacing distribution (histogram)

        Parameters:
        spacings (array): spacings between unfolded neighbouring energy levels
        theory (function): theoretical distribution (default: wigner)
        bins (int): number of bins (default: 50)
        title (str): title of the plot (default: "")
    """
    plt.hist(spacings, density=True, bins=bins, range=(0,5), rwidth=0.8, label="Numerical NNSD")

    x = np.linspace(0,5,100)
    
    plt.plot(x, theory(x), label=theory.__name__)
    
    plt.xlabel("$s$")
    plt.ylabel("$p(s)$")
    plt.title("NNSD " + title)
    plt.legend()
    plt.ylim(0)         # Show the horizontal axis
    plt.xlim(0, 5)

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

def number_variance(sigma2, theory=rmt_statistics.goe_number_variance, title="", file=None):
    """ Plots a graph of the number variance

        Parameters:
        sigma2 (array): calculated number variance
        theory (function): theoretical distribution (default: wigner_number_variance)
        bins (int): number of bins (default: 50)
        title (str): title of the plot (default: "")
    """
    plt.plot(sigma2, label="Number variance")
    x = np.linspace(1, len(sigma2), 100)
    
    plt.plot(x, theory(x), label=theory.__name__)
    
    plt.xlabel("$L$")
    plt.ylabel("$\\Sigma^2(L)$")
    plt.title("Number variance " + title)
    plt.legend()
    plt.ylim(0)         # Show the horizontal axis

    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()
