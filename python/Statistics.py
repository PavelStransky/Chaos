import numpy as np
import matplotlib.pyplot as plt


def strech_spectrum(spectrum):
    """ Stretch the spectrum to get an exact average level density 1 """
    spectrum = (spectrum - spectrum[0]) / (spectrum[-1] - spectrum[0]) * (len(spectrum) - 1)
    print(f"Range of streched energies: {spectrum[0]} - {spectrum[-1]}")
    return spectrum


def level_spacing(spectrum, shift=1):
    """ Calculates the spacing between nearest levels.

        Parameters:
        shift: 1 for the nearest neigbour spacing, 2 for the next-to-nearest neigbour spacing etc (default: 1)
     """
    result = np.roll(spectrum, -shift) - spectrum
    return result[:-shift]

def poisson(s):
    """ Poisson NNSD """
    return np.exp(-s)


def wigner(s):
    """ Wigner NNSD """
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s**2)


def polynomial_unfolding(spectrum, degree):
    p = np.polynomial.Polynomial.fit(spectrum, range(len(spectrum)), degree)

    plt.plot(spectrum, range(len(spectrum)), label="Data")
    plt.plot(*p.linspace(), label=f"Polynomial of degree {degree}")
    plt.title("Polynomial Unfolding")
    plt.legend()
    plt.show()

    return p(spectrum)

