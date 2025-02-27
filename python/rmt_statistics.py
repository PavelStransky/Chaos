import numpy as np
import matplotlib.pyplot as plt


def strech_spectrum(spectrum):
    """ Stretch the spectrum to get an exact average level density 1 """
    if spectrum.ndim == 1:
        return (spectrum - spectrum[0]) / (spectrum[-1] - spectrum[0]) * (len(spectrum) + 0)
    
    result = []
    for ev in spectrum:
        result.append((ev - ev[0]) / (ev[-1] - ev[0]) * (len(ev) + 1))

    return np.array(result)


def level_spacing(spectrum, shift=1):
    """ Calculates the spacing between nearest levels.

        Parameters:
        shift: 1 for the nearest neigbour spacing, 2 for the next-to-nearest neigbour spacing etc (default: 1)
     """

    if spectrum.ndim == 1:
        result = np.roll(spectrum, -shift) - spectrum
        return result[:-shift]

    result = []
    for ev in spectrum:
        r = np.roll(ev, -shift) - ev
        result.append(r[:-shift])

    return np.array(result).flatten()

def poisson(s):
    """ Poisson NNSD """
    return np.exp(-s)


def wigner(s):
    """ Wigner NNSD """
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s**2)

def goe_number_variance(l):
    """ GOE number variance """
    return 2 / np.pi**2 * (np.log(2 * np.pi * l) + 0.5772156649 - np.pi**2 / 8 + 1)

def polynomial_unfolding(spectrum, degree, figure=True):
    all_states = sorted(spectrum.flatten())
    p = np.polynomial.Polynomial.fit(all_states, range(len(all_states)), degree)

    if figure:
        plt.plot(all_states, range(len(all_states)), label="Data")
        plt.plot(*p.linspace(), label=f"Polynomial of degree {degree}")
        plt.title("Polynomial Unfolding")
        plt.legend()
        plt.show()

    if spectrum.ndim == 1:
        return p(spectrum)
    
    result = []
    for ev in spectrum:
        result.append(p(ev) / len(spectrum))

    return np.array(result)

def number_variance(spectrum, i, max_l=100):
    result = []

    for l in range(1, max_l):        
        v = []
        for ev in spectrum:
            v.append((ev[i+l] - ev[i] - l)**2)

        result.append(np.mean(v))

    return np.array(result)

def cut_edges(spectrum, num_states):
    if spectrum.ndim == 1:
        return spectrum[num_states:-num_states]
    
    result = []
    for ev in spectrum:
        result.append(ev[num_states:-num_states])
    
    return np.array(result)