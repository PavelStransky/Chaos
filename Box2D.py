# Energy levels of 2D rectangular potential
# E = K [(n/a)^2 + (m/b)^2]
# K = (pi hbar / (2 M))^2

# Mean level density
# rho = (2 a b M) / (pi hbar^2)

import numpy as np

class Parameters:
    def __init__(this, a, b, hbar=1, M=1):
        this.a = a
        this.b = b
        this.hbar = hbar
        this.M = M

    def __str__(this):
        return f"(a, b) = ({this.a}, {this.b})"

def LevelDensity(parameters, E=0):
    return parameters.a * parameters.b * parameters.M / (np.pi * parameters.hbar**2) * E**0


def CummulativeLevelDensity(parameters, E):
    return E * LevelDensity(parameters, E)          # This is valid in the 2D Box only


def CalculateSpectrum(parameters, numStates=100000):
    """ Calculates the lowest numStates energies

        Arguments:
        parameters -- a, b: dimensions of the box
                      M: mass of the particle
                      hbar: Planck constant
    """
    numStates = int(numStates)

    K = (np.pi * parameters.hbar / (2 * parameters.M))**2

    rho = LevelDensity(parameters)                  # Level density (to estimate maximum energy)
    maxEnergy = 1.5 * numStates / rho               # 1.1 is a Pišvejc (Bulgarian) constant 
                                                    # (to be on the safe side and have always slightly more levels than necessary)
    def Energy(n, m):
        return K * ((n / parameters.a)**2 + (m / parameters.b)**2)

    spectrum = []
    for n in range(1, numStates):
        for m in range(1, numStates):
            energy = Energy(n, m)
            if energy <= maxEnergy:
                spectrum.append(energy)
            else:
                break

    print(f"Final number of calculated states: {len(spectrum)}")

    spectrum = sorted(spectrum)
    
    return np.array(spectrum[0:numStates])          # We return just the desired number of states


def Unfolding(spectrum):
    """ Norm the spectrum to get level density 1 
        (2D box has a constant level density, so the unfolding is a trivial stretching of the original levels)
    """
    return (spectrum - spectrum[0]) / (spectrum[-1] - spectrum[0]) * (len(spectrum) - 1)
