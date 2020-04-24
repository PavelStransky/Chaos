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

    def V(this):
        return this.a * this.b

    def __str__(this):
        return f"(a, b) = ({this.a}, {this.b})"


def LevelDensity(parameters, E=0):
    return parameters.V() * parameters.M / (np.pi * parameters.hbar**2) * E**0


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

    rho = LevelDensity(parameters)                  # Level density (to estimate maximum energy)
    maxEnergy = 1.1 * numStates / rho               # 1.5 is a Pišvejc (Bulgarian) constant 
                                                    # (to be on the safe side and have always slightly more levels than necessary)
    K = (np.pi * parameters.hbar / (2 * parameters.M))**2

    def Energy(n, m):
        return K * ((n / parameters.a)**2 + (m / parameters.b)**2)

    p = np.sqrt(2 * parameters.M * maxEnergy) / (np.pi * parameters.hbar)
    maxn = int(p * parameters.a) + 1
    maxm = int(p * parameters.b) + 1

    print(f"MaxEnergy = {maxEnergy}, MaxIndices = ({maxn}, {maxm})")

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



