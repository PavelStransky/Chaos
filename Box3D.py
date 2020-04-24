# Energy levels of 3D rectangular potential
# E = K [(n/a)^2 + (m/b)^2 + (p/c)^2]
# K = (pi hbar / (2 M))^2

# Mean level density
# rho = M / (pi^2 hbar^3) sqrt(M / 2) sqrt(E)

import numpy as np


class Parameters:
    def __init__(this, a, b, c, hbar=1, M=1):
        this.a = a
        this.b = b
        this.c = c
        this.hbar = hbar
        this.M = M

    def V(this):
        return this.a * this.b * this.c

    def __str__(this):
        return f"(a, b, c) = ({this.a}, {this.b}, {this.c})"


def LevelDensity(parameters, E):
    return parameters.V() * parameters.M / (np.pi**2 * parameters.hbar**3) * np.sqrt(parameters.M * E / 2)


def CummulativeLevelDensity(parameters, E):
    return parameters.V() * parameters.M / (3 * np.pi**2 * parameters.hbar**3) * np.sqrt(2 * parameters.M * E**3)


def CalculateSpectrum(parameters, numStates=100000):
    """ Calculates the lowest numStates energies

        Arguments:
        parameters -- a, b, c: dimensions of the box
                      M: mass of the particle
                      hbar: Planck constant
    """
    numStates = int(numStates)

    maxEnergy = 1.1 * parameters.hbar**2 / parameters.M * (3 * np.pi**2 * numStates / (parameters.V() * np.sqrt(2)))**(2/3)
                                                # 1.5 is a Pišvejc (Bulgarian) constant 
                                                # Estimate of the maximum energy by solving N(E) = numStates

    K = (np.pi * parameters.hbar)**2 / (2 * parameters.M)

    def Energy(n, m, p):
        return K * ((n / parameters.a)**2 + (m / parameters.b)**2 + (p / parameters.c)**2)

    p = np.sqrt(2 * parameters.M * maxEnergy) / (np.pi * parameters.hbar)
    maxn = int(p * parameters.a) + 1
    maxm = int(p * parameters.b) + 1
    maxp = int(p * parameters.c) + 1

    print(f"MaxEnergy = {maxEnergy}, MaxIndices = ({maxn}, {maxm}, {maxp})")

    spectrum = []
    for n in range(1, maxn):
        for m in range(1, maxm):
            for p in range(1, maxp):
                energy = Energy(n, m, p)
                if energy <= maxEnergy:
                    spectrum.append(energy)
                else:
                    break

    print(f"Final number of calculated states: {len(spectrum)}")

    spectrum = sorted(spectrum)
    
    return np.array(spectrum[0:numStates]) 


def Unfolding(parameters, spectrum):
    return CummulativeLevelDensity(parameters, spectrum)

