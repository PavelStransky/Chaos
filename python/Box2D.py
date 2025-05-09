# Energy levels of the 2D rectangular potential
# E = K [(n/a)^2 + (m/b)^2]

# Mean level density (given by the the Weyl formula)
# rho = (2 a b M) / (pi hbar^2)

import numpy as np

import rmt_statistics, plots

def dimensional_parameter(hbar=1, M=1):
    """ Calculates the dimensional parameter (pi hbar)^2 / (2 M))
        
        Parameters:
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return (np.pi * hbar)**2 / (2 * M)


def level_density(e, a=1, b=np.sqrt(np.pi / 3), hbar=1, M=1):
    """ Calculates the mean level density of the rectangular 2D infinite potential well.
        
        Parameters:
        e (float): energy
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return a * b * M / (2 * np.pi * hbar**2) * e**0

def cummulative_level_density(e, a=1, b=np.sqrt(np.pi / 3), hbar=1, M=1):
    """ Calculates the cummulative level density of the rectangular 2D infinite potential well.
        
        Parameters:
        e (float): energy
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return a * b * M / (2 * np.pi * hbar**2) * e

def spectrum(num_states, a=1, b=np.sqrt(np.pi / 3), hbar=1, M=1):
    """ Calculates the lowest num_states energy levels of the rectangular 2D infinite potential well.
        
        Parameters:
        num_states (int): number of energy levels to calculate (starting from the ground state)
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """

    # 1.1 is a Pišvejc (Bulgarian) constant 
    # Estimate of the maximum energy by solving N(E) = num_states       
    max_energy = 1.1 * num_states / (a * b * M / (2 * np.pi * hbar**2))

    K = dimensional_parameter(hbar, M)

    p = np.sqrt(max_energy / K)

    # Estimate of the maximum values of the quantum numbers
    maxn = int(p * a) + 1
    maxm = int(p * b) + 1

    print(f"Maximum energy = {max_energy}, maximum indices = ({maxn}, {maxm})")

    spectrum = []

    for n in range(1, maxn):
        for m in range(1, maxm):
            energy = K * ((n / a)**2 + (m / b)**2)
            if energy <= max_energy:
                spectrum.append(energy)
            else:
                break

    print(f"Final number of calculated states: {len(spectrum)}")
    
    spectrum = np.array(sorted(spectrum)[0:num_states])

    print(f"Range of energies: ({min(spectrum)}, {max(spectrum)})")

    return spectrum

def demonstrate(num_states=100000, a=1, b=np.sqrt(np.pi / 3)):
    energies = spectrum(num_states, a, b)

    unfolded = cummulative_level_density(energies, a, b)
    spacings = rmt_statistics.level_spacing(unfolded)

    title = f"2D box (a, b) = ({a}, {b})"

    plots.level_density(energies, lambda x: level_density(x, a, b), title=title)
    plots.cummulative_level_density(energies, lambda x: cummulative_level_density(x, a, b), title=title)
    plots.nnsd(spacings, rmt_statistics.poisson, title=title)

if __name__ == "__main__":
    demonstrate()
    demonstrate(a=1, b=7/4)