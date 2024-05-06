# Energy levels of the 3D rectangular potential
# E = K [(n/a)^2 + (m/b)^2 + (p/c)^2]

# Mean level density
# rho = a b c M / (pi^2 hbar^3) sqrt(M / 2) sqrt(E)

import numpy as np

def dimensional_parameter(hbar=1, M=1):
    """ Calculates the dimensional parameter (pi hbar)^2 / (2 M))
        
        Parameters:
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return (np.pi * hbar)**2 / (2 * M)

def level_density(e, a=1, b=np.sqrt(np.pi / 3), c=np.sqrt(np.exp(3 / 20)), hbar=1, M=1):
    """ Calculates the mean level density of the rectangular 3D infinite potential well.
        
        Parameters:
        e (float): energy
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        c (float): third dimension of the well (default: sqrt(exp(3) / 20))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return a * b * c * M / (np.pi**2 * hbar**3) * np.sqrt(M / 2) * np.sqrt(e)

def cummulative_level_density(e, a=1, b=np.sqrt(np.pi / 3), c=np.sqrt(np.exp(3 / 20)), hbar=1, M=1):
    """ Calculates the cummulative level density of the rectangular 3D infinite potential well. 
    
        Parameters:
        e (float): energy
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        c (float): third dimension of the well (default: sqrt(exp(3) / 20))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """
    return 2 * a * b * c * M / (3 * np.pi**2 * hbar**3) * np.sqrt(M / 2) * np.sqrt(e**3)

def spectrum(num_states, a=1, b=np.sqrt(np.pi / 3), c=np.sqrt(np.exp(3 / 20)), hbar=1, M=1):
    """ Calculates the lowest num_states energy levels of the rectangular 3D infinite potential well.
        
        Parameters:
        num_states (int): number of energy levels to calculate (starting from the ground state)
        a (float): width of the well (default: 1)
        b (float): height of the well (default: sqrt(pi / 3))
        c (float): third dimension of the well (default: sqrt(exp(3) / 20))
        hbar (float): reduced Planck constant (default: 1)
        M (float): mass of the particle (default: 1)
    """

    # 1.1 is a Pišvejc (Bulgarian) constant 
    # Estimate of the maximum energy by solving N(E) = num_states
    max_energy = 1.1 * (3 * np.pi**2 * hbar**3 * num_states / (a * b * c * M * np.sqrt(2 * M)))**(2/3)

    K = dimensional_parameter(hbar, M)

    p = np.sqrt(2 * max_energy / K)

    # Estimate of the maximum values of the quantum numbers
    maxn = int(p * a) + 1
    maxm = int(p * b) + 1
    maxp = int(p * c) + 1

    print(f"Maximum energy = {max_energy}, maximum indices = ({maxn}, {maxm}, {maxp})")

    spectrum = []
    for n in range(1, maxn):
        for m in range(1, maxm):
            for p in range(1, maxp):
                energy = K * ((n / a)**2 + (m / b)**2 + (p / c)**2)
                if energy <= max_energy:
                    spectrum.append(energy)
                else:
                    break

    print(f"Final number of calculated states: {len(spectrum)}")

    spectrum = np.array(sorted(spectrum)[0:num_states])
    
    print(f"Range of energies: ({min(spectrum)}, {max(spectrum)})")

    return spectrum

