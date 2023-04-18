# Energy levels of 3D rectangular potential
# E = K [(n/a)^2 + (m/b)^2 + (p/c)^2]
# K = (pi hbar / (2 M))^2

# Mean level density
# rho = M / (pi^2 hbar^3) sqrt(M / 2) sqrt(E)

import numpy as np

class Box3D:
    def __init__(this, a, b, c, hbar=1, M=1):
        this.a = a
        this.b = b
        this.c = c
        this.hbar = hbar
        this.M = M

        this.K = (np.pi * this.hbar)**2 / (2 * this.M)
        this.spectrum = np.array([])

    def volume(this):
        return this.a * this.b * this.c

    def __str__(this):
        return f"3D box (a, b, c) = ({this.a}, {this.b}, {this.c}), {len(this.spectrum)} levels."

    def level_density(this, e):
        return this.volume() * this.M / (np.pi**2 * this.hbar**3) * np.sqrt(this.M * e / 2)

    def cummulative_level_density(this, e):
        return this.volume() * this.M / (np.pi**2 * this.hbar**3) * np.sqrt(2 * this.M * e**3)

    def energy_level(this, n, m, p):
        return this.K * ((n / this.a)**2 + (m / this.b)**2 + (p / this.c)**2)

    def calculate_spectrum(this, num_states):
        """ Calculates the lowest num_states energy levels """
        num_states = int(num_states)

        max_energy = 1.1 * this.hbar**2 / this.M * (3 * np.pi**2 * num_states / (this.volume() * np.sqrt(2)))**(2/3)
                                                    # 1.1 is a Pišvejc (Bulgarian) constant 
                                                    # Estimate of the maximum energy by solving N(E) = num_states

        p = np.sqrt(2 * this.M * max_energy) / (np.pi * this.hbar)
        maxn = int(p * this.a) + 1
        maxm = int(p * this.b) + 1
        maxp = int(p * this.c) + 1

        print(f"MaxEnergy = {max_energy}, MaxIndices = ({maxn}, {maxm}, {maxp})")

        this.spectrum = []
        for n in range(1, maxn):
            for m in range(1, maxm):
                for p in range(1, maxp):
                    energy = this.energy_level(n, m, p)
                    if energy <= max_energy:
                        this.spectrum.append(energy)
                    else:
                        break

        print(f"Final number of calculated states: {len(this.spectrum)}")
        this.spectrum = np.array(sorted(this.spectrum)[0:num_states])
        
        return this.spectrum

