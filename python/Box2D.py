# Energy levels of 2D rectangular potential
# E = K [(n/a)^2 + (m/b)^2]
# K = (pi hbar / (2 M))^2

# Mean level density
# rho = (2 a b M) / (pi hbar^2)

import numpy as np

class Box2D:
    def __init__(this, a, b, hbar=1, M=1):
        this.a = a
        this.b = b
        this.hbar = hbar
        this.M = M

        this.K = (np.pi * this.hbar)**2 / (2 * this.M)
        this.spectrum = np.array([])

    def volume(this):
        return this.a * this.b

    def __str__(this):
        return f"2D box (a, b) = ({this.a}, {this.b}), {len(this.spectrum)} levels."

    def level_density(this, e=0):
        return this.volume() * 0.5 * this.M / (np.pi * this.hbar**2) * e**0

    def cummulative_level_density(this, e):
        return e * this.level_density()             # This is valid in the 2D Box only

    def energy_level(this, n, m):
        return this.K * ((n / this.a)**2 + (m / this.b)**2)

    def calculate_spectrum(this, num_states):
        """ Calculates the lowest num_states energy levels """
        num_states = int(num_states)

        rho = this.level_density()                  # Level density (to estimate maximum energy)
        max_energy = 1.1 * num_states / rho           # 1.1 is a Pišvejc (Bulgarian) constant 
                                                    # (to be on the safe side and have always slightly more levels than necessary)

        p = np.sqrt(2 * this.M * max_energy) / (np.pi * this.hbar)
        maxn = int(p * this.a) + 1
        maxm = int(p * this.b) + 1

        print(f"MaxEnergy = {max_energy}, MaxIndices = ({maxn}, {maxm})")

        this.spectrum = []
        for n in range(1, maxn):
            for m in range(1, maxm):
                energy = this.energy_level(n, m)
                if energy <= max_energy:
                    this.spectrum.append(energy)
                else:
                    break

        print(f"Final number of calculated states: {len(this.spectrum)}")
        this.spectrum = np.array(sorted(this.spectrum)[0:num_states])
        
        return this.spectrum