import numpy as np
import matplotlib.pyplot as plt

generator = np.random.default_rng()

class GOE:
    def __init__(this, size, sigma=1):
        this.size = size
        this.sigma = sigma

        if this.sigma <= 0:         # Special normalization (levels in the range from -1 to +1)
            this.sigma = 0.5 / np.sqrt(this.size) 

    def __str__(this):
        return f"GOE ({this.size} x {this.size}), sigma = {this.sigma}."
    
    def generate(this):
        diagonal = generator.normal(scale=this.sigma * np.sqrt(2), size=this.size)       
        matrix = generator.normal(scale=this.sigma, size=(this.size, this.size))
        np.fill_diagonal(matrix, diagonal)

        return matrix
    
    def calculate_spectrum(this, num_matrices=1):
        """ Generates spectrum of num_matrices realizations of the GOE matrices """
        ev = []
        for _ in range(num_matrices):
            matrix = this.generate()
            spectrum = np.linalg.eigvalsh(matrix)
            ev.append(spectrum)

        spectrum = np.array(ev).flatten()
        return np.array(sorted(spectrum))