import numpy as np
import matplotlib.pyplot as plt

import time

generator = np.random.default_rng()

class GOE:
    def __init__(this, size, sigma=1):
        """ Constructor of the GOE class.
            Parameters:
            size (int): size of the matrix
            sigma (float): standard deviation of the Gaussian distribution 
            (if negative, the deviation is calculated so that the levels lie in interval (-1, 1))
        """
        this.size = size
        this.sigma = sigma

        if this.sigma <= 0:         # Special normalization 
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
    