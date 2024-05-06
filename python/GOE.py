import numpy as np
import matplotlib.pyplot as plt

import time

import statistics, plots

generator = np.random.default_rng()

def generate_goe(size, sigma=1):
    """ Generates a Gaussian Orthogonal Ensemble (GOE) matrix.
        
        Parameters:
        size (int): size of the matrix
        sigma (float): standard deviation of the Gaussian distribution
        (if negative, the deviation is calculated so that the levels lie in interval (-1, 1))
    """

    # Special normalization (for the levels to lie in the interval (-1, 1))
    if sigma <= 0:
        sigma = 0.5 / np.sqrt(size) 

    diagonal = generator.normal(scale=sigma * np.sqrt(2), size=size)       
    matrix = generator.normal(scale=sigma, size=(size, size))
    np.fill_diagonal(matrix, diagonal)

    return matrix

def calculate_spectrum(size, num_matrices=1, sigma=1):
    """ Generates spectrum of num_matrices realizations of the GOE matrices.

        Parameters:
        size (int): size of the matrix
        num_matrices (int): number of matrices to generate (default: 1)
        sigma (float): standard deviation of the Gaussian distribution
        (if negative, the deviation is calculated so that the levels lie in interval (-1, 1))
    """        
    
    ev = []
    for _ in range(num_matrices):
        matrix = generate_goe(size, sigma)
        spectrum = np.linalg.eigvalsh(matrix)
        ev.append(spectrum)

    spectrum = np.array(ev).flatten()
    return np.array(sorted(spectrum))


def demonstrate(size=1000):
    energies = calculate_spectrum(size)

    title = f"GOE (size = {size})"

    plots.level_density(energies, title=title)

    for polynomial_order in range(1, 20, 2):
        unfolded = statistics.polynomial_unfolding(energies, polynomial_order)
        plots.level_density(unfolded, title=f"Unfolded ({polynomial_order} order polynomial)" + title)

        spacings = statistics.level_spacing(unfolded)
        plots.nnsd(spacings, statistics.wigner, title=title)

if __name__ == "__main__":
    demonstrate(10000)