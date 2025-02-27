import numpy as np
import matplotlib.pyplot as plt

import time

import rmt_statistics, plots

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

    matrix = generator.normal(scale=sigma, size=(size, size))

    return (matrix + matrix.T) / 2

def calculate_spectrum(size, num_matrices=1, sigma=1):
    """ Generates spectrum of num_matrices realizations of the GOE matrices.

        Parameters:
        size (int): size of the matrix
        num_matrices (int): number of matrices to generate (default: 1)
        sigma (float): standard deviation of the Gaussian distribution
        (if negative, the deviation is calculated so that the levels lie in interval (-1, 1))
    """        
    
    spectrum = []
    for i in range(num_matrices):
        print(i)
        matrix = generate_goe(size, sigma)
        spectrum.append(np.linalg.eigvalsh(matrix))
    
    if num_matrices == 1:
        spectrum = spectrum[0]
    else:
        spectrum = np.array(spectrum)

    return spectrum


def demonstrate(size=1000, num_matrices=1):
    spectrum = calculate_spectrum(size, num_matrices)
    # spectrum = rmt_statistics.cut_edges(spectrum, size // 5)

    title = f"GOE (size={size}, samples={num_matrices})"

    #plt.figure(figsize=(12, 8))
    plt.rcParams["figure.figsize"] = (12,8)

    plots.level_density(spectrum, title=title)

    path=f"d:/results/number_variance/"

    for polynomial_order in [3,5,7,9,11,21,101]:
        file = f"{size}_{num_matrices}_{polynomial_order}_"
        title_polynomial = title + f"Unfolded ({polynomial_order} order polynomial)"

        def unfolding(spectrum):
            # unfolded = rmt_statistics.polynomial_unfolding(spectrum, polynomial_order, figure=False)
            unfolded = []
            for s in spectrum:
                unfolded.append(rmt_statistics.polynomial_unfolding(s, polynomial_order, figure=False))
            unfolded = np.array(unfolded)
            return unfolded

        unfolded = unfolding(spectrum)
        plots.level_density(unfolded, title=title_polynomial, file=path + file + "ld.png")

        spacings = rmt_statistics.level_spacing(unfolded)
        plots.nnsd(spacings, rmt_statistics.wigner, title=title_polynomial, file=path + file + "nnsd.png")

        sigma2 = rmt_statistics.number_variance(unfolded, size // 5, max_l=size // 5)
        plots.number_variance(sigma2, rmt_statistics.goe_number_variance, title=title_polynomial, file=path + file + "nv.png")

        ############
        unfolded = unfolding(rmt_statistics.cut_edges(spectrum, size // 5))
        file = file + "cut_"
        title_polynomial = title_polynomial + ", cut"
        plots.level_density(unfolded, title=title_polynomial, file=path + file + "ld.png")

        spacings = rmt_statistics.level_spacing(unfolded)
        plots.nnsd(spacings, rmt_statistics.wigner, title=title_polynomial, file=path + file + "nnsd.png")

        sigma2 = rmt_statistics.number_variance(unfolded, size // 5, max_l=size // 5)
        plots.number_variance(sigma2, rmt_statistics.goe_number_variance, title=title_polynomial, file=path + file + "nv.png")

        ############
        unfolded = rmt_statistics.strech_spectrum(unfolded)        
        file = file + "stretch_"
        title_polynomial = title_polynomial + ", stretch"
        plots.level_density(unfolded, title=title_polynomial, file=path + file + "ld.png")

        spacings = rmt_statistics.level_spacing(unfolded)
        plots.nnsd(spacings, rmt_statistics.wigner, title=title_polynomial, file=path + file + "nnsd.png")

        sigma2 = rmt_statistics.number_variance(unfolded, size // 5, max_l=size // 5)
        plots.number_variance(sigma2, rmt_statistics.goe_number_variance, title=title_polynomial, file=path + file + "nv.png")

if __name__ == "__main__":
    demonstrate(5000, 2000)
