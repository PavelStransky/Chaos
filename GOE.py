import numpy as np
import matplotlib.pyplot as plt

generator = np.random.default_rng()

def GOE(size, sigma=1):
    if sigma <= 0:
        sigma = 0.5 / np.sqrt(size) 

    diagonal = generator.normal(scale=sigma * np.sqrt(2), size=size)
    matrix = generator.normal(scale=sigma, size=(size,size))
    np.fill_diagonal(matrix, diagonal)
    return matrix

def Ensemble(size, num, sigma=1):
    ev = []
    for _ in range(num):
        matrix = GOE(size, sigma=sigma)
        spectrum = np.linalg.eigvalsh(matrix)
        ev.append(spectrum)

    spectrum = np.array(ev).flatten()
    return np.array(sorted(spectrum))

#spectrum = Ensemble(5000, 100, sigma=-1)
#plt.hist(spectrum, bins=100, rwidth=0.8, histtype='step')
#plt.show()
