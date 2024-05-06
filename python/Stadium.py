# https://medium.com/@natsunoyuki/quantum-chaos-with-python-f4adfe4ab62c

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla

def schrodinger2D(Vfun2D, params, Ne, E0=0.0, findpsi=False):
    V, x, y = Vfun2D(params)        # Get the potential function
    dx = x[1] - x[0]  
    dy = y[1] - y[0]

    Nx = len(x)
    Ny = len(y)

    # Create the 2D Hamiltonian matrix
    # First, the derivatives in the x direction.
    # Note that instead of using arrays, we use sparse matrices
    # in order to reduce computational resource consumption.
    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / (dx ** 2)    # Next, the derivatives in the y direction.
    Hy = sparse.lil_matrix(2 * np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / (dy ** 2)    # Combine both x and y Hilbert spaces using Kronecker products.
    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)  

    # Re-convert to sparse matrix lil form.
    H = H.tolil()    # And add the potential energy.
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]    

    # Convert to sparse matrix csc form, 
    # and solve the eigenvalue problem
    H = H.tocsc()  
    [evl, evt] = sla.eigs(H, k=Ne, sigma=E0)
            
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x, y
    

def eval_wavefunctions(xmin, xmax, Nx,
                       ymin, ymax, Ny,
                       Vfun, params, neigs, E0, findpsi):
    
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, 
                      Vfun, params, neigs, E0, findpsi)    # Get eigen energies
    
    evl = H[0]
    indices = np.argsort(evl)
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))    # Get eigen wave functions
    evt = H[1]     
    
    plt.figure(figsize = (8, 8))
    # Unpack the vector into 2 dimensions for plotting:
    for n in range(neigs):
        psi = evt[:, n]  
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI)**2
        plt.subplot(2, int(neigs/2), n + 1)    
        plt.pcolormesh(np.flipud(PSI), cmap = 'jet')
        plt.axis('equal')
        plt.axis('off')
    plt.show()


def twoD_to_oneD(Nx, Ny, F):
    # From a 2D matrix F return a 1D vector V.
    V = np.zeros(Nx * Ny)
    vindex = 0
    for i in range(Ny):
        for j in range(Nx):
            V[vindex] = F[i, j]
            vindex = vindex + 1                      
    return V

def oneD_to_twoD(Nx, Ny, psi):
    # From a 1D vector psi return a 2D matrix PSI.
    vindex = 0
    PSI = np.zeros([Ny, Nx], dtype='complex')
    for i in range(Ny):
        for j in range(Nx):
            PSI[i, j] = psi[vindex]
            vindex = vindex + 1 
    return PSI


def Vfun(params):
    r, l, v0, Ny, full = params

    ymin = 0
    ymax = 0.5 * l + r
    xmin = 0
    xmax = r
    Nx = int(Ny * 2 * r / (2.0 * r + l))    

    if full:
        ymin = -ymax
        xmin = -xmax

    print(f"Dimensions: ({Nx}, {Ny})")

    x = np.linspace(xmin, xmax, Nx)  
    y = np.linspace(ymin, ymax, Ny)

    X, Y = np.meshgrid(x, y)    
    F = np.zeros([Ny, Nx])    
    
    for i in range(Nx):
        for j in range(Ny):
            if not full and i == 0 or j == 0:
                F[j, i] = v0

            if abs(x[i]) == r or abs(y[j]) == r + 0.5 * l:
                F[j, i] = v0

            cond_0 = (abs(y[j]) - 0.5 * l) > 0
            cond_1 = np.sqrt((abs(y[j])-0.5 * l)**2 + x[i]**2) >= r
            if cond_0 and cond_1:
                F[j, i] = v0    

    plt.contourf(y, x, np.transpose(F))
    plt.show()

    V = twoD_to_oneD(Nx, Ny, F) # Fold the 2D matrix to a 1D array.
    return V, x, y
   
def stadium_energies(r=1, l=2, v0=1E6, Ny=250, Ne=1000, full=True):
    """
        r = stadium radius
        l = stadium length
        v0 = stadium wall potential
        ny - number of cells in the y direction
        ne - number of calculated eigenstates
        full - true if all symmetry states are calculated; false if only one symmetry class is calculated
    """

    params = [r, l, v0, Ny, full]
    H = schrodinger2D(Vfun, params, Ne, findpsi=False)    # Get eigen energies    

    return np.real(H)

#stadium_wavefunctions_plot(1, 2, 1e6, 100)
#stadium_energies()