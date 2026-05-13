# ---------------------------------------------------------------------------
# schmidt.py contains the functions related to the definition of the 
# Schmidt polynomials, which makes it the most important script. It is very 
# important to reduce the number of loops and avoid repeating calculations, 
# thus if you see some improvement contact us!
# ---------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------
# This function creates all K and S for a given number of multipoles, the 
# definitions for the recurssive relations can be found in the pdfs in docs/
# ---------------------------------------------------------------------------
def KandS(NPOL):
    K = np.zeros([NPOL, NPOL])
    S = np.zeros([NPOL, NPOL])
    S[0, 0] = 1
    delta = 0
    for n in range(1, NPOL):
        for m in range(0, n + 1):
            K[n, m] = ((n - 1) ** 2 - m ** 2) / (2 * n - 1) / (2 * n - 3)
            if n == 1:
                K[n, m] = 0
            if m == 1:
                delta = 1
            else:
                delta = 0
            if m == 0:
                S[n, 0] = (2 * n - 1) * S[n - 1, 0] / n
            else:
                S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1) * (delta + 1) / (n + m))
    return K, S

# ---------------------------------------------------------------------------
# This function creates all Schmidt quasi-normalized Legendre polynomials for 
# a given theta grid and up to a given number of multipoles. This functions 
# needs a set of already calculated K and S. The recurrsive formulae can also
# be found in docs/
# ---------------------------------------------------------------------------
def Schmidtcoefficients(NPOL,
                        Ntheta,
                        theta,
                        K,
                        S):
    
    Pgauss = np.zeros([NPOL, NPOL, Ntheta])
    derivPgauss = np.zeros([NPOL, NPOL, Ntheta])
    P = np.zeros([NPOL, NPOL, Ntheta])
    derivP = np.zeros([NPOL, NPOL, Ntheta])
    
    Pgauss[0, 0, :] = 1
    derivPgauss[0, 0, :] = 0

    for i in range(1, NPOL):
        for j in range(0, i + 1):
            if i == j:
                Pgauss[i, i, :] = np.sin(theta) * \
                      Pgauss[i - 1, i - 1, :]
                derivPgauss[i, i, :] = np.sin(theta) * \
                      derivPgauss[i - 1, i - 1, :] + np.cos(theta) * \
                      Pgauss[i - 1, i - 1, :]
            elif i == 1:
                Pgauss[1, 0, :] = np.cos(theta)
                derivPgauss[1, 0, :] = - np.sin(theta)
            else:
                Pgauss[i, j, :] = np.cos(theta) *  \
                      Pgauss[i - 1, j, :] - K[i, j] * Pgauss[i - 2, j, :]
                derivPgauss[i, j, :] = np.cos(theta) * derivPgauss[i - 1, j, :] - \
                      np.sin(theta) * Pgauss[i - 1, j, :] - \
                      K[i, j] * derivPgauss[i - 2, j, :]
    
    for ntheta in range(0, Ntheta):
        P[:, :, ntheta] = S[:, :] * Pgauss[:, :, ntheta]
        derivP[:, :, ntheta] = S[:, :] * derivPgauss[:, :, ntheta]

    return P, derivP

# ---------------------------------------------------------------------------
# This function creates the potential spherical harmonic expansion for the 
# magnetic field and its components, for one valur of r, thera and phi. This
# it needs to loop for all degree and order of the Schimdt polynomials. 
# It needs as an input a specific r, theta, phi of a grid, and a the Schmidt
# quasi-normalized Legendre polynomials for the same grid. This function 
# needs to be looped trough all the grid to obtain a complete map. The
# expressions can also be found in docs/
# ---------------------------------------------------------------------------
def potentialfunction(radius, 
                      ntheta, 
                      phi, 
                      theta, 
                      NPOL, 
                      P, 
                      derivP, 
                      const,
                      g, 
                      h):
    poten, fr, ftheta, fphi = 0, 0, 0, 0
    for n in range(1, NPOL):
        suma, sumatheta, sumaphi = 0, 0, 0
        for m in range(0, n + 1):

            suma += P[n, m, ntheta] * \
                (g[n, m] * np.cos(m * phi) + h[n, m] * np.sin(m * phi))
            
            sumatheta += derivP[n, m, ntheta] * \
                (g[n, m] * np.cos(m * phi) + h[n, m] * np.sin(m * phi))
            
            sumaphi += m * P[n, m, ntheta] * \
                (- g[n, m] * np.sin(m * phi) + h[n, m] * np.cos(m * phi))
            
        poten += suma * (1 / radius) ** (n + 1)
        fr += suma * (n + 1) * (1 / radius) ** (n + 2)
        ftheta += - sumatheta * (1 / radius) ** (n + 2)
        fphi += - sumaphi * (1 / radius) ** (n + 2) / np.sin(theta)
    return poten / const, fr / const, ftheta / const, fphi / const

# ---------------------------------------------------------------------------
# This is exactly the same function as potentialfunction but for external
# sources, as the sum slightly varies in the radial direction and thus in 
# all derivatives.
# ---------------------------------------------------------------------------
def potentialfunctionexternal(radius, 
                              ntheta, 
                              phi, 
                              theta, 
                              NPOL_EXT,
                              P, 
                              derivP, 
                              const, 
                              G, 
                              H):
    poten, fr, ftheta, fphi = 0, 0, 0, 0
    for n in range(1, NPOL_EXT):
        suma, sumatheta, sumaphi = 0, 0, 0
        for m in range(0, n + 1):

            suma += P[n, m, ntheta] * \
                  (G[n, m] * np.cos(m * phi) + H[n, m] * np.sin(m * phi))
            
            sumatheta += derivP[n, m, ntheta] * \
                  (G[n, m] * np.cos(m * phi) + H[n, m] * np.sin(m * phi))
            
            sumaphi += m * P[n, m, ntheta] * \
                  (- G[n, m] * np.sin(m * phi) + H[n, m] * np.cos(m * phi))
            
        poten += suma * radius ** n
        fr += - suma * n * radius ** (n - 1)
        ftheta += - sumatheta * radius ** (n - 1)
        fphi += - sumaphi * radius ** (n - 1) / np.sin(theta)
    return poten / const, fr / const, ftheta / const, fphi / const
# ---------------------------------------------------------------------------