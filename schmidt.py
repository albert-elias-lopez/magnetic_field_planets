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
# magnetic field and its components on the whole (r, theta, phi) grid at once,
# summing over all degrees and orders of the Schmidt polynomials. It needs the
# Schmidt quasi-normalized Legendre polynomials for the same theta grid. The
# expressions can also be found in docs/. With external=True the expansion
# for external sources is used (G and H), where the radial dependence is r^n
# instead of r^-(n+1). The output arrays have dimension [Nr, Ntheta, Nphi].
# ---------------------------------------------------------------------------
def potentialfunction(radius, 
                      theta, 
                      phi, 
                      NPOL, 
                      P, 
                      derivP, 
                      const,
                      g, 
                      h,
                      external=False):
    radius = np.atleast_1d(radius)[:, None, None]
    m = np.arange(NPOL)
    cosm, sinm = np.cos(np.outer(m, phi)), np.sin(np.outer(m, phi))
    shape = (radius.shape[0], len(theta), len(phi))
    poten, fr, ftheta, fphi = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for n in range(1, NPOL):
        # Angular sums over the order m (matrices of dimension Ntheta x Nphi)
        cosine = g[n, :NPOL, None] * cosm + h[n, :NPOL, None] * sinm
        sine = m[:, None] * (- g[n, :NPOL, None] * sinm + h[n, :NPOL, None] * cosm)
        suma = P[n, :NPOL, :].T @ cosine
        sumatheta = derivP[n, :NPOL, :].T @ cosine
        sumaphi = P[n, :NPOL, :].T @ sine
        if external:
            poten += suma * radius ** n
            fr += - suma * n * radius ** (n - 1)
            ftheta += - sumatheta * radius ** (n - 1)
            fphi += - sumaphi * radius ** (n - 1) / np.sin(theta)[:, None]
        else:
            poten += suma * (1 / radius) ** (n + 1)
            fr += suma * (n + 1) * (1 / radius) ** (n + 2)
            ftheta += - sumatheta * (1 / radius) ** (n + 2)
            fphi += - sumaphi * (1 / radius) ** (n + 2) / np.sin(theta)[:, None]
    return poten / const, fr / const, ftheta / const, fphi / const

# ---------------------------------------------------------------------------
# This function adds the internal and (if NPOL_EXT != 0) external fields and
# also returns the modulus. All arrays have dimension [Nr, Ntheta, Nphi].
# ---------------------------------------------------------------------------
def totalfield(radius, 
               theta, 
               phi, 
               NPOL, 
               NPOL_EXT, 
               P, 
               derivP, 
               const, 
               g, 
               h, 
               G, 
               H):
    potential, fieldr, fieldtheta, fieldphi = \
        potentialfunction(radius, theta, phi, NPOL, P, derivP, const, g, h)
    if NPOL_EXT != 0:
        external = potentialfunction(radius, theta, phi, NPOL_EXT, P, derivP, const, G, H, external=True)
        potential, fieldr, fieldtheta, fieldphi = potential + external[0], fieldr + external[1], \
            fieldtheta + external[2], fieldphi + external[3]
    fieldmod = np.sqrt(fieldr ** 2 + fieldtheta ** 2 + fieldphi ** 2)
    return potential, fieldr, fieldtheta, fieldphi, fieldmod
# ---------------------------------------------------------------------------
