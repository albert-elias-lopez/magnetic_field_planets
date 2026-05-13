# ---------------------------------------------------------------------------
# main_movie_Earth.py is very similar to movie.py. It can be used to plot one 
# magnitude many times over at different radii. Similarly, you only need to 
# use the command:
# python main.py
# or run it in any python IDE/code editor. Here you don't have to play with 
# anything (maybe only resolution and radius). It plots every all available 
# years. Once all snapshots have been created, you can join them in an
# movie file using ffmpeg:
# ffmpeg -framerate 4 -i Earth_fieldr_Mollweide_r_%03d.png Earth_fieldr_Mollweide.mp4
#----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveplots, lowes_spec
# Grid resolution
Ntheta = 50     # Latitudinal points (North-South direction)
Nphi = 2*Ntheta  # Longitudial points (East-West direction)
Nr = 1           # Radial points (change only to generate 3D output)

# Radius considered in the map plot, and name of the corresponding images 
# This should be the actual radius in kilometers (6371.2 for Earth),
# but we renormalize to 1, since r/a is what matters.
rc = 1.00

# Raw data for Earth is located in folder data/
planet = "Earth"
years = np.linspace(1900,2020,25)

# Definition of the spherical grid matrices
phi    = np.linspace(0, 2*np.pi, num=Nphi)
theta  = np.linspace(np.pi / Ntheta, np.pi * (1 - 1 /  Ntheta), num=Ntheta)
radius  = np.linspace(rc,rc+0.00001, num=1)
 
#----------------------------------------------------------------------------
# Switches to save projections in plane and Mollweide projections. Coastlines
#  are included in Earth plots.
planeproj, mollweideproj = True, True
# If you have successfully installed the ccrs library you can put the Earth 
# coastline in the Earth plane projections also, using the boolean ccrs_library
ccrs_library = True

# ATTENTION: To plot using the Mollweide projection you need the ccrs library.
# The combination mollweideproj=True, ccrs_library=False will crash if you have 
# not installed this library

# Switch to save the Lowes spectrum for the given radius
lowes = True

magnitude_name = "fieldr"

if magnitude_name == "potential": index = 0
elif magnitude_name == "fieldr": index = 1
elif magnitude_name == "fieldtheta": index = 2
elif magnitude_name == "fieldphi": index = 3
elif magnitude_name == "fieldmod": index = 4
else:
    print("There is no option for " + planet + " (maybe you had a typo)")
    raise SystemExit

#----------------------------------------------------------------------------
# The planet can only be Earth
NPOL,NPOL_EXT=14,0
const=1e5 

#----------------------------------------------------------------------------
# Plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["figure.autolayout"] = True

#----------------------------------------------------------------------------

for frame, year in enumerate(years):
    year = int(year)

    # The following function reads the g's and h's constants and puts them 
    # in NPOL x NPOL matrices, depending on the planet and the year:
    print("------------------------------------------------------------")
    print("Reading Schmidt constants (g,h,G,H):")
    g, h, G, H = reader.reader(planet, year, NPOL, NPOL_EXT)

    # This function defines the K and S matrices with dimension NPOL x NPOL
    print("------------------------------------------------------------")
    print("Calculating K and S recursively:")
    K, S = schmidt.KandS(NPOL)

    # This function defines the Gaussian-normalized and the Schmidt quasi-normalized
    # associated Legendre polynomials for the given theta resolution
    print("------------------------------------------------------------")
    print("Calculating Schmidt quasi-normalized polynomials recursively:")
    P, derivP = schmidt.Schmidtcoefficients(NPOL, Ntheta, theta, K, S)

    # Initialize all components of the magnetic field (spherical, cartesian and modulus)
    potential, potential_EXT = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    fieldr, fieldr_EXT = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    fieldtheta, fieldtheta_EXT = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    fieldphi, fieldphi_EXT = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    fieldmod = np.zeros([Ntheta, Nphi])

    # Loops for all r, theta and phi and obtaining all the corresponding potential and fields
    for j in range(0, Ntheta):
        for k in range(0, Nphi):
            potential[j, k], fieldr[j, k], fieldtheta[j, k], fieldphi[j, k] = \
                schmidt.potentialfunction(radius[0], j, phi[k], theta[j], NPOL, P, derivP, const, g, h)
            if NPOL_EXT != 0:
                potential_EXT[j, k], fieldr_EXT[j, k], fieldtheta_EXT[j, k], fieldphi_EXT[j, k] = \
                    schmidt.potentialfunctionexternal(radius[0], j, phi[k], theta[j], NPOL_EXT, P, derivP, const, G, H)
                potential[j, k] += potential_EXT[j, k]
                fieldr[j, k] += fieldr_EXT[j, k]
                fieldtheta[j, k] += fieldtheta_EXT[j, k]
                fieldphi[j, k] += fieldphi_EXT[j, k]
            fieldmod[j, k] = np.sqrt(fieldr[j, k] ** 2 + fieldtheta[j, k] ** 2 + fieldphi[j, k] ** 2)

    all_magnitudes = [potential, fieldr, fieldtheta, fieldphi, fieldmod]
    
    print("------------------------------------------------------------")
    print("Saving plots for " + planet + " r = " + str(rc))
    
    # Saves plane projection of the magnitudes
    if planeproj: saveplots.plot_1(planet,
                                   rc,
                                   frame,
                                   phi,
                                   theta,
                                   all_magnitudes[index],
                                   index,
                                   ccrs_library,
                                   year=year,
                                   years=True,
                                   plane=True,
                                   Mollweide=False)

    # Saves Mollweide projection of the magnitudes
    if mollweideproj: saveplots.plot_1(planet,
                                       rc,
                                       frame,
                                       phi,
                                       theta,
                                       all_magnitudes[index],
                                       index,
                                       ccrs_library,
                                       year=year,
                                       years=True,
                                       plane=False,
                                       Mollweide=True)

    # Obtain the Lowes spectrum for a the given plotted radius and plot it
    if lowes:
        Rn = lowes_spec.lowes_spec(NPOL, radius, g, h)
        lowes_spec.plot_lowes(planet, rc, frame, Rn, movie=True, year=year, years=True)
    
    
print("------------------------------------------------------------")
print("---  Created by: A.Elias, The IMAGINE PROJECT, ICE-CSIC  ---")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------

