# ---------------------------------------------------------------------------
# main_movie.py is very similar to movie.py. It can be used to plot one 
# magnitude many times over at different radii. Similarly, you only need to 
# use the command:
# python main_movie.py
# or run it in any python IDE/code editor. Now, you only need to play with 
# the first 20ish lines. Once all snapshots have been created, you can join
# them in an movie file using ffmpeg:
# ffmpeg -framerate 4 -i Earth_fieldr_Mollweide_r_%03d.png Earth_fieldr_Mollweide_r_.mp4
#----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveplots, lowes_spec

# Plot resolution
Ntheta = 50     # Latitudinal points (North-South direction)
Nphi = 2*Ntheta  # Longitudial points (East-West direction)
Nr = 400        # Radial points which will be the number of movie frames

# Radial interval (in radial planetary units) to plot Nr number of plots
rc_i,rc_o = 0.80, 1.30 
rc_i,rc_o = 0.55, 1.30 

# Planet (or satellite) to choose. Raw data is located in folder data/
planet, year = "Earth", 2020
# You can choose either Earth, Jupiter, Jupiter_2021, Saturn, Neptune, Uranus,
# Mercury and Ganymede or My_own. Anything else will make the code stop.  If 
# you choose Earth, you also need to choose a year, which can only be: 1900, 
# 1905, 1910, ..., to 2020.

# Definition of the spherical grid matrices
phi    = np.linspace(0, 2*np.pi, num=Nphi)
theta  = np.linspace(np.pi / Ntheta, np.pi * (1 - 1 /  Ntheta), num=Ntheta)
# To calculate curvature/curl it is recommended to use a fixed value
# theta  = np.linspace(np.pi / 20, np.pi * (1 - 1 / 20), num=Ntheta)
# to avoid doing operations too close to the axis.
rc     = np.linspace(rc_o, rc_i, num=Nr)


#----------------------------------------------------------------------------
# Switches to save projections in plane and Mollweide projections. Coastlines
#  are included in Earth plots.
planeproj, mollweideproj = False, True
# If you have successfully installed the ccrs library you can put the Earth 
# coastline in the Earth plane projections also, using the boolean ccrs_library
ccrs_library = True

# ATTENTION: To plot using the Mollweide projection you need the ccrs library.
# The combination mollweideproj=True, ccrs_library=False will crash if you have 
# not installed this library

# Switch to save the Lowes spectrum for the given radius
lowes = True

#----------------------------------------------------------------------------
# For the radial movies only choose one magnitude to plot. The usual ones 
# are the radial magnetic field, or its modulus. Below you can see the options

magnitude_name = "fieldr"

if magnitude_name == "potential": index = 0
elif magnitude_name == "fieldr": index = 1
elif magnitude_name == "fieldtheta": index = 2
elif magnitude_name == "fieldphi": index = 3
elif magnitude_name == "fieldmod": index = 4
else:
    print("There is no option for " + magnitude_name + " (maybe you had a typo)")
    raise SystemExit

#----------------------------------------------------------------------------
# Depending on the planet you choose, the data will have different 
# multipole definition and will be normalized in nT or G.
# Constants const are used to go from nanotesla to gauss (all plots have gauss
# as main units). This and the total 
if planet=="Earth":
    NPOL,NPOL_EXT=14,0
    const=1e5  # To go from nanotesla to gauss (usually the plots are using gauss) if necessary
elif planet=="Jupiter":
    NPOL,NPOL_EXT=11,0
    const=1e5
elif planet=="Jupiter_2021":
    NPOL,NPOL_EXT=31,2
    const=1e5
elif planet=="Saturn":
    NPOL,NPOL_EXT=7,0
    const=1e5
elif planet=="Uranus":
    NPOL,NPOL_EXT=4,0
    const=1      # Uranus and Neptune coefficients are already in Gauss 
elif planet=="Neptune":
    NPOL,NPOL_EXT=4,0
    const=1
elif planet=="Mercury":
    NPOL,NPOL_EXT=4,2
    const=1e5
elif planet=="Ganymede":
    NPOL,NPOL_EXT=3,0
    const=1e5
elif planet=="My_own":
    NPOL,NPOL_EXT=7,0
    const=1e5
else:
    print("There is no option for " + planet + " (maybe you had a typo)")
    raise SystemExit

#----------------------------------------------------------------------------
# The following function reads the g's and h's constants and puts them 
# in NPOL x NPOL matrices, depending on the planet and the year:
print("------------------------------------------------------------")
print("Reading Schmidt constants (g,h,G,H):")
g, h, G, H = reader.reader(planet, year, NPOL, NPOL_EXT)


# This function defines the K and S matrices with dimension NPOL x NPOL
print("------------------------------------------------------------")
print("Calculating K and S recursively:")
K, S = schmidt.KandS(NPOL)

# Plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["figure.autolayout"] = True

for frame, radii in enumerate(rc):

    radius  = np.linspace(radii,radii+0.00001, num=1)

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
                schmidt.potentialfunction(radius[:], j, phi[k], theta[j], NPOL, P, derivP, const, g, h)
            if NPOL_EXT != 0:
                potential_EXT[j, k], fieldr_EXT[j, k], fieldtheta_EXT[j, k], fieldphi_EXT[j, k] = \
                    schmidt.potentialfunctionexternal(radius[:], j, phi[k], theta[j], NPOL_EXT, P, derivP, const, G, H)
                potential[j, k] += potential_EXT[j, k]
                fieldr[j, k] += fieldr_EXT[j, k]
                fieldtheta[j, k] += fieldtheta_EXT[j, k]
                fieldphi[j, k] += fieldphi_EXT[j, k]
            fieldmod[j, k] = np.sqrt(fieldr[j, k] ** 2 + fieldtheta[j, k] ** 2 + fieldphi[j, k] ** 2)

    all_magnitudes = [potential, fieldr, fieldtheta, fieldphi, fieldmod]
    
    
    print("------------------------------------------------------------")
    print("Saving plots for " + planet + " r = " + str(radii))
    
    # Saves plane projection of the magnitudes
    
    if planeproj: saveplots.plot_1(planet,
                                   radii,
                                   frame,
                                   phi,
                                   theta,
                                   all_magnitudes[index],
                                   index, ccrs_library,
                                   plane=True,
                                   Mollweide=False)

    # Saves Mollweide projection of the magnitudes
    if mollweideproj: saveplots.plot_1(planet,
                                       radii,
                                       frame,
                                       phi,
                                       theta,
                                       all_magnitudes[index],
                                       index, ccrs_library,
                                       plane=False,
                                       Mollweide=True)

    # Obtain the Lowes spectrum for a the given plotted radius and plot it
    if lowes:
        Rn = lowes_spec.lowes_spec(NPOL, radius, g, h)
        lowes_spec.plot_lowes(planet, radii, frame, Rn, movie=True)

    
print("------------------------------------------------------------")
print("---  Created by: A.Elias, The IMAGINE PROJECT, ICE-CSIC  ---")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------
