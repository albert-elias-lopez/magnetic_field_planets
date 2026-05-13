# ---------------------------------------------------------------------------
# main.py has all the inputs to produce the planetary magentic field plots.
# You only need to use the command:
# python main.py
# or run it in any python IDE/code editor.
# You only need to play with the first 50ish lines, but feel free to tweak
# and try, probably there are some improvements to be made.
#----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveoutput, saveplots, lowes_spec, magnitudes
#----------------------------------------------------------------------------
# Grid resolution
Ntheta = 100     # Latitudinal points (North-South direction)
Nphi = 2*Ntheta  # Longitudial points (East-West direction)
Nr = 1           # Radial points (change only to generate 3D output)

# Radius considered in the map plot, and name of the corresponding images 
# This should be the actual radius in kilometers (6371.2/72492 for
# Earth/Jupiter), but we renormalize to 1, since r/a is what matters.
rc = 1.0

# String used for naming the output files
rc_file = '%.2f'%rc
rc_file = rc_file.replace(".","_") 

# Planet (or satellite) to choose. Raw data is located in folder data/
planet, planet_name, year = "My_own", r"$\mathrm{Jupiter \, dipole}$", 2021
# planet, planet_name = "Jupiter", r"$\mathrm{Jupiter}$"
# planet, planet_name = "Jupiter_2021", r"$\mathrm{Jupiter \, \, 2021}$"
# planet, planet_name = "Saturn", r"$\mathrm{Saturn}$"
# planet, planet_name = "Neptune", r"$\mathrm{Neptune}$"
# planet, planet_name = "Uranus", r"$\mathrm{Uranus}$"
# planet, planet_name = "Mercury", r"$\mathrm{Mercury}$"
# planet, planet_name = "Ganymede", r"$\mathrm{Ganymede}$"
# planet, planet_name = "My_own", r"$\mathrm{My \, \, own}$"
# You can choose either Earth, Jupiter, Jupiter_2021, Saturn, Neptune, Uranus,
# Mercury and Ganymede or My_own. Anything else will make the code stop.  If 
# you choose Earth, you also need to choose a year, which can only be: 1900, 
# 1905, 1910, ..., to 2020.

# Switch to print the Brms and Bmax values in the terminal
print_bcomponents = True 

# Definition of the spherical grid matrices
phi    = np.linspace(0, 2*np.pi, num=Nphi)
theta  = np.linspace(np.pi / Ntheta, np.pi * (1 - 1 /  Ntheta), num=Ntheta)
# To calculate curvature/curl it is recommended to use a fixed value
# theta  = np.linspace(np.pi / 20, np.pi * (1 - 1 / 20), num=Ntheta)
# to avoid doing operations too close to the axis.

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
lowes = False
# Switch to save the Lowes spectrum for a number of radii
multiple_lowes_r, lowes_radii = False, np.array([1.20,1.15,1.10,1.05,1.00,0.95,0.90,0.85,0.80])
# np.array([1.20,1.15,1.10,1.05,1.00,0.95,0.90,0.85,0.80])
# np.array([1.45,1.30,1.15,1.00,0.85,0.70,0.55])
# Switch to plot the curl, divergence and curvature of the magnetic field
plot_magnitudes = False

# To calulate derivatives and laplacians in a given radius we need at least 
# two other shells of points (you do not need to touch this at first!).
if plot_magnitudes:
    Nr = 3     
    dr = 0.001 # Radial distance between the three shells (in units set to 1)
    radius = np.linspace(rc * (1. - dr), rc * (1. + dr), num=Nr)
else: radius = np.linspace(rc, rc + 3 * rc, num=Nr)

# Switch to saves a csv file with potencial, Br, Btheta, Bphi and Bmod. 
# Use with only 1 Nr, for spherical plots
# Switch to Saves a vtu file (Paraview for 3D visualization) with Bx, By, Bz. 
# Use more than 1 Nr, as cells are used
filecvs, filevtu = False, False

#----------------------------------------------------------------------------
# Depending on the planet you choose, the data will have different 
# multipole definition and will be normalized in nT or G.
# Constants const are used to go from nanotesla to gauss (all plots have gauss
# as main units). This and the total 
if planet=="Earth":
    NPOL,NPOL_EXT=14,0
    const=1e5  # To 
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
    print("There is no option for " + planet + " (maybe you made a typo)")
    raise SystemExit

#----------------------------------------------------------------------------
# The following function reads the g's and h's constants and puts them 
# in NPOL x NPOL matrices, depending on the planet and the year:
print("------------------------------------------------------------")
print("Reading Schmidt constants (g,h,G,H)")
g, h, G, H = reader.reader(planet, year, NPOL, NPOL_EXT)

# This function defines the K and S matrices with dimension NPOL x NPOL
print("------------------------------------------------------------")
print("Calculating K and S recursively")
K, S = schmidt.KandS(NPOL)

# This function defines the Gaussian-normalized and the Schmidt quasi-normalized
# associated Legendre polynomials for the given theta resolution
print("------------------------------------------------------------")
print("Calculating Schmidt quasi-normalized polynomials recursively")
P, derivP = schmidt.Schmidtcoefficients(NPOL, Ntheta, theta, K, S)

#----------------------------------------------------------------------------

# Initialize all components of the magnetic field (spherical, cartesian and modulus)
potential, potential_EXT = np.zeros([Nr, Ntheta, Nphi]), np.zeros([Nr, Ntheta, Nphi])
fieldr, fieldr_EXT = np.zeros([Nr, Ntheta, Nphi]), np.zeros([Nr, Ntheta, Nphi])
fieldtheta, fieldtheta_EXT = np.zeros([Nr, Ntheta, Nphi]), np.zeros([Nr, Ntheta, Nphi])
fieldphi, fieldphi_EXT = np.zeros([Nr, Ntheta, Nphi]), np.zeros([Nr, Ntheta, Nphi])
fieldmod = np.zeros([Nr, Ntheta, Nphi])
fieldx = np.zeros([Nr, Ntheta, Nphi])
fieldy = np.zeros([Nr, Ntheta, Nphi])
fieldz = np.zeros([Nr, Ntheta, Nphi])

print("------------------------------------------------------------")
print("Obtaining the magnetic field potential and components")
# Main loop (for all r, theta phi) which obtains the potential and field components
for j in range(0, Ntheta):
    for k in range(0, Nphi):
        potential[:, j, k], fieldr[:, j, k], fieldtheta[:, j, k], fieldphi[:, j, k] = \
            schmidt.potentialfunction(radius[:], j, phi[k], theta[j], NPOL, P, derivP, const, g, h)
        if NPOL_EXT != 0:
            potential_EXT[:, j, k], fieldr_EXT[:, j, k], fieldtheta_EXT[:, j, k], fieldphi_EXT[:, j, k] = \
                schmidt.potentialfunctionexternal(radius[:], j, phi[k], theta[j], NPOL_EXT, P, derivP, const, G, H)
            
            potential[:, j, k] += potential_EXT[:, j, k]
            fieldr[:, j, k] += fieldr_EXT[:, j, k]
            fieldtheta[:, j, k] += fieldtheta_EXT[:, j, k]
            fieldphi[:, j, k] += fieldphi_EXT[:, j, k]

        fieldmod[:, j, k] = np.sqrt(fieldr[:, j, k] ** 2 +  \
                                    fieldtheta[:, j, k] ** 2 +  \
                                    fieldphi[:, j, k] ** 2)

        fieldx[:, j, k] = np.cos(phi[k]) * np.sin(theta[j]) * fieldr[:, j, k] + \
                          np.cos(phi[k]) * np.cos(theta[j]) * fieldtheta[:, j, k] - \
                          np.sin(phi[j]) * fieldphi[:, j, k]
        
        fieldy[:, j, k] = np.sin(phi[k]) * np.sin(theta[j]) * fieldr[:, j, k] + \
                          np.sin(phi[k]) * np.cos(theta[j]) * fieldtheta[:, j, k] + \
                          np.cos(phi[j]) * fieldphi[:, j, k]
        
        fieldz[:, j, k] = np.cos(theta[j]) * fieldr[:, j, k] - \
                          np.sin(theta[j]) * fieldtheta[:, j, k]
        
brms = np.zeros([Ntheta, Nphi])
for j in range(1, Ntheta - 1):
    brms[j, :] = fieldmod[0, j, :] * np.sin(theta[j])
brms = np.sum(brms)/Nphi/np.sum(np.sin(theta))
if print_bcomponents:
    print("Brms: " + str(brms))
    print("Bmax: " + str(np.max(fieldmod)))

#----------------------------------------------------------------------------
# Prints a cvs files with all calculated spherical quantities
if filecvs: saveoutput.savecsv(Nr, 
                               Ntheta, 
                               Nphi, 
                               radius, 
                               phi, 
                               theta, 
                               potential, 
                               fieldr, 
                               fieldtheta, 
                               fieldphi, 
                               fieldmod, 
                               planet, 
                               year)

# Prints a vtu file with the cartesian magnetic field, for Paraview 3D
# visualization
if filevtu: saveoutput.savevtu(Nr, 
                               Ntheta, 
                               Nphi, 
                               radius, 
                               phi, 
                               theta, 
                               fieldx, 
                               fieldy, 
                               fieldz, 
                               planet, 
                               year)

#----------------------------------------------------------------------------
# Plot parameters
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 25
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["figure.autolayout"] = True
#plt.rcParams['text.usetex'] = True

print("------------------------------------------------------------")
print("Saving plots for " + planet + " r = " + str(rc))

#----------------------------------------------------------------------------
# Saves plane projection of the magnitudes
if planeproj: saveplots.plot_all(planet, 
                                 rc, 
                                 rc_file, 
                                 phi, 
                                 theta, 
                                 potential, 
                                 fieldr, 
                                 fieldtheta, 
                                 fieldphi, 
                                 fieldmod, 
                                 planet_name=planet_name,
                                 ccrs_library=ccrs_library, 
                                 plane=True, 
                                 Mollweide=False,
                                 show_title=False,
                                 show_colorbar_title=True)

# Saves Mollweide projection of the magnitudes
if mollweideproj: saveplots.plot_all(planet, 
                                     rc, 
                                     rc_file, 
                                     phi, 
                                     theta, 
                                     potential, 
                                     fieldr, 
                                     fieldtheta, 
                                     fieldphi, 
                                     fieldmod,
                                     planet_name=planet_name,
                                     ccrs_library=ccrs_library, 
                                     plane=False, 
                                     Mollweide=True,
                                     show_title=False,
                                     show_colorbar_title=True)

# Calculates the curl, divergence, and curvature and plots them
if plot_magnitudes: magnitudes.printMagnitudes(planet, 
                                               Ntheta, 
                                               Nphi, 
                                               radius, 
                                               rc, 
                                               rc_file,
                                               dr, 
                                               phi, 
                                               theta, 
                                               fieldr, 
                                               fieldtheta, 
                                               fieldphi, 
                                               fieldmod, 
                                               plane=True, 
                                               Mollweide=True)

# Obtain the Lowes spectrum for a the given plotted radius and plot it
if lowes:
    Rn = lowes_spec.lowes_spec(NPOL, rc, g, h)
    lowes_spec.plot_lowes(planet, rc, rc_file, Rn)

# Obtian many Lowes spectrum ad different radii and plot them
if multiple_lowes_r:
    Rn = np.zeros([len(lowes_radii), NPOL])
    for r in range(0,len(lowes_radii)):
        Rn[r,:] = lowes_spec.lowes_spec(NPOL, lowes_radii[r], g, h)
    lowes_spec.plot_multiple_lowes(planet, lowes_radii, Rn)

print("------------------------------------------------------------")
print("---  Created by: A.Elias, The IMAGINE PROJECT, ICE-CSIC  ---")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------
