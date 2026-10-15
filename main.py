# ---------------------------------------------------------------------------
# main.py has all the inputs to produce the planetary magentic field plots.
# You only need to use the command:
# python main.py --planet Jupiter_2021
# (see python main.py --help for the planet, year, radius and multipole
# options) or run it in any python IDE/code editor.
# You only need to play with the first 50ish lines, but feel free to tweak
# and try, probably there are some improvements to be made.
#----------------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveoutput, saveplots, lowes_spec, magnitudes
#----------------------------------------------------------------------------
# Grid resolution (the latitudinal points are chosen with --ntheta below)
Nr = 1           # Radial points (change only to generate 3D output)

# The planet, year, radius and multipoles are chosen from the terminal, e.g.:
# python main.py --planet Earth --year 1950 --rc 0.83
# python main.py --planet Jupiter_2021 --nmax 18 --no-external
# Without arguments the defaults below are used (Earth 2020 at r = 1). Use
# python main.py --help to see all the options.
parser = argparse.ArgumentParser(
    description="Planetary magnetic field maps",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="examples:\n"
           "  python main.py\n"
           "  python main.py --planet Earth --year 2020 --rc 0.55\n"
           "  python main.py --planet Jupiter --rc 0.85\n"
           "  python main.py --planet Jupiter_2021 --nmax 18 --no-external\n"
           "  python main.py --planet Saturn --rc 1.2\n"
           "  python main.py --planet Mercury --no-external\n"
           "  python main.py --planet Ganymede\n"
           "  python main.py --planet Jupiter_2021 --nmax 18 --ntheta 300\n"
           "  python main.py --planet Jupiter_2021 --title\n"
           "  python main.py --planet Earth --year 1950 --title \"Earth 1950\"\n"
           "  python main.py --planet Saturn --title '$\\mathrm{Saturn \\, (Cassini)}$'\n"
           "  python main.py --planet Jupiter_2021 --lowes-units nT\n"
           "  python main.py --planet Saturn --no-latex\n"
           "\n"
           "full examples (all the options):\n"
           "  python main.py --planet Earth --year 1980 --rc 0.55 --nmax 8 --ntheta 200 "
           "--lowes-units nT --title '$\\mathrm{Earth \\, 1980 \\, (n \\leq 8)}$'\n"
           "  python main.py --planet Jupiter_2021 --rc 0.85 --nmax 18 --no-external --ntheta 300 "
           "--lowes-units nT --title '$\\mathrm{Jupiter \\, (JRM33, \\, n \\leq 18)}$'\n"
           "  python main.py --planet Mercury --rc 1.0 --nmax 3 --no-external --ntheta 150 --title --no-latex")
# Planet (or satellite) to choose. Raw data is located in folder data/
parser.add_argument("--planet", default="Earth", choices=reader.PLANET_NAMES,
                    help="planet or satellite (default: Earth)")
# Only Earth has a table with many years (IGRF): 1900, 1905, ..., 2020
parser.add_argument("--year", type=int, choices=reader.EARTH_YEARS, metavar="YEAR",
                    help="year of the Earth model, 1900, 1905, ..., 2020 (default: 2020)")
# Radius considered in the map plot, and name of the corresponding images.
# This should be the actual radius in kilometers (6371.2/72492 for
# Earth/Jupiter), but we renormalize to 1, since r/a is what matters.
parser.add_argument("--rc", type=float, default=1.0,
                    help="radius in planetary radii (default: 1.0)")
# Maximum degree of the multipoles to include (None uses all the table, except
# for Jupiter_2021, which uses 13, as the paper says it is reliable up to 13
# and useful up to 18) and switch to remove the external field G and H
# coefficients (only for Jupiter_2021 and Mercury).
parser.add_argument("--nmax", type=int, default=None,
                    help="maximum multipole degree (default: the planet's default)")
parser.add_argument("--no-external", dest="external", action="store_false",
                    help="do not include the external field (only for Jupiter_2021 and Mercury)")
# Switch to show a title on top of the plots. Without TEXT the planet name is
# used (with the year for Earth), otherwise TEXT (LaTeX math works, e.g. 
# '$\mathrm{Jupiter}$', with single quotes so the terminal keeps the dollars)
parser.add_argument("--title", nargs="?", const="", default=None, metavar="TEXT",
                    help="show a title on the plots: the planet name, or TEXT if given "
                         "(default: no title)")
# Grid resolution: latitudinal points (North-South direction). The 
# longitudinal points (East-West direction) are twice as many
parser.add_argument("--ntheta", type=int, default=100, metavar="N",
                    help="latitudinal grid points, the longitudinal ones are 2N "
                         "(default: 100)")
# Units of the Lowes spectrum: G^2 like the rest of the plots, or nT^2 as in
# many papers
parser.add_argument("--lowes-units", choices=["G", "nT"], default="G",
                    help="units of the Lowes spectrum, G^2 or nT^2 (default: G)")
# Switch to use the Computer Modern fonts of matplotlib instead of LaTeX
# (faster, and LaTeX does not need to be installed)
parser.add_argument("--no-latex", dest="latex", action="store_false",
                    help="use the matplotlib fonts instead of LaTeX (faster, LaTeX is "
                         "not needed)")
args = parser.parse_args()
if args.ntheta < 3: parser.error("--ntheta must be at least 3")
if args.nmax is not None and args.nmax < 1: parser.error("--nmax must be at least 1")

planet, rc, nmax, external = args.planet, args.rc, args.nmax, args.external
Ntheta = args.ntheta  # Latitudinal points (North-South direction)
Nphi = 2*Ntheta       # Longitudial points (East-West direction)
if planet == "Earth":
    year = 2020 if args.year is None else args.year
    planet_name = r"$\mathrm{Earth \, " + str(year) + "}$"
elif args.year is not None:
    parser.error("--year can only be used with --planet Earth")
else:
    year, planet_name = None, reader.PLANET_NAMES[planet]
show_title = args.title is not None
if args.title: planet_name = args.title

# String used for naming the output files
rc_file = '%.2f'%rc
rc_file = rc_file.replace(".","_")

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
lowes = True
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
# multipole definition and will be normalized in nT or G. The constant const 
# is used to go from nanotesla to gauss (all plots have gauss as main units).
NPOL, NPOL_EXT, const = reader.planet_settings(planet, nmax, external)

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

print("------------------------------------------------------------")
print("Obtaining the magnetic field potential and components")
# Potential and field components in all the grid, dimension [Nr, Ntheta, Nphi]
potential, fieldr, fieldtheta, fieldphi, fieldmod = \
    schmidt.totalfield(radius, theta, phi, NPOL, NPOL_EXT, P, derivP, const, g, h, G, H)

# Cartesian components of the magnetic field
sintheta, costheta = np.sin(theta)[None, :, None], np.cos(theta)[None, :, None]
sinphi, cosphi = np.sin(phi)[None, None, :], np.cos(phi)[None, None, :]
fieldx = cosphi * sintheta * fieldr + cosphi * costheta * fieldtheta - sinphi * fieldphi
fieldy = sinphi * sintheta * fieldr + sinphi * costheta * fieldtheta + cosphi * fieldphi
fieldz = costheta * fieldr - sintheta * fieldtheta

# Values at the sphere r = rc: root mean square of |B| from the multipolar sum
# (the sum of the Lowes spectrum is the mean of |B|^2 over the sphere, exact
# and independent of the grid, internal field only) and maximum of |B| on the
# grid, strength of the dipole (n = 1) and axial dipole (n = 1, m = 0) fields
# at the equator and the dipole fraction, the first term of the Lowes spectrum
# over its sum
i_rc = np.argmin(np.abs(radius - rc))
brms = np.sqrt(np.sum(lowes_spec.lowes_spec(NPOL, rc, g, h, const, "G")))
bmax = np.max(fieldmod[i_rc])
Rn = lowes_spec.lowes_spec(NPOL, rc, g, h, const, args.lowes_units)
bdip, bdip_axi = lowes_spec.dipole_field(rc, g, h, const)
fdip = Rn[1] / np.sum(Rn)
# Dipole fraction up to degree 13 (the usual one in the literature, since the
# crustal field dominates above it on Earth). Only for the tables that reach
# degree 13 (Earth and Jupiter_2021), read again so it does not depend on --nmax
if reader.planet_settings(planet, 13)[0] == 14:
    g13, h13, _, _ = reader.reader(planet, year, 14, 0)
    Rn13 = lowes_spec.lowes_spec(14, rc, g13, h13, const)
    fdip13 = Rn13[1] / np.sum(Rn13)
else: fdip13 = None
if print_bcomponents:
    print("------------------------------------------------------------")
    print("Field at r = %.3f R_P" % rc)
    print("  Brms     = %.6f G   (internal field, sqrt of the sum of R_n)" % brms)
    print("  Bmax     = %.6f G" % bmax)
    print("  Bdip     = %.6f G   (dipole at the magnetic equator, n = 1)" % bdip)
    print("  Bdip_axi = %.6f G   (axial dipole at the equator, n = 1, m = 0)" % bdip_axi)
    print("  f_dip    = %.6f     (R_1 / sum of R_n of the Lowes spectrum)" % fdip)
    if fdip13 is not None:
        print("  f_dip_13 = %.6f     (R_1 / sum of R_n up to n = 13)" % fdip13)

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
# LaTeX fonts (--no-latex uses the Computer Modern fonts of matplotlib,
# which is faster and does not need LaTeX installed)
saveplots.latex_fonts(usetex=args.latex)

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
                                 show_title=show_title,
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
                                     show_title=show_title,
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
if lowes: lowes_spec.plot_lowes(planet, rc, rc_file, Rn, units=args.lowes_units)

# Obtian many Lowes spectrum ad different radii and plot them
if multiple_lowes_r:
    Rn = np.zeros([len(lowes_radii), NPOL])
    for r in range(0,len(lowes_radii)):
        Rn[r,:] = lowes_spec.lowes_spec(NPOL, lowes_radii[r], g, h, const, args.lowes_units)
        print("f_dip(r = %.2f R_P) = %.6f" % (lowes_radii[r], Rn[r,1]/np.sum(Rn[r,:])))
    lowes_spec.plot_multiple_lowes(planet, lowes_radii, Rn, units=args.lowes_units)

print("------------------------------------------------------------")
print("------------  Created by: Albert Elias-López  --------------")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------
