# ---------------------------------------------------------------------------
# main_movie.py is very similar to main.py. It can be used to plot one 
# magnitude many times over at different radii. Similarly, you only need to 
# use the command:
# python main_movie.py --planet Jupiter_2021 --nmax 18
# (see python main_movie.py --help for all the options) or run it in any 
# python IDE/code editor. Once all snapshots have been created, you can join
# them in a movie file (and its reverse) using ffmpeg. The commands are printed
# at the end, for example:
# ffmpeg -framerate 6 -i Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p Jupiter_2021_fieldr_Mollweide.mp4
# ffmpeg -framerate 6 -i Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -vf reverse Jupiter_2021_fieldr_Mollweide_reversed.mp4
#----------------------------------------------------------------------------
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveplots, lowes_spec

# Magnitudes that can be plotted in the movie (by the order used in saveplots)
MAGNITUDES = ["potential", "fieldr", "fieldtheta", "fieldphi", "fieldmod"]

# The planet, year, radii, multipoles and resolution are chosen from the 
# terminal. Without arguments the defaults below are used (JRM33 up to degree
# 18, from r = 2.00 to 0.80 in 241 frames). Use python main_movie.py --help to
# see all the options.
parser = argparse.ArgumentParser(
    description="Movie of a planetary magnetic field map at decreasing radii",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="examples:\n"
           "  python main_movie.py\n"
           "  python main_movie.py --planet Jupiter_2021 --nmax 13\n"
           "  python main_movie.py --planet Jupiter_2021 --nmax 18 --external\n"
           "  python main_movie.py --planet Jupiter --rmin 0.85 --rmax 1.2\n"
           "  python main_movie.py --planet Earth --year 1950 --rmin 0.55 --rmax 1.0\n"
           "  python main_movie.py --planet Saturn --magnitude fieldmod\n"
           "  python main_movie.py --planet Jupiter_2021 --frames 25 --ntheta 60\n"
           "  python main_movie.py --planet Jupiter_2021 --no-title\n"
           "  python main_movie.py --planet Saturn --title '$\\mathrm{Saturn \\, (Cassini)}$'\n"
           "  python main_movie.py --planet Jupiter_2021 --lowes-units nT --no-latex\n"
           "\n"
           "full examples (all the options):\n"
           "  python main_movie.py --planet Jupiter_2021 --nmax 18 --no-external --rmin 0.8 --rmax 2.0 "
           "--frames 241 --ntheta 180 --magnitude fieldr --lowes-units nT --title '$\\mathrm{Jupiter \\, (JRM33, \\, n \\leq 18)}$'\n"
           "  python main_movie.py --planet Earth --year 2020 --nmax 13 --rmin 0.55 --rmax 1.3 "
           "--frames 151 --ntheta 120 --magnitude fieldmod --no-title\n"
           "  python main_movie.py --planet Mercury --nmax 3 --external --rmin 0.9 --rmax 1.5 "
           "--frames 61 --ntheta 100 --magnitude fieldtheta --no-latex")
# Planet (or satellite) to choose. Raw data is located in folder data/
parser.add_argument("--planet", default="Jupiter_2021", choices=reader.PLANET_NAMES,
                    help="planet or satellite (default: Jupiter_2021)")
# Only Earth has a table with many years (IGRF): 1900, 1905, ..., 2020
parser.add_argument("--year", type=int, choices=reader.EARTH_YEARS, metavar="YEAR",
                    help="year of the Earth model, 1900, 1905, ..., 2020 (default: 2020)")
# Maximum degree of the multipoles to include (degrees above the table are 
# ignored, so 18 uses the whole table for all planets except Jupiter_2021, 
# where the paper says it is reliable up to 13 and useful up to 18) and switch
# to include the external field G and H coefficients (only for Jupiter_2021 
# and Mercury).
parser.add_argument("--nmax", type=int, default=18,
                    help="maximum multipole degree (default: 18)")
parser.add_argument("--external", action=argparse.BooleanOptionalAction, default=False,
                    help="include the external field, only Jupiter_2021 and Mercury "
                         "(default: --no-external)")
# Radial interval (in radial planetary units) to plot frames number of plots. 
# The movie goes from the outer radius rmax to the inner radius rmin.
parser.add_argument("--rmin", type=float, default=0.80,
                    help="inner radius, the last frame (default: 0.80)")
parser.add_argument("--rmax", type=float, default=2.00,
                    help="outer radius, the first frame (default: 2.00)")
parser.add_argument("--frames", type=int, default=241, metavar="N",
                    help="number of radii, which will be the movie frames (default: 241)")
# Grid resolution: latitudinal points (North-South direction). The 
# longitudinal points (East-West direction) are twice as many
parser.add_argument("--ntheta", type=int, default=180, metavar="N",
                    help="latitudinal grid points, the longitudinal ones are 2N "
                         "(default: 180)")
# For the radial movies only one magnitude is plotted. The usual ones are the
# radial magnetic field, or its modulus.
parser.add_argument("--magnitude", default="fieldr", choices=MAGNITUDES,
                    help="magnitude to plot (default: fieldr)")
# The title is the planet name and the maximum degree, or TEXT (LaTeX math 
# works, e.g. '$\mathrm{Jupiter}$', with single quotes so the terminal keeps 
# the dollars). --no-title removes it.
title = parser.add_mutually_exclusive_group()
title.add_argument("--title", metavar="TEXT",
                   help="title of the plots (default: the planet name and maximum degree)")
title.add_argument("--no-title", dest="show_title", action="store_false",
                   help="do not show a title on the plots")
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
if args.nmax is not None and args.nmax < 1: parser.error("--nmax must be at least 1")
if args.ntheta < 3: parser.error("--ntheta must be at least 3")
if args.frames < 1: parser.error("--frames must be at least 1")
if not 0 < args.rmin <= args.rmax: parser.error("the radii must be 0 < --rmin <= --rmax")

planet, nmax, external = args.planet, args.nmax, args.external
if planet == "Earth":
    year = 2020 if args.year is None else args.year
    planet_name = r"$\mathrm{Earth \, " + str(year) + "}$"
elif args.year is not None:
    parser.error("--year can only be used with --planet Earth")
else:
    year, planet_name = None, reader.PLANET_NAMES[planet]
show_title = args.show_title

# Plot resolution
Ntheta = args.ntheta  # Latitudinal points (North-South direction)
Nphi = 2*Ntheta       # Longitudial points (East-West direction)
Nr = args.frames      # Radial points which will be the number of movie frames

# Radial interval (in radial planetary units) to plot Nr number of plots. The
# movie goes from the outer radius rc_o to the inner radius rc_i.
rc_i,rc_o = args.rmin, args.rmax

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

# Magnitude to plot and its index
magnitude_name = args.magnitude
index = MAGNITUDES.index(magnitude_name)

#----------------------------------------------------------------------------
# Depending on the planet you choose, the data will have different 
# multipole definition and will be normalized in nT or G. The constant const 
# is used to go from nanotesla to gauss (all plots have gauss as main units).
NPOL, NPOL_EXT, const = reader.planet_settings(planet, nmax, external)

# Folder for the frames, with the year for Earth, the maximum degree and if 
# the external field is included (e.g. Jupiter_2021_movie_n18), so different
# movies are not mixed
folder = planet + ("_" + str(year) if planet == "Earth" else "") + \
         "_movie_n" + str(NPOL - 1) + ("_ext" if NPOL_EXT != 0 else "")
if args.title is not None: planet_name = args.title
else: planet_name = planet_name + r" $(n \leq " + str(NPOL - 1) + ")$"

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
# LaTeX fonts (--no-latex uses the Computer Modern fonts of matplotlib,
# which is faster and does not need LaTeX installed)
saveplots.latex_fonts(usetex=args.latex)

# This function defines the Gaussian-normalized and the Schmidt quasi-normalized
# associated Legendre polynomials for the given theta resolution
print("------------------------------------------------------------")
print("Calculating Schmidt quasi-normalized polynomials recursively:")
P, derivP = schmidt.Schmidtcoefficients(NPOL, Ntheta, theta, K, S)

for frame, radii in enumerate(rc):

    # Potential and field components for this radius
    potential, fieldr, fieldtheta, fieldphi, fieldmod = [magnitude[0] for magnitude in \
        schmidt.totalfield(radii, theta, phi, NPOL, NPOL_EXT, P, derivP, const, g, h, G, H)]

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
                                   planet_name=planet_name,
                                   plane=True,
                                   Mollweide=False,
                                   show_title=show_title,
                                   folder=folder)

    # Saves Mollweide projection of the magnitudes
    if mollweideproj: saveplots.plot_1(planet,
                                       radii,
                                       frame,
                                       phi,
                                       theta,
                                       all_magnitudes[index],
                                       index, ccrs_library,
                                       planet_name=planet_name,
                                       plane=False,
                                       Mollweide=True,
                                       show_title=show_title,
                                       folder=folder)

    # Obtain the Lowes spectrum for a the given plotted radius and plot it
    if lowes:
        Rn = lowes_spec.lowes_spec(NPOL, radii, g, h, const, args.lowes_units)
        lowes_spec.plot_lowes(planet, radii, frame, Rn, movie=True, folder=folder,
                              units=args.lowes_units)

    
print("------------------------------------------------------------")
print("To make the movie (from r = %.2f to %.2f) and its reverse use:" % (rc_o, rc_i))
for projection, switch in [("", planeproj), ("_Mollweide", mollweideproj)]:
    if not switch: continue
    movie = os.path.join(folder, planet + "_" + magnitude_name + projection)
    for reverse, name in [("", ".mp4"), ("reverse,", "_reversed.mp4")]:
        print("ffmpeg -framerate 6 -i " + movie + "_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p " + \
              "-vf \"" + reverse + "pad=ceil(iw/2)*2:ceil(ih/2)*2\" " + movie + name)
print("------------------------------------------------------------")
print("---  Created by: A.Elias, The IMAGINE PROJECT, ICE-CSIC  ---")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------
