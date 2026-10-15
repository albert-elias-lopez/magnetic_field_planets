# ---------------------------------------------------------------------------
# main_movie_Earth.py is very similar to main.py. It can be used to plot one 
# magnitude many times over for the different years of the Earth model 
# (IGRF), every 5 years from 1900 to 2020. Similarly, you only need to use 
# the command:
# python main_movie_Earth.py
# (see python main_movie_Earth.py --help for all the options) or run it in any
# python IDE/code editor. Once all snapshots have been created, you can join
# them in a movie file using ffmpeg. The commands are printed at the end, for 
# example:
# ffmpeg -framerate 4 -i Earth_movie_years_r_1_000/Earth_fieldr_Mollweide_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p Earth_movie_years_r_1_000/Earth_fieldr_Mollweide.mp4
#----------------------------------------------------------------------------
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reader, schmidt, saveplots, lowes_spec

# Magnitudes that can be plotted in the movie (by the order used in saveplots)
MAGNITUDES = ["potential", "fieldr", "fieldtheta", "fieldphi", "fieldmod"]

# The years, radius, multipoles and resolution are chosen from the terminal.
# Without arguments the defaults below are used (all the years at r = 1). Use
# python main_movie_Earth.py --help to see all the options.
parser = argparse.ArgumentParser(
    description="Movie of Earth's magnetic field map over the years (IGRF, 1900-2020)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="examples:\n"
           "  python main_movie_Earth.py\n"
           "  python main_movie_Earth.py --first-year 1950 --last-year 2000\n"
           "  python main_movie_Earth.py --rc 0.55\n"
           "  python main_movie_Earth.py --nmax 8\n"
           "  python main_movie_Earth.py --ntheta 150\n"
           "  python main_movie_Earth.py --magnitude fieldmod\n"
           "  python main_movie_Earth.py --lowes-units nT --no-latex\n"
           "\n"
           "full examples (all the options):\n"
           "  python main_movie_Earth.py --first-year 1900 --last-year 2020 --rc 1.0 --nmax 13 "
           "--ntheta 50 --magnitude fieldr --lowes-units nT\n"
           "  python main_movie_Earth.py --first-year 1950 --last-year 2020 --rc 0.55 --nmax 8 "
           "--ntheta 150 --magnitude fieldmod --no-latex")
# Years of the movie (the IGRF table has 1900, 1905, ..., 2020)
parser.add_argument("--first-year", type=int, default=1900, choices=reader.EARTH_YEARS,
                    metavar="YEAR", help="first year, 1900, 1905, ..., 2020 (default: 1900)")
parser.add_argument("--last-year", type=int, default=2020, choices=reader.EARTH_YEARS,
                    metavar="YEAR", help="last year, 1900, 1905, ..., 2020 (default: 2020)")
# Radius considered in the map plot, and name of the corresponding images 
# This should be the actual radius in kilometers (6371.2 for Earth),
# but we renormalize to 1, since r/a is what matters.
parser.add_argument("--rc", type=float, default=1.0,
                    help="radius in Earth radii (default: 1.0)")
# Maximum degree of the multipoles to include (the IGRF table goes up to 13)
parser.add_argument("--nmax", type=int, default=None,
                    help="maximum multipole degree (default: 13, the whole table)")
# Grid resolution: latitudinal points (North-South direction). The 
# longitudinal points (East-West direction) are twice as many
parser.add_argument("--ntheta", type=int, default=50, metavar="N",
                    help="latitudinal grid points, the longitudinal ones are 2N "
                         "(default: 50)")
# Only one magnitude is plotted. The usual ones are the radial magnetic field,
# or its modulus.
parser.add_argument("--magnitude", default="fieldr", choices=MAGNITUDES,
                    help="magnitude to plot (default: fieldr)")
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
if args.first_year > args.last_year: parser.error("--first-year must be before --last-year")
if args.ntheta < 3: parser.error("--ntheta must be at least 3")
if args.rc <= 0: parser.error("--rc must be positive")

# Grid resolution
Ntheta = args.ntheta  # Latitudinal points (North-South direction)
Nphi = 2*Ntheta       # Longitudial points (East-West direction)
Nr = 1                # Radial points (change only to generate 3D output)

# Radius considered in the map plot
rc = args.rc

# Raw data for Earth is located in folder data/
planet = "Earth"
years = np.arange(args.first_year, args.last_year + 1, 5)

# Definition of the spherical grid matrices
phi    = np.linspace(0, 2*np.pi, num=Nphi)
theta  = np.linspace(np.pi / Ntheta, np.pi * (1 - 1 /  Ntheta), num=Ntheta)
 
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

# Magnitude to plot and its index
magnitude_name = args.magnitude
index = MAGNITUDES.index(magnitude_name)

#----------------------------------------------------------------------------
# The planet can only be Earth
NPOL, NPOL_EXT, const = reader.planet_settings(planet, args.nmax)

# Folder for the frames with the radius (e.g. Earth_movie_years_r_1_000), and
# the maximum degree if it is not the whole table, so movies are not mixed
folder = "Earth_movie_years_r_" + ('%.3f'%rc).replace(".","_") + \
         ("_n" + str(NPOL - 1) if NPOL - 1 < reader.planet_settings(planet)[0] - 1 else "")

# K and S matrices and the Schmidt quasi-normalized polynomials (the same
# for all years)
K, S = schmidt.KandS(NPOL)
P, derivP = schmidt.Schmidtcoefficients(NPOL, Ntheta, theta, K, S)

#----------------------------------------------------------------------------
# Plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["figure.autolayout"] = True
# LaTeX fonts (--no-latex uses the Computer Modern fonts of matplotlib,
# which is faster and does not need LaTeX installed)
saveplots.latex_fonts(usetex=args.latex)

#----------------------------------------------------------------------------

for frame, year in enumerate(years):
    year = int(year)

    # The following function reads the g's and h's constants and puts them 
    # in NPOL x NPOL matrices, depending on the planet and the year:
    print("------------------------------------------------------------")
    print("Reading Schmidt constants (g,h,G,H):")
    g, h, G, H = reader.reader(planet, year, NPOL, NPOL_EXT)

    # Potential and field components for this year
    potential, fieldr, fieldtheta, fieldphi, fieldmod = [magnitude[0] for magnitude in \
        schmidt.totalfield(rc, theta, phi, NPOL, NPOL_EXT, P, derivP, const, g, h, G, H)]

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
                                   Mollweide=False,
                                   folder=folder)

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
                                       Mollweide=True,
                                       folder=folder)

    # Obtain the Lowes spectrum for a the given plotted radius and plot it
    if lowes:
        Rn = lowes_spec.lowes_spec(NPOL, rc, g, h, const, args.lowes_units)
        lowes_spec.plot_lowes(planet, rc, frame, Rn, movie=True, year=year, years=True,
                             folder=folder, units=args.lowes_units)
    
    
print("------------------------------------------------------------")
print("To make the movie (from %d to %d) use:" % (years[0], years[-1]))
for projection, switch in [("", planeproj), ("_Mollweide", mollweideproj)]:
    if not switch: continue
    movie = os.path.join(folder, planet + "_" + magnitude_name + projection)
    print("ffmpeg -framerate 4 -i " + movie + "_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p " + \
          "-vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" " + movie + ".mp4")
print("------------------------------------------------------------")
print("---  Created by: A.Elias, The IMAGINE PROJECT, ICE-CSIC  ---")
print("------------------------------------------------------------")

#----------------------------------------------------------------------------

