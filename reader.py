#--------------------------------------------------------------------------------
# reader.py reads the files in data/ depending on which planet is chosen. Tables
# from Jupiter and Earth are the public files (some spaces have been added),
# the others have been constructed manually from data on "old" Voyager papers
# or a reduced table from Saturn (it has few non-zero multipoles).
#--------------------------------------------------------------------------------

import os
import numpy as np

# Folder with the data tables, so the scripts can be run from any directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Available planets (or satellites) and the names used in the plot titles
PLANET_NAMES = {"Earth":        r"$\mathrm{Earth}$",
                "Jupiter":      r"$\mathrm{Jupiter}$",
                "Jupiter_2021": r"$\mathrm{Jupiter \, (JRM33)}$",
                "Saturn":       r"$\mathrm{Saturn}$",
                "Uranus":       r"$\mathrm{Uranus}$",
                "Neptune":      r"$\mathrm{Neptune}$",
                "Mercury":      r"$\mathrm{Mercury}$",
                "Ganymede":     r"$\mathrm{Ganymede}$",
                "My_own":       r"$\mathrm{My \, \, own}$"}

# Years available for Earth (IGRF-13 table): 1900, 1905, ..., 2020
EARTH_YEARS = list(range(1900, 2021, 5))

#--------------------------------------------------------------------------------
# This function gives the size of the multipole matrices and the constant used 
# to go from the table units to gauss (all plots have gauss as main units). 
# NPOL (NPOL_EXT) is the maximum internal (external) degree plus one. You can
# choose the maximum internal degree nmax (None uses the default of each 
# planet, a larger value than the table uses the whole table) and whether the
# external G and H coefficients are included (only Jupiter_2021 and Mercury).
#--------------------------------------------------------------------------------
def planet_settings(planet, 
                    nmax=None, 
                    external=True):
    # Maximum internal degree of each table, the default degree used when 
    # nmax=None, maximum external degree and the constant to go to gauss
    if planet == "Earth":
        table_nmax, default_nmax, NPOL_EXT, const = 13, 13, 0, 1e5
    elif planet == "Jupiter":
        table_nmax, default_nmax, NPOL_EXT, const = 10, 10, 0, 1e5
    elif planet == "Jupiter_2021":
        # On the 2021 related paper, they say that the first 13 multipoles are 
        # reliable and there is useful information up to degree and order 18. 
        # The table goes up to degree 30, but the degrees above 18 are not 
        # constrained and they blow up below the surface. 
        table_nmax, default_nmax, NPOL_EXT, const = 30, 13, 2, 1e5
    elif planet == "Saturn":
        table_nmax, default_nmax, NPOL_EXT, const = 6, 6, 0, 1e5
    elif planet == "Uranus":
        table_nmax, default_nmax, NPOL_EXT, const = 3, 3, 0, 1   # Uranus and Neptune are already in Gauss
    elif planet == "Neptune":
        table_nmax, default_nmax, NPOL_EXT, const = 3, 3, 0, 1
    elif planet == "Mercury":
        table_nmax, default_nmax, NPOL_EXT, const = 3, 3, 2, 1e5
    elif planet == "Ganymede":
        table_nmax, default_nmax, NPOL_EXT, const = 2, 2, 0, 1e5
    elif planet == "My_own":
        table_nmax, default_nmax, NPOL_EXT, const = 6, 6, 0, 1e5
    else:
        print("There is no option for " + planet + " (maybe you made a typo)")
        raise SystemExit
    if nmax is None: nmax = default_nmax
    NPOL = min(nmax, table_nmax) + 1
    if not external: NPOL_EXT = 0
    return NPOL, NPOL_EXT, const

#--------------------------------------------------------------------------------
# This function reads one table. Each line has a coefficient type (g, h, G or H)
# in column type_col, its degree and order in the next two columns and its value
# in column value_col. Coefficients with a degree that does not fit in the
# matrices (n >= NPOL or n >= NPOL_EXT) are skipped.
#--------------------------------------------------------------------------------
def read_table(filename,
               skip,
               type_col,
               value_col,
               g,
               h,
               G,
               H):
    coefficients = {'g': g, 'h': h, 'G': G, 'H': H}
    with open(os.path.join(DATA_DIR, filename), "r") as file:
        lines = file.readlines()[skip:]
    for line in lines:
        file_list = line.split()
        if len(file_list) <= max(type_col + 2, value_col): continue
        matrix = coefficients.get(file_list[type_col])
        if matrix is None: continue
        n, m = int(file_list[type_col + 1]), int(file_list[type_col + 2])
        if n < matrix.shape[0]:
            matrix[n, m] = float(file_list[value_col])

def reader(planet,
           year,
           NPOL,
           NPOL_EXT):
#--------------------------------------------------------------------------------
    # Initialize the multipole constant matricies to be filled with data
    g = np.zeros([NPOL, NPOL])
    h = np.zeros([NPOL, NPOL])
    G = np.zeros([NPOL_EXT, NPOL_EXT])
    H = np.zeros([NPOL_EXT, NPOL_EXT])
#--------------------------------------------------------------------------------
    if planet == "Earth":
        # The column corresponding to that year (1900, 1905, ..., 2020)
        index = int((year - 1900) / 5 + 3)
        read_table("igrf13coeffs.txt", 4, 0, index, g, h, G, H)
    elif planet == "Jupiter":
        read_table("grl57087-sup-0005-2018GL077312-ds01.txt", 1, 3, 1, g, h, G, H)
    elif planet == "Jupiter_2021":
        read_table("2021JE007055-sup-0002-Table+SI-S01.txt", 1, 3, 1, g, h, G, H)
    elif planet == "Saturn":
        read_table("saturn_models_dougherty18.txt", 2, 0, 4, g, h, G, H)
    elif planet == "Uranus":
        read_table("uranus_model_connerney87.txt", 2, 0, 3, g, h, G, H)
    elif planet == "Neptune":
        read_table("neptune_models_selesnick92.txt", 2, 0, 7, g, h, G, H)
    elif planet == "Mercury":
        read_table("mercury_model_toepfet21.txt", 2, 0, 3, g, h, G, H)
    elif planet == "Ganymede":
        read_table("ganymede_models_weber22.txt", 2, 0, 4, g, h, G, H)
    elif planet == "My_own":
        read_table("my_own_planet.txt", 2, 0, 3, g, h, G, H)

    return g,h,G,H
#--------------------------------------------------------------------------------
