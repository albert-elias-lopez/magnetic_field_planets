# -----------------------------------------------------------------------------------
# lowes_spec.py has the function that calculates the Lowes spectrum at a 
# given radius, and also has two other functions which plot one or several 
# spectra depending on the options given in main.py
#------------------------------------------------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# This function obtains the Lowes spectrum for a set of g and h's and and a given
# radius, in G^2 (units="G", default) or nT^2 (units="nT", used in many papers).
# Its sum over n is the mean of |B|^2 over the sphere of radius rc.
# -----------------------------------------------------------------------------------
def lowes_spec(NPOL,    # Maximum degree and order of the expansion
               rc,      # Radius for which the Lowes spectrum to be calculated
               g,       # Values for the g constants in the Schmidt polynomial
               h,       # Values for the h constants in the Schmidt polynomial
               const=1.,   # Constant to go from the table units to gauss (see
                           # reader.planet_settings), 1 if they are in gauss
               units="G"): # Units of the spectrum, "G" (G^2) or "nT" (nT^2)
    Rn = np.zeros(NPOL)
    for n in range(0,NPOL):
        for m in range(0,n+1):
            Rn[n] = Rn[n] + (g[n,m]**2+h[n,m]**2)
        Rn[n] = float(rc)**(-2*n-4)*(n+1)*Rn[n]
    Rn = Rn / const**2
    if units == "nT": Rn = Rn * 1e10
    return Rn

# -----------------------------------------------------------------------------------
# This function gives the strength of the dipole field (n = 1, all m) and of the
# axial dipole field (n = 1, m = 0) at radius rc, in gauss: the dipole moment
# over rc^3, which is the field at the (magnetic) equator, e.g. 4.2 G for
# Jupiter at the surface. The field at the poles is twice this value, and the
# root mean square over the sphere sqrt(2) times this value.
# -----------------------------------------------------------------------------------
def dipole_field(rc, g, h, const=1.):
    bdip = np.sqrt(g[1,0]**2 + g[1,1]**2 + h[1,1]**2) / const / float(rc)**3
    bdip_axi = np.abs(g[1,0]) / const / float(rc)**3
    return bdip, bdip_axi

# Label of the y-axis of the Lowes spectrum plots
def lowes_label(units):
    return r'$R_n \, \mathrm{(nT^2)}$' if units == "nT" else r'$R_n \, \mathrm{(G^2)}$'

# -----------------------------------------------------------------------------------
# This function plots one Lowes spectrum for a given radius and saves it to 
# the correspondinf folder. It has logical varibles to include the  radial movie
# options, and more specifically the Earth movie option.
# -----------------------------------------------------------------------------------
def plot_lowes(planet,             # String for the directory name
               rc,                 # Float for the x-label plotting
               rc_file,            # String for file name radius/frame (normal/movie)
               Rn,                 # Precalculated Lewis spectrum
               movie=False,        # Switch to same in movie folders
               year=None,          # Year for Earth movie plots (used in plot title)
               years=False,        # Switch to save the plot into the correct movie
                                   # folder for the Earth yearly movie
               folder=None,        # Folder to save the plot (default depends on
                                   # the options above)
               units="G"):         # Units of Rn, "G" (G^2) or "nT" (nT^2)
    plt.clf()
    plt.grid(ls='--', alpha=.5)
    plt.scatter(np.arange(1, len(Rn)), Rn[1:])
    plt.xlabel('Harmonic degree (n) at $r =$' + '%.3f'%rc + '$R_P$')
    plt.ylabel(lowes_label(units))
    plt.yscale('log')
    plt.xlim(-0.2, len(Rn) - 0.5)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if movie:
        # In the movies rc_file is the frame number
        file = planet + '_lowes_spectrum_r_' + '%03d'%rc_file + '.png'
        if years:
            plt.title(str(year))
            rc_file_true = '%.3f'%rc
            rc_file_true = rc_file_true.replace(".","_")
            default_folder = planet + "_movie_years_r_" + rc_file_true
        else:
            default_folder = planet + "_movie"
    else:
        file = planet + '_lowes_spectrum_r_' + rc_file + '.png'
        default_folder = planet
    if folder is None: folder = default_folder
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, file))
    print("Saving " + os.path.join(folder, file))

# -----------------------------------------------------------------------------------
# This function plots multiple Lowes spectra at different radii as described
# in main.py. They already have to be calculated and set in a matrix-like array Rn.
# -----------------------------------------------------------------------------------
def plot_multiple_lowes(planet,        # String used for saving plots in the right folder name
                        lowes_radii,   # Set of radii for which to plot the Lowes spectra on the same plot 
                        Rn,            # All different precaluculated and joint in a single array
                        units="G"):    # Units of Rn, "G" (G^2) or "nT" (nT^2)
    plt.figure(figsize=(10, 10))
    multipoles = np.arange(1, np.shape(Rn)[1])
    colors = plt.cm.jet(np.linspace(0.10,0.90, len(lowes_radii)))
    #plt.grid(ls='--', alpha=.5)
    for r in range(0, len(lowes_radii)):
        plt.plot(multipoles,
                 Rn[r,1:], marker='o',
                 color=colors[r],
                 label=r'$r \, = \, $' + f'${{{lowes_radii[r]:.2f}}}$' + r' $\mathrm{R_P}$')
    plt.xlabel(r'$\mathrm{Harmonic \, degree \, (n)}$')
    plt.ylabel(lowes_label(units))
    plt.legend(loc='best')
    plt.yscale('log')
    if planet == "Jupiter_2021": plt.xticks(np.arange(0, multipoles[-1] + 1, 2))
    plt.xlim(0,multipoles[-1]+1)
    os.makedirs(planet, exist_ok=True)
    plt.savefig(planet + "/" + planet + '_lowes_spectrum.png')
    print("Saving " + planet + "/" + planet + '_lowes_spectrum.png')
    plt.close()
# -----------------------------------------------------------------------------------
