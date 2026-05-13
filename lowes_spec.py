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
# radius
# -----------------------------------------------------------------------------------
def lowes_spec(NPOL,    # Maximum degree and order of the expansion
               rc,      # Radius for which the Lowes spectrum to be calculated
               g,       # Values for the g constants in the Schmidt polynomial
               h):      # Values for the h constants in the Schmidt polynomial
    Rn = np.zeros(NPOL)
    for n in range(0,NPOL):
        for m in range(0,n+1):
            Rn[n] = Rn[n] + (g[n,m]**2+h[n,m]**2)
        Rn[n] = rc**(-2*n-4)*(n+1)*Rn[n]
    return Rn

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
               years=False):       # Switch to save the plot into the correct movie
                                   # folder for the Earth yearly movie
    plt.clf()
    plt.grid(ls='--', alpha=.5)
    plt.scatter(np.arange(1, len(Rn)), Rn[1:])
    plt.xlabel('Harmonic degree (n) at $r =$' + '%.3f'%rc + '$R_P$')
    plt.ylabel(r'$R_n \mathrm{(nT)}^2$')
    plt.yscale('log')
    if planet == "Jupiter_2021": plt.xlim(-0.2,18.5)

    if movie:
        if years:
            plt.title(str(year))
            rc_file_true = '%.3f'%rc
            rc_file_true = rc_file_true.replace(".","_")
            try: plt.savefig(planet + "_movie_years_r_" + rc_file_true + \
                            "/" + planet + '_lowes_spectrum_r_' + \
                            '%03d'%rc_file + '.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "_movie_years_r_" + rc_file_true + \
                            "/" + planet + '_lowes_spectrum_r_' + \
                            '%03d'%rc_file + '.png')
            print("Saving " + planet + "_movie_years_r_" + rc_file_true + \
                  "/" + planet + '_lowes_spectrum_r_' + \
                  '%03d'%rc_file + '.png')
        else:
            try: plt.savefig(planet + "_movie/" + planet + '_lowes_spectrum_r_' \
                        + '%03d'%rc_file + '.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "_movie/" + planet + '_lowes_spectrum_r_' \
                        + '%03d'%rc_file + '.png')
            print("Saving " + planet + "_movie/" + planet + '_lowes_spectrum_r_' \
                        + '%03d'%rc_file + '.png')
    else:
        try: plt.savefig(planet + "/" + planet + '_lowes_spectrum_r_' + \
                        rc_file + '.png')
        except:
            os.mkdir(planet)
            plt.savefig(planet + "/" + planet + '_lowes_spectrum_r_' + \
                        rc_file + '.png')
        print("Saving " + planet + "/" + planet + '_lowes_spectrum_r_' + \
                    rc_file + '.png')

# -----------------------------------------------------------------------------------
# This function plots multiple Lowes spectra at different radii as described
# in main.py. They already have to be calculated and set in a matrix-like array Rn.
# -----------------------------------------------------------------------------------
def plot_multiple_lowes(planet,        # String used for saving plots in the right folder name
                        lowes_radii,   # Set of radii for which to plot the Lowes spectra on the same plot 
                        Rn):           # All different precaluculated and joint in a single array
    plt.clf()
    plt.figure(figsize=(10, 10))
    multipoles = np.arange(1, np.shape(Rn)[1])
    if planet == "Jupiter_2021":
        Rn = Rn[:,0:19]
        multipoles = multipoles[0:18]
    colors = plt.cm.jet(np.linspace(0.10,0.90, len(lowes_radii)))
    #plt.grid(ls='--', alpha=.5)
    for r in range(0, len(lowes_radii)):
        plt.plot(multipoles,
                 Rn[r,1:], marker='o',
                 color=colors[r],
                 label=r'$r \, = \, $' + f'${{{lowes_radii[r]:.2f}}}$' + r' $\mathrm{R_P}$')
    plt.xlabel(r'$\mathrm{Harmonic \, degree \, (n)}$')
    plt.ylabel(r'$R_n \mathrm{(nT)}^2$')
    plt.legend(loc='best')
    plt.yscale('log')
    if planet == "Jupiter_2021": plt.xticks([0,2,4,6,8,10,12,14,16,18])
    plt.xlim(0,multipoles[-1]+1)
    plt.savefig(planet + "/" + planet + '_lowes_spectrum.png')
    print("Saving " + planet + "/" + planet + '_lowes_spectrum.png')
# -----------------------------------------------------------------------------------