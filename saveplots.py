# ---------------------------------------------------------------------------
# saveplots.py contains two functions that either plot the given magnitudes
# in a plane projections or Mollweide projection. Take into account that
# for the Molleweide projection and the coastline in the plane Earth
# projection you need to have cartopy installed.

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import shutil

# Colormaps for signed magnitudes, the modulus and the angles
cmap_signed = matplotlib.colormaps['RdBu_r'].resampled(255)
cmap_modulus = matplotlib.colormaps['inferno'].resampled(255)
cmap_angles = matplotlib.colormaps['PRGn'].resampled(255)
# The declination goes from -180 to 180 degrees, so a cyclic colormap is used
cmap_cyclic = matplotlib.colormaps['twilight_shifted'].resampled(255)

# ---------------------------------------------------------------------------
# This function sets LaTeX fonts in all the plots. With usetex=True the text
# is rendered by LaTeX (it needs a LaTeX installation and it is slower), 
# otherwise (or if LaTeX is not found) matplotlib uses the Computer Modern
# fonts, which look almost the same.
# ---------------------------------------------------------------------------
def latex_fonts(usetex=True):
    if usetex and shutil.which('latex') is None:
        print("LaTeX not found, using the Computer Modern fonts of matplotlib")
        usetex = False
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['font.family'] = 'serif'
    if not usetex:
        plt.rcParams['font.serif'] = ['cmr10']
        plt.rcParams['mathtext.fontset'] = 'cm'
        # cmr10 has no minus sign, so the tick labels are written as math text
        plt.rcParams['axes.formatter.use_mathtext'] = True

# ---------------------------------------------------------------------------
# This function draws one magnitude in the current figure, either in the plane
# or in the Mollweide projection, and saves it in folder/file.png. Signed
# magnitudes use a symmetric color scale, the modulus (modulus=True) goes from
# its minimum to its maximum.
# ---------------------------------------------------------------------------
def draw(planet,
         Phi,
         Theta,
         magnitude,
         cmap,
         modulus,
         label,
         folder,
         file,
         ccrs_library=False,
         Mollweide=False,
         planet_name=None,
         show_title=False,
         show_colorbar_title=True,
         title=None,
         ticksize=18,
         labelsize=None,
         titlesize="x-large",
         colorbar_location="right"):
    plt.clf()
    if Mollweide:
        import cartopy.crs as ccrs
        ax = plt.axes(projection=ccrs.Mollweide())
        transform = {'transform': ccrs.PlateCarree()}
        pad = .01
    else:
        if ccrs_library:
            import cartopy.crs as ccrs
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:
            ax = plt.axes()
        ax.set_xticks([-180,-120,-60,0,60,120,180,240,300,360])
        ax.set_yticks([-90,-60,-30,0,30,60,90])
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        transform = {}
        pad = .15
    if modulus:
        levels = np.linspace(np.min(magnitude), np.max(magnitude), 31)
    else:
        limit = np.max(np.absolute(magnitude))
        levels = np.linspace(-limit, limit, 31)
    # A constant magnitude (e.g. B_phi of an axisymmetric field) has no range
    if levels[0] == levels[-1]: levels = 30
    if planet == "Earth":
        x, magnitude = Phi, np.flip(magnitude, 1)
    else:
        x = - Phi + 180
    if cmap is cmap_cyclic:
        # Contours would interpolate across the jump from 180 to -180 degrees
        # Cartopy fails to wrap the cells at the edges of the Mollweide map for
        # some resolutions (e.g. Ntheta = 40), which is avoided by moving the
        # edges 0.001 degrees inwards
        if Mollweide: x = np.clip(x, np.min(x) + 1e-3, np.max(x) - 1e-3)
        plt.pcolormesh(x, Theta, magnitude, cmap=cmap, vmin=-180, vmax=180,
                       shading='auto', **transform)
    else:
        # extend='both' makes the color bar end in triangles
        plt.contourf(x, Theta, magnitude, cmap=cmap, levels=levels, extend='both',
                     **transform)
    if planet == "Earth" and (Mollweide or ccrs_library): ax.coastlines()
    # The color bar goes on the right, or at the bottom (used in the movies)
    if colorbar_location == "bottom":
        cbar = plt.colorbar(orientation="horizontal",
                            pad=.05 if Mollweide else .2,
                            shrink=0.75,
                            aspect=30,
                            extend='both')
    else:
        cbar = plt.colorbar(orientation="vertical",
                            pad=pad,
                            shrink=0.9,
                            aspect=15,
                            extend='both')
    # Only the ticks inside the color range are kept, otherwise the ticks
    # outside it are drawn on the triangles, overlapping with the last ones
    vmin, vmax = cbar.norm.vmin, cbar.norm.vmax
    nticks = 9 if colorbar_location == "bottom" else 7
    ticks = matplotlib.ticker.MaxNLocator(nticks, symmetric=not modulus).tick_values(vmin, vmax)
    tolerance = 1e-9 * (vmax - vmin)
    cbar.set_ticks(ticks[(ticks >= vmin - tolerance) & (ticks <= vmax + tolerance)])
    if show_title: plt.title(planet_name, pad=20, fontsize=titlesize)
    if title is not None: plt.title(title, fontsize=titlesize)
    if show_colorbar_title: cbar.set_label(label, fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)
    os.makedirs(folder, exist_ok=True)
    print(os.path.join(folder, file + '.png'))
    plt.savefig(os.path.join(folder, file + '.png'))

# ---------------------------------------------------------------------------
# Names that will appear as the color bar title of each plot
# ---------------------------------------------------------------------------
def colorbar_names(rc):
    radius = r'$\, \mathrm{at} \, r = \, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$'
    return [r'$\mathrm{Potential \, (G \cdot 1 \, R_P)}$' + radius,
            r'$\mathrm{B_r \, (G)}$' + radius,
            r'$\mathrm{B_\theta \, (G)}$' + radius, # In case of error, try $B_{\theta}$
            r'$\mathrm{B_\phi \, (G)}$' + radius,
            r'$\mathrm{|B| \, (G)}$' + radius,
            r'$\mathrm{Magnetic \, declination \, D \, (^{\circ})}$' + radius,
            r'$\mathrm{Magnetic \, inclination \, I \, (^{\circ})}$' + radius]

def colormap(index):
    if index == 4: return cmap_modulus, True
    elif index == 5: return cmap_cyclic, False
    elif index == 6: return cmap_angles, False
    else: return cmap_signed, False

# ---------------------------------------------------------------------------
# This function plots the potential, the magnetic field compnents and its
# magnitude.
# ATTENTION: ccrs_library=True or Mollweide=True, means that you should have
# installed the ccrs library, otherwise the function will crash.
# ---------------------------------------------------------------------------

def plot_all(planet,
             rc,
             rc_file,
             phi,
             theta,
             potential,
             fieldr,
             fieldtheta,
             fieldphi,
             fieldmod,
             planet_name=r"$\mathrm{Earth}$",
             ccrs_library=False,
             plane=True,
             Mollweide=False,
             show_title=False,
             show_colorbar_title=True):

    # -----------------------------------------------------------------------
    # The grid for the plots needs to be in degrees not radians
    Phi, Theta = np.meshgrid(360 * (1 - phi / 2 / np.pi),
                             - 180 * theta / np.pi + 90)
    names = colorbar_names(rc)

    # Names that will recieve the given png files
    files = [planet + '_r_' + rc_file + '_potential',
             planet + '_r_' + rc_file + '_fieldr',
             planet + '_r_' + rc_file + '_fieldtheta',
             planet + '_r_' + rc_file + '_fieldphi',
             planet + '_r_' + rc_file + '_fieldmod',
             planet + '_r_' + rc_file + '_declination',
             planet + '_r_' + rc_file + '_inclination']
    # Set of magnitudes to be plotted (by the same order of names and files).
    # The North, East and down components are -Btheta, Bphi and -Br.
    declination = np.rad2deg(np.arctan2(fieldphi, -fieldtheta))
    inclination = np.rad2deg(np.arctan2(-fieldr, np.sqrt(fieldtheta**2+fieldphi**2)))
    magnitudes = [potential, fieldr, fieldtheta, fieldphi, fieldmod, declination, inclination]
    # -----------------------------------------------------------------------

    for index, magnitude in enumerate(magnitudes):
        cmap, modulus = colormap(index)
        if plane:
            draw(planet, Phi, Theta, magnitude[0, :, :], cmap, modulus, names[index],
                 planet, files[index], ccrs_library=ccrs_library, Mollweide=False,
                 planet_name=planet_name, show_title=show_title,
                 show_colorbar_title=show_colorbar_title)
        if Mollweide:
            draw(planet, Phi, Theta, magnitude[0, :, :], cmap, modulus, names[index],
                 planet, files[index] + '_Mollweide', Mollweide=True,
                 planet_name=planet_name, show_title=show_title,
                 show_colorbar_title=show_colorbar_title)

# ---------------------------------------------------------------------------
# This function is almost the same but it only makes one plot, which you
# choose by the integer "index". This is used for looping to obtain a movie
# for different radii or different years (Earth) and to only plot one
# quantity. The frames are saved in folder (by default planet_movie/ or
# Earth_movie_years_r_*/ for the Earth yearly movie).
# ---------------------------------------------------------------------------

def plot_1(planet,
           rc,
           frame,
           phi,
           theta,
           magnitude,
           index,
           ccrs_library,
           planet_name=r"$\mathrm{Earth}$",
           year=None,
           years=False,
           plane=True,
           Mollweide=False,
           show_title=False,
           show_colorbar_title=True,
           folder=None,
           ticksize=20,
           labelsize=28,
           titlesize=30,
           colorbar_location="bottom"):

    rc_file = '%.3f'%rc
    rc_file = rc_file.replace(".","_")
    if folder is None:
        folder = "Earth_movie_years_r_" + rc_file if years else planet + "_movie"

# ---------------------------------------------------------------------------
    Phi, Theta = np.meshgrid(360 * (1 - phi / 2 / np.pi),
                             - 180 * theta / np.pi + 90)
    names = colorbar_names(rc)
    quantities = ['potential', 'fieldr', 'fieldtheta', 'fieldphi', 'fieldmod']
    title = str(year) if years else None
    cmap, modulus = colormap(index)
# ---------------------------------------------------------------------------
    if plane:
        draw(planet, Phi, Theta, magnitude, cmap, modulus, names[index], folder,
             planet + '_' + quantities[index] + '_r_' + '%03d'%frame,
             ccrs_library=ccrs_library, Mollweide=False, planet_name=planet_name,
             show_title=show_title, show_colorbar_title=show_colorbar_title,
             title=title, ticksize=ticksize, labelsize=labelsize,
             titlesize=titlesize, colorbar_location=colorbar_location)
    if Mollweide:
        draw(planet, Phi, Theta, magnitude, cmap, modulus, names[index], folder,
             planet + '_' + quantities[index] + '_Mollweide_r_' + '%03d'%frame,
             Mollweide=True, planet_name=planet_name,
             show_title=show_title, show_colorbar_title=show_colorbar_title,
             title=title, ticksize=ticksize, labelsize=labelsize,
             titlesize=titlesize, colorbar_location=colorbar_location)

# ---------------------------------------------------------------------------
