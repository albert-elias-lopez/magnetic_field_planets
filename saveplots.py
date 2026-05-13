# ---------------------------------------------------------------------------
# saveplots.py contains two functions that either plot the given magnitudes 
# in a plane projections or Mollweide projection. Take into account that 
# for the Molleweide projection and the coastline in the plane Earth 
# projection you need to have cartopy installed.

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

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
    # Names that will appear as the color bar title of each plot
    names = [r'$\mathrm{Potential \, (G · 1 \, R_P) \, at} \, r = \, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$\mathrm{B_r \, (G) \, at} \, r =\, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{B_\theta \, (G) \, at} \, r =\, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', # In case of error, try $B_{\theta}$
             r'$\mathrm{B_\phi \, (G) \, at} \, r =\, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{|B| \, (G) \, at} \, r =\, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{Magnetic \, declination \, D \, (^{\circ}) \, at} \, r = \, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{Magnetic \, inclination \, I \, (^{\circ}) \, at} \, r = \, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$']
    
    # Names that will recieve the given png files
    files = [planet + '_r_' + rc_file + '_potential', 
             planet + '_r_' + rc_file + '_fieldr', 
             planet + '_r_' + rc_file + '_fieldtheta', 
             planet + '_r_' + rc_file + '_fieldphi',
             planet + '_r_' + rc_file + '_fieldmod',
             planet + '_r_' + rc_file + '_declination',
             planet + '_r_' + rc_file + '_inclination']
    # Set of magnitudes to be plotted (by the same order of names and files)
    declination = np.rad2deg(np.arctan(-fieldphi/fieldtheta))
    inclination = np.rad2deg(np.arctan(-fieldr/np.sqrt(fieldtheta**2+fieldphi**2)))
    magnitudes = [potential, fieldr, fieldtheta, fieldphi, fieldmod, declination, inclination]
    # -----------------------------------------------------------------------
    
    if plane:
        for index, magnitude in enumerate(magnitudes):
            plt.clf()
            if ccrs_library:
                import cartopy.crs as ccrs
                ax = plt.axes(projection=ccrs.PlateCarree())
            else:
                ax = plt.axes()
            ax.set_xticks([-180,-120,-60,0,60,120,180,240,300,360])
            ax.set_yticks([-90,-60,-30,0,30,60,90])
            ax.set_ylabel("Latitude")
            ax.set_xlabel("Longitude")
            limit = max(np.absolute(np.max(magnitude)), 
                        np.absolute(np.min(magnitude)))
            if index==4: 
                realmap = cm.get_cmap('inferno', 255)
                vmin, vmax = np.min(magnitude), np.max(magnitude)
            elif index==5 or index==6: 
                realmap = cm.get_cmap('PRGn', 255)
                vmin, vmax = -limit, limit
            else:
                realmap = cm.get_cmap('RdBu_r', 255)
                vmin, vmax = -limit, limit
            if planet=="Earth":
                plt.contourf(Phi,
                             Theta,
                             np.flip(magnitude[0, :, :],1),
                             cmap=realmap, vmin=vmin,
                             vmax=vmax,
                             levels=30)
                if ccrs_library: ax.coastlines()
            else:
                plt.contourf(- Phi + 180,
                             Theta, 
                             magnitude[0, :, :],
                             cmap=realmap,
                             vmin=vmin,
                             vmax=vmax,
                             levels=30)
            cbar = plt.colorbar(orientation="vertical", 
                                pad=.15, 
                                shrink=0.8,
                                aspect=15)
            if show_title: plt.title(planet_name,pad=20)
            if show_colorbar_title: cbar.set_label(names[index])
            cbar.ax.tick_params(labelsize=18)
            print(planet + "/" + files[index] + '.png')
            try: plt.savefig(planet + "/" + files[index] + '.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "/" + files[index] + '.png')
    # -----------------------------------------------------------------------
    if Mollweide:
        import cartopy.crs as ccrs
        for index, magnitude in enumerate(magnitudes):
            plt.clf()
            ax = plt.axes(projection=ccrs.Mollweide())
            limit = max(np.absolute(np.max(magnitude)),
                        np.absolute(np.min(magnitude)))
            if index==4: 
                realmap = cm.get_cmap('inferno', 255)
                vmin, vmax = np.min(magnitude), np.max(magnitude)
            elif index==5 or index==6: 
                realmap = cm.get_cmap('PRGn', 255)
                vmin, vmax = -limit, limit
            else:
                realmap = cm.get_cmap('RdBu_r', 255)
                vmin, vmax = -limit, limit
            if planet=="Earth":
                plt.contourf(Phi,
                            Theta,
                            np.flip(magnitude[0, :, :],1),
                            cmap=realmap,
                            levels=30,
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree())
                ax.coastlines()
            else:
                plt.contourf(- Phi + 180,
                            Theta,
                            magnitude[0, :, :],
                            cmap=realmap,
                            levels=30,
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree())
            
            cbar = plt.colorbar(orientation="vertical",
                                pad=.01,
                                shrink=0.8,
                                aspect=15)
            if show_title: plt.title(planet_name,pad=20)
            if show_colorbar_title: cbar.set_label(names[index])
            cbar.ax.tick_params(labelsize=18)
            print(planet + "/" + files[index] + '_Mollweide.png')
            try: plt.savefig(planet + "/" + files[index] + '_Mollweide.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "/" + files[index] + '_Mollweide.png')

# ---------------------------------------------------------------------------
# This function is almost the same but it only makes one plot, which you 
# choose by the integer "index". This is used for looping to obtain a movie
# for different radii or different years (Earth) and to only plot one
# quantity.
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
           show_colorbar_title=True):

    rc_file = '%.3f'%rc
    rc_file = rc_file.replace(".","_")

# ---------------------------------------------------------------------------
    Phi, Theta = np.meshgrid(360 * (1 - phi / 2 / np.pi),
                             - 180 * theta / np.pi + 90)    
    names = [r'$\mathrm{Potential \, (G · 1 \, R_P) \, at} \, r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$\mathrm{B_r \, (G) \, at} \, r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{B_\theta \, (G) \, at} \, r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', # In case of error, try $B_{\theta}$
             r'$\mathrm{B_\phi \, (G) \, at} \, r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$',
             r'$\mathrm{|B| \, (G) \, at} \, r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$']
# --------------------------------------------------------------------------- 
    if plane:
        files = [planet + '_potential_r_' + '%03d'%frame, 
                 planet + '_fieldr_r_' + '%03d'%frame, 
                 planet + '_fieldtheta_r_' + '%03d'%frame, 
                 planet + '_fieldphi_r_' + '%03d'%frame,
                 planet + '_fieldmod_r_' + '%03d'%frame]
        plt.clf()
        if ccrs_library:
            import cartopy.crs as ccrs
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:
            ax = plt.axes()
        ax.set_xticks([-180,-120,-60,0,60,120,180,240,300,360])
        ax.set_yticks([-90,-60,-30,0,30,60,90])
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        limit = max(np.absolute(np.max(magnitude)),
                    np.absolute(np.min(magnitude)))
        if index==4: 
            realmap = cm.get_cmap('inferno', 255)
            vmin, vmax = np.min(magnitude), np.max(magnitude)
        else:
            realmap = cm.get_cmap('RdBu_r', 255)
            vmin, vmax = -limit, limit
        if planet=="Earth":
            plt.contourf(Phi,
                         Theta,
                         np.flip(magnitude,1),
                         cmap=realmap, vmin=vmin,
                         vmax=vmax,
                         levels=30)
            if ccrs_library: ax.coastlines()
        else:
            plt.contourf(- Phi + 180,
                         Theta, 
                         magnitude,
                         cmap=realmap,
                         vmin=vmin,
                         vmax=vmax,
                         levels=30)
        cbar = plt.colorbar(orientation="vertical",
                            pad=.15,
                            shrink=0.8,
                            aspect=15)
        if show_title: plt.title(planet_name,pad=20)
        if show_colorbar_title: cbar.set_label(names[index])
        cbar.ax.tick_params(labelsize=18)
        if years:
            plt.title(str(year))
            print("Earth_movie_years_r_" + rc_file + "/" + \
                  files[index] + '.png')
        else:
            print(planet + "_movie/" + files[index] + '.png')
        try: 
            if years:
                plt.savefig("Earth_movie_years_r_" + rc_file + "/" + \
                            files[index] + '.png')
            else:
                plt.savefig(planet + "_movie/" + files[index] + '.png')
        except:
            if years:
                os.mkdir("Earth_movie_years_r_" + rc_file + "/")
                plt.savefig("Earth_movie_years_r_" + rc_file + "/" + \
                            files[index] + '.png')
            else:
                os.mkdir(planet + "_movie/")
                plt.savefig(planet + "_movie/" + files[index] + '.png')

# ---------------------------------------------------------------------------
    if Mollweide:
        files = [planet + '_potential_Mollweide_r_' + '%03d'%frame, 
                 planet + '_fieldr_Mollweide_r_' + '%03d'%frame, 
                 planet + '_fieldtheta_Mollweide_r_' + '%03d'%frame, 
                 planet + '_fieldphi_Mollweide_r_' + '%03d'%frame,
                 planet + '_fieldmod_Mollweide_r_' + '%03d'%frame]
        import cartopy.crs as ccrs
        plt.clf()
        ax = plt.axes(projection=ccrs.Mollweide())
        limit = max(np.absolute(np.max(magnitude)),
                    np.absolute(np.min(magnitude)))
        
        if index==4: 
            realmap = cm.get_cmap('inferno', 255)
            vmin, vmax = np.min(magnitude), np.max(magnitude)
        else:
            realmap = cm.get_cmap('RdBu_r', 255)
            vmin, vmax = -limit, limit
        if planet=="Earth":
            plt.contourf(Phi,
                        Theta,
                        np.flip(magnitude,1),
                        cmap=realmap,
                        levels=30,
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree())
            ax.coastlines()
        else:
            plt.contourf(- Phi + 180,
                        Theta,
                        magnitude,
                        cmap=realmap,
                        levels=30,
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree())
        cbar = plt.colorbar(orientation="vertical",
                            pad=.01,
                            shrink=0.8,
                            aspect=15)
        if show_title: plt.title(planet_name,pad=20)
        if show_colorbar_title: cbar.set_label(names[index])
        cbar.ax.tick_params(labelsize=18)
        if years:
            plt.title(str(year))
            print("Earth_movie_years_r_" + rc_file + "/" + \
                  files[index] + '.png')
        else:
            print(planet + "_movie/" + files[index] + '.png')
        try: 
            if years:
                plt.savefig("Earth_movie_years_r_" + rc_file + "/" \
                            + files[index] + '.png')
            else:
                plt.savefig(planet + "_movie/" + files[index] + '.png')
        except:
            if years:
                os.mkdir("Earth_movie_years_r_" + rc_file + "/")
                plt.savefig("Earth_movie_years_r_" + rc_file + "/" + \
                            files[index] + '.png')
            else:
                os.mkdir(planet + "_movie/")
                plt.savefig(planet + "_movie/" + files[index] + '.png')

# ---------------------------------------------------------------------------