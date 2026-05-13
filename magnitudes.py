#------------------------------------------------------------------------------------
# magnitudes.py calculates and plots all components and madulus of the curl, 
# curvature and divergence. Others can be added. Both divergence and curl are 
# calculated to evaluate if they are close to 0 or not, as they should vanish 
# as they are constructed from a potential (in a current-free zone).
#------------------------------------------------------------------------------------

from statistics import mean
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

cmap1 = cm.get_cmap('RdBu_r', 255)
cmap2 = cm.get_cmap('inferno', 255)
#cmap2 = cm.get_cmap('YlGn', 255)

def printMagnitudes(planet,            # String for the directory name
                    Ntheta,            # Number of latitudinal points
                    Nphi,              # Number of longitudinal points
                    radius,            # Array of radii for the three shells
                    rc,                # Value of the middle radii
                    rc_file,           # String for file name (radius)
                    dr,                # Radial distance between the shells
                    phi,               # Longitudinal grid
                    theta,             # Latitudinal grid
                    fieldr,            # Radial magnetic field
                    fieldtheta,        # Latitudinal magnetic field
                    fieldphi,          # Longitudinal magnetic field
                    fieldmod,          # Magnetic field modulus
                    plane=True,        # Switch to ask for plane projection plots
                    Mollweide=False):  # Switch to ask for Mollweide projections

    fieldmod = np.sqrt(fieldr ** 2 + fieldtheta ** 2 + fieldphi ** 2)
    directionr = fieldr / fieldmod
    directionphi = fieldphi / fieldmod
    directiontheta = fieldtheta / fieldmod
#------------------------------------------------------------------------------------
    # Initializes all possible derivatives
    derivrfieldr, derivrdirectionr = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivrfieldtheta, derivrdirectiontheta = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivrfieldphi, derivrdirectionphi = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])

    derivthetafieldr, derivthetadirectionr = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivthetafieldtheta, derivthetadirectiontheta = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivthetafieldphi, derivthetadirectionphi = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])

    derivphifieldr, derivphidirectionr = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivphifieldtheta, derivphidirectiontheta = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])
    derivphifieldphi, derivphidirectionphi = np.zeros([Ntheta, Nphi]), np.zeros([Ntheta, Nphi])

    divd2 = np.zeros([Ntheta,Nphi])
    divergence = np.zeros([Ntheta, Nphi])
    volume = np.zeros([Ntheta, Nphi])
#------------------------------------------------------------------------------------
    # Calculates all the field and direction derivatives.
    h_r = radius[2] - radius[1]
    h_theta = rc * (theta[2] - theta[1])
    for j in range(1, Ntheta - 1):
        h_phi = rc * np.sin(theta[j]) * (phi[2] - phi[1])
        i = 1

        for k in range(0, Nphi):
            volume[j,k] = h_r*h_phi*h_theta
            divd2[j,k] = h_r**(2) + h_theta**(2) + h_phi**(2)

            derivrfieldr[j, k] = .5 * (fieldr[i + 1, j, k] - fieldr[i - 1, j, k]) / h_r
            derivrdirectionr[j, k] = .5 * (directionr[i + 1, j, k] - directionr[i - 1, j, k]) / h_r
            derivrfieldtheta[j, k] = .5 * (fieldtheta[i + 1, j, k] - fieldtheta[i - 1, j, k]) / h_r
            derivrdirectiontheta[j, k] = .5 * (directiontheta[i + 1, j, k] - directiontheta[i - 1, j, k]) / h_r
            derivrfieldphi[j, k] = .5 * (fieldphi[i + 1, j, k] - fieldphi[i - 1, j, k]) / h_r
            derivrdirectionphi[j, k] = .5 * (directionphi[i + 1, j, k] - directionphi[i - 1, j, k]) / h_r

            derivthetafieldr[j, k] = .5 * (fieldr[i, j + 1, k] - fieldr[i, j - 1, k]) / h_theta
            derivthetadirectionr[j, k] = .5 * (directionr[i, j + 1, k] - directionr[i, j - 1, k]) / h_theta
            derivthetafieldtheta[j, k] = .5 * (fieldtheta[i, j + 1, k] - fieldtheta[i, j - 1, k]) / h_theta
            derivthetadirectiontheta[j, k] = .5 * (directiontheta[i, j + 1, k] - directiontheta[i, j - 1, k]) / h_theta
            derivthetafieldphi[j, k] = .5 * (fieldphi[i, j + 1, k] - fieldphi[i, j - 1, k]) / h_theta
            derivthetadirectionphi[j, k] = .5 * (directionphi[i, j + 1, k] - directionphi[i, j - 1, k]) / h_theta

            if k == 0 or k == (Nphi-1):
                derivphifieldr[j, 0] = .5 * (fieldr[i, j, 1] - fieldr[i, j, Nphi - 2]) / h_phi
                derivphidirectionr[j, 0] = .5 * (directionr[i, j, 1] - directionr[i, j, Nphi - 2]) / h_phi
                derivphifieldtheta[j, 0] = .5 * (fieldtheta[i, j, 1] - fieldtheta[i, j, Nphi - 2]) / h_phi
                derivphidirectiontheta[j, 0] = .5 * (directiontheta[i, j, 1] - directiontheta[i, j, Nphi - 2]) / h_phi
                derivphifieldphi[j, 0] = .5 * (fieldphi[i, j, 1] - fieldphi[i, j, Nphi - 2]) / h_phi
                derivphidirectionphi[j, 0] = .5 * (directionphi[i, j, 1] - directionphi[i, j, Nphi - 2]) / h_phi

                derivphifieldr[j, Nphi-1] = .5 * (fieldr[i, j, 1] - fieldr[i, j, Nphi - 2]) / h_phi
                derivphidirectionr[j, Nphi-1] = .5 * (directionr[i, j, 1] - directionr[i, j, Nphi - 2]) / h_phi
                derivphifieldtheta[j, Nphi-1] = .5 * (fieldtheta[i, j, 1] - fieldtheta[i, j, Nphi - 2]) / h_phi
                derivphidirectiontheta[j, Nphi-1] = .5 * (directiontheta[i, j, 1] - directiontheta[i, j, Nphi - 2]) / h_phi
                derivphifieldphi[j, Nphi-1] = .5 * (fieldphi[i, j, 1] - fieldphi[i, j, Nphi - 2]) / h_phi
                derivphidirectionphi[j, Nphi-1] = .5 * (directionphi[i, j, 1] - directionphi[i, j, Nphi - 2]) / h_phi
            else:
                derivphifieldr[j, k] = .5 * (fieldr[i, j, k + 1] - fieldr[i, j, k - 1]) / h_phi
                derivphidirectionr[j, k] = .5 * (directionr[i, j, k + 1] - directionr[i, j, k - 1]) / h_phi
                derivphifieldtheta[j, k] = .5 * (fieldtheta[i, j, k + 1] - fieldtheta[i, j, k - 1]) / h_phi
                derivphidirectiontheta[j, k] = .5 * (directiontheta[i, j, k + 1] - directiontheta[i, j, k - 1]) / h_phi
                derivphifieldphi[j, k] = .5 * (fieldphi[i, j, k + 1] - fieldphi[i, j, k - 1]) / h_phi
                derivphidirectionphi[j, k] = .5 * (directionphi[i, j, k + 1] - directionphi[i, j, k - 1]) / h_phi
            
            divergence[j, k] = 0.5/rc**2 * ( (rc + dr)**2*fieldr[i+1,j,1] - (rc - dr)**2*fieldr[i-1,j,1] ) / h_r + \
                        0.5/(rc*np.sin(theta[j])) * ( fieldtheta[i,j+1,k]*np.sin(theta[j+1]) - fieldtheta[i,j-1,k]*np.sin(theta[j-1]) )/ h_theta + \
                        derivphifieldphi[j, k] / rc / np.sin(theta[j])
            
#------------------------------------------------------------------------------------
    curlr = np.zeros([Ntheta, Nphi])
    curltheta = np.zeros([Ntheta, Nphi])
    curlphi = np.zeros([Ntheta, Nphi])
    curlmod = np.zeros([Ntheta, Nphi])
    curvaturer = np.zeros([Ntheta, Nphi])
    curvaturetheta = np.zeros([Ntheta, Nphi])
    curvaturephi = np.zeros([Ntheta, Nphi])
    curvaturemod = np.zeros([Ntheta, Nphi])

    for j in range(1, Ntheta - 1):
        i = 1

        divergence[j, :] = 2. * fieldr[i, j, :] / rc + derivrfieldr[j, :] + \
                        fieldtheta[i, j, :] * np.cos(theta[j]) / rc / np.sin(theta[j]) + \
                        derivthetafieldtheta[j, :] / rc + \
                        derivphifieldphi[j, :] / rc / np.sin(theta[j])

        curlr[j, :] = fieldphi[i, j, :] / rc / np.tan(theta[j]) + \
                      derivthetafieldphi[j, :] / rc - \
                      derivphifieldtheta[j, :] / rc / np.sin(theta[j])
        
        curltheta[j, :] = derivphifieldr[j, :] / rc / np.sin(theta[j]) - \
                          fieldphi[i, j, :] / rc - \
                          derivrfieldphi[j, :]
        
        curlphi[j, :] = fieldtheta[i, j, :] / rc + \
                        derivrfieldtheta[j, :] - \
                        derivthetafieldr[j, :] / rc

        curvaturer[j, :] = directionr[i, j, :] * derivrdirectionr[j, :] + \
                        directiontheta[i, j, :] * derivthetadirectionr[j, :] / rc + \
                        directionphi[i, j, :] * derivphidirectionr[j, :] / rc / np.sin(theta[j]) - \
                        (directiontheta[i, j, :] ** 2 + directionphi[i, j, :] ** 2) / rc

        curvaturetheta[j, :] = directionr[i, j, :] * derivrdirectiontheta[j, :] + \
                            directiontheta[i, j, :] * derivthetadirectiontheta[j, :] / rc + \
                            directionphi[i, j, :] * derivphidirectiontheta[j, :] / rc / np.sin(theta[j]) + \
                            directiontheta[i, j, :] * directionr[i, j, :] / rc - \
                            directionphi[i, j, :] ** 2 / rc / np.tan(theta[j])

        curvaturephi[j, :] = directionr[i, j, :] * derivrdirectionphi[j, :] + \
                            directiontheta[i, j, :] * derivthetadirectionphi[j, :] / rc + \
                            directionphi[i, j, :] * derivphidirectionphi[j, :] / rc / np.sin(theta[j]) + \
                            directionphi[i, j, :] * directionr[i, j, :] / rc + \
                            directionphi[i, j, :] * directiontheta[i, j, :] / rc / np.tan(theta[j])
#------------------------------------------------------------------------------------
    curlmod = np.sqrt(curlr ** 2 + curltheta ** 2 + curlphi ** 2)
    curvaturemod = np.sqrt(curvaturer ** 2 + curvaturetheta ** 2 + curvaturephi ** 2)

    field_L2 = np.sum(volume*fieldmod**2)
    div_L2 = np.sum(volume*divergence**2*divd2)
    curl_L2 = np.sum(volume*curlmod**2*divd2)

    meancurvature = np.zeros([Ntheta, Nphi])
    for j in range(1, Ntheta - 1):
        meancurvature[j, :] = curvaturemod[j, :] * np.sin(theta[j])
    meancurvature = np.sum(meancurvature)/Nphi/np.sum(np.sin(theta))
#------------------------------------------------------------------------------------
    print("------------------------------------------------------------")
    print("Integrated magnitudes for dr =" + str(dr) + " and (Ntheta,Nphi)=(" \
          + str(Ntheta) + "," + str(Nphi) + "):")
    print("Volume:     ", np.sum(volume))
    print("B^2:        ",field_L2)
    print("(Div*h)^2:  ",div_L2)
    print("(Curl*h)^2: ",curl_L2)
    print("<kappa>:    ", meancurvature)
    
    Phi, Theta = np.meshgrid(360 * (1 - phi / 2 / np.pi), 
                             - 180 * theta / np.pi + 90)
    names = [r'$\nabla \cdot B h$ (Gauss) at $r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$|\nabla x B|h$ (Gauss) at $r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$(\nabla x B)_r h$ (Gauss) at $r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$(\nabla x B)_\theta h$ (Gauss) at $r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$(\nabla x B)_\phi h$ (Gauss) at $r =$' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$', 
             r'$|\kappa| \, \mathrm{(1/R_J) \, at} \, r = \, $' + f'${{{rc:.2f}}}$' + r'$\, \mathrm{R_P}$']
    files = [planet+ '_r_' + rc_file +'_div',
             planet+ '_r_' + rc_file+'_curl_mod',
             planet+ '_r_' + rc_file+'_curlr',
             planet+ '_r_' + rc_file+'_curlth',
             planet+ '_r_' + rc_file+'_curlphi',
             planet+ '_r_' + rc_file+'_k_mod']
    magnitudes = [divergence*np.sqrt(divd2), 
                  curlmod*np.sqrt(divd2), 
                  curlr*np.sqrt(divd2), 
                  curltheta*np.sqrt(divd2), 
                  curlphi*np.sqrt(divd2), 
                  curvaturemod]

    maxf = [ 100.,10., 1., 1., 1.,13.]
    minf = [-100.,0.,-1.,-1.,-1.,  0.]
    
    if plane:
        for index, magnitude in enumerate(magnitudes):
            plt.clf()
            ax = plt.axes()
            ax.set_xticks([-180,-120,-60,0,60,120,180,240,300,360])
            ax.set_yticks([-90,-60,-30,0,30,60,90])
            ax.set_ylabel("Latitude")
            ax.set_xlabel("Longitude")

            magnitude[np.where(magnitude > maxf[index])] = maxf[index]
            magnitude[np.where(magnitude < minf[index])] = minf[index]
            limit = max(np.absolute(np.max(magnitude)),
                        np.absolute(np.min(magnitude)))
            if index == 1 or index == 5:
                map = cmap2
                vmin, vmax = np.min(magnitude), np.max(magnitude)
            else:
                map = cmap1
                vmin, vmax = -limit, limit
            if planet=="Earth":
                plt.contourf(Phi,
                             Theta,
                             np.flip(magnitude,1),
                             cmap=map,
                             vmin=vmin,
                             vmax=vmax,
                             levels=30)
            else:
                plt.contourf(- Phi + 180,
                             Theta,
                             magnitude,
                             cmap=map,
                             vmin=vmin,
                             vmax=vmax,
                             levels=30)
            plt.gca().invert_xaxis()
            cbar = plt.colorbar(orientation="vertical",
                                pad=.01,
                                shrink=0.8,
                                aspect=15)
            cbar.set_label(names[index])
            cbar.ax.tick_params(labelsize=11)
            print("Saving " + planet + "/" + files[index] + '.png')
            try: plt.savefig(planet + "/" + files[index] + '.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "/" + files[index] + '.png')
    if Mollweide:
        import cartopy.crs as ccrs
        for index, magnitude in enumerate(magnitudes):
            plt.clf()
            ax = plt.axes(projection=ccrs.Mollweide())
            limit = max(np.absolute(np.max(magnitude)),
                        np.absolute(np.min(magnitude)))
            if index == 1 or index == 5:
                map = cmap2
                vmin, vmax = np.min(magnitude), np.max(magnitude)
            else:
                map = cmap1
                vmin, vmax = -limit, limit
            if planet=="Earth":
                plt.contourf(Phi,
                            Theta,
                            np.flip(magnitude,1),
                            cmap=map,
                            levels=30,
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree())
                ax.coastlines()
            else:
                plt.contourf(- Phi + 180,
                            Theta,
                            magnitude,
                            cmap=map,
                            levels=30,
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree())
            cbar = plt.colorbar(orientation="vertical",
                                pad=.01,
                                shrink=0.8,
                                aspect=15)
            cbar.set_label(names[index])
            cbar.ax.tick_params(labelsize=18)
            print("Saving " + planet + "/" + files[index] + '_Mollweide.png')
            try: plt.savefig(planet + "/" + files[index] + '_Mollweide.png')
            except:
                os.mkdir(planet)
                plt.savefig(planet + "/" + files[index] + '_Mollweide.png')
#------------------------------------------------------------------------------------
