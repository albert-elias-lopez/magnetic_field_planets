# ---------------------------------------------------------------------------
# saveoutput.py saves cvs giles or vtu files for more advanced 3D 
# visualizations, such as paraview. This should be checked for mistakes.
# --------------------------------------------------------------------------- 

import numpy as np
import csv

def savecsv(Nr,
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
            year):
    
    if planet == "Earth":
        names = ['Earthfieldpot_' + str(year) + '.csv', 
                 'Earthfieldr_' + str(year) + '.csv',
                 'Earthfieldtheta_' + str(year) + '.csv', 
                 'Earthfieldphi_' + str(year) + '.csv',
                 'Earthfieldmod_' + str(year) + '.csv']
        
    elif planet == "Jupiter":
        names = ['Jupiterfieldpot.csv', 
                 'Jupiterfieldr.csv', 
                 'Jupiterfieldtheta.csv', 
                 'Jupiterfieldphi.csv',
                 'Jupiterfieldmod.csv']
        
    else:
        names = ['error.csv',
                 'error.csv',
                 'error.csv',
                 'error.csv',
                 'error.csv']

    heads = [["X", "Y", "Z", "Pot"], 
             ["X", "Y", "Z", "Br"], 
             ["X", "Y", "Z", "Btheta"], 
             ["X", "Y", "Z", "Bphi"],
             ["X", "Y", "Z", "Bmod"]]
    
    magntidues = [potential, fieldr, fieldtheta, fieldphi, fieldmod]

    printvalues = np.zeros([Nr, Ntheta, Nphi])
    for number in range(0, 5):
        outputfile = names[number]
        head = heads[number]
        printvalues = magntidues[number]
        with open(outputfile, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(head)
            for i in range(0, Nr):
                for j in range(0, Ntheta):
                    for k in range(0, Nphi):
                        values = [radius[i] * np.cos(phi[k]) * np.sin(theta[j]),
                                  radius[i] * np.sin(phi[k]) * np.sin(theta[j]), 
                                  radius[i] * np.cos(theta[j]),
                                  printvalues[i, j, k]]
                        writer.writerow(values)

# ---------------------------------------------------------------------------
def savevtu(Nr, 
            Ntheta, 
            Nphi, 
            radius, 
            phi, 
            theta, 
            fieldx, 
            fieldy, 
            fieldz, 
            planet, 
            year):
    
    cellx = np.zeros([Nr, Ntheta, Nphi])
    celly = np.zeros([Nr, Ntheta, Nphi])
    cellz = np.zeros([Nr, Ntheta, Nphi])

    for i in range(0, Nr - 1):
        for j in range(0, Ntheta - 1):
            for k in range(0, Nphi - 1):
                cellx[i, j, k] = 0.125 * (
                            fieldx[i, j, k] + fieldx[i, j + 1, k] + fieldx[i, j + 1, k + 1] + fieldx[i, j, k + 1] +
                            fieldx[i + 1, j, k] + fieldx[i + 1, j + 1, k] + fieldx[i + 1, j + 1, k + 1] + fieldx[
                                i + 1, j, k + 1])
                celly[i, j, k] = 0.125 * (
                            fieldy[i, j, k] + fieldy[i, j + 1, k] + fieldy[i, j + 1, k + 1] + fieldy[i, j, k + 1] +
                            fieldy[i + 1, j, k] + fieldy[i + 1, j + 1, k] + fieldy[i + 1, j + 1, k + 1] + fieldy[
                                i + 1, j, k + 1])
                cellz[i, j, k] = 0.125 * (
                            fieldz[i, j, k] + fieldz[i, j + 1, k] + fieldz[i, j + 1, k + 1] + fieldz[i, j, k + 1] +
                            fieldz[i + 1, j, k] + fieldz[i + 1, j + 1, k] + fieldz[i + 1, j + 1, k + 1] + fieldz[
                                i + 1, j, k + 1])

    N_Points = Nr * Ntheta * Nphi
    N_Cells = (Nr - 1) * (Ntheta - 1) * (Nphi - 1)
    id_nodes = np.zeros([Nr + 1, Ntheta, Nphi], dtype=int)

    if planet == "Earth":
        name = 'Earthfieldvec_' + str(year) + '.vtu'
    elif planet == "Jupiter":
        name = 'Jupiterfieldvec.vtu'
    else:
        name = 'error.vtu'
    with open(name, 'w') as file:
        file.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        file.write('  <UnstructuredGrid>\n')
        file.write('    <Piece NumberOfPoints="{}" NumberOfCells="{}">\n'.format(N_Points, N_Cells))
        file.write('      <Points>\n')
        file.write('        <DataArray type="Float32" format="ascii" NumberOfComponents="3">\n')
        identif = 0
        for i in range(0, Nr):
            for j in range(0, Ntheta):
                for k in range(0, Nphi):
                    id_nodes[i, j, k] = identif
                    file.write('        {}   {}   {}\n'.format(radius[i] * np.cos(phi[k]) * np.sin(theta[j]),
                                                               radius[i] * np.sin(phi[k]) * np.sin(theta[j]),
                                                               radius[i] * np.cos(theta[j])))
                    identif += 1
        file.write('        </DataArray>\n')
        file.write('      </Points>\n')
        file.write('      <Cells>\n')
        file.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(0, Nr - 1):
            for j in range(0, Ntheta - 1):
                for k in range(0, Nphi - 1):
                    file.write('        {}    {}    {}    {}    {}    {}    {}    {}\n'.format(id_nodes[i, j, k],
                                                                                               id_nodes[i, j + 1, k],
                                                                                               id_nodes[
                                                                                                   i, j + 1, k + 1],
                                                                                               id_nodes[i, j, k + 1],
                                                                                               id_nodes[i + 1, j, k],
                                                                                               id_nodes[
                                                                                                   i + 1, j + 1, k],
                                                                                               id_nodes[
                                                                                                   i + 1, j + 1, k + 1],
                                                                                               id_nodes[
                                                                                                   i + 1, j, k + 1]))
        file.write('        </DataArray>\n')
        file.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, N_Cells + 1):
            file.write('      {}\n'.format(i * 8))
        file.write('        </DataArray>\n')
        file.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(1, N_Cells + 1):
            file.write('      {}\n'.format(12))
        file.write('        </DataArray>\n')
        file.write('      </Cells>\n')
        file.write('      <PointData>\n')
        file.write('      </PointData>\n')
        file.write('      <CellData>\n')
        file.write('        <DataArray Name="B" NumberOfComponents="3" type="Float32" format="ascii">')
        for i in range(0, Nr - 1):
            for j in range(0, Ntheta - 1):
                for k in range(0, Nphi - 1):
                    file.write('        {}    {}    {}\n'.format(cellx[i, j, k], celly[i, j, k], cellz[i, j, k]))
        file.write('        </DataArray>\n')
        file.write('      </CellData>\n')
        file.write('    </Piece>\n')
        file.write('  </UnstructuredGrid>\n')
        file.write('</VTKFile>\n')

# ---------------------------------------------------------------------------

