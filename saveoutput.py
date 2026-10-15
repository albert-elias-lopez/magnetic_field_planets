# ---------------------------------------------------------------------------
# saveoutput.py saves cvs giles or vtu files for more advanced 3D
# visualizations, such as paraview.
# ---------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------
# Name of the output files (with the year for Earth)
def output_name(planet, year):
    return planet + '_' + str(year) if planet == "Earth" else planet

# ---------------------------------------------------------------------------
# Cartesian coordinates of the grid, with dimension [Nr, Ntheta, Nphi]
def cartesian_grid(radius, phi, theta):
    R, Theta, Phi = np.meshgrid(radius, theta, phi, indexing='ij')
    return R * np.cos(Phi) * np.sin(Theta), \
           R * np.sin(Phi) * np.sin(Theta), \
           R * np.cos(Theta)

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

    name = output_name(planet, year)
    names = [name + '_fieldpot.csv',
             name + '_fieldr.csv',
             name + '_fieldtheta.csv',
             name + '_fieldphi.csv',
             name + '_fieldmod.csv']
    heads = ["Pot", "Br", "Btheta", "Bphi", "Bmod"]
    magnitudes = [potential, fieldr, fieldtheta, fieldphi, fieldmod]

    x, y, z = cartesian_grid(radius, phi, theta)
    for outputfile, head, magnitude in zip(names, heads, magnitudes):
        values = np.column_stack([x.ravel(), y.ravel(), z.ravel(), magnitude.ravel()])
        np.savetxt(outputfile, values, delimiter=',', header='X,Y,Z,' + head, comments='')
        print("Saving " + outputfile)

# ---------------------------------------------------------------------------
# Saves the cartesian magnetic field in a vtu file (unstructured grid of
# hexahedra). B is given at the grid points (PointData) and as the average of
# the eight corners of each cell (CellData). It needs Nr > 1 to have cells.
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

    if Nr < 2:
        print("The vtu file needs Nr > 1 (cells between radial shells), not saved")
        return

    # Field at the points and averaged in the cells
    field = np.stack([fieldx, fieldy, fieldz], axis=-1)
    cells = 0.125 * (field[:-1, :-1, :-1] + field[:-1, 1:, :-1] + field[:-1, 1:, 1:] + field[:-1, :-1, 1:] +
                     field[1:, :-1, :-1] + field[1:, 1:, :-1] + field[1:, 1:, 1:] + field[1:, :-1, 1:])

    # Point ids and hexahedra connectivity (four points in one shell and the
    # same four in the next one)
    x, y, z = cartesian_grid(radius, phi, theta)
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    ids = np.arange(Nr * Ntheta * Nphi).reshape(Nr, Ntheta, Nphi)
    connectivity = np.stack([ids[:-1, :-1, :-1], ids[:-1, 1:, :-1], ids[:-1, 1:, 1:], ids[:-1, :-1, 1:],
                             ids[1:, :-1, :-1], ids[1:, 1:, :-1], ids[1:, 1:, 1:], ids[1:, :-1, 1:]],
                            axis=-1).reshape(-1, 8)
    N_Points = Nr * Ntheta * Nphi
    N_Cells = (Nr - 1) * (Ntheta - 1) * (Nphi - 1)

    name = output_name(planet, year) + '_fieldvec.vtu'
    with open(name, 'w') as file:
        file.write('<?xml version="1.0"?>\n')
        file.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        file.write('  <UnstructuredGrid>\n')
        file.write('    <Piece NumberOfPoints="{}" NumberOfCells="{}">\n'.format(N_Points, N_Cells))
        file.write('      <Points>\n')
        file.write('        <DataArray type="Float32" format="ascii" NumberOfComponents="3">\n')
        np.savetxt(file, points, fmt='%.7g')
        file.write('        </DataArray>\n')
        file.write('      </Points>\n')
        file.write('      <Cells>\n')
        file.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        np.savetxt(file, connectivity, fmt='%d')
        file.write('        </DataArray>\n')
        file.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        np.savetxt(file, 8 * np.arange(1, N_Cells + 1), fmt='%d')
        file.write('        </DataArray>\n')
        file.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(file, np.full(N_Cells, 12), fmt='%d')
        file.write('        </DataArray>\n')
        file.write('      </Cells>\n')
        file.write('      <PointData Vectors="B">\n')
        file.write('        <DataArray Name="B" NumberOfComponents="3" type="Float32" format="ascii">\n')
        np.savetxt(file, field.reshape(-1, 3), fmt='%.7g')
        file.write('        </DataArray>\n')
        file.write('      </PointData>\n')
        file.write('      <CellData Vectors="B_cell">\n')
        file.write('        <DataArray Name="B_cell" NumberOfComponents="3" type="Float32" format="ascii">\n')
        np.savetxt(file, cells.reshape(-1, 3), fmt='%.7g')
        file.write('        </DataArray>\n')
        file.write('      </CellData>\n')
        file.write('    </Piece>\n')
        file.write('  </UnstructuredGrid>\n')
        file.write('</VTKFile>\n')
    print("Saving " + name)

# ---------------------------------------------------------------------------
