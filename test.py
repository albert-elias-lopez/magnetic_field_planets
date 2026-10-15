# ---------------------------------------------------------------------------
# test.py checks that the code gives the right physics and that all the
# scripts run. It has two groups of tests:
#  - physics: the Schmidt polynomials, the field of a dipole and of an
#    external uniform field, B = -grad(V), div(B) = 0, the Lowes spectrum and
#    the tables of every planet (no plots are made, it takes a few seconds)
#  - smoke: runs main.py, main_movie.py and main_movie_Earth.py (and the csv,
#    vtu and curl/divergence/curvature outputs) with a small grid in a
#    temporary folder, which is deleted at the end
# Use the command:
# python test.py
# (see python test.py --help for the options). It is also called by setup.py,
# and the test_* functions can be run with pytest too.
# ---------------------------------------------------------------------------
import os
import sys
import time
import shutil
import argparse
import tempfile
import traceback
import subprocess
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import reader, schmidt, lowes_spec

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# Field on a (r, theta, phi) grid for the given coefficients, as main.py does
def field(radius, theta, phi, g, h, G=None, H=None, const=1.):
    NPOL = g.shape[0]
    if G is None: G, H = np.zeros([0, 0]), np.zeros([0, 0])
    K, S = schmidt.KandS(NPOL)
    P, derivP = schmidt.Schmidtcoefficients(NPOL, len(theta), theta, K, S)
    return schmidt.totalfield(radius, theta, phi, NPOL, G.shape[0], P, derivP,
                              const, g, h, G, H)

# Random coefficients (in nT) up to degree nmax, with h[n, 0] = 0 and m <= n
def random_coefficients(nmax, seed):
    rng = np.random.default_rng(seed)
    g, h = rng.normal(size=(2, nmax + 1, nmax + 1)) * 1e4
    mask = np.tril(np.ones([nmax + 1, nmax + 1]))
    mask[0, :] = 0
    g, h = g * mask, h * mask
    h[:, 0] = 0
    return g, h

# Gauss-Legendre points in theta (exact integrals over the sphere)
def gauss_theta(N):
    x, w = np.polynomial.legendre.leggauss(N)
    return np.arccos(x), w

def check_close(name, value, expected, rtol=1e-10, atol=0.):
    value, expected = np.asarray(value, dtype=float), np.asarray(expected, dtype=float)
    if not np.allclose(value, expected, rtol=rtol, atol=atol):
        error = np.max(np.abs(value - expected))
        raise AssertionError("%s: maximum difference %.3e (rtol=%g, atol=%g)"
                             % (name, error, rtol, atol))

# ---------------------------------------------------------------------------
# Physics tests
# ---------------------------------------------------------------------------
# The first Schmidt quasi-normalized polynomials have known formulas
def test_schmidt_low_degrees():
    theta = np.linspace(0.05, np.pi - 0.05, 37)
    c, s = np.cos(theta), np.sin(theta)
    K, S = schmidt.KandS(4)
    P, derivP = schmidt.Schmidtcoefficients(4, len(theta), theta, K, S)
    exact = {(0, 0): np.ones_like(c),
             (1, 0): c,
             (1, 1): s,
             (2, 0): (3 * c ** 2 - 1) / 2,
             (2, 1): np.sqrt(3) * s * c,
             (2, 2): np.sqrt(3) / 2 * s ** 2,
             (3, 0): (5 * c ** 3 - 3 * c) / 2,
             (3, 1): np.sqrt(3 / 8) * s * (5 * c ** 2 - 1),
             (3, 2): np.sqrt(15) / 2 * c * s ** 2,
             (3, 3): np.sqrt(5 / 8) * s ** 3}
    for (n, m), value in exact.items():
        check_close("P_%d^%d" % (n, m), P[n, m], value, atol=1e-13)

# The derivatives dP/dtheta agree with finite differences
def test_schmidt_derivatives():
    NPOL, dt = 31, 1e-6
    theta = np.linspace(0.05, np.pi - 0.05, 41)
    K, S = schmidt.KandS(NPOL)
    _, derivP = schmidt.Schmidtcoefficients(NPOL, len(theta), theta, K, S)
    Pplus, _ = schmidt.Schmidtcoefficients(NPOL, len(theta), theta + dt, K, S)
    Pminus, _ = schmidt.Schmidtcoefficients(NPOL, len(theta), theta - dt, K, S)
    check_close("dP/dtheta", derivP, (Pplus - Pminus) / (2 * dt), rtol=1e-5, atol=1e-5)

# Schmidt normalization up to degree 30 (the whole JRM33 table):
# int P_n^m P_k^m sin(theta) dtheta = 2 (2 - delta_m0) / (2n + 1) delta_nk
def test_schmidt_normalization():
    NPOL = 31
    theta, w = gauss_theta(64)
    K, S = schmidt.KandS(NPOL)
    P, _ = schmidt.Schmidtcoefficients(NPOL, len(theta), theta, K, S)
    for m in range(NPOL):
        Pm = P[m:, m, :]
        integral = (Pm * w) @ Pm.T
        n = np.arange(m, NPOL)
        expected = np.diag(2 * (2 - (m == 0)) / (2 * n + 1))
        check_close("normalization m=%d" % m, integral, expected, atol=1e-11)

# A pure dipole g10: V = g10 cos(theta) / r^2, Br = 2 g10 cos(theta) / r^3,
# Btheta = g10 sin(theta) / r^3 and Bphi = 0 (in gauss, with const = 1e5)
def test_dipole():
    g, h = np.zeros([2, 2]), np.zeros([2, 2])
    g[1, 0], const = -29404.8, 1e5
    radius = np.array([0.55, 1., 2.3])
    theta, phi = np.linspace(0.1, np.pi - 0.1, 25), np.linspace(0, 2 * np.pi, 30)
    V, Br, Bt, Bp, Bmod = field(radius, theta, phi, g, h, const=const)
    r, t = radius[:, None, None], theta[None, :, None]
    b0 = g[1, 0] / const
    check_close("V", V, b0 * np.cos(t) / r ** 2 * np.ones_like(V))
    check_close("Br", Br, 2 * b0 * np.cos(t) / r ** 3 * np.ones_like(Br))
    check_close("Btheta", Bt, b0 * np.sin(t) / r ** 3 * np.ones_like(Bt))
    check_close("Bphi", Bp, 0, atol=1e-15)
    check_close("|B|", Bmod, abs(b0) / r ** 3 * np.sqrt(1 + 3 * np.cos(t) ** 2) * np.ones_like(Bmod))

# An external G10 is a uniform field along z: V = G10 r cos(theta), B = -G10 z
def test_external_uniform_field():
    g, h = np.zeros([2, 2]), np.zeros([2, 2])
    G, H = np.zeros([2, 2]), np.zeros([2, 2])
    G[1, 0] = 20.
    radius = np.array([0.5, 1., 3.])
    theta, phi = np.linspace(0.1, np.pi - 0.1, 25), np.linspace(0, 2 * np.pi, 30)
    V, Br, Bt, Bp, _ = field(radius, theta, phi, g, h, G, H)
    t = theta[None, :, None]
    check_close("V", V, G[1, 0] * radius[:, None, None] * np.cos(t) * np.ones_like(V))
    check_close("Bz", np.cos(t) * Br - np.sin(t) * Bt, - G[1, 0], rtol=1e-12)
    check_close("Bx,y", np.sin(t) * Br + np.cos(t) * Bt, 0, atol=1e-12)
    check_close("Bphi", Bp, 0, atol=1e-12)

# All the components are B = -grad(V), for random internal and external
# coefficients (this checks the theta and phi derivatives and their signs)
def test_field_is_minus_grad_potential():
    g, h = random_coefficients(8, seed=1)
    G, H = random_coefficients(3, seed=2)
    theta, phi = np.linspace(0.3, 2.8, 11), np.linspace(0.1, 6.1, 13)
    r, d = np.array([0.9, 1.3]), 1e-5
    V, Br, Bt, Bp, _ = field(r, theta, phi, g, h, G, H)
    dVdr = (field(r + d, theta, phi, g, h, G, H)[0] - field(r - d, theta, phi, g, h, G, H)[0]) / (2 * d)
    dVdt = (field(r, theta + d, phi, g, h, G, H)[0] - field(r, theta - d, phi, g, h, G, H)[0]) / (2 * d)
    dVdp = (field(r, theta, phi + d, g, h, G, H)[0] - field(r, theta, phi - d, g, h, G, H)[0]) / (2 * d)
    R, sinT = r[:, None, None], np.sin(theta)[None, :, None]
    scale = np.max(np.abs(Br))
    check_close("Br", Br, - dVdr, atol=1e-6 * scale)
    check_close("Btheta", Bt, - dVdt / R, atol=1e-6 * scale)
    check_close("Bphi", Bp, - dVdp / (R * sinT), atol=1e-6 * scale)

# div(B) = 0 (the potential solves Laplace's equation), numerically
def test_divergence_free():
    g, h = random_coefficients(8, seed=3)
    G, H = random_coefficients(3, seed=4)
    theta, phi = np.linspace(0.3, 2.8, 11), np.linspace(0.1, 6.1, 13)
    r, d = np.array([0.9, 1.3]), 1e-5
    R, sinT = r[:, None, None], np.sin(theta)[None, :, None]
    Br = lambda rr: field(rr, theta, phi, g, h, G, H)[1]
    Bt = lambda tt: field(r, tt, phi, g, h, G, H)[2]
    Bp = lambda pp: field(r, theta, pp, g, h, G, H)[3]
    div = ((r + d)[:, None, None] ** 2 * Br(r + d) - (r - d)[:, None, None] ** 2 * Br(r - d)) / (2 * d * R ** 2) \
        + (np.sin(theta + d)[None, :, None] * Bt(theta + d)
           - np.sin(theta - d)[None, :, None] * Bt(theta - d)) / (2 * d * R * sinT) \
        + (Bp(phi + d) - Bp(phi - d)) / (2 * d * R * sinT)
    scale = np.max(np.abs(field(r, theta, phi, g, h, G, H)[4])) / r.min()
    check_close("div(B)", div, 0, atol=1e-6 * scale)

# The vectorized grid calculation agrees with a direct point-by-point sum
def test_grid_matches_point_by_point():
    g, h = random_coefficients(6, seed=5)
    G, H = random_coefficients(2, seed=6)
    radius, theta, phi = np.array([0.8, 1.7]), np.linspace(0.2, 2.9, 7), np.linspace(0, 6.2, 9)
    V, Br, Bt, Bp, _ = field(radius, theta, phi, g, h, G, H)
    K, S = schmidt.KandS(7)
    P, dP = schmidt.Schmidtcoefficients(7, len(theta), theta, K, S)
    for i, r in enumerate(radius):
        for j, t in enumerate(theta):
            for k, p in enumerate(phi):
                v = br = bt = bp = 0.
                for n in range(1, 7):
                    for m in range(n + 1):
                        cos, sin = np.cos(m * p), np.sin(m * p)
                        a = g[n, m] * cos + h[n, m] * sin
                        b = m * (- g[n, m] * sin + h[n, m] * cos)
                        v += r ** (- n - 1) * a * P[n, m, j]
                        br += (n + 1) * r ** (- n - 2) * a * P[n, m, j]
                        bt -= r ** (- n - 2) * a * dP[n, m, j]
                        bp -= r ** (- n - 2) * b * P[n, m, j] / np.sin(t)
                        if n < 3:
                            a = G[n, m] * cos + H[n, m] * sin
                            b = m * (- G[n, m] * sin + H[n, m] * cos)
                            v += r ** n * a * P[n, m, j]
                            br -= n * r ** (n - 1) * a * P[n, m, j]
                            bt -= r ** (n - 1) * a * dP[n, m, j]
                            bp -= r ** (n - 1) * b * P[n, m, j] / np.sin(t)
                check_close("point (%d,%d,%d)" % (i, j, k),
                            [V[i, j, k], Br[i, j, k], Bt[i, j, k], Bp[i, j, k]],
                            [v, br, bt, bp], rtol=1e-11, atol=1e-9)

# Lowes spectrum of a dipole: R1 = 2 g10^2 / r^6
def test_lowes_dipole():
    g, h = np.zeros([3, 3]), np.zeros([3, 3])
    g[1, 0] = 410244.7
    for r in [0.8, 1., 2.]:
        Rn = lowes_spec.lowes_spec(3, r, g, h)
        check_close("Rn(r=%g)" % r, Rn, [0, 2 * g[1, 0] ** 2 / r ** 6, 0])

# The sum of the Lowes spectrum is the mean of |B|^2 over the sphere
def test_lowes_is_mean_square_field():
    nmax, r, const = 10, 0.9, 1e5
    g, h = random_coefficients(nmax, seed=7)
    theta, w = gauss_theta(2 * nmax + 2)
    Nphi = 4 * nmax + 4
    phi = 2 * np.pi * np.arange(Nphi) / Nphi
    Bmod = field(r, theta, phi, g, h, const=const)[4][0]
    mean_B2 = np.sum(w[:, None] * Bmod ** 2) / (2 * Nphi)
    Rn = lowes_spec.lowes_spec(nmax + 1, r, g, h)
    check_close("<B^2>", mean_B2 * const ** 2, np.sum(Rn), rtol=1e-10)

# The tables are read correctly (some published coefficients, in nT)
def test_reader_known_coefficients():
    g, h, _, _ = reader.reader("Earth", 2020, 14, 0)
    check_close("IGRF 2020", [g[1, 0], g[1, 1], h[1, 1]], [-29404.8, -1450.9, 4652.5])
    g, _, _, _ = reader.reader("Earth", 1900, 14, 0)
    check_close("IGRF 1900 g10", g[1, 0], -31543)
    g, _, _, _ = reader.reader("Jupiter", None, 11, 0)
    check_close("JRM09 g10", g[1, 0], 410244.7)
    g, _, _, _ = reader.reader("Jupiter_2021", None, 31, 2)
    check_close("JRM33 g10", g[1, 0], 410993.4)
    # A degree above the matrices is skipped, not an error
    g, _, _, _ = reader.reader("Jupiter_2021", None, 3, 0)
    assert g.shape == (3, 3)

# Every planet (and every Earth year) can be read, with finite coefficients,
# a non-zero dipole and no coefficient with m > n
def test_reader_all_planets():
    for planet in reader.PLANET_NAMES:
        years = reader.EARTH_YEARS if planet == "Earth" else [None]
        for year in years:
            NPOL, NPOL_EXT, _ = reader.planet_settings(planet, nmax=50)
            g, h, G, H = reader.reader(planet, year, NPOL, NPOL_EXT)
            name = planet + ("" if year is None else " " + str(year))
            assert g[1, 0] != 0, name + ": g10 is zero"
            for M in (g, h, G, H):
                assert np.all(np.isfinite(M)), name + ": not finite coefficients"
                assert not np.any(np.triu(M, 1)), name + ": coefficients with m > n"

# Surface field of every planet in gauss (this checks the const units, Uranus
# and Neptune tables are in gauss, the others in nT)
def test_surface_field_all_planets():
    expected = {"Earth": (0.2, 0.7), "Jupiter": (1, 25), "Jupiter_2021": (1, 25),
                "Saturn": (0.1, 1), "Uranus": (5e-3, 2), "Neptune": (5e-3, 2),
                "Mercury": (1e-3, 1e-2), "Ganymede": (5e-3, 3e-2), "My_own": (1e-3, 1e3)}
    theta, phi = np.linspace(0.01, np.pi - 0.01, 90), np.linspace(0, 2 * np.pi, 180)
    for planet in reader.PLANET_NAMES:
        NPOL, NPOL_EXT, const = reader.planet_settings(planet)
        g, h, G, H = reader.reader(planet, 2020, NPOL, NPOL_EXT)
        Bmod = field(1., theta, phi, g, h, G, H, const=const)[4]
        low, high = expected[planet]
        assert low < Bmod.min() and Bmod.max() < high, \
            "%s: |B| from %.3g to %.3g G, expected within (%g, %g)" \
            % (planet, Bmod.min(), Bmod.max(), low, high)

# IGRF 2020 at the surface: |B| from about 0.22 G (South Atlantic Anomaly)
# to about 0.67 G (near the South magnetic pole)
def test_earth_2020_surface_values():
    NPOL, NPOL_EXT, const = reader.planet_settings("Earth")
    g, h, G, H = reader.reader("Earth", 2020, NPOL, NPOL_EXT)
    theta, phi = np.linspace(0.005, np.pi - 0.005, 361), np.linspace(0, 2 * np.pi, 721)
    Bmod = field(1., theta, phi, g, h, G, H, const=const)[4][0]
    assert 0.215 < Bmod.min() < 0.235, "minimum |B| = %.4f G" % Bmod.min()
    assert 0.655 < Bmod.max() < 0.68, "maximum |B| = %.4f G" % Bmod.max()
    # The minimum is in the South Atlantic (latitude -50 to -10, longitude -90 to 0)
    j, k = np.unravel_index(np.argmin(Bmod), Bmod.shape)
    lat, lon = 90 - np.degrees(theta[j]), (np.degrees(phi[k]) + 180) % 360 - 180
    assert -50 < lat < -10 and -90 < lon < 0, "minimum at lat %.0f lon %.0f" % (lat, lon)

PHYSICS_TESTS = [test_schmidt_low_degrees,
                 test_schmidt_derivatives,
                 test_schmidt_normalization,
                 test_dipole,
                 test_external_uniform_field,
                 test_field_is_minus_grad_potential,
                 test_divergence_free,
                 test_grid_matches_point_by_point,
                 test_lowes_dipole,
                 test_lowes_is_mean_square_field,
                 test_reader_known_coefficients,
                 test_reader_all_planets,
                 test_surface_field_all_planets,
                 test_earth_2020_surface_values]

# ---------------------------------------------------------------------------
# Smoke tests: the scripts run in a temporary folder with a small grid and
# without a display. The plots are not checked, only that they are made.
# ---------------------------------------------------------------------------
# LaTeX can be installed but not work (e.g. missing fonts or permissions), so
# it is tested by rendering a small figure
def latex_works():
    code = ("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt;"
            "plt.rcParams['text.usetex'] = True; plt.title(r'$B_r \\, \\mathrm{(G)}$');"
            "plt.savefig(__import__('io').BytesIO(), format='png')")
    if shutil.which("latex") is None: return False
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0

# Runs a script as python script.py args in folder. With usetex=False the
# scripts use the Computer Modern fonts of matplotlib instead of LaTeX
def run_script(script, args, folder, usetex):
    wrapper = ("import sys, runpy; sys.path.insert(0, %r); import saveplots;"
               "latex_fonts = saveplots.latex_fonts;"
               "saveplots.latex_fonts = lambda usetex=True: latex_fonts(usetex=usetex and %r);"
               "sys.argv = [%r] + %r;"
               "runpy.run_path(%r, run_name='__main__')"
               % (REPO, usetex, script, args, os.path.join(REPO, script)))
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-c", wrapper], cwd=folder, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines() or ["no error message"]
        return "python %s %s\n  -> %s" % (script, " ".join(args), lines[-1])

# Runs several scripts as (script, args) and fails with all their errors
def run_scripts(commands, folder, usetex):
    errors = [run_script(script, args, folder, usetex) for script, args in commands]
    errors = [error for error in errors if error]
    if errors:
        raise AssertionError("%d of %d commands failed:\n" % (len(errors), len(commands))
                             + "\n".join(errors))

def count_png(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".png")]) if os.path.isdir(folder) else 0

def smoke_main(folder, usetex):
    commands = []
    for planet in reader.PLANET_NAMES:
        args = ["--planet", planet, "--ntheta", "16"]
        if planet == "Earth": args += ["--year", "1950", "--title", "--rc", "0.9"]
        if planet == "Jupiter_2021": args += ["--nmax", "18", "--no-external"]
        commands.append(("main.py", args))
    run_scripts(commands, folder, usetex)
    # Plane and Mollweide maps of 7 magnitudes (11 for Earth) and the Lowes spectrum
    for planet in reader.PLANET_NAMES:
        n = count_png(os.path.join(folder, planet))
        assert n >= 15, "main.py --planet %s made %d png files" % (planet, n)

def smoke_movies(folder, usetex):
    run_scripts([("main_movie.py", ["--planet", "Jupiter_2021", "--frames", "2", "--ntheta", "16",
                                    "--rmin", "0.9", "--rmax", "1.1"]),
                 ("main_movie.py", ["--planet", "Earth", "--year", "1950", "--frames", "2",
                                    "--ntheta", "16", "--magnitude", "fieldmod", "--no-title"]),
                 ("main_movie.py", ["--planet", "Mercury", "--external", "--frames", "1",
                                    "--ntheta", "16", "--title", "Mercury"]),
                 ("main_movie_Earth.py", ["--first-year", "2015", "--last-year", "2020",
                                          "--ntheta", "16"])], folder, usetex)
    folders = [f for f in os.listdir(folder) if "movie" in f]
    assert len(folders) >= 4, "movie folders made: " + ", ".join(folders)
    for f in folders:
        assert count_png(os.path.join(folder, f)) >= 2, "no frames in " + f

# The outputs that main.py has switched off by default (csv, vtu, the curl,
# divergence and curvature plots and the Lowes spectra at several radii)
def smoke_other_outputs(folder, usetex):
    code = r"""
import numpy as np
import reader, schmidt, saveoutput, saveplots, lowes_spec, magnitudes
saveplots.latex_fonts(usetex=%r)
planet, Ntheta, rc, dr = "Earth", 16, 1., 0.001
Nphi = 2 * Ntheta
NPOL, NPOL_EXT, const = reader.planet_settings(planet)
g, h, G, H = reader.reader(planet, 2020, NPOL, NPOL_EXT)
phi = np.linspace(0, 2 * np.pi, Nphi)
theta = np.linspace(np.pi / Ntheta, np.pi * (1 - 1 / Ntheta), Ntheta)
radius = np.linspace(rc * (1 - dr), rc * (1 + dr), 3)
K, S = schmidt.KandS(NPOL)
P, derivP = schmidt.Schmidtcoefficients(NPOL, Ntheta, theta, K, S)
V, Br, Bt, Bp, Bmod = schmidt.totalfield(radius, theta, phi, NPOL, NPOL_EXT, P, derivP, const, g, h, G, H)
saveoutput.savecsv(3, Ntheta, Nphi, radius, phi, theta, V, Br, Bt, Bp, Bmod, planet, 2020)
saveoutput.savevtu(3, Ntheta, Nphi, radius, phi, theta, Br, Bt, Bp, planet, 2020)
magnitudes.printMagnitudes(planet, Ntheta, Nphi, radius, rc, "1_00", dr, phi, theta,
                           Br, Bt, Bp, Bmod, plane=True, Mollweide=True)
radii = np.array([1.2, 1.0, 0.8])
lowes_spec.plot_multiple_lowes(planet, radii, np.array([lowes_spec.lowes_spec(NPOL, r, g, h) for r in radii]))
""" % usetex
    script = os.path.join(folder, "other_outputs.py")
    with open(script, "w") as file:
        file.write(code)
    run_scripts([(script, [])], folder, usetex)
    files = os.listdir(folder)
    for end in ["_fieldr.csv", ".vtu"]:
        assert any(f.endswith(end) for f in files), "no *%s file" % end
    assert os.path.isfile(os.path.join(folder, "Earth", "Earth_lowes_spectrum.png"))
    assert count_png(os.path.join(folder, "Earth")) >= 4, "no curl/divergence/curvature plots"

SMOKE_TESTS = [smoke_main, smoke_movies, smoke_other_outputs]

# ---------------------------------------------------------------------------
# Runs a list of tests and prints PASS/FAIL for each one
# ---------------------------------------------------------------------------
def run(tests, *args, verbose=False):
    failed = []
    for test in tests:
        start = time.time()
        try:
            test(*args)
            status, message = "PASS", ""
        except Exception as error:
            status, message = "FAIL", str(error) if not verbose else traceback.format_exc()
            failed.append(test.__name__)
        print("  %s  %-36s %5.1f s" % (status, test.__name__, time.time() - start), flush=True)
        if message: print("        " + message.replace("\n", "\n        "))
    return failed

def main():
    parser = argparse.ArgumentParser(
        description="Physics and smoke tests of the magnetic field code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python test.py\n"
               "  python test.py --physics\n"
               "  python test.py --smoke --no-latex\n"
               "  python test.py --smoke --keep smoke_output")
    parser.add_argument("--physics", action="store_true", help="only run the physics tests")
    parser.add_argument("--smoke", action="store_true", help="only run the smoke tests")
    parser.add_argument("--no-latex", dest="latex", action="store_false",
                        help="smoke tests with the matplotlib fonts instead of LaTeX (faster)")
    parser.add_argument("--keep", metavar="FOLDER",
                        help="save the smoke test plots in FOLDER instead of deleting them")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the full error tracebacks")
    args = parser.parse_args()
    physics, smoke = args.physics or not args.smoke, args.smoke or not args.physics

    failed = []
    if physics:
        print("------------------------------------------------------------")
        print("Physics tests")
        failed += run(PHYSICS_TESTS, verbose=args.verbose)
    if smoke:
        print("------------------------------------------------------------")
        usetex = args.latex and latex_works()
        if args.latex and not usetex:
            print("LaTeX does not work, the smoke tests use the matplotlib fonts")
        print("Smoke tests (%s, it can take a few minutes)"
              % ("LaTeX fonts" if usetex else "matplotlib fonts"))
        import matplotlib
        matplotlib.use("Agg")
        if args.keep:
            folder = os.path.abspath(args.keep)
            os.makedirs(folder, exist_ok=True)
        else:
            folder = tempfile.mkdtemp(prefix="magnetic_field_test_")
        try:
            failed += run(SMOKE_TESTS, folder, usetex, verbose=args.verbose)
        finally:
            if args.keep: print("Plots saved in " + folder)
            else: shutil.rmtree(folder, ignore_errors=True)
    print("------------------------------------------------------------")
    if failed:
        print("%d test(s) failed: %s" % (len(failed), ", ".join(failed)))
    else:
        print("All tests passed")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
