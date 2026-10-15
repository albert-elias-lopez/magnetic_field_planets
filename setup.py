# ---------------------------------------------------------------------------
# setup.py checks that your environment has everything the code needs:
# Python, numpy, matplotlib, cartopy (and its coastlines), LaTeX, ffmpeg and
# the data tables. Then it runs test.py (physics and smoke tests). Use:
# python setup.py
# (see python setup.py --help for the options). This is not a package
# installer, the code is run directly from this folder.
# ---------------------------------------------------------------------------
import os
import sys
import shutil
import argparse
import subprocess
import importlib

REPO = os.path.dirname(os.path.abspath(__file__))

# Python packages: (import name, pip name, required, what it is used for)
PACKAGES = [("numpy", "numpy", True, "all the calculations"),
            ("matplotlib", "matplotlib", True, "all the plots"),
            ("cartopy", "cartopy", True, "Mollweide maps and Earth coastlines")]

# pip and setuptools run setup.py with these commands, which this file is not for
if len(sys.argv) > 1 and sys.argv[1] in ["install", "develop", "egg_info", "bdist_wheel",
                                         "sdist", "build", "dist_info", "--name", "--version"]:
    print("This setup.py only checks the environment, the code is run directly from this "
          "folder (python main.py). Use python setup.py --install to install the packages.")
    sys.exit(1)

results = []

def report(status, name, message=""):
    results.append(status)
    print("  %-4s  %-22s %s" % (status, name, message), flush=True)

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
def check_python():
    version = "%d.%d.%d" % sys.version_info[:3]
    if sys.version_info >= (3, 9):
        report("PASS", "Python", version + " (" + sys.executable + ")")
    else:
        report("FAIL", "Python", version + ", at least 3.9 is needed")

def check_packages():
    missing = []
    for module, pip_name, required, use in PACKAGES:
        try:
            version = importlib.import_module(module).__version__
            report("PASS", module, version)
        except Exception as error:
            missing.append(pip_name)
            report("FAIL" if required else "WARN", module,
                   "not found, needed for %s (%s)" % (use, type(error).__name__))
    return missing

# Cartopy downloads the Natural Earth coastlines the first time they are used
# (it needs internet then), which the Earth plots need
def check_coastlines():
    try:
        import cartopy.feature
        feature = cartopy.feature.NaturalEarthFeature("physical", "coastline", "110m")
        next(iter(feature.geometries()))
        report("PASS", "cartopy coastlines")
    except ImportError:
        report("FAIL", "cartopy coastlines", "cartopy is not installed")
    except Exception as error:
        report("FAIL", "cartopy coastlines",
               "needed for the Earth plots, cartopy downloads them the first time "
               "(needs internet): " + str(error).splitlines()[0][:150])

# The plots use LaTeX fonts. Without LaTeX the matplotlib fonts are used, but
# if LaTeX is installed and does not work the plots crash
def check_latex():
    if shutil.which("latex") is None:
        report("WARN", "LaTeX", "not found, the plots will use the matplotlib fonts "
                                "(install e.g. texlive-latex-extra, texlive-fonts-recommended, "
                                "dvipng and cm-super)")
        return
    try:
        import test
        works = test.latex_works()
    except Exception:
        works = False
    if works:
        report("PASS", "LaTeX", shutil.which("latex"))
    else:
        report("FAIL", "LaTeX", "installed but it cannot render a plot, so the plots will "
                                "crash (check texlive-latex-extra, texlive-fonts-recommended, "
                                "dvipng, cm-super and the permissions of ~/.cache/matplotlib)")

# ffmpeg joins the movie frames in a movie file (it is not needed otherwise)
def check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        report("WARN", "ffmpeg", "not found, only needed to join the movie frames in a movie")
        return
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
    except Exception:
        ok = False
    if ok:
        report("PASS", "ffmpeg", result.stdout.split("\n")[0][:60])
    else:
        report("WARN", "ffmpeg", ffmpeg + " cannot be run, only needed to join the movie frames")

# All the tables in data/ can be read
def check_data():
    try:
        import reader
    except ImportError as error:
        report("FAIL", "data tables", "reader.py cannot be imported (%s)" % error)
        return
    failed = []
    for planet in reader.PLANET_NAMES:
        try:
            NPOL, NPOL_EXT, _ = reader.planet_settings(planet)
            reader.reader(planet, 2020, NPOL, NPOL_EXT)
        except Exception as error:
            failed.append("%s (%s)" % (planet, error))
    if failed:
        report("FAIL", "data tables", "cannot read: " + ", ".join(failed))
    else:
        report("PASS", "data tables", "%d planets in %s" % (len(reader.PLANET_NAMES), reader.DATA_DIR))

def install(packages):
    command = [sys.executable, "-m", "pip", "install"] + packages
    print("------------------------------------------------------------")
    print(" ".join(command))
    return subprocess.run(command).returncode == 0

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Checks the environment and runs the tests of the magnetic field code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python setup.py\n"
               "  python setup.py --install\n"
               "  python setup.py --quick\n"
               "  python setup.py --no-tests\n"
               "  python setup.py --no-latex")
    parser.add_argument("--install", action="store_true",
                        help="install the missing Python packages with pip")
    parser.add_argument("--quick", action="store_true",
                        help="only run the physics tests, not the smoke tests (no plots)")
    parser.add_argument("--no-tests", dest="tests", action="store_false",
                        help="only check the environment, do not run test.py")
    parser.add_argument("--no-latex", dest="latex", action="store_false",
                        help="smoke tests with the matplotlib fonts instead of LaTeX (faster)")
    args = parser.parse_args()
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    print("------------------------------------------------------------")
    print("Environment")
    check_python()
    missing = check_packages()
    if missing and args.install:
        if install(missing):
            importlib.invalidate_caches()
            results.clear()
            print("------------------------------------------------------------")
            print("Environment (after installing)")
            check_python()
            missing = check_packages()
    check_coastlines()
    check_latex()
    check_ffmpeg()
    check_data()

    failed = results.count("FAIL")
    print("------------------------------------------------------------")
    if missing and not args.install:
        print("Install the missing packages with python setup.py --install, or with")
        print("  pip install " + " ".join(missing))
        print("  conda install -c conda-forge " + " ".join(missing))
    if failed:
        print("%d environment check(s) failed, %d warning(s)" % (failed, results.count("WARN")))
    else:
        print("Environment ready (%d warning(s))" % results.count("WARN"))

    # The tests need numpy and matplotlib (and cartopy for the smoke tests)
    if not args.tests: return 1 if failed else 0
    if missing:
        print("The tests are not run until the missing packages are installed")
        return 1
    command = [sys.executable, os.path.join(REPO, "test.py")]
    if args.quick: command.append("--physics")
    if not args.latex: command.append("--no-latex")
    sys.stdout.flush()
    tests = subprocess.run(command).returncode
    return 1 if failed or tests else 0

if __name__ == "__main__":
    sys.exit(main())
