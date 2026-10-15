# magnetic_field_planets

This is an open-source repository used to reconstruct the planetary magnetic field with the usual Schmidt semi-normalised spherical harmonic coefficients for Earth, Jupiter, Saturn, Neptune, Uranus, Mercury and Ganymede. All data tables in the directory data/ can be found (or constructed) with the open scientific data from the corresponding space missions.

To run the general code in terminal you only need to run:

    python main.py

which by default plots Earth (2020) at r = 1. The planet, year (only for Earth, which uses the IGRF table: 1900, 1905, ..., 2020), radius and multipoles are chosen with arguments, for example:

    python main.py --planet Earth --year 1950 --rc 0.83
    python main.py --planet Jupiter_2021 --nmax 18 --no-external

The planets available are Earth, Jupiter, Jupiter_2021, Saturn, Uranus, Neptune, Mercury, Ganymede and My_own (use python main.py --help to see all the options). Depending on your system you may need python3 instead of python, and you can also use anaconda, jupyter, etc. This will make surface (in latitude and longitude) plots of whatever planet and radius you chose. You should play around with the first 50ish lines in main.py. The files main_movie.py is for plotting recursively the magnetic field at different radii to observe the changes in a gif or mp4. Similarly, main_movie_Earth.py plots recursively the every-5-year update of Earth's magnetic field. In main_movie.py (and main.py) you can choose the maximum degree of the multipoles to include (e.g. 18 for the Juno model JRM33, Jupiter_2021) and whether the external field is included. Both movie scripts also take arguments (see python main_movie.py --help and python main_movie_Earth.py --help for examples), e.g. python main_movie.py --planet Jupiter_2021 --nmax 18 --rmin 0.8 --rmax 2.0 --frames 241 or python main_movie_Earth.py --first-year 1950 --last-year 2020 --rc 0.55, and print the ffmpeg commands to join the frames in a movie (and its reverse for main_movie.py). 

To enable all the plotting options (Mollweide projection and Earth's coastlines) you need the cartopy library, which ideally the following command will be enough:

pip install cartopy

If you use anaconda you should use:

conda install -c conda-forge cartopy

For specific solution look in https://scitools.org.uk/cartopy/docs/latest/installing.html, or as the powerful Mr Google. Some ubuntu version are a bit more complex. For example, I had to fight a bit:

sudo apt-get install libproj-dev proj-data proj-bin  
sudo apt-get install libgeos-dev  
sudo pip install cython  
sudo pip install cartopy

Feel free to use and change!
