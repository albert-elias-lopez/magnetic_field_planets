# magnetic_field_planets

This is an open-source repository used to reconstruct the planetary magnetic field with the usual Schmidt semi-normalised spherical harmonic coefficients for Earth, Jupiter, Saturn, Neptune, Uranus, Mercury and Ganymede. All data tables in the directory data/ can be found (or constructed) with the open scientific data from the corresponding space missions.

To run the general code in terminal you only need to run:

    python main.py

(or python3, depends on your system). Otherwise use anaconda, jupyter, etc. This will make surface (in latitude and longitude) plots of whatever planet and radius you chose. You should play around with the first 50ish lines in main.py. The files main_movie.py is for plotting recursively the magnetic field at different radii to observe the changes in a gif or mp4. Similarly, main_movie_Earth.py plots recursively the every-5-year update of Earth's magnetic field. 

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
