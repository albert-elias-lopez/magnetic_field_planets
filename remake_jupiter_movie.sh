#!/bin/bash
# Remakes the Jupiter (JRM33, n <= 18) radial movie, from r = 2.0 to 0.8 in 241
# frames, and its reverse at 6 frames per second
set -e
cd "$(dirname "$0")"

# Movie frames
python main_movie.py

# Movie
ffmpeg -y -framerate 6 -i Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide.mp4

# Reversed movie
ffmpeg -y -framerate 6 -i Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide_r_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -vf "reverse,pad=ceil(iw/2)*2:ceil(ih/2)*2" Jupiter_2021_movie_n18/Jupiter_2021_fieldr_Mollweide_reversed.mp4

# Static maps with title
python main.py --planet Jupiter_2021 --title
