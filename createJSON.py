from imdb import Cinemagoer
import json
import os

ia = Cinemagoer()

top250_movies = ia.get_top250_movies()

# Fetch genres information for each movie
movie_genres = {}
for movie in top250_movies:
    info = ia.get_movie(movie.movieID)
    movie_genres[info.movieID] = {
        'genres': info.get('genres', []),
        'plot_outline': info.get('plot outline')
    }

# Specify the directory where you want to save the file
output_directory = ''

# Save movie genres to a JSON file in the specified directory
output_file = os.path.join(output_directory, 'top250_movies.json')
with open(output_file, 'w') as json_file:
    json.dump(movie_genres, json_file)

print(f"Movie genres saved to {output_file}")
