import datetime
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MovieRating:
    movieId: int

    rating: float

    timestamp: datetime


@dataclass
class Movie:
    dbId: int

    title: str

    genre: list[str]

    releaseDate: datetime

    popularity: float


@dataclass
class User:
    userId: int

    ratings: dict

    genres: dict

    Location: str = ''

    def rate_movie(self, movie: Movie, rating: MovieRating):
        self.ratings[movie.movieId] = rating
        for genre in movie.genre:
            self.genres[genre] = rating

    def top_genres(self, n=5):
        # calc the avg rating per genre
        genre_avg = {g: sum(r)/len(r) for g, r in self.genres.items()}
        top = sorted(genre_avg.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, _ in top[:n]]

    def avg_rating(self):
        if not self.ratings:
            return 0
        return sum(self.ratings.values()) / len(self.ratings)
