import pandas as pd

def load_data():
    try:
        ratings = pd.read_csv("data/ratings.csv")
        movies = pd.read_csv("data/movies.csv")
        tags = pd.read_csv("data/tags.csv")
        links = pd.read_csv("data/links.csv")

        print("Data loaded successfully!")

        return ratings, movies, tags, links

    except Exception as e:
        print("Error loading data:", e)
