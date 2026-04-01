# Project 1 Pipeline: Movie Recommendation Dataset and Analysis

This notebook creates a secondary relational dataset from the MovieLens “latest small” dataset and uses it to build a movie recommendation pipeline. The notebook first cleans and normalizes the raw data into multiple related tables. Then it loads the processed data into DuckDB, runs SQL queries for exploration, and implements both a baseline recommendation model and a collaborative filtering model. The goal is to generate movie recommendations in a way that is structured, explainable, and aligned with the relational model.

#### Imports and Set up


```python
import os
import logging
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Create folders in the project root, not inside pipeline/
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../figures", exist_ok=True)
os.makedirs("../logs", exist_ok=True)

# Write logs to the top-level logs folder
logging.basicConfig(
    filename="../logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Pipeline started")
```

#### Raw Data Loading

In this section, the raw MovieLens files are loaded from the `data/raw` folder. Error handling is included so that the notebook clearly reports if any file is missing or cannot be read.


```python
# Load raw MovieLens data from the top-level data/raw folder
try:
    movies = pd.read_csv("../data/raw/movies.csv")
    ratings = pd.read_csv("../data/raw/ratings.csv")
    tags = pd.read_csv("../data/raw/tags.csv")
    links = pd.read_csv("../data/raw/links.csv")
    logging.info("Raw data loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Missing file: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error while loading raw data: {e}")
    raise
```


```python
print("movies:", movies.shape)
print("ratings:", ratings.shape)
print("tags:", tags.shape)
print("links:", links.shape)

display(movies.head())
display(ratings.head())
display(tags.head())
display(links.head())
```

    movies: (9742, 3)
    ratings: (100836, 4)
    tags: (3683, 4)
    links: (9742, 3)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>60756</td>
      <td>funny</td>
      <td>1445714994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>60756</td>
      <td>Highly quotable</td>
      <td>1445714996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>60756</td>
      <td>will ferrell</td>
      <td>1445714992</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>89774</td>
      <td>Boxing story</td>
      <td>1445715207</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>89774</td>
      <td>MMA</td>
      <td>1445715200</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>imdbId</th>
      <th>tmdbId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>113497</td>
      <td>8844.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113228</td>
      <td>15602.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>114885</td>
      <td>31357.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>113041</td>
      <td>11862.0</td>
    </tr>
  </tbody>
</table>
</div>


#### Processed Dataset Creation

In this section, the raw MovieLens files are transformed into a more normalized relational dataset. A users table is created from unique user IDs, release year is extracted from movie titles, and the multi-valued genres field is normalized into separate `genres` and `movie_genres` tables. This produces a secondary dataset with seven related tables that can support recommendation analysis.


```python
# Create a users table from all unique user IDs that appear in ratings and tags
users = pd.DataFrame({
    "userId": sorted(pd.concat([ratings["userId"], tags["userId"]]).unique())
})

print("users:", users.shape)
display(users.head())
```

    users: (610, 1)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Copy the movies table and extract the release year from the title
movies_clean = movies.copy()
movies_clean["releaseYear"] = movies_clean["title"].str.extract(r"\((\d{4})\)")
movies_clean["releaseYear"] = pd.to_numeric(movies_clean["releaseYear"], errors="coerce")

print("movies_clean:", movies_clean.shape)
display(movies_clean.head())
```

    movies_clean: (9742, 4)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>releaseYear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>1995.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Split the multi-valued genres field into one genre per row
movie_genres = movies_clean[["movieId", "genres"]].copy()
movie_genres["genreName"] = movie_genres["genres"].str.split("|")
movie_genres = movie_genres.explode("genreName")
movie_genres = movie_genres[["movieId", "genreName"]].drop_duplicates()

print("movie_genres step 1:", movie_genres.shape)
display(movie_genres.head())
```

    movie_genres step 1: (22084, 2)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>genreName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Children</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Create a separate genres table with one unique row per genre
genres = pd.DataFrame({
    "genreName": sorted(movie_genres["genreName"].dropna().unique())
})

genres["genreId"] = range(1, len(genres) + 1)
genres = genres[["genreId", "genreName"]]

print("genres:", genres.shape)
display(genres.head())
```

    genres: (20, 2)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genreId</th>
      <th>genreName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>(no genres listed)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Children</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Replace genre names with genre IDs in the bridge table
movie_genres = movie_genres.merge(genres, on="genreName", how="left")
movie_genres = movie_genres[["movieId", "genreId"]].drop_duplicates()

print("movie_genres final:", movie_genres.shape)
display(movie_genres.head())
```

    movie_genres final: (22084, 2)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>genreId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Keep the final versions of the processed tables
movies_final = movies_clean[["movieId", "title", "releaseYear"]].copy()
ratings_final = ratings.copy()
tags_final = tags.copy()
links_final = links.copy()

print("users:", users.shape)
print("movies_final:", movies_final.shape)
print("ratings_final:", ratings_final.shape)
print("tags_final:", tags_final.shape)
print("links_final:", links_final.shape)
print("genres:", genres.shape)
print("movie_genres:", movie_genres.shape)
```

    users: (610, 1)
    movies_final: (9742, 3)
    ratings_final: (100836, 4)
    tags_final: (3683, 4)
    links_final: (9742, 3)
    genres: (20, 2)
    movie_genres: (22084, 2)
    

#### Save Processed Data

The processed relational tables are saved as CSV files in the `data/processed` folder. These files form the secondary dataset used in the rest of the project.


```python
# Save processed tables to the top-level data/processed folder
try:
    users.to_csv("../data/processed/users.csv", index=False)
    movies_final.to_csv("../data/processed/movies.csv", index=False)
    ratings_final.to_csv("../data/processed/ratings.csv", index=False)
    tags_final.to_csv("../data/processed/tags.csv", index=False)
    links_final.to_csv("../data/processed/links.csv", index=False)
    genres.to_csv("../data/processed/genres.csv", index=False)
    movie_genres.to_csv("../data/processed/movie_genres.csv", index=False)
    logging.info("Processed data saved successfully")
except Exception as e:
    logging.error(f"Error saving processed data: {e}")
    raise
```

#### Load Processed Data into DuckDB

This section loads the processed CSV files into DuckDB using Python. DuckDB makes it easy to run SQL queries directly on the relational dataset without needing a separate database server.


```python
# Connect to DuckDB and load the processed CSV files as tables
con = duckdb.connect()

con.execute("CREATE OR REPLACE TABLE users AS SELECT * FROM '../data/processed/users.csv'")
con.execute("CREATE OR REPLACE TABLE movies AS SELECT * FROM '../data/processed/movies.csv'")
con.execute("CREATE OR REPLACE TABLE ratings AS SELECT * FROM '../data/processed/ratings.csv'")
con.execute("CREATE OR REPLACE TABLE tags AS SELECT * FROM '../data/processed/tags.csv'")
con.execute("CREATE OR REPLACE TABLE links AS SELECT * FROM '../data/processed/links.csv'")
con.execute("CREATE OR REPLACE TABLE genres AS SELECT * FROM '../data/processed/genres.csv'")
con.execute("CREATE OR REPLACE TABLE movie_genres AS SELECT * FROM '../data/processed/movie_genres.csv'")

logging.info("Processed CSV files loaded into DuckDB")
print(con.execute("SHOW TABLES").fetchall())
```

    [('genres',), ('links',), ('movie_genres',), ('movies',), ('ratings',), ('tags',), ('users',)]
    

#### Exploratory SQL Queries

Before building recommendation models, SQL queries are used to explore the dataset. These queries help show patterns in the data and prepare for the recommendation analysis.


```python
# Count how many ratings are associated with each genre
query1 = con.execute("""
SELECT g.genreName, COUNT(*) AS rating_count
FROM ratings r
JOIN movie_genres mg ON r.movieId = mg.movieId
JOIN genres g ON mg.genreId = g.genreId
GROUP BY g.genreName
ORDER BY rating_count DESC
LIMIT 10
""").fetchdf()

query1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genreName</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Drama</td>
      <td>41928</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Comedy</td>
      <td>39053</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action</td>
      <td>30635</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thriller</td>
      <td>26452</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adventure</td>
      <td>24161</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Romance</td>
      <td>18124</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sci-Fi</td>
      <td>17243</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Crime</td>
      <td>16681</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fantasy</td>
      <td>11834</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Children</td>
      <td>9208</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Find the highest rated movies with at least 50 ratings
query2 = con.execute("""
SELECT m.title, AVG(r.rating) AS avg_rating, COUNT(*) AS num_ratings
FROM ratings r
JOIN movies m ON r.movieId = m.movieId
GROUP BY m.title
HAVING COUNT(*) > 50
ORDER BY avg_rating DESC
LIMIT 10
""").fetchdf()

query2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>avg_rating</th>
      <th>num_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.429022</td>
      <td>317</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Godfather, The (1972)</td>
      <td>4.289062</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fight Club (1999)</td>
      <td>4.272936</td>
      <td>218</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cool Hand Luke (1967)</td>
      <td>4.271930</td>
      <td>57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>4.268041</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rear Window (1954)</td>
      <td>4.261905</td>
      <td>84</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Godfather: Part II, The (1974)</td>
      <td>4.259690</td>
      <td>129</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Departed, The (2006)</td>
      <td>4.252336</td>
      <td>107</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Goodfellas (1990)</td>
      <td>4.250000</td>
      <td>126</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Casablanca (1942)</td>
      <td>4.240000</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



#### Baseline Recommendation Model
A baseline recommendation model was implemented using average ratings and popularity. Movies are ranked based on their average rating, with a minimum number of ratings required to make the recommendations more reliable.

#### Why This Approach Was Used
This model provides a simple starting point for recommendations and helps identify movies that are widely liked by many users. However, it does not account for individual user preferences, so the same movies are recommended to everyone.


```python
# Recommend movies that are highly rated and have enough ratings to be reliable
recommendations = con.execute("""
SELECT 
    m.movieId,
    m.title,
    AVG(r.rating) AS avg_rating,
    COUNT(r.rating) AS num_ratings
FROM ratings r
JOIN movies m ON r.movieId = m.movieId
GROUP BY m.movieId, m.title
HAVING COUNT(r.rating) > 50
ORDER BY avg_rating DESC
LIMIT 10
""").fetchdf()

recommendations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>avg_rating</th>
      <th>num_ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.429022</td>
      <td>317</td>
    </tr>
    <tr>
      <th>1</th>
      <td>858</td>
      <td>Godfather, The (1972)</td>
      <td>4.289062</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2959</td>
      <td>Fight Club (1999)</td>
      <td>4.272936</td>
      <td>218</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1276</td>
      <td>Cool Hand Luke (1967)</td>
      <td>4.271930</td>
      <td>57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>4.268041</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>904</td>
      <td>Rear Window (1954)</td>
      <td>4.261905</td>
      <td>84</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1221</td>
      <td>Godfather: Part II, The (1974)</td>
      <td>4.259690</td>
      <td>129</td>
    </tr>
    <tr>
      <th>7</th>
      <td>48516</td>
      <td>Departed, The (2006)</td>
      <td>4.252336</td>
      <td>107</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1213</td>
      <td>Goodfellas (1990)</td>
      <td>4.250000</td>
      <td>126</td>
    </tr>
    <tr>
      <th>9</th>
      <td>912</td>
      <td>Casablanca (1942)</td>
      <td>4.240000</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



#### Basline Visualization Rationale

A horizontal bar chart is used because movie titles are easier to read on the y-axis. This makes it easier to compare the average ratings of the recommended movies and present the results clearly.


```python
# Visualize the top baseline recommendations
plt.figure(figsize=(10, 6))
plt.barh(recommendations["title"], recommendations["avg_rating"])
plt.xlabel("Average Rating")
plt.ylabel("Movie Title")
plt.title("Top Recommended Movies Based on Average Rating")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../figures/top_recommended_movies.png", bbox_inches="tight")
plt.show()
```


    
![png](project1_pipeline_files/project1_pipeline_23_0.png)
    


#### Collaborative Filtering Model

A collaborative filtering approach was implemented to generate personalized recommendations. Instead of recommending the same movies to every user, this method identifies users with similar rating patterns and recommends movies those similar users liked.

#### Why This Works

This method captures patterns in user behavior and makes recommendations more personalized than the baseline model. Cosine similarity is used to compare users based on their rating vectors.


```python
# Create a user-movie rating matrix
user_movie = ratings_final.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print(user_movie.shape)
user_movie.head()
```

    (610, 9724)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9724 columns</p>
</div>




```python
# Measure similarity between users based on rating patterns
user_similarity = cosine_similarity(user_movie)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie.index,
    columns=user_movie.index
)

user_similarity_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>userId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
      <th>610</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.027283</td>
      <td>0.059720</td>
      <td>0.194395</td>
      <td>0.129080</td>
      <td>0.128152</td>
      <td>0.158744</td>
      <td>0.136968</td>
      <td>0.064263</td>
      <td>0.016875</td>
      <td>...</td>
      <td>0.080554</td>
      <td>0.164455</td>
      <td>0.221486</td>
      <td>0.070669</td>
      <td>0.153625</td>
      <td>0.164191</td>
      <td>0.269389</td>
      <td>0.291097</td>
      <td>0.093572</td>
      <td>0.145321</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.027283</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.003726</td>
      <td>0.016614</td>
      <td>0.025333</td>
      <td>0.027585</td>
      <td>0.027257</td>
      <td>0.000000</td>
      <td>0.067445</td>
      <td>...</td>
      <td>0.202671</td>
      <td>0.016866</td>
      <td>0.011997</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028429</td>
      <td>0.012948</td>
      <td>0.046211</td>
      <td>0.027565</td>
      <td>0.102427</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.059720</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.002251</td>
      <td>0.005020</td>
      <td>0.003936</td>
      <td>0.000000</td>
      <td>0.004941</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.005048</td>
      <td>0.004892</td>
      <td>0.024992</td>
      <td>0.000000</td>
      <td>0.010694</td>
      <td>0.012993</td>
      <td>0.019247</td>
      <td>0.021128</td>
      <td>0.000000</td>
      <td>0.032119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.194395</td>
      <td>0.003726</td>
      <td>0.002251</td>
      <td>1.000000</td>
      <td>0.128659</td>
      <td>0.088491</td>
      <td>0.115120</td>
      <td>0.062969</td>
      <td>0.011361</td>
      <td>0.031163</td>
      <td>...</td>
      <td>0.085938</td>
      <td>0.128273</td>
      <td>0.307973</td>
      <td>0.052985</td>
      <td>0.084584</td>
      <td>0.200395</td>
      <td>0.131746</td>
      <td>0.149858</td>
      <td>0.032198</td>
      <td>0.107683</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.129080</td>
      <td>0.016614</td>
      <td>0.005020</td>
      <td>0.128659</td>
      <td>1.000000</td>
      <td>0.300349</td>
      <td>0.108342</td>
      <td>0.429075</td>
      <td>0.000000</td>
      <td>0.030611</td>
      <td>...</td>
      <td>0.068048</td>
      <td>0.418747</td>
      <td>0.110148</td>
      <td>0.258773</td>
      <td>0.148758</td>
      <td>0.106435</td>
      <td>0.152866</td>
      <td>0.135535</td>
      <td>0.261232</td>
      <td>0.060792</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 610 columns</p>
</div>




```python
# Select a target user and find the most similar users
target_user = 1

similar_users = user_similarity_df[target_user].sort_values(ascending=False)
similar_users = similar_users.drop(target_user)
top_similar_users = similar_users.head(5)

top_similar_users
```




    userId
    266    0.357408
    313    0.351562
    368    0.345127
    57     0.345034
    91     0.334727
    Name: 1, dtype: float64




```python
# Get highly rated movies from similar users
similar_users_ids = top_similar_users.index

similar_users_ratings = ratings_final[
    ratings_final["userId"].isin(similar_users_ids)
]

liked_movies = similar_users_ratings[
    similar_users_ratings["rating"] >= 4
]
```


```python
# Remove movies the target user has already rated
target_user_movies = ratings_final[
    ratings_final["userId"] == target_user
]["movieId"]

recommendation_candidates = liked_movies[
    ~liked_movies["movieId"].isin(target_user_movies)
]
```


```python
# Rank candidate movies by how often similar users rated them highly
recommendations_cf = (
    recommendation_candidates
    .groupby("movieId")
    .size()
    .reset_index(name="score")
    .sort_values("score", ascending=False)
    .head(10)
)

recommendations_cf = recommendations_cf.merge(
    movies_final[["movieId", "title"]],
    on="movieId"
)

recommendations_cf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>score</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1036</td>
      <td>5</td>
      <td>Die Hard (1988)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200</td>
      <td>5</td>
      <td>Aliens (1986)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1242</td>
      <td>4</td>
      <td>Glory (1989)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1221</td>
      <td>4</td>
      <td>Godfather: Part II, The (1974)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>858</td>
      <td>4</td>
      <td>Godfather, The (1972)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1610</td>
      <td>4</td>
      <td>Hunt for Red October, The (1990)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>589</td>
      <td>4</td>
      <td>Terminator 2: Judgment Day (1991)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>541</td>
      <td>4</td>
      <td>Blade Runner (1982)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2762</td>
      <td>4</td>
      <td>Sixth Sense, The (1999)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>32</td>
      <td>4</td>
      <td>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</td>
    </tr>
  </tbody>
</table>
</div>



#### Visualization Rationale

A horizontal bar chart is again used here because it makes the movie titles readable and allows the recommendation scores to be compared clearly. This chart helps show which movies are the strongest personalized recommendations for the selected user.


```python
# Visualize the collaborative filtering recommendations
plt.figure(figsize=(10, 6))
plt.barh(recommendations_cf["title"], recommendations_cf["score"])
plt.xlabel("Recommendation Score")
plt.ylabel("Movie Title")
plt.title("Collaborative Filtering Recommendations for User 1")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../figures/collaborative_filtering_recommendations.png", bbox_inches="tight")
plt.show()

logging.info("Pipeline completed successfully")
```


    
![png](project1_pipeline_files/project1_pipeline_32_0.png)
    

