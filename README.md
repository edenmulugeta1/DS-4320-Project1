# DS-4320-Project1
DS 4320 Project 1 repository for a movie recommendation system using the relational model and MovieLens data.

DS-4320-Project1
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   └── processed/
│       ├── users.csv
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       ├── links.csv
│       ├── genres.csv
│       └── movie_genres.csv
│
├── pipeline/
│   ├── project1_pipeline.ipynb
│   ├── project1_pipeline.md
│   ├── build_dataset.py
│   ├── load_duckdb.py
│   ├── recommender.py
│   └── sql/
│       ├── create_tables.sql
│       └── analysis_queries.sql
│
├── press_release/
│   └── press_release.md
│
├── background_readings/
│   ├── reading_01.pdf
│   ├── reading_02.pdf
│   ├── reading_03.pdf
│   ├── reading_04.pdf
│   └── reading_05.pdf
│
├── figures/
│   └── top10_recommendations.png
│
└── logs/
    └── pipeline.log
