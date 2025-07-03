# RecommendIt

RecommendIt is a content-based movie recommendation API built with FastAPI and scikit-learn. It uses TF-IDF vectorization and cosine similarity to suggest movies based on title, genre, and release year. It also supports fuzzy matching of movie titles using Levenshtein distance.

## Features

- Recommend similar movies based on title and metadata
- Optional filters by genre and release year
- Fuzzy title matching with Levenshtein distance
- Built with FastAPI and scikit-learn
- Dockerized and ready to run

## How to Run

1.	Build the Docker image:

```bash
docker build -t recommendit .
```

2.	Run the container:

```bash
docker run -p 8000:8000 recommendit
```

3.	Make a test request:
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception"}'
```

Receive a JSON response with top 5 movie recommendations.
