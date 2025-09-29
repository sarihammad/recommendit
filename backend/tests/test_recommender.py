import pandas as pd
from app.recommender import Recommender

sample = pd.DataFrame({
    "title": ["Inception", "Interstellar", "The Matrix"],
    "overview": [
        "A thief who steals corporate secrets through dream-sharing technology.",
        "A team of explorers travel through a wormhole in space.",
        "A computer hacker learns about the true nature of reality."
    ],
    "genres": ["Sci-Fi", "Sci-Fi", "Action"],
    "keywords": ["dream, heist", "space, wormhole", "hacker, simulation"],
    "cast": ["Leonardo DiCaprio", "Matthew McConaughey", "Keanu Reeves"],
    "director": ["Christopher Nolan", "Christopher Nolan", "Wachowski"],
    "release_date": ["2010-07-16", "2014-11-07", "1999-03-31"]
})

sample["combined"] = (
    sample["title"] + " " +
    sample["overview"] + " " +
    sample["genres"] + " " +
    sample["keywords"] + " " +
    sample["cast"] + " " +
    sample["director"]
).str.lower()

def test_recommend():
    """
    Unit test for the Recommender class using a small hardcoded movie dataset.
    Tests similarity-based recommendations using TF-IDF and cosine similarity.
    """
    r = Recommender(sample)
    recs = r.recommend("Inception")
    titles = [rec["title"] for rec in recs]
    assert isinstance(recs, list)
    assert "Interstellar" in titles
    assert len(recs) > 0