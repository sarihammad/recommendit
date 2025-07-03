from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, movie_data):
        """
        Initialize with preprocessed movie data and compute TF-IDF matrix.
        """
        self.movies = movie_data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(movie_data["combined"])

    def _get_index_by_title(self, title: str):
        match_row = self.movies[self.movies["title"].str.lower() == title.lower()]
        if match_row.empty:
            return None
        return match_row.index[0]

    def _apply_filters(self, df, genre, year):
        if genre:
            df = df[df["genres"].str.contains(genre, case=False, na=False)]
        if year:
            df = df[df["release_date"].str.contains(str(year), na=False)]
        return df

    def recommend(self, title, top_k=5, genre=None, year=None):
        """
        Return a list of recommended movies similar to the given title,
        optionally filtered by genre and release year.
        """
        idx = self._get_index_by_title(title)
        if idx is None:
            return []

        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_indices = sims.argsort()[::-1][1:]  # exclude the movie itself

        filtered = self.movies.iloc[sim_indices]
        filtered = self._apply_filters(filtered, genre, year)

        return filtered[["title", "overview", "genres"]].head(top_k).to_dict(orient="records")