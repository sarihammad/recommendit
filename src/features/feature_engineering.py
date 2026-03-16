"""
Feature engineering for MovieLens 1M dataset.
Builds user features, item features, and interaction features for the recommendation pipeline.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

# MovieLens 1M genre list (18 genres in order)
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
N_GENRES = len(GENRES)  # 18


class FeatureEngineer:
    """Builds and manages all features for the RecommendIt pipeline."""

    def __init__(self, data_dir: str = "data/ml-1m"):
        self.data_dir = Path(data_dir)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user_features: Optional[pd.DataFrame] = None
        self.item_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Data Loading                                                         #
    # ------------------------------------------------------------------ #

    def load_data(self) -> None:
        """Load all MovieLens 1M data files."""
        logger.info("Loading MovieLens 1M data from %s", self.data_dir)

        self.ratings_df = pd.read_csv(
            self.data_dir / "ratings.dat",
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        self.ratings_df["timestamp"] = pd.to_datetime(self.ratings_df["timestamp"], unit="s")

        self.users_df = pd.read_csv(
            self.data_dir / "users.dat",
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
            encoding="latin-1",
        )

        self.movies_df = pd.read_csv(
            self.data_dir / "movies.dat",
            sep="::",
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

        logger.info(
            "Loaded %d ratings, %d users, %d movies",
            len(self.ratings_df),
            len(self.users_df),
            len(self.movies_df),
        )

    # ------------------------------------------------------------------ #
    # Genre Encoding                                                       #
    # ------------------------------------------------------------------ #

    def _encode_genres(self, genre_str: str) -> np.ndarray:
        """Convert pipe-separated genre string to multi-hot vector."""
        vec = np.zeros(len(GENRES), dtype=np.float32)
        for genre in genre_str.split("|"):
            idx = GENRE_TO_IDX.get(genre)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    # ------------------------------------------------------------------ #
    # User Features                                                        #
    # ------------------------------------------------------------------ #

    def build_user_features(self) -> pd.DataFrame:
        """
        Build user-level features:
        - avg_rating: mean rating given by user
        - rating_count: total number of ratings (log-scaled)
        - genre_preferences: weighted genre vector based on liked items
        - recency_score: how recently the user was active (0-1)
        - gender_encoded: 0/1 for M/F
        - age_group: normalized age bucket
        """
        logger.info("Building user features...")

        # Base rating stats
        user_stats = self.ratings_df.groupby("user_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            last_timestamp=("timestamp", "max"),
        ).reset_index()

        # Recency score: normalize last activity time to [0, 1]
        ts_min = user_stats["last_timestamp"].min()
        ts_max = user_stats["last_timestamp"].max()
        ts_range = (ts_max - ts_min).total_seconds()
        if ts_range > 0:
            user_stats["recency_score"] = (
                (user_stats["last_timestamp"] - ts_min).dt.total_seconds() / ts_range
            ).astype(np.float32)
        else:
            user_stats["recency_score"] = 1.0

        user_stats["log_rating_count"] = np.log1p(user_stats["rating_count"]).astype(np.float32)
        user_stats.drop(columns=["last_timestamp"], inplace=True)

        # Genre preference vectors: average genre vector of liked items (rating >= 4)
        movie_genre_vecs = {
            row["item_id"]: self._encode_genres(row["genres"])
            for _, row in self.movies_df.iterrows()
        }

        liked = self.ratings_df[self.ratings_df["rating"] >= 4].copy()
        liked["genre_vec"] = liked["item_id"].map(movie_genre_vecs)
        liked = liked.dropna(subset=["genre_vec"])

        genre_pref_rows = []
        for user_id, group in liked.groupby("user_id"):
            vecs = np.stack(group["genre_vec"].values)
            weights = group["rating"].values - 3  # weight by how much above threshold
            weighted = vecs * weights[:, None]
            avg_vec = weighted.mean(axis=0)
            # Normalize
            norm = np.linalg.norm(avg_vec)
            if norm > 0:
                avg_vec = avg_vec / norm
            genre_pref_rows.append({"user_id": user_id, "genre_pref": avg_vec})

        genre_pref_df = pd.DataFrame(genre_pref_rows)

        # User demographic features
        demo = self.users_df[["user_id", "gender", "age", "occupation"]].copy()
        demo["gender_encoded"] = (demo["gender"] == "F").astype(np.float32)
        demo["age_normalized"] = (demo["age"] / demo["age"].max()).astype(np.float32)
        demo["occupation_normalized"] = (demo["occupation"] / demo["occupation"].max()).astype(np.float32)
        demo.drop(columns=["gender", "age", "occupation"], inplace=True)

        # Merge all user features
        user_features = user_stats.merge(demo, on="user_id", how="left")
        user_features = user_features.merge(genre_pref_df, on="user_id", how="left")

        # Fill missing genre preferences with zero vectors
        user_features["genre_pref"] = user_features["genre_pref"].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(len(GENRES), dtype=np.float32)
        )

        self.user_features = user_features
        logger.info("Built user features for %d users", len(user_features))
        return user_features

    # ------------------------------------------------------------------ #
    # Item Features                                                        #
    # ------------------------------------------------------------------ #

    def build_item_features(self) -> pd.DataFrame:
        """
        Build item-level features:
        - avg_rating: mean rating received
        - rating_count: total number of ratings
        - genre_vector: multi-hot genre encoding (18 dims)
        - popularity_score: normalized rating count
        - year: extracted from title (normalized)
        """
        logger.info("Building item features...")

        item_stats = self.ratings_df.groupby("item_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            rating_stddev=("rating", "std"),
        ).reset_index()
        item_stats["rating_stddev"] = item_stats["rating_stddev"].fillna(0.0)

        # Popularity score: log-normalized rating count
        item_stats["log_rating_count"] = np.log1p(item_stats["rating_count"]).astype(np.float32)
        max_log_count = item_stats["log_rating_count"].max()
        item_stats["popularity_score"] = (item_stats["log_rating_count"] / max_log_count).astype(np.float32)

        # Movie metadata and genre vectors
        movies = self.movies_df.copy()
        movies["genre_vector"] = movies["genres"].apply(self._encode_genres)

        # Extract year from title (format: "Movie Title (YYYY)")
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
        year_min = movies["year"].min()
        year_max = movies["year"].max()
        movies["year_normalized"] = ((movies["year"] - year_min) / (year_max - year_min + 1e-8)).astype(np.float32)
        movies["year_normalized"] = movies["year_normalized"].fillna(0.5)

        item_features = item_stats.merge(
            movies[["item_id", "title", "genre_vector", "year_normalized"]],
            on="item_id",
            how="left",
        )

        # Fill missing genre vectors (items with no stats)
        item_features["genre_vector"] = item_features["genre_vector"].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(len(GENRES), dtype=np.float32)
        )

        self.item_features = item_features
        logger.info("Built item features for %d items", len(item_features))
        return item_features

    # ------------------------------------------------------------------ #
    # Training Pairs                                                       #
    # ------------------------------------------------------------------ #

    def build_training_pairs(
        self,
        ratings_df: Optional[pd.DataFrame] = None,
        n_negatives: int = 4,
        test_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build positive/negative training pairs for learning-to-rank.

        Strategy:
          - Positives: user-item pairs with rating >= 4
          - Negatives: randomly sampled items the user has NOT rated
          - Test split: last 10% of each user's interactions by timestamp

        Returns:
            (train_pairs_df, test_pairs_df) with columns:
            [user_id, item_id, label, rating, query_id]
        """
        if ratings_df is None:
            ratings_df = self.ratings_df

        logger.info("Building training pairs with %d negatives per positive...", n_negatives)

        all_items = set(ratings_df["item_id"].unique())
        pairs = []

        # Sort by timestamp for temporal split
        ratings_sorted = ratings_df.sort_values(["user_id", "timestamp"])

        for user_id, group in ratings_sorted.groupby("user_id"):
            rated_items = set(group["item_id"].values)
            positive_items = group[group["rating"] >= 4]["item_id"].values
            if len(positive_items) == 0:
                continue

            unrated_items = list(all_items - rated_items)
            if len(unrated_items) < n_negatives:
                continue

            neg_sample = np.random.choice(unrated_items, size=len(positive_items) * n_negatives, replace=False)

            for pos_item in positive_items:
                pairs.append({
                    "user_id": user_id,
                    "item_id": pos_item,
                    "label": 1,
                    "rating": group[group["item_id"] == pos_item]["rating"].values[0],
                })

            for neg_item in neg_sample:
                pairs.append({
                    "user_id": user_id,
                    "item_id": neg_item,
                    "label": 0,
                    "rating": 0,
                })

        pairs_df = pd.DataFrame(pairs)
        pairs_df["query_id"] = pairs_df["user_id"].astype("category").cat.codes

        # Temporal train/test split: last 10% of positive interactions per user
        # For simplicity, split query groups
        unique_queries = pairs_df["query_id"].unique()
        np.random.shuffle(unique_queries)
        n_test = max(1, int(len(unique_queries) * test_ratio))
        test_queries = set(unique_queries[:n_test])

        train_df = pairs_df[~pairs_df["query_id"].isin(test_queries)].copy()
        test_df = pairs_df[pairs_df["query_id"].isin(test_queries)].copy()

        logger.info(
            "Training pairs: %d train, %d test (queries: %d train, %d test)",
            len(train_df), len(test_df),
            len(train_df["query_id"].unique()), len(test_df["query_id"].unique()),
        )
        return train_df, test_df

    # ------------------------------------------------------------------ #
    # Interaction Features                                                 #
    # ------------------------------------------------------------------ #

    def build_interaction_features(
        self,
        pairs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build interaction features by joining user and item features onto pairs.
        Returns a flat DataFrame with all features concatenated.
        """
        if self.user_features is None or self.item_features is None:
            raise RuntimeError("Call build_user_features() and build_item_features() first.")

        # User scalar features
        user_scalar_cols = ["avg_rating", "log_rating_count", "recency_score",
                            "gender_encoded", "age_normalized", "occupation_normalized"]
        user_scalar = self.user_features[["user_id"] + user_scalar_cols].copy()

        # Item scalar features
        item_scalar_cols = ["avg_rating", "log_rating_count", "popularity_score",
                            "rating_stddev", "year_normalized"]
        item_scalar = self.item_features[["item_id"] + item_scalar_cols].copy()
        item_scalar = item_scalar.rename(columns={
            "avg_rating": "item_avg_rating",
            "log_rating_count": "item_log_rating_count",
        })

        merged = pairs_df[["user_id", "item_id", "label", "query_id"]].merge(
            user_scalar, on="user_id", how="left"
        ).merge(item_scalar, on="item_id", how="left")

        # Interaction features
        merged["rating_diff"] = merged["avg_rating"] - merged["item_avg_rating"]
        merged["user_item_popularity_ratio"] = (
            merged["log_rating_count"] / (merged["item_log_rating_count"] + 1e-8)
        )

        # Expand user genre preferences
        user_genre_prefs = self.user_features[["user_id", "genre_pref"]].copy()
        item_genre_vecs = self.item_features[["item_id", "genre_vector"]].copy()

        genre_pref_mat = np.stack(user_genre_prefs["genre_pref"].values)
        user_genre_df = pd.DataFrame(
            genre_pref_mat,
            columns=[f"user_genre_{i}" for i in range(len(GENRES))],
        )
        user_genre_df["user_id"] = user_genre_prefs["user_id"].values

        genre_vec_mat = np.stack(item_genre_vecs["genre_vector"].values)
        item_genre_df = pd.DataFrame(
            genre_vec_mat,
            columns=[f"item_genre_{i}" for i in range(len(GENRES))],
        )
        item_genre_df["item_id"] = item_genre_vecs["item_id"].values

        merged = merged.merge(user_genre_df, on="user_id", how="left")
        merged = merged.merge(item_genre_df, on="item_id", how="left")

        # Genre affinity: dot product of user genre pref and item genre vector
        user_genre_cols = [f"user_genre_{i}" for i in range(len(GENRES))]
        item_genre_cols = [f"item_genre_{i}" for i in range(len(GENRES))]
        merged["genre_affinity"] = (
            merged[user_genre_cols].values * merged[item_genre_cols].values
        ).sum(axis=1)

        merged = merged.fillna(0.0)
        return merged

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_features(self, output_dir: str = "data/features") -> None:
        """Save user and item feature DataFrames as parquet."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.user_features is not None:
            # Expand genre_pref array into separate columns for parquet compatibility
            user_feat = self.user_features.copy()
            genre_mat = np.stack(user_feat["genre_pref"].values)
            genre_df = pd.DataFrame(
                genre_mat, columns=[f"genre_pref_{i}" for i in range(len(GENRES))]
            )
            user_feat = pd.concat(
                [user_feat.drop(columns=["genre_pref"]).reset_index(drop=True), genre_df],
                axis=1,
            )
            user_feat.to_parquet(out / "user_features.parquet", index=False)
            logger.info("Saved user features to %s", out / "user_features.parquet")

        if self.item_features is not None:
            item_feat = self.item_features.copy()
            genre_mat = np.stack(item_feat["genre_vector"].values)
            genre_df = pd.DataFrame(
                genre_mat, columns=[f"genre_vec_{i}" for i in range(len(GENRES))]
            )
            item_feat = pd.concat(
                [item_feat.drop(columns=["genre_vector"]).reset_index(drop=True), genre_df],
                axis=1,
            )
            item_feat.to_parquet(out / "item_features.parquet", index=False)
            logger.info("Saved item features to %s", out / "item_features.parquet")

    def load_features(self, features_dir: str = "data/features") -> None:
        """Load saved feature DataFrames from parquet."""
        feat_dir = Path(features_dir)

        user_path = feat_dir / "user_features.parquet"
        item_path = feat_dir / "item_features.parquet"

        if user_path.exists():
            user_feat = pd.read_parquet(user_path)
            # Reconstruct genre_pref column
            genre_cols = [f"genre_pref_{i}" for i in range(len(GENRES))]
            if all(c in user_feat.columns for c in genre_cols):
                user_feat["genre_pref"] = list(user_feat[genre_cols].values.astype(np.float32))
                user_feat.drop(columns=genre_cols, inplace=True)
            self.user_features = user_feat
            logger.info("Loaded user features: %d users", len(user_feat))

        if item_path.exists():
            item_feat = pd.read_parquet(item_path)
            genre_cols = [f"genre_vec_{i}" for i in range(len(GENRES))]
            if all(c in item_feat.columns for c in genre_cols):
                item_feat["genre_vector"] = list(item_feat[genre_cols].values.astype(np.float32))
                item_feat.drop(columns=genre_cols, inplace=True)
            self.item_features = item_feat
            logger.info("Loaded item features: %d items", len(item_feat))

    def get_feature_columns(self) -> List[str]:
        """Return the list of scalar feature columns used for ranking."""
        return [
            "avg_rating", "log_rating_count", "recency_score",
            "gender_encoded", "age_normalized", "occupation_normalized",
            "item_avg_rating", "item_log_rating_count", "popularity_score",
            "rating_stddev", "year_normalized",
            "rating_diff", "user_item_popularity_ratio", "genre_affinity",
        ] + [f"user_genre_{i}" for i in range(len(GENRES))] \
          + [f"item_genre_{i}" for i in range(len(GENRES))]
