"""
Tests for feature engineering and feature store.
Uses synthetic data to avoid requiring the actual MovieLens dataset.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer, GENRES, N_GENRES
from src.features.feature_store import RedisFeatureStore


# ------------------------------------------------------------------ #
# Fixtures: Synthetic MovieLens Data                                  #
# ------------------------------------------------------------------ #

def make_synthetic_data(
    n_users: int = 50,
    n_items: int = 100,
    n_ratings: int = 2000,
) -> tuple:
    """Generate synthetic ratings, users, and movies DataFrames."""
    np.random.seed(42)

    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    item_ids = np.random.randint(1, n_items + 1, size=n_ratings)
    ratings = np.random.randint(1, 6, size=n_ratings).astype(float)
    timestamps = pd.date_range("2003-01-01", periods=n_ratings, freq="1h")

    ratings_df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })

    genders = np.random.choice(["M", "F"], n_users)
    ages = np.random.choice([1, 18, 25, 35, 45, 50, 56], n_users)
    occupations = np.random.randint(0, 21, n_users)
    users_df = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "gender": genders,
        "age": ages,
        "occupation": occupations,
        "zip_code": ["12345"] * n_users,
    })

    genre_pool = GENRES[:5]
    movie_genres = [
        "|".join(np.random.choice(genre_pool, size=np.random.randint(1, 4), replace=False).tolist())
        for _ in range(n_items)
    ]
    years = np.random.randint(1990, 2004, n_items)
    movies_df = pd.DataFrame({
        "item_id": range(1, n_items + 1),
        "title": [f"Movie {i} ({y})" for i, y in zip(range(1, n_items + 1), years)],
        "genres": movie_genres,
    })

    return ratings_df, users_df, movies_df


@pytest.fixture
def synthetic_fe():
    """FeatureEngineer pre-loaded with synthetic data."""
    fe = FeatureEngineer(data_dir="data/ml-1m")  # Won't actually load from disk
    fe.ratings_df, fe.users_df, fe.movies_df = make_synthetic_data()
    return fe


# ------------------------------------------------------------------ #
# User Feature Tests                                                   #
# ------------------------------------------------------------------ #

class TestUserFeatures:
    def test_user_features_shape(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        assert len(user_feats) > 0
        assert "user_id" in user_feats.columns
        assert "avg_rating" in user_feats.columns
        assert "log_rating_count" in user_feats.columns
        assert "recency_score" in user_feats.columns
        assert "genre_pref" in user_feats.columns

    def test_user_features_no_nan_scalars(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        scalar_cols = ["avg_rating", "log_rating_count", "recency_score",
                       "gender_encoded", "age_normalized"]
        for col in scalar_cols:
            assert not user_feats[col].isna().any(), f"NaN found in column {col}"

    def test_avg_rating_in_bounds(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        assert (user_feats["avg_rating"] >= 1.0).all()
        assert (user_feats["avg_rating"] <= 5.0).all()

    def test_recency_score_normalized(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        assert (user_feats["recency_score"] >= 0.0).all()
        assert (user_feats["recency_score"] <= 1.0).all()

    def test_genre_pref_is_array(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        for val in user_feats["genre_pref"].values:
            assert isinstance(val, np.ndarray), f"Expected np.ndarray, got {type(val)}"
            assert val.shape == (N_GENRES,), f"Expected shape ({N_GENRES},), got {val.shape}"

    def test_genre_pref_normalized(self, synthetic_fe):
        user_feats = synthetic_fe.build_user_features()
        for val in user_feats["genre_pref"].values:
            norm = np.linalg.norm(val)
            assert norm <= 1.01, f"Genre pref not normalized: norm={norm}"


# ------------------------------------------------------------------ #
# Item Feature Tests                                                   #
# ------------------------------------------------------------------ #

class TestItemFeatures:
    def test_item_features_shape(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        assert len(item_feats) > 0
        assert "item_id" in item_feats.columns
        assert "avg_rating" in item_feats.columns
        assert "popularity_score" in item_feats.columns
        assert "genre_vector" in item_feats.columns

    def test_item_features_no_nan_scalars(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        scalar_cols = ["avg_rating", "log_rating_count", "popularity_score", "year_normalized"]
        for col in scalar_cols:
            assert not item_feats[col].isna().any(), f"NaN in column {col}"

    def test_genre_vector_shape(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        for val in item_feats["genre_vector"].values:
            assert isinstance(val, np.ndarray)
            assert val.shape == (N_GENRES,)

    def test_genre_vector_binary(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        for val in item_feats["genre_vector"].values:
            assert set(val.tolist()).issubset({0.0, 1.0}), "Genre vector should be binary"

    def test_popularity_score_normalized(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        assert (item_feats["popularity_score"] >= 0.0).all()
        assert (item_feats["popularity_score"] <= 1.0).all()

    def test_year_normalized_range(self, synthetic_fe):
        item_feats = synthetic_fe.build_item_features()
        assert (item_feats["year_normalized"] >= 0.0).all()
        assert (item_feats["year_normalized"] <= 1.0).all()


# ------------------------------------------------------------------ #
# Training Pairs Tests                                                 #
# ------------------------------------------------------------------ #

class TestTrainingPairs:
    def test_pairs_have_positives_and_negatives(self, synthetic_fe):
        synthetic_fe.build_user_features()
        synthetic_fe.build_item_features()
        train_pairs, test_pairs = synthetic_fe.build_training_pairs(n_negatives=2)
        assert len(train_pairs) > 0
        assert len(test_pairs) > 0
        assert 1 in train_pairs["label"].values
        assert 0 in train_pairs["label"].values

    def test_pairs_columns(self, synthetic_fe):
        train_pairs, _ = synthetic_fe.build_training_pairs(n_negatives=2)
        for col in ["user_id", "item_id", "label", "query_id"]:
            assert col in train_pairs.columns, f"Missing column: {col}"

    def test_negative_ratio(self, synthetic_fe):
        n_neg = 3
        train_pairs, _ = synthetic_fe.build_training_pairs(n_negatives=n_neg)
        label_counts = train_pairs["label"].value_counts()
        # Negatives should be ~n_neg times the number of positives
        pos_count = label_counts.get(1, 0)
        neg_count = label_counts.get(0, 0)
        assert neg_count > 0
        ratio = neg_count / max(pos_count, 1)
        # Allow some tolerance for users with few positive items
        assert 1.0 <= ratio <= n_neg + 1


# ------------------------------------------------------------------ #
# Interaction Features Tests                                           #
# ------------------------------------------------------------------ #

class TestInteractionFeatures:
    def test_interaction_features_no_nan(self, synthetic_fe):
        synthetic_fe.build_user_features()
        synthetic_fe.build_item_features()
        train_pairs, _ = synthetic_fe.build_training_pairs(n_negatives=2)
        interaction_df = synthetic_fe.build_interaction_features(train_pairs)
        scalar_feat_cols = [c for c in interaction_df.columns
                            if c not in {"user_id", "item_id", "label", "query_id", "rating"}
                            and interaction_df[c].dtype in [np.float64, np.float32, float]]
        for col in scalar_feat_cols:
            nan_count = interaction_df[col].isna().sum()
            assert nan_count == 0, f"NaN found in interaction feature column: {col}"

    def test_genre_affinity_range(self, synthetic_fe):
        synthetic_fe.build_user_features()
        synthetic_fe.build_item_features()
        train_pairs, _ = synthetic_fe.build_training_pairs(n_negatives=2)
        df = synthetic_fe.build_interaction_features(train_pairs)
        assert "genre_affinity" in df.columns
        assert (df["genre_affinity"] >= -1.01).all()
        assert (df["genre_affinity"] <= 1.01).all()


# ------------------------------------------------------------------ #
# Feature Store Tests                                                  #
# ------------------------------------------------------------------ #

class TestFeatureStore:
    def test_in_memory_store_user(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")  # Intentionally invalid
        assert not store.is_redis_available

        features = {"avg_rating": 4.2, "log_rating_count": 3.5, "genre_pref": [0.1] * N_GENRES}
        store.store_user_features(user_id=42, features=features)
        retrieved = store.get_user_features(user_id=42)

        assert retrieved is not None
        assert abs(retrieved["avg_rating"] - 4.2) < 1e-4
        assert len(retrieved["genre_pref"]) == N_GENRES

    def test_in_memory_store_item(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")

        features = {"avg_rating": 3.8, "popularity_score": 0.65, "genre_vector": [1.0, 0.0] * 9}
        store.store_item_features(item_id=101, features=features)
        retrieved = store.get_item_features(item_id=101)

        assert retrieved is not None
        assert abs(retrieved["avg_rating"] - 3.8) < 1e-4

    def test_batch_item_fetch(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")

        for iid in [1, 2, 3]:
            store.store_item_features(iid, {"avg_rating": float(iid), "popularity_score": 0.5})

        batch = store.get_item_features_batch([1, 2, 3, 99])  # 99 doesn't exist
        assert batch[1] is not None
        assert batch[2] is not None
        assert batch[3] is not None
        assert batch[99] is None

    def test_missing_user_returns_none(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")
        result = store.get_user_features(user_id=999999)
        assert result is None

    def test_recommendation_cache(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")
        recs = [{"item_id": i, "title": f"Movie {i}", "score": 1.0 - i / 10,
                  "rank": i, "retrieval_score": 0.9, "genres": ["Action"]}
                for i in range(1, 6)]
        store.cache_recommendations(user_id=1, recommendations=recs)
        retrieved = store.get_cached_recommendations(user_id=1)
        assert retrieved is not None
        assert len(retrieved) == 5
        assert retrieved[0]["item_id"] == 1

    def test_bulk_load_from_dataframe(self, synthetic_fe):
        synthetic_fe.build_user_features()
        synthetic_fe.build_item_features()

        with tempfile.TemporaryDirectory() as tmpdir:
            synthetic_fe.save_features(tmpdir)

            user_feat_flat = pd.read_parquet(Path(tmpdir) / "user_features.parquet")
            item_feat_flat = pd.read_parquet(Path(tmpdir) / "item_features.parquet")

        store = RedisFeatureStore(redis_url="redis://localhost:9999")
        store.load_all_features(user_feat_flat, item_feat_flat)

        # Verify at least one user/item is retrievable
        sample_user_id = int(user_feat_flat["user_id"].iloc[0])
        sample_item_id = int(item_feat_flat["item_id"].iloc[0])

        u_feat = store.get_user_features(sample_user_id)
        i_feat = store.get_item_features(sample_item_id)

        assert u_feat is not None, "User features not found after bulk load"
        assert i_feat is not None, "Item features not found after bulk load"

    def test_flush(self):
        store = RedisFeatureStore(redis_url="redis://localhost:9999")
        store.store_user_features(1, {"x": 1})
        store.flush()
        assert store.get_user_features(1) is None


# ------------------------------------------------------------------ #
# Parquet Persistence Tests                                            #
# ------------------------------------------------------------------ #

class TestFeaturePersistence:
    def test_save_and_reload_features(self, synthetic_fe):
        synthetic_fe.build_user_features()
        synthetic_fe.build_item_features()

        with tempfile.TemporaryDirectory() as tmpdir:
            synthetic_fe.save_features(tmpdir)

            # Verify parquet files exist
            assert (Path(tmpdir) / "user_features.parquet").exists()
            assert (Path(tmpdir) / "item_features.parquet").exists()

            # Reload into a fresh FeatureEngineer
            fe2 = FeatureEngineer()
            fe2.load_features(tmpdir)

        assert fe2.user_features is not None
        assert fe2.item_features is not None
        assert len(fe2.user_features) == len(synthetic_fe.user_features)
        assert len(fe2.item_features) == len(synthetic_fe.item_features)

        # Verify genre_pref arrays are reconstructed correctly
        for val in fe2.user_features["genre_pref"].values:
            assert isinstance(val, np.ndarray)
            assert val.shape == (N_GENRES,)
