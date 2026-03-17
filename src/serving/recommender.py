"""
End-to-end recommendation inference pipeline.
Combines two-tower ANN retrieval, Redis feature lookup, and LightGBM re-ranking.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import settings
from src.features.feature_engineering import FeatureEngineer, N_GENRES, GENRE_TO_IDX
from src.features.feature_store import RedisFeatureStore
from src.models.faiss_index import FAISSIndex
from src.models.ranker import LightGBMRanker
from src.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    item_id: int
    title: str
    score: float
    rank: int
    retrieval_score: float = 0.0
    genres: List[str] = field(default_factory=list)


class LatencyTracker:
    """Tracks p50/p99 latency using a rolling window."""

    def __init__(self, window: int = 1000):
        self.window = window
        self._samples: List[float] = []

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        if len(self._samples) > self.window:
            self._samples.pop(0)

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        return float(np.percentile(self._samples, p))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        return len(self._samples)


class RecommendationPipeline:
    """
    Production-grade recommendation pipeline.

    Flow:
      1. Check Redis cache for pre-computed recommendations
      2. Get user embedding from two-tower model
      3. FAISS ANN search → top-K candidates (default 500)
      4. Batch fetch item features from Redis feature store
      5. Build ranking feature matrix
      6. LightGBM score all candidates
      7. Return top-K sorted by score
    """

    def __init__(
        self,
        model_path: str = None,
        index_path: str = None,
        ranker_path: str = None,
        redis_url: str = None,
        data_dir: str = None,
        top_k_candidates: int = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.ranker_path = ranker_path or settings.RANKER_MODEL_PATH
        self.redis_url = redis_url or settings.REDIS_URL
        self.data_dir = data_dir or settings.DATA_DIR
        self.top_k_candidates = top_k_candidates or settings.TOP_K_CANDIDATES

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[TwoTowerModel] = None
        self.faiss_index: Optional[FAISSIndex] = None
        self.ranker: Optional[LightGBMRanker] = None
        self.feature_store: Optional[RedisFeatureStore] = None
        self.feature_engineer: Optional[FeatureEngineer] = None

        # In-memory fallbacks
        self._item_titles: Dict[int, str] = {}
        self._item_genres: Dict[int, List[str]] = {}
        self._popularity_fallback: List[int] = []  # Pre-ranked popular items for cold start

        # Latency tracking
        self.latency_tracker = LatencyTracker(window=1000)
        self.retrieval_latency = LatencyTracker(window=1000)
        self.ranking_latency = LatencyTracker(window=1000)

        # Cache stats
        self._cache_hits = 0
        self._cache_misses = 0

        self._loaded = False

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load all model artifacts and connect to services."""
        logger.info("Loading recommendation pipeline...")
        t_start = time.time()

        self._load_model()
        self._load_index()
        self._load_ranker()
        self._connect_feature_store()
        self._load_item_metadata()
        self._build_popularity_fallback()

        self._loaded = True
        elapsed = time.time() - t_start
        logger.info("Pipeline loaded in %.2fs", elapsed)

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Two-tower model not found: {self.model_path}")
        self.model = TwoTowerModel.load(self.model_path, device=self.device)
        self.model.eval()
        logger.info("Loaded two-tower model (dim=%d)", self.model.embed_dim)

    def _load_index(self) -> None:
        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.faiss_index = FAISSIndex.load(self.index_path)
        logger.info("Loaded FAISS index: %s", self.faiss_index.stats())

    def _load_ranker(self) -> None:
        if not Path(self.ranker_path).exists():
            raise FileNotFoundError(f"LightGBM ranker not found: {self.ranker_path}")
        self.ranker = LightGBMRanker.load(self.ranker_path)
        logger.info("Loaded LightGBM ranker (%d features)", self.ranker.n_features)

    def _connect_feature_store(self) -> None:
        self.feature_store = RedisFeatureStore(
            redis_url=self.redis_url,
            ttl=settings.FEATURE_CACHE_TTL_SECONDS,
        )
        logger.info(
            "Feature store: %s",
            "Redis" if self.feature_store.is_redis_available else "in-memory fallback",
        )

    def _load_item_metadata(self) -> None:
        """Load item titles and genre strings for response enrichment."""
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()
        for _, row in fe.movies_df.iterrows():
            iid = int(row["item_id"])
            self._item_titles[iid] = str(row["title"])
            self._item_genres[iid] = str(row["genres"]).split("|")
        logger.info("Loaded metadata for %d items", len(self._item_titles))

    def _build_popularity_fallback(self) -> None:
        """Pre-compute popularity-ranked list for cold-start users."""
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()
        popularity = (
            fe.ratings_df.groupby("item_id")["rating"]
            .count()
            .sort_values(ascending=False)
        )
        self._popularity_fallback = popularity.index.tolist()
        logger.info("Built popularity fallback list (%d items)", len(self._popularity_fallback))

    # ------------------------------------------------------------------ #
    # User Embedding                                                       #
    # ------------------------------------------------------------------ #

    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Return L2-normalized user embedding from the two-tower model."""
        try:
            emb = self.model.get_user_embedding(user_id, device=self.device)
            return emb.astype(np.float32)
        except Exception as exc:
            logger.warning("Failed to get embedding for user %d: %s", user_id, exc)
            return None

    # ------------------------------------------------------------------ #
    # Feature Assembly                                                     #
    # ------------------------------------------------------------------ #

    def _build_ranking_features(
        self,
        user_features: Dict[str, Any],
        item_features_batch: Dict[int, Optional[Dict[str, Any]]],
        candidate_item_ids: List[int],
    ) -> pd.DataFrame:
        """
        Assemble the ranking feature matrix for all candidates.
        Combines user-level, item-level, and derived interaction features.
        """
        rows = []
        for item_id in candidate_item_ids:
            item_feat = item_features_batch.get(item_id) or {}
            row = {
                "item_id": item_id,
                # User features
                "avg_rating": float(user_features.get("avg_rating", 3.5)),
                "log_rating_count": float(user_features.get("log_rating_count", 0.0)),
                "recency_score": float(user_features.get("recency_score", 0.5)),
                "gender_encoded": float(user_features.get("gender_encoded", 0.0)),
                "age_normalized": float(user_features.get("age_normalized", 0.3)),
                "occupation_normalized": float(user_features.get("occupation_normalized", 0.3)),
                # Item features
                "item_avg_rating": float(item_feat.get("avg_rating", 3.5)),
                "item_log_rating_count": float(item_feat.get("log_rating_count", 0.0)),
                "popularity_score": float(item_feat.get("popularity_score", 0.0)),
                "rating_stddev": float(item_feat.get("rating_stddev", 0.0)),
                "year_normalized": float(item_feat.get("year_normalized", 0.5)),
            }

            # Interaction features
            row["rating_diff"] = row["avg_rating"] - row["item_avg_rating"]
            row["user_item_popularity_ratio"] = (
                row["log_rating_count"] / (row["item_log_rating_count"] + 1e-8)
            )

            # Genre features
            user_genre_pref = user_features.get("genre_pref", [0.0] * N_GENRES)
            item_genre_vec = item_feat.get("genre_vector", [0.0] * N_GENRES)

            for i in range(N_GENRES):
                row[f"user_genre_{i}"] = float(user_genre_pref[i]) if i < len(user_genre_pref) else 0.0
                row[f"item_genre_{i}"] = float(item_genre_vec[i]) if i < len(item_genre_vec) else 0.0

            row["genre_affinity"] = sum(
                row[f"user_genre_{i}"] * row[f"item_genre_{i}"] for i in range(N_GENRES)
            )

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Main Recommendation Entry Point                                     #
    # ------------------------------------------------------------------ #

    def get_recommendations(
        self,
        user_id: int,
        k: int = None,
        use_cache: bool = True,
    ) -> List[RecommendationResult]:
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: integer user ID
            k: number of results to return (default: settings.TOP_K_RESULTS)
            use_cache: whether to check/populate Redis recommendation cache

        Returns:
            List of RecommendationResult sorted by score (best first)
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        k = k or settings.TOP_K_RESULTS
        t_pipeline_start = time.time()

        # --- Cache check ---
        if use_cache:
            cached = self.feature_store.get_cached_recommendations(user_id)
            if cached is not None:
                self._cache_hits += 1
                results = [
                    RecommendationResult(**item) for item in cached
                ]
                return results[:k]
        self._cache_misses += 1

        # --- User embedding ---
        user_emb = self._get_user_embedding(user_id)
        if user_emb is None:
            logger.warning("No embedding for user %d — falling back to popularity", user_id)
            return self._popularity_recommendations(k)

        # --- FAISS retrieval ---
        t_retrieval = time.time()
        distances, candidate_item_ids = self.faiss_index.search(
            user_emb, k=self.top_k_candidates
        )
        retrieval_ms = (time.time() - t_retrieval) * 1000
        self.retrieval_latency.record(retrieval_ms)
        retrieval_score_map = dict(zip(candidate_item_ids.tolist(), distances.tolist()))

        # --- Feature fetch ---
        user_features = self.feature_store.get_user_features(user_id) or {}
        item_features_batch = self.feature_store.get_item_features_batch(
            candidate_item_ids.tolist()
        )

        # --- Ranking ---
        t_ranking = time.time()
        features_df = self._build_ranking_features(
            user_features, item_features_batch, candidate_item_ids.tolist()
        )

        ranker_feature_cols = [
            c for c in self.ranker.feature_names if c in features_df.columns
        ]
        # Ensure all required feature columns are present (fill missing with 0)
        for col in self.ranker.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0.0

        scores = self.ranker.predict(features_df)
        ranking_ms = (time.time() - t_ranking) * 1000
        self.ranking_latency.record(ranking_ms)

        features_df["score"] = scores
        features_df["retrieval_score"] = features_df["item_id"].map(retrieval_score_map).fillna(0.0)

        # --- Sort and take top-K ---
        top_k_df = features_df.nlargest(k, "score").reset_index(drop=True)

        results = []
        for rank, (_, row) in enumerate(top_k_df.iterrows(), start=1):
            item_id = int(row["item_id"])
            results.append(
                RecommendationResult(
                    item_id=item_id,
                    title=self._item_titles.get(item_id, f"Item {item_id}"),
                    score=float(row["score"]),
                    rank=rank,
                    retrieval_score=float(row["retrieval_score"]),
                    genres=self._item_genres.get(item_id, []),
                )
            )

        # --- Cache results ---
        if use_cache and results:
            self.feature_store.cache_recommendations(
                user_id,
                [
                    {
                        "item_id": r.item_id,
                        "title": r.title,
                        "score": r.score,
                        "rank": r.rank,
                        "retrieval_score": r.retrieval_score,
                        "genres": r.genres,
                    }
                    for r in results
                ],
                ttl=settings.CACHE_TTL_SECONDS,
            )

        total_ms = (time.time() - t_pipeline_start) * 1000
        self.latency_tracker.record(total_ms)
        logger.debug(
            "user=%d | candidates=%d | total=%.1fms (retrieval=%.1fms, ranking=%.1fms)",
            user_id, len(candidate_item_ids), total_ms, retrieval_ms, ranking_ms,
        )

        return results

    # ------------------------------------------------------------------ #
    # Cold Start                                                           #
    # ------------------------------------------------------------------ #

    def _popularity_recommendations(self, k: int) -> List[RecommendationResult]:
        """
        Popularity-based fallback for cold-start users (new users with no history).
        Returns the globally most popular items.
        """
        results = []
        for rank, item_id in enumerate(self._popularity_fallback[:k], start=1):
            results.append(
                RecommendationResult(
                    item_id=int(item_id),
                    title=self._item_titles.get(int(item_id), f"Item {item_id}"),
                    score=1.0 - (rank / (k + 1)),
                    rank=rank,
                    retrieval_score=0.0,
                    genres=self._item_genres.get(int(item_id), []),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        return {
            "total_requests": total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(total_requests, 1),
            "latency_p50_ms": round(self.latency_tracker.p50, 2),
            "latency_p99_ms": round(self.latency_tracker.p99, 2),
            "retrieval_p50_ms": round(self.retrieval_latency.p50, 2),
            "retrieval_p99_ms": round(self.retrieval_latency.p99, 2),
            "ranking_p50_ms": round(self.ranking_latency.p50, 2),
            "ranking_p99_ms": round(self.ranking_latency.p99, 2),
        }
