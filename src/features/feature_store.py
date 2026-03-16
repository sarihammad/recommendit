"""
Redis-backed online feature store for serving user and item features.
Falls back to an in-memory dict when Redis is unavailable.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

logger = logging.getLogger(__name__)

USER_FEATURE_PREFIX = "user:feat:"
ITEM_FEATURE_PREFIX = "item:feat:"


def _serialize(data: Dict[str, Any]) -> bytes:
    """Serialize feature dict using msgpack if available, otherwise JSON."""
    # Convert numpy arrays/scalars to Python native types
    clean = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v
    if MSGPACK_AVAILABLE:
        return msgpack.packb(clean, use_bin_type=True)
    return json.dumps(clean).encode("utf-8")


def _deserialize(data: bytes) -> Dict[str, Any]:
    """Deserialize bytes back to feature dict."""
    if MSGPACK_AVAILABLE:
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception:
            pass
    return json.loads(data.decode("utf-8"))


class RedisFeatureStore:
    """
    Online feature store backed by Redis.
    Supports bulk loading from DataFrames and individual lookups.
    Automatically falls back to an in-memory store when Redis is unavailable.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self._client: Optional[Any] = None
        self._memory_store: Dict[str, bytes] = {}
        self._use_redis = False
        self._connect()

    def _connect(self) -> None:
        if not REDIS_AVAILABLE:
            logger.warning("redis package not available. Using in-memory fallback.")
            return
        try:
            client = redis.from_url(self.redis_url, socket_connect_timeout=2)
            client.ping()
            self._client = client
            self._use_redis = True
            logger.info("Connected to Redis at %s", self.redis_url)
        except Exception as exc:
            logger.warning(
                "Cannot connect to Redis (%s). Falling back to in-memory store.", exc
            )
            self._use_redis = False

    @property
    def is_redis_available(self) -> bool:
        return self._use_redis

    # ------------------------------------------------------------------ #
    # Low-level get/set                                                    #
    # ------------------------------------------------------------------ #

    def _set(self, key: str, value: bytes) -> None:
        if self._use_redis:
            self._client.setex(key, self.ttl, value)
        else:
            self._memory_store[key] = value

    def _get(self, key: str) -> Optional[bytes]:
        if self._use_redis:
            return self._client.get(key)
        return self._memory_store.get(key)

    def _mget(self, keys: List[str]) -> List[Optional[bytes]]:
        if self._use_redis:
            return self._client.mget(keys)
        return [self._memory_store.get(k) for k in keys]

    # ------------------------------------------------------------------ #
    # User Features                                                        #
    # ------------------------------------------------------------------ #

    def store_user_features(self, user_id: int, features: Dict[str, Any]) -> None:
        key = f"{USER_FEATURE_PREFIX}{user_id}"
        self._set(key, _serialize(features))

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        key = f"{USER_FEATURE_PREFIX}{user_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return _deserialize(raw)

    # ------------------------------------------------------------------ #
    # Item Features                                                        #
    # ------------------------------------------------------------------ #

    def store_item_features(self, item_id: int, features: Dict[str, Any]) -> None:
        key = f"{ITEM_FEATURE_PREFIX}{item_id}"
        self._set(key, _serialize(features))

    def get_item_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        key = f"{ITEM_FEATURE_PREFIX}{item_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return _deserialize(raw)

    def get_item_features_batch(self, item_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
        """Batch fetch item features for a list of item IDs."""
        keys = [f"{ITEM_FEATURE_PREFIX}{iid}" for iid in item_ids]
        raws = self._mget(keys)
        result = {}
        for item_id, raw in zip(item_ids, raws):
            result[item_id] = _deserialize(raw) if raw is not None else None
        return result

    # ------------------------------------------------------------------ #
    # Bulk Loading                                                         #
    # ------------------------------------------------------------------ #

    def load_all_features(
        self,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        batch_size: int = 500,
    ) -> None:
        """
        Bulk-load all user and item features from DataFrames into the store.
        Processes in batches to avoid memory spikes for large catalogs.
        """
        logger.info(
            "Loading features: %d users, %d items",
            len(user_features_df),
            len(item_features_df),
        )

        # Identify genre preference columns in user features
        user_genre_pref_cols = [c for c in user_features_df.columns if c.startswith("genre_pref_")]
        user_scalar_cols = [
            c for c in user_features_df.columns
            if c not in {"user_id"} and c not in user_genre_pref_cols
        ]

        # Load user features
        for start in range(0, len(user_features_df), batch_size):
            chunk = user_features_df.iloc[start : start + batch_size]
            pipe_data = {}
            for _, row in chunk.iterrows():
                feat: Dict[str, Any] = {col: row[col] for col in user_scalar_cols}
                if user_genre_pref_cols:
                    feat["genre_pref"] = [float(row[c]) for c in user_genre_pref_cols]
                key = f"{USER_FEATURE_PREFIX}{int(row['user_id'])}"
                pipe_data[key] = _serialize(feat)

            if self._use_redis:
                pipe = self._client.pipeline()
                for key, val in pipe_data.items():
                    pipe.setex(key, self.ttl, val)
                pipe.execute()
            else:
                self._memory_store.update(pipe_data)

        logger.info("Loaded %d user feature records", len(user_features_df))

        # Identify genre vector columns in item features
        item_genre_vec_cols = [c for c in item_features_df.columns if c.startswith("genre_vec_")]
        item_scalar_cols = [
            c for c in item_features_df.columns
            if c not in {"item_id", "title"} and c not in item_genre_vec_cols
        ]

        # Load item features
        for start in range(0, len(item_features_df), batch_size):
            chunk = item_features_df.iloc[start : start + batch_size]
            pipe_data = {}
            for _, row in chunk.iterrows():
                feat: Dict[str, Any] = {col: row[col] for col in item_scalar_cols}
                if "title" in item_features_df.columns:
                    feat["title"] = str(row["title"])
                if item_genre_vec_cols:
                    feat["genre_vector"] = [float(row[c]) for c in item_genre_vec_cols]
                key = f"{ITEM_FEATURE_PREFIX}{int(row['item_id'])}"
                pipe_data[key] = _serialize(feat)

            if self._use_redis:
                pipe = self._client.pipeline()
                for key, val in pipe_data.items():
                    pipe.setex(key, self.ttl, val)
                pipe.execute()
            else:
                self._memory_store.update(pipe_data)

        logger.info("Loaded %d item feature records", len(item_features_df))

    # ------------------------------------------------------------------ #
    # Cache Operations                                                     #
    # ------------------------------------------------------------------ #

    def cache_recommendations(self, user_id: int, recommendations: List[Dict], ttl: int = 300) -> None:
        """Cache final recommendation results for a user."""
        key = f"recs:{user_id}"
        data = _serialize({"recs": recommendations})
        if self._use_redis:
            self._client.setex(key, ttl, data)
        else:
            self._memory_store[key] = data

    def get_cached_recommendations(self, user_id: int) -> Optional[List[Dict]]:
        """Retrieve cached recommendation results if they exist."""
        key = f"recs:{user_id}"
        raw = self._get(key)
        if raw is None:
            return None
        result = _deserialize(raw)
        return result.get("recs")

    def flush(self) -> None:
        """Clear all stored features (useful for testing)."""
        if self._use_redis:
            self._client.flushdb()
        else:
            self._memory_store.clear()

    def stats(self) -> Dict[str, Any]:
        """Return basic store statistics."""
        if self._use_redis:
            info = self._client.info("keyspace")
            db_info = info.get("db0", {})
            return {
                "backend": "redis",
                "url": self.redis_url,
                "keys": db_info.get("keys", 0),
            }
        return {
            "backend": "in-memory",
            "keys": len(self._memory_store),
        }
