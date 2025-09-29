"""Redis cache utilities for the recommendation system."""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
import redis
import structlog
from .config import config

logger = structlog.get_logger(__name__)

class RedisCache:
    """Redis cache wrapper for recommendation system."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=False  # Keep binary for pickle
        )
        self.ttl = config.REDIS_TTL
        
    def _get_key(self, prefix: str, domain: str, identifier: str) -> str:
        """Generate Redis key."""
        return f"recsys:{prefix}:{domain}:{identifier}"
    
    def set_user_candidates(self, domain: str, user_id: str, candidates: List[Dict[str, Any]]) -> bool:
        """Cache user candidates."""
        try:
            key = self._get_key("candidates", domain, user_id)
            data = pickle.dumps(candidates)
            return self.redis_client.setex(key, self.ttl, data)
        except Exception as e:
            logger.error("Failed to cache user candidates", user_id=user_id, domain=domain, error=str(e))
            return False
    
    def get_user_candidates(self, domain: str, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached user candidates."""
        try:
            key = self._get_key("candidates", domain, user_id)
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to get cached user candidates", user_id=user_id, domain=domain, error=str(e))
            return None
    
    def set_item_features(self, domain: str, item_id: str, features: Dict[str, Any]) -> bool:
        """Cache item features."""
        try:
            key = self._get_key("features", domain, item_id)
            data = json.dumps(features, default=str)
            return self.redis_client.setex(key, self.ttl, data)
        except Exception as e:
            logger.error("Failed to cache item features", item_id=item_id, domain=domain, error=str(e))
            return False
    
    def get_item_features(self, domain: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get cached item features."""
        try:
            key = self._get_key("features", domain, item_id)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error("Failed to get cached item features", item_id=item_id, domain=domain, error=str(e))
            return None
    
    def set_user_features(self, domain: str, user_id: str, features: Dict[str, Any]) -> bool:
        """Cache user features."""
        try:
            key = self._get_key("user_features", domain, user_id)
            data = json.dumps(features, default=str)
            return self.redis_client.setex(key, self.ttl, data)
        except Exception as e:
            logger.error("Failed to cache user features", user_id=user_id, domain=domain, error=str(e))
            return False
    
    def get_user_features(self, domain: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user features."""
        try:
            key = self._get_key("user_features", domain, user_id)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error("Failed to get cached user features", user_id=user_id, domain=domain, error=str(e))
            return None
    
    def set_popular_items(self, domain: str, items: List[Dict[str, Any]]) -> bool:
        """Cache popular items."""
        try:
            key = self._get_key("popular", domain, "items")
            data = pickle.dumps(items)
            return self.redis_client.setex(key, self.ttl, data)
        except Exception as e:
            logger.error("Failed to cache popular items", domain=domain, error=str(e))
            return False
    
    def get_popular_items(self, domain: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached popular items."""
        try:
            key = self._get_key("popular", domain, "items")
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to get cached popular items", domain=domain, error=str(e))
            return None
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """Increment a counter."""
        try:
            key = f"recsys:counter:{name}"
            return self.redis_client.incr(key, value)
        except Exception as e:
            logger.error("Failed to increment counter", name=name, error=str(e))
            return 0
    
    def set_latency(self, operation: str, latency_ms: float) -> bool:
        """Store latency measurement."""
        try:
            key = f"recsys:latency:{operation}"
            self.redis_client.lpush(key, latency_ms)
            # Keep only last 1000 measurements
            self.redis_client.ltrim(key, 0, 999)
            return True
        except Exception as e:
            logger.error("Failed to store latency", operation=operation, error=str(e))
            return False
    
    def get_latency_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get latency statistics."""
        try:
            key = f"recsys:latency:{operation}"
            data = self.redis_client.lrange(key, 0, -1)
            if not data:
                return None
            
            latencies = [float(x) for x in data]
            latencies.sort()
            
            n = len(latencies)
            return {
                "count": n,
                "p50": latencies[n // 2],
                "p95": latencies[int(n * 0.95)],
                "p99": latencies[int(n * 0.99)],
                "mean": sum(latencies) / n,
                "min": latencies[0],
                "max": latencies[-1]
            }
        except Exception as e:
            logger.error("Failed to get latency stats", operation=operation, error=str(e))
            return None
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False
    
    def clear_domain_cache(self, domain: str) -> bool:
        """Clear all cache entries for a domain."""
        try:
            pattern = f"recsys:*:{domain}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error("Failed to clear domain cache", domain=domain, error=str(e))
            return False

# Global cache instance
cache = RedisCache() 